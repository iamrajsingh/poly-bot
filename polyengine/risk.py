"""Position sizing (fractional Kelly) and portfolio-level risk limits."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

from .config import Settings
from .models import Order, OrderBook, Position, Side, Signal
from .orderbook import stats, sweep


def kelly_fraction(fair: float, price: float) -> float:
    """Optimal bankroll fraction for buying a $1 binary share at `price` with win prob `fair`."""
    if price <= 0 or price >= 1 or fair <= price:
        return 0.0
    return (fair - price) / (1 - price)


@dataclass
class RiskState:
    bankroll: float  # cash + marked value of positions
    cash: float
    positions: list[Position]
    open_orders: list[Order]
    realized_today: float
    unrealized: float


class RiskManager:
    def __init__(self, settings: Settings):
        self.s = settings

    def kill_switch(self) -> bool:
        return os.path.exists(self.s.kill_switch_file)

    def halted(self, st: RiskState) -> str | None:
        if self.kill_switch():
            return f"kill switch file '{self.s.kill_switch_file}' present"
        day_pnl = st.realized_today + min(0.0, st.unrealized)
        if day_pnl <= -self.s.max_daily_loss_pct * st.bankroll:
            return f"daily loss limit hit ({day_pnl:.2f})"
        return None

    def size(self, sig: Signal, book: OrderBook, st: RiskState, event_slug: str,
             min_order_size: float) -> tuple[float, str]:
        """Return (shares, reason). 0 shares means don't trade."""
        if (h := self.halted(st)):
            return 0, h

        if sum(1 for p in st.positions if p.strategy != "arb") >= self.s.max_open_positions:
            return 0, "max open positions"
        if any(p.market_id == sig.market_id for p in st.positions):
            return 0, "already positioned in this market"
        if any(o.market_id == sig.market_id and o.side == Side.BUY for o in st.open_orders):
            return 0, "entry order already working"

        exposure = sum(p.cost for p in st.positions) + sum(
            o.remaining * o.price for o in st.open_orders if o.side == Side.BUY)
        room_total = self.s.max_total_exposure_pct * st.bankroll - exposure
        event_exp = sum(p.cost for p in st.positions if event_slug and p.event_slug == event_slug)
        room_event = self.s.max_event_exposure_pct * st.bankroll - event_exp

        k = kelly_fraction(sig.fair, sig.price)
        usd = st.bankroll * self.s.kelly_fraction * k * sig.confidence
        usd = min(usd, self.s.max_position_usd, self.s.max_position_pct_bankroll * st.bankroll,
                  room_total, room_event, st.cash)

        # Liquidity: never be more than a fraction of what's visibly there.
        bs = stats(book)
        if sig.use_taker:
            fill = sweep(book, Side.BUY, usd=usd, limit=sig.price)
            depth_cap = self.s.max_book_participation * sum(l.price * l.size for l in book.asks
                                                            if l.price <= sig.price)
        else:
            depth_cap = self.s.max_book_participation * (bs.bid_depth + bs.ask_depth)
            fill = None
        usd = min(usd, depth_cap)
        if fill is not None and fill.shares > 0:
            usd = min(usd, fill.cost)

        if usd < self.s.min_order_usd:
            return 0, f"size ${usd:.2f} below minimum (kelly={k:.3f}, conf={sig.confidence:.2f})"
        shares = math.floor(usd / sig.price * 100) / 100
        if shares < min_order_size:
            return 0, f"{shares} shares below market minimum {min_order_size}"
        return shares, f"kelly={k:.3f} -> ${usd:.2f}"
