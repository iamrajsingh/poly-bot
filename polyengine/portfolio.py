"""Position accounting (fills -> positions -> realized PnL) and exit rules."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .execution.base import Broker
from .models import Market, Order, OrderBook, Position, Side
from .orderbook import stats, taker_fee_per_share
from .storage import Store

log = logging.getLogger(__name__)


@dataclass
class ExitPlan:
    reason: str
    price: float
    taker: bool


class Portfolio:
    def __init__(self, settings: Settings, store: Store, broker: Broker):
        self.s = settings
        self.store = store
        self.broker = broker

    def positions(self) -> list[Position]:
        return self.store.positions()

    def position(self, token_id: str) -> Position | None:
        return next((p for p in self.positions() if p.token_id == token_id), None)

    def apply_fill(self, order: Order, shares: float, price: float, market: Market | None,
                   fair: float = 0.0) -> None:
        """Book `shares` filled at `price` for `order`."""
        if shares <= 0:
            return
        with self.store.transaction():
            fee = 0.0 if order.post_only else shares * taker_fee_per_share(price, order.fee_rate, order.fee_exponent)
            pos = self.position(order.token_id)
            if order.side == Side.BUY:
                if pos is None:
                    pos = Position(market_id=order.market_id, token_id=order.token_id, outcome=order.outcome,
                                   shares=0, avg_price=0,
                                   event_slug=market.event_slug if market else "",
                                   question=market.question if market else "", fair_at_entry=fair,
                                   strategy="arb" if order.purpose == "arb" else "directional")
                # Fees are folded into cost basis so realized PnL is net.
                new_cost = pos.cost + shares * price + fee
                pos.shares += shares
                pos.avg_price = new_cost / pos.shares
                cash = -(shares * price + fee)
                realized = 0.0
            else:
                if pos is None:
                    log.error("sell fill with no position for %s", order.token_id)
                    return
                shares = min(shares, pos.shares)
                realized = shares * (price - pos.avg_price) - fee
                pos.shares -= shares
                pos.realized_pnl += realized
                cash = shares * price - fee
            self.broker.adjust_cash(cash)
            self.store.save_position(pos)
            note = f"{order.side.value} {order.purpose}" + (f": {order.reason}" if order.reason else "")
            self.store.ledger("fill", order.market_id, order.token_id, shares if order.side == Side.BUY else -shares,
                              price, fee, cash, realized, note, question=pos.question, strategy=pos.strategy,
                              outcome=order.outcome)
        log.info("FILL %s %s %.2f @ %.3f fee=%.4f realized=%+.2f", order.side.value, order.outcome, shares,
                 price, fee, realized)

    def settle(self, pos: Position, payout: float) -> None:
        """Market resolved: each share pays `payout` (1.0 or 0.0; 0.5 on a 50-50 resolution)."""
        realized = pos.shares * (payout - pos.avg_price)
        cash = pos.shares * payout
        shares = pos.shares
        with self.store.transaction():
            self.broker.adjust_cash(cash)
            self.store.ledger("settle", pos.market_id, pos.token_id, -shares, payout, 0.0, cash, realized,
                              "resolution", question=pos.question, strategy=pos.strategy, outcome=pos.outcome)
            pos.shares = 0
            self.store.save_position(pos)
        log.info("SETTLE %s %s %.2f sh @ %.2f realized=%+.2f", pos.outcome, pos.question[:60], shares,
                 payout, realized)

    @staticmethod
    def mark(pos: Position, book: OrderBook | None) -> float:
        """Conservative mark: best bid (what we could actually sell at)."""
        if book is None or book.best_bid is None:
            return pos.avg_price
        return book.best_bid

    def exit_plan(self, pos: Position, book: OrderBook, fair: float | None,
                  market: Market | None) -> ExitPlan | None:
        st = stats(book)
        bid, ask = st.best_bid, st.best_ask
        if bid is None:
            return None
        tick = book.tick_size or 0.01
        pnl_pct = (bid - pos.avg_price) / pos.avg_price if pos.avg_price else 0.0
        # Profit-taking exits rest as makers (no fee): undercut the ask by a tick, or join it.
        if ask is None:
            maker_px = round(bid + tick, 4)
        else:
            maker_px = round(ask - tick if ask - tick > bid else ask, 4)

        # 1. Hard stop: losses beyond tolerance -> get out now.
        if pnl_pct <= -self.s.stop_loss_pct:
            return ExitPlan(f"hard stop {pnl_pct:+.0%}", max(bid - 2 * tick, tick), True)
        # 2. Thesis broken: refreshed fair value says we're now on the wrong side.
        if fair is not None and fair - bid < self.s.thesis_break_edge:
            return ExitPlan(f"thesis broken: fair {fair:.3f} < bid {bid:.3f}", bid, True)
        # 3. Edge captured: market has converged to our fair value.
        if fair is not None and bid >= fair - self.s.exit_edge_buffer and pnl_pct > 0:
            return ExitPlan(f"edge captured: bid {bid:.3f} ~ fair {fair:.3f}", maker_px, False)
        # 4. Take profit.
        if pnl_pct >= self.s.take_profit_pct:
            return ExitPlan(f"take profit {pnl_pct:+.0%}", maker_px, False)
        # 5. Close to resolution with little edge left: avoid resolution/oracle risk.
        hrs = market.hours_to_resolution if market else None
        if hrs is not None and hrs < self.s.exit_hours_before_resolution:
            if fair is None or fair - bid < self.s.min_edge:
                return ExitPlan(f"{hrs:.1f}h to resolution, edge {((fair or bid) - bid):+.3f}", bid, True)
        return None
