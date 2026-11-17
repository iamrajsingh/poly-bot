"""Basket arbitrage on Polymarket neg-risk events (no LLM, no forecasting).

In a neg-risk event the outcome markets are mutually exclusive: at most one resolves YES.

* NO basket: buy 1 NO share in *every* open market of the event. At most one leg loses, so the
  basket pays at least N-1 (N if none resolves YES). If the total cost after fees is below N-1,
  the profit is locked in whatever happens. This holds even for "augmented" events, where
  outcomes can be added later - new markets don't affect the legs we hold.
* YES basket: buy 1 YES in every market; pays exactly 1 *only if the listed outcomes are
  exhaustive*. Augmented events (e.g. "2028 nominee", where an unlisted person can win) are NOT
  exhaustive, so a YES sum below 1 there is a trap, not an arb. Off by default; never on augmented events.

Mispricings like this are rare on liquid events. They appear briefly after news, when one
leg reprices and the rest lag, so the scanner runs often and only fetches books for near-arbs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import Settings
from .models import Market, OrderBook, Side
from .orderbook import sweep, taker_fee_per_share
from .venues.polymarket import Polymarket, _jsonlist, _parse_dt, parse_market

log = logging.getLogger(__name__)


@dataclass
class Leg:
    market: Market
    token_id: str
    outcome: str  # "YES" | "NO"
    book: OrderBook
    limit: float = 0.0  # worst price needed for the planned size


@dataclass
class Basket:
    event_slug: str
    title: str
    kind: str  # "NO" | "YES"
    legs: list[Leg]
    sets: float  # shares bought in each leg
    cost: float  # USD incl. taker fees
    payout_min: float  # guaranteed USD at resolution
    days: float

    @property
    def profit(self) -> float:
        return self.payout_min - self.cost

    @property
    def ret(self) -> float:
        return self.profit / self.cost if self.cost else 0.0

    @property
    def apr(self) -> float:
        return self.ret * 365 / max(self.days, 1.0)


def _gamma_num(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class BasketScanner:
    def __init__(self, s: Settings, pm: Polymarket):
        self.s = s
        self.pm = pm
        self.last_scanned = 0
        self.near_misses: list[dict] = []  # closest-to-arb NO baskets from the last scan (dashboard)

    async def neg_risk_events(self) -> list[dict]:
        out: list[dict] = []
        for offset in range(0, self.s.arb_scan_events, 100):
            rows = await self.pm.http.get_json(f"{self.pm.gamma}/events", {
                "active": "true", "closed": "false", "order": "volume24hr", "ascending": "false",
                "limit": min(100, self.s.arb_scan_events - offset), "offset": offset})
            if not rows:
                break
            out.extend(e for e in rows if e.get("negRisk"))
        return out

    def _open_markets(self, ev: dict) -> list[dict] | None:
        """All open markets of the event, or None if the basket can't be completed."""
        ms = ev.get("markets") or []
        if any(m.get("closed") and _gamma_num((_jsonlist(m.get("outcomePrices")) or [0])[0]) == 1.0 for m in ms):
            return None  # already resolved YES somewhere: event is over
        open_ = [m for m in ms if not m.get("closed")]
        if len(open_) < 2 or len(open_) > self.s.arb_max_legs:
            return None
        if any(not m.get("enableOrderBook") or not m.get("acceptingOrders", True) for m in open_):
            return None  # a leg we can't buy breaks the hedge
        return open_

    def _prefilter(self, kind: str, ms: list[dict]) -> bool:
        """Cheap check on Gamma's cached top-of-book before fetching any order books."""
        n = len(ms)
        if kind == "NO":
            cost = sum(1 - (_gamma_num(m.get("bestBid")) or 0.0) for m in ms)
            return cost < (n - 1) * (1 - self.s.arb_min_return) + 0.01
        asks = [_gamma_num(m.get("bestAsk")) for m in ms]
        return None not in asks and sum(asks) < (1 - self.s.arb_min_return) + 0.01

    async def scan(self) -> list[Basket]:
        found: list[Basket] = []
        events = await self.neg_risk_events()
        self.last_scanned = len(events)
        gaps = []
        for ev in events:
            ms = self._open_markets(ev)
            if not ms:
                continue
            no_cost = sum(1 - (_gamma_num(m.get("bestBid")) or 0.0) for m in ms)
            gaps.append({"event": ev.get("slug", ""), "title": ev.get("title", ""), "legs": len(ms),
                         "no_cost": no_cost, "payout": len(ms) - 1, "gap": (len(ms) - 1) - no_cost})
            kinds = ["NO"]
            if self.s.arb_yes_baskets and not ev.get("negRiskAugmented"):
                kinds.append("YES")
            for kind in kinds:
                if not self._prefilter(kind, ms):
                    continue
                try:
                    b = await self.evaluate(ev, ms, kind)
                except Exception as e:  # noqa: BLE001
                    log.debug("basket %s failed: %s", ev.get("slug"), e)
                    continue
                if b:
                    found.append(b)
        self.near_misses = sorted(gaps, key=lambda g: -g["gap"])[:10]
        return found

    async def evaluate(self, ev: dict, raw_markets: list[dict], kind: str) -> Basket | None:
        for m in raw_markets:
            m.setdefault("events", [{"slug": ev.get("slug"), "title": ev.get("title")}])
        markets = [parse_market(m) for m in raw_markets]
        if any(m is None for m in markets):
            return None
        tokens = [(m.no_token if kind == "NO" else m.yes_token) for m in markets]
        books = await asyncio.gather(*(self.pm.book(t) for t in tokens))
        legs = [Leg(m, t, kind, b) for m, t, b in zip(markets, tokens, books)]
        n = len(legs)
        payout_per_set = (n - 1) if kind == "NO" else 1.0
        end = _parse_dt(ev.get("endDate"))
        days = (end - datetime.now(timezone.utc)).total_seconds() / 86400 if end else 365.0

        def cost_for(sets: float) -> float | None:
            total = 0.0
            for leg in legs:
                f = sweep(leg.book, Side.BUY, shares=sets)
                if not f.complete:
                    return None
                rate, exp = self.s.fee_for(leg.market)
                # fee is per fill level; using VWAP is a close, slightly conservative approximation
                total += f.cost + sets * taker_fee_per_share(f.avg_price, rate, exp)
            return total

        def ok(sets: float) -> bool:
            c = cost_for(sets)
            if c is None:
                return False
            profit = payout_per_set * sets - c
            return profit / c >= self.s.arb_min_return and (profit / c) * 365 / max(days, 1) >= self.s.arb_min_apr

        lo = max(m.min_order_size for m in markets)
        if not ok(lo):
            return None
        # Per-set profit only falls as size grows (we walk deeper into each book): binary search.
        hi = lo
        while ok(hi * 2) and cost_for(hi * 2) <= self.s.arb_max_usd:
            hi *= 2
        top = hi * 2
        while top - hi > 1:
            mid = (hi + top) / 2
            c = cost_for(mid)
            if c is not None and c <= self.s.arb_max_usd and ok(mid):
                hi = mid
            else:
                top = mid
        sets = float(int(hi))
        cost = cost_for(sets)
        if cost is None or cost > self.s.arb_max_usd:
            return None
        for leg in legs:
            leg.limit = sweep(leg.book, Side.BUY, shares=sets).worst_price
        return Basket(ev.get("slug", ""), ev.get("title", ""), kind, legs, sets, cost,
                      payout_per_set * sets, days)

