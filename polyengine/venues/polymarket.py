"""Polymarket public data: Gamma (market discovery) + CLOB (order books, price history)."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from ..http import Http
from ..models import Level, Market, OrderBook

log = logging.getLogger(__name__)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def _jsonlist(v) -> list:
    if isinstance(v, list):
        return v
    try:
        return json.loads(v or "[]")
    except (TypeError, json.JSONDecodeError):
        return []


def parse_market(m: dict) -> Market | None:
    """Gamma market dict -> Market. Returns None for anything that isn't a tradable binary market."""
    outcomes = _jsonlist(m.get("outcomes"))
    tokens = _jsonlist(m.get("clobTokenIds"))
    prices = _jsonlist(m.get("outcomePrices"))
    if len(outcomes) != 2 or len(tokens) != 2 or not m.get("enableOrderBook"):
        return None
    # Polymarket binaries are ["Yes","No"]; some markets use team names - keep index 0 as "YES".
    ev = (m.get("events") or [{}])[0]
    try:
        yes_price = float(prices[0]) if prices else float(m.get("lastTradePrice") or 0.5)
    except (TypeError, ValueError):
        yes_price = 0.5
    return Market(
        id=str(m["id"]),
        condition_id=m.get("conditionId", ""),
        question=m.get("question", ""),
        description=m.get("description", "") or "",
        slug=m.get("slug", ""),
        event_slug=ev.get("slug", ""),
        event_title=ev.get("title", ""),
        end_date=_parse_dt(m.get("endDate")),
        yes_token=str(tokens[0]),
        no_token=str(tokens[1]),
        yes_price=yes_price,
        liquidity=float(m.get("liquidityNum") or m.get("liquidity") or 0),
        volume_24h=float(m.get("volume24hr") or 0),
        tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
        min_order_size=float(m.get("orderMinSize") or 5),
        neg_risk=bool(m.get("negRisk")),
        category=(m.get("category") or ev.get("category") or ""),
        accepting_orders=bool(m.get("acceptingOrders", True)),
        outcomes=(str(outcomes[0]), str(outcomes[1])),
        **_fees(m),
    )


def _fees(m: dict) -> dict:
    """Taker fee schedule. Gamma exposes `feeSchedule` ({rate, exponent, ...}), `feeType`, `feesEnabled`."""
    sched = m.get("feeSchedule")
    if m.get("feesEnabled") is False or m.get("feeType") == "zero_fees":
        return {"fee_rate": 0.0, "fee_exponent": 1.0}
    if isinstance(sched, dict) and sched.get("rate") is not None:
        return {"fee_rate": float(sched["rate"]), "fee_exponent": float(sched.get("exponent") or 1.0)}
    return {"fee_rate": None, "fee_exponent": 1.0}  # unknown -> config default applies


class Polymarket:
    def __init__(self, http: Http, gamma_host: str, clob_host: str):
        self.http = http
        self.gamma = gamma_host.rstrip("/")
        self.clob = clob_host.rstrip("/")

    async def active_markets(self, limit: int = 300) -> list[Market]:
        out: list[Market] = []
        page = 100
        for offset in range(0, limit, page):
            rows = await self.http.get_json(f"{self.gamma}/markets", {
                "active": "true", "closed": "false", "archived": "false",
                "order": "volume24hr", "ascending": "false",
                "limit": min(page, limit - offset), "offset": offset,
            })
            if not rows:
                break
            for r in rows:
                mk = parse_market(r)
                if mk:
                    out.append(mk)
        return out

    async def market(self, id_or_slug: str) -> Market | None:
        params = {"id": id_or_slug} if id_or_slug.isdigit() else {"slug": id_or_slug}
        rows = await self.http.get_json(f"{self.gamma}/markets", params)
        if not rows and not id_or_slug.isdigit():
            # maybe an event slug: return its most liquid binary market
            evs = await self.http.get_json(f"{self.gamma}/events", {"slug": id_or_slug})
            rows = (evs[0].get("markets") if evs else []) or []
            for r in rows:
                r.setdefault("events", [{"slug": evs[0].get("slug"), "title": evs[0].get("title")}])
            rows.sort(key=lambda r: float(r.get("liquidityNum") or 0), reverse=True)
        for r in rows:
            mk = parse_market(r)
            if mk:
                return mk
        return None

    async def resolution(self, market_id: str) -> float | None:
        """YES payout (1.0 / 0.0 / 0.5) once the market has resolved, else None.

        Note: Gamma's `?id=` filter hides closed markets unless `closed=true` is passed, and the
        `/markets/{id}` path can lag behind the list endpoint - so query the list with closed=true.
        """
        rows = await self.http.get_json(f"{self.gamma}/markets", {"id": market_id, "closed": "true"})
        if not rows:
            return None
        r = rows[0]
        if str(r.get("umaResolutionStatus", "")).lower() not in ("resolved", ""):
            return None
        prices = [float(x) for x in _jsonlist(r.get("outcomePrices"))]
        if len(prices) == 2 and prices[0] in (0.0, 0.5, 1.0) and abs(sum(prices) - 1) < 1e-6:
            return prices[0]
        return None

    async def book(self, token_id: str) -> OrderBook:
        d = await self.http.get_json(f"{self.clob}/book", {"token_id": token_id})
        # CLOB returns bids ascending and asks descending (best last) - normalise to best-first.
        bids = sorted((Level(float(x["price"]), float(x["size"])) for x in d.get("bids", [])),
                      key=lambda l: l.price, reverse=True)
        asks = sorted((Level(float(x["price"]), float(x["size"])) for x in d.get("asks", [])),
                      key=lambda l: l.price)
        lt = d.get("last_trade_price")
        return OrderBook(token_id=token_id, bids=bids, asks=asks,
                         tick_size=float(d.get("tick_size") or 0.01),
                         last_trade=float(lt) if lt not in (None, "") else None)

    async def price_history(self, token_id: str, interval: str = "1d", fidelity: int = 60) -> list[tuple[int, float]]:
        d = await self.http.get_json(f"{self.clob}/prices-history",
                                     {"market": token_id, "interval": interval, "fidelity": fidelity})
        return [(int(x["t"]), float(x["p"])) for x in d.get("history", [])]
