"""Kalshi public market data (CFTC-regulated, real-money -> high-signal cross-check).

Kalshi lists far too many open markets (100k+, mostly sports props) to snapshot. Instead we keep
the ~15k *series* list in memory, fuzzy-match a question to series titles, and fetch only the
open markets of the best-matching series.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property

from rapidfuzz import fuzz, process, utils

from ..http import Http

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"


@dataclass
class KalshiMarket:
    ticker: str
    event_ticker: str
    title: str
    subtitle: str
    rules: str
    yes_bid: float
    yes_ask: float
    last: float
    volume: float
    open_interest: float
    close_time: str

    @property
    def mid(self) -> float | None:
        if self.yes_bid > 0 and self.yes_ask > 0 and self.yes_ask >= self.yes_bid:
            return (self.yes_bid + self.yes_ask) / 2
        return self.last or None

    @property
    def spread(self) -> float:
        return self.yes_ask - self.yes_bid if self.yes_ask and self.yes_bid else 1.0

    @cached_property
    def close_dt(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.close_time.replace("Z", "+00:00"))
        except ValueError:
            return None

    @property
    def search_text(self) -> str:
        return f"{self.title} {self.subtitle}"


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _parse(m: dict) -> KalshiMarket:
    return KalshiMarket(
        ticker=m["ticker"], event_ticker=m.get("event_ticker", ""), title=m.get("title", ""),
        subtitle=m.get("yes_sub_title") or "", rules=(m.get("rules_primary") or "")[:800],
        yes_bid=_f(m.get("yes_bid_dollars")), yes_ask=_f(m.get("yes_ask_dollars")),
        last=_f(m.get("last_price_dollars")), volume=_f(m.get("volume_fp")),
        open_interest=_f(m.get("open_interest_fp")), close_time=m.get("close_time", ""),
    )


class Kalshi:
    def __init__(self, http: Http, series_cache_seconds: int = 600):
        self.http = http
        self.series: list[dict] = []
        self._series_text: list[str] = []
        self._series_markets: dict[str, tuple[float, list[KalshiMarket]]] = {}
        self._ttl = series_cache_seconds
        self.refreshed_at = 0.0

    async def refresh(self) -> None:
        """Reload the series catalogue (cheap: one request)."""
        d = await self.http.get_json(f"{BASE}/series", {})
        series = [s for s in d.get("series", []) if not s.get("ticker", "").startswith("KXMVE")]
        if series:
            self.series = series
            self._series_text = [f"{s.get('title', '')} {' '.join(s.get('tags') or [])}" for s in series]
            self.refreshed_at = time.time()
        log.info("kalshi: %d series loaded", len(self.series))

    def match_series(self, query: str, limit: int = 8, cutoff: float = 60) -> list[str]:
        if not self.series:
            return []
        hits = process.extract(query, self._series_text, scorer=fuzz.token_set_ratio, limit=limit,
                               score_cutoff=cutoff, processor=utils.default_process)
        return [self.series[i]["ticker"] for _, _, i in hits]

    async def series_markets(self, series_ticker: str) -> list[KalshiMarket]:
        hit = self._series_markets.get(series_ticker)
        if hit and time.time() - hit[0] < self._ttl:
            return hit[1]
        try:
            d = await self.http.get_json(f"{BASE}/markets", {
                "series_ticker": series_ticker, "status": "open", "limit": 200, "mve_filter": "exclude"})
        except Exception as e:  # noqa: BLE001 - venue outages must never break the engine
            log.debug("kalshi series %s failed: %s", series_ticker, e)
            return hit[1] if hit else []
        ms = [_parse(m) for m in d.get("markets", []) if m.get("market_type", "binary") == "binary"]
        self._series_markets[series_ticker] = (time.time(), ms)
        return ms

    async def quote(self, ticker: str) -> KalshiMarket | None:
        try:
            d = await self.http.get_json(f"{BASE}/markets/{ticker}")
        except Exception as e:  # noqa: BLE001
            log.debug("kalshi quote %s failed: %s", ticker, e)
            return None
        m = d.get("market") or {}
        if not m or m.get("status") not in ("active", "open"):
            return None
        return _parse(m)
