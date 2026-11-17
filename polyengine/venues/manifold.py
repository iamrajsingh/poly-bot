"""Manifold Markets (play money - useful as a crowd signal, weighted low)."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..http import Http

log = logging.getLogger(__name__)

BASE = "https://api.manifold.markets/v0"


@dataclass
class ManifoldMarket:
    id: str
    question: str
    prob: float
    url: str
    liquidity: float
    volume: float
    close_time_ms: int
    unique_bettors: int

    @property
    def search_text(self) -> str:
        return self.question


class Manifold:
    def __init__(self, http: Http):
        self.http = http

    async def search(self, term: str, limit: int = 10) -> list[ManifoldMarket]:
        try:
            rows = await self.http.get_json(f"{BASE}/search-markets", {
                "term": term, "limit": limit, "filter": "open", "contractType": "BINARY", "sort": "liquidity",
            })
        except Exception as e:  # noqa: BLE001
            log.debug("manifold search failed: %s", e)
            return []
        out = []
        for r in rows or []:
            if r.get("outcomeType") != "BINARY" or r.get("isResolved") or r.get("probability") is None:
                continue
            out.append(ManifoldMarket(
                id=r["id"], question=r.get("question", ""), prob=float(r["probability"]),
                url=r.get("url", ""), liquidity=float(r.get("totalLiquidity") or 0),
                volume=float(r.get("volume") or 0), close_time_ms=int(r.get("closeTime") or 0),
                unique_bettors=int(r.get("uniqueBettorCount") or 0),
            ))
        return out

    async def quote(self, market_id: str) -> tuple[float, int] | None:
        """(probability, unique bettors) for an open binary market."""
        try:
            d = await self.http.get_json(f"{BASE}/market/{market_id}")
        except Exception:  # noqa: BLE001
            return None
        if d.get("probability") is None or d.get("isResolved"):
            return None
        return float(d["probability"]), int(d.get("uniqueBettorCount") or 0)
