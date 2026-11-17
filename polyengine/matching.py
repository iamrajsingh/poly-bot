"""Find the same proposition on other venues.

Two stages: cheap fuzzy retrieval (Kalshi snapshot + Manifold search) narrows thousands of
markets to a handful of candidates; then Claude confirms which candidates resolve on the *same*
event - or its exact inverse - reading both sets of resolution rules. Results are cached in
SQLite, so each market is only checked about once a day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

from rapidfuzz import fuzz, process

from .config import Settings
from .llm import LLM
from .models import Market
from .research.news import keywords
from .storage import Store
from .venues.kalshi import Kalshi
from .venues.manifold import Manifold

log = logging.getLogger(__name__)

MATCH_TOOL = {
    "name": "submit_matches",
    "description": "Report which candidate markets are equivalent to the target market.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "matches": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {"type": "string"},
                        "relation": {"type": "string", "enum": ["same", "inverse", "unrelated"]},
                        "confidence": {"type": "number"},
                        "caveat": {"type": "string"},
                    },
                    "required": ["candidate_id", "relation", "confidence", "caveat"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["matches"],
        "additionalProperties": False,
    },
}

SYSTEM = """You compare prediction-market contracts across exchanges. For each candidate decide:
- "same": candidate YES resolves YES in exactly the same real-world circumstances as the target YES.
- "inverse": candidate YES happens exactly when the target resolves NO.
- "unrelated": anything else - different threshold, date window, entity, resolution source, \
or merely correlated/related events.
Be strict - these are "unrelated":
- a different deadline, threshold, or date window ("by date X" vs "on date X");
- a candidate vs a party ("Will Smith win?" vs "Will Republicans win?");
- a conditional market ("If X is the nominee, will X win?");
- the opposing side of a contest that can have a third outcome. In an election, "Party A wins" is \
NOT the inverse of "Party B wins", because an independent or third party can win. "inverse" needs \
outcomes that are exhaustive and exclusive, e.g. the two teams of a match that cannot draw.
Confidence is 0-1. Note any small wording differences in `caveat`. \
Call submit_matches once with a verdict for every candidate."""


@dataclass
class Match:
    venue: str
    ref: str
    title: str
    relation: str  # same | inverse
    confidence: float
    caveat: str


class Matcher:
    def __init__(self, settings: Settings, llm: LLM, store: Store, kalshi: Kalshi, manifold: Manifold):
        self.s = settings
        self.llm = llm
        self.store = store
        self.kalshi = kalshi
        self.manifold = manifold

    async def matches(self, market: Market, max_age_hours: float = 24) -> list[Match]:
        cached = self.store.get_matches(market.id, max_age_hours)
        if cached is not None:
            return [Match(**m) for m in cached]

        cands = await self._candidates(market)
        if not cands:
            self.store.save_matches(market.id, [])
            return []

        out = await self.judge(market, cands)
        found = self.accept(out, cands)
        if out is not None:  # don't cache a failed call - retry next cycle
            self.store.save_matches(market.id, [f.__dict__ for f in found])
        log.info("market %s: %d cross-venue matches (%d candidates)", market.id, len(found), len(cands))
        return found

    @staticmethod
    def prompt(market: Market, cands: dict[str, tuple[str, str, str]]) -> str:
        listing = "\n\n".join(
            f"<candidate id=\"{cid}\" venue=\"{venue}\">\n{text}\n</candidate>"
            for cid, (venue, _ref, text) in cands.items()
        )
        return (
            f"<target venue=\"polymarket\">\nQuestion: {market.question}\n"
            f"Target YES = outcome \"{market.outcomes[0]}\"; NO = \"{market.outcomes[1]}\"\n"
            f"Closes: {market.end_date}\nRules: {market.description[:2500]}\n</target>\n\n{listing}"
        )

    async def judge(self, market: Market, cands: dict[str, tuple[str, str, str]], model: str | None = None,
                    effort: str | None = None) -> dict | None:
        """One model call: a verdict for every candidate (raw tool input), or None on failure."""
        return await self.llm.run_until_tool(
            purpose=f"match:{market.id}", system=SYSTEM, user=self.prompt(market, cands), tools=[MATCH_TOOL],
            final_tool="submit_matches", effort=effort or self.s.match_effort, max_turns=3,
            model=model or self.s.match_model or self.s.anthropic_model,
        )

    @staticmethod
    def accept(out: dict | None, cands: dict[str, tuple[str, str, str]], min_conf: float = 0.7) -> list[Match]:
        found: list[Match] = []
        for m in (out or {}).get("matches", []):
            cid = m.get("candidate_id")
            if cid not in cands or m.get("relation") not in ("same", "inverse"):
                continue
            conf = float(m.get("confidence", 0))
            if conf < min_conf:
                continue
            venue, ref, text = cands[cid]
            found.append(Match(venue, ref, text.splitlines()[0][:200], m["relation"], conf, m.get("caveat", "")))
        return found

    async def _candidates(self, market: Market) -> dict[str, tuple[str, str, str]]:
        """candidate_id -> (venue, ref, description text)."""
        out: dict[str, tuple[str, str, str]] = {}
        query = f"{market.event_title} {market.question}"

        kw = keywords(f"{market.question} {market.event_title}", 8)
        pool: list = []
        for series in self.kalshi.match_series(kw):
            pool.extend(await self.kalshi.series_markets(series))
        pool = self._near_date(market, pool)
        if pool:
            hits = process.extract(query, [k.search_text for k in pool], scorer=fuzz.token_set_ratio,
                                   limit=20, score_cutoff=55)
            per_event: dict[str, int] = {}
            for _, _score, idx in hits:
                k = pool[idx]
                if per_event.get(k.event_ticker, 0) >= 3:  # keep candidates diverse across events
                    continue
                per_event[k.event_ticker] = per_event.get(k.event_ticker, 0) + 1
                out[f"k{len(out)}"] = ("kalshi", k.ticker,
                                       f"{k.title} [{k.subtitle}] (closes {k.close_time})\nRules: {k.rules}")
                if len(out) >= 8:
                    break

        n = len(out)
        mf = await self.manifold.search(keywords(market.question, 6), limit=8)
        mf = [m for m in mf if fuzz.token_set_ratio(market.question, m.question) >= 55][:4]
        for i, m in enumerate(mf, start=n):
            out[f"m{i}"] = ("manifold", m.id, f"{m.question} (bettors: {m.unique_bettors})")
        return out

    @staticmethod
    def _near_date(market: Market, pool: list, window_days: int = 45) -> list:
        """Prefer Kalshi markets closing near the Polymarket end date (falls back to all)."""
        if not market.end_date:
            return pool
        lo = market.end_date - timedelta(days=window_days)
        hi = market.end_date + timedelta(days=window_days)
        near = [k for k in pool if k.close_dt and lo <= k.close_dt <= hi]
        return near or pool
