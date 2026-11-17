"""Free news sources: Google News RSS + GDELT. Gives the forecaster fresh, multi-outlet headlines."""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from urllib.parse import quote_plus

import feedparser
from rapidfuzz import fuzz

from ..http import Http

log = logging.getLogger(__name__)

_STOP = set("""will the a an of in on at to by for be is are was were and or with before after
during than this that from as it its into over under between does do did has have had any
who what when which whether market resolve yes no end win wins next there""".split())
_DATE = re.compile(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{0,2},?\s*\d{0,4}\b|\b\d{4}-\d{2}-\d{2}\b", re.I)


def keywords(question: str, max_words: int = 7) -> str:
    q = _DATE.sub(" ", question)
    words = re.findall(r"[\w$%.'-]+", q)  # \w is unicode-aware: keeps "München", "Flávio"
    kept = [w.strip(".'") for w in words if w.lower() not in _STOP and len(w) > 1]
    return " ".join(kept[:max_words])


@dataclass
class Headline:
    title: str
    source: str
    published: str
    url: str


async def google_news(http: Http, query: str, limit: int = 12) -> list[Headline]:
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}+when:14d&hl=en-US&gl=US&ceid=US:en"
    try:
        text = await http.get_text(url)
    except Exception as e:  # noqa: BLE001
        log.debug("google news failed: %s", e)
        return []
    feed = feedparser.parse(text)
    out = []
    for e in feed.entries[:limit]:
        src = getattr(getattr(e, "source", None), "title", "") or ""
        out.append(Headline(e.get("title", ""), src, e.get("published", ""), e.get("link", "")))
    return out


async def gdelt(http: Http, query: str, limit: int = 12) -> list[Headline]:
    try:
        d = await http.get_json("https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": f"{query} sourcelang:english", "mode": "artlist", "format": "json",
            "maxrecords": limit, "timespan": "7d", "sort": "datedesc",
        }, retries=1)
    except Exception as e:  # noqa: BLE001 - GDELT rate-limits aggressively and returns non-JSON on error
        log.debug("gdelt failed: %s", e)
        return []
    return [Headline(a.get("title", ""), a.get("domain", ""), a.get("seendate", ""), a.get("url", ""))
            for a in (d or {}).get("articles", [])]


async def headlines(http: Http, question: str, limit: int = 20) -> list[Headline]:
    q = keywords(question, 5)
    if not q:
        return []
    results = list(await asyncio.gather(google_news(http, q), gdelt(http, q)))
    if len(results[0]) < 3:  # search engines AND the terms - fall back to a broader query
        results.append(await google_news(http, keywords(question, 3)))
    merged: list[Headline] = []
    for h in [x for r in results for x in r]:
        if h.title and not any(fuzz.token_set_ratio(h.title, m.title) > 85 for m in merged):
            merged.append(h)
    return merged[:limit]
