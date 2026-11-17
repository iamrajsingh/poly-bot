"""Turn multi-source probabilities + order books into a tradable signal.

Fair value = weighted mean of all sources in log-odds space. The Polymarket mid is itself a
source with a large weight, so the fair value only moves far from the market when several
independent sources (other exchanges, the blind LLM forecast) pull the same way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import Settings
from .models import Market, OrderBook, Signal, VenueQuote, logit, sigmoid
from .orderbook import BookStats, stats, taker_fee_per_share

AGREE_EPS = 0.015  # a source must sit at least this far beyond the market price to count as "agreeing"


def ensemble(quotes: list[VenueQuote]) -> tuple[float, float]:
    """Return (fair P(YES), weighted std-dev of the sources)."""
    tw = sum(q.weight for q in quotes)
    if tw <= 0:
        raise ValueError("no weighted quotes")
    x = sum(q.weight * logit(q.prob) for q in quotes) / tw
    fair = sigmoid(x)
    var = sum(q.weight * (q.prob - fair) ** 2 for q in quotes) / tw
    return fair, math.sqrt(var)


def merge_by_venue(quotes: list[VenueQuote]) -> list[VenueQuote]:
    """One quote per venue. Several matched contracts on one venue (e.g. a race's "Democrat wins"
    and its inverse "Republican wins") are one source of information, not several - counting them
    separately would let a single venue satisfy the "independent sources must agree" rule."""
    by: dict[str, list[VenueQuote]] = {}
    for q in quotes:
        by.setdefault(q.venue, []).append(q)
    out = []
    for venue, qs in by.items():
        if len(qs) == 1:
            out.append(qs[0])
            continue
        tw = sum(q.weight for q in qs) or 1.0
        prob = sigmoid(sum(q.weight * logit(q.prob) for q in qs) / tw)
        out.append(VenueQuote(venue, prob, max(q.weight for q in qs), ref=", ".join(q.ref for q in qs),
                              title=qs[0].title, liquidity=sum(q.liquidity for q in qs),
                              note=f"{len(qs)} contracts merged: " + " / ".join(f"{q.prob:.3f}" for q in qs)))
    return out


def _round_down(p: float, tick: float) -> float:
    return math.floor(p / tick + 1e-9) * tick


@dataclass
class Candidate:
    outcome: str
    token_id: str
    fair: float
    book: OrderBook
    st: BookStats


def evaluate(market: Market, yes_book: OrderBook, no_book: OrderBook, quotes: list[VenueQuote],
             llm_confidence: float | None, s: Settings) -> tuple[Signal | None, str]:
    """Best entry (YES or NO) for this market, or (None, reason)."""
    ys = stats(yes_book)
    if ys.mid is None or ys.spread is None:
        return None, "empty book"
    if ys.spread > s.max_spread:
        return None, f"spread {ys.spread:.3f} > {s.max_spread}"

    fair_yes, disp = ensemble(quotes)
    pm_yes = ys.mid
    others = [q for q in quotes if q.venue != "polymarket"]
    if not others:
        return None, "no independent sources"

    fee_rate, fee_exp = s.fee_for(market)
    best: Signal | None = None
    why = "no edge"
    for c in (Candidate("YES", market.yes_token, fair_yes, yes_book, ys),
              Candidate("NO", market.no_token, 1 - fair_yes, no_book, stats(no_book))):
        bb, ba = c.st.best_bid, c.st.best_ask
        if bb is None or ba is None:
            continue
        tick = c.book.tick_size or market.tick_size
        pm_price = pm_yes if c.outcome == "YES" else 1 - pm_yes

        # Independent sources that agree this outcome is underpriced.
        agree = sum(1 for q in others
                    if (q.prob if c.outcome == "YES" else 1 - q.prob) > pm_price + AGREE_EPS)
        disagree = sum(1 for q in others
                       if (q.prob if c.outcome == "YES" else 1 - q.prob) < pm_price - AGREE_EPS)

        # Confidence: LLM self-assessed quality, cross-source agreement, low dispersion.
        agree_ratio = agree / len(others)
        conf = (0.4 * (llm_confidence if llm_confidence is not None else 0.5)
                + 0.35 * agree_ratio
                + 0.25 * max(0.0, 1 - disp / 0.15))
        if disagree:
            conf *= 0.7

        # Taker: cross the spread now, pay the fee.
        taker_edge = c.fair - ba - taker_fee_per_share(ba, fee_rate, fee_exp)
        # Maker: improve the bid by a tick (no fee), but never pay more than fair - min_edge.
        maker_px = min(bb + tick if bb + tick < ba else bb, c.fair - s.min_edge)
        maker_px = round(_round_down(maker_px, tick), 4)
        maker_edge = c.fair - maker_px if maker_px >= tick else -1

        if taker_edge >= s.min_edge + s.taker_edge_premium:
            price, edge, taker = ba, taker_edge, True
        elif maker_edge >= s.min_edge and maker_px >= bb:
            price, edge, taker = maker_px, maker_edge, False
        else:
            continue

        if not (s.min_price <= price <= s.max_price):
            why = f"{c.outcome} price {price:.3f} outside [{s.min_price}, {s.max_price}]"
            continue
        if agree < s.min_agreeing_sources:
            why = f"{c.outcome}: only {agree} source(s) agree (need {s.min_agreeing_sources})"
            continue
        if conf < s.min_confidence:
            why = f"{c.outcome}: confidence {conf:.2f} < {s.min_confidence}"
            continue

        sig = Signal(market_id=market.id, token_id=c.token_id, outcome=c.outcome, fair=c.fair,
                     price=price, edge=edge, confidence=conf, agreeing_sources=agree,
                     quotes=quotes, use_taker=taker,
                     reason=f"fair {c.fair:.3f} vs {'ask' if taker else 'bid'} {price:.3f}, "
                            f"{agree}/{len(others)} sources agree, disp {disp:.3f}")
        if best is None or sig.edge * sig.confidence > best.edge * best.confidence:
            best = sig
    return best, (best.reason if best else why)


def timing_ok(sig: Signal, book: OrderBook, history: list[tuple[int, float]], s: Settings) -> tuple[bool, str]:
    """Entry timing filters. Returns (ok, reason). Deferring is cheap - we re-check next cycle."""
    st = stats(book)
    # Book pressure: heavy resting asks vs bids near the touch means sellers are leaning on it.
    if st.imbalance < -s.adverse_imbalance:
        return False, f"adverse book imbalance {st.imbalance:+.2f}"
    # Taking liquidity while microprice sits below the mid -> price likely to tick down; wait.
    if sig.use_taker and st.microprice is not None and st.mid is not None and st.microprice < st.mid - 0.25 * (st.spread or 0):
        return False, "microprice below mid; wait for a better taker entry"
    # Momentum: don't buy into a sharp move against us (often informed flow / news we haven't seen).
    if history:
        now_t = history[-1][0]
        past = [p for t, p in history if t <= now_t - s.momentum_lookback_hours * 3600]
        if past:
            p0, p1 = past[-1], history[-1][1]
            move = p1 - p0  # history is for the token we're buying
            if move < -s.max_adverse_momentum:
                return False, f"falling {move:+.3f} over {s.momentum_lookback_hours}h"
    return True, "ok"
