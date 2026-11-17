"""The learning loop: score every source against real outcomes and improve the next decision.

Each analysis stores a snapshot of every source's probability. Once a market resolves we learn,
per source:
  * Brier score / log loss (lower is better), averaged per market so long-watched markets don't dominate;
  * skill vs the Polymarket price on the *same* markets (> 1 = better than the market);
  * "when it disagreed with the market by > 5 points, how often was it closer to the truth?";
  * a reliability table (forecast bucket -> how often it actually happened).

That feeds back three ways, all shrunk towards "no change" while evidence is thin, so the engine
adapts from the first few resolutions without being swung by a lucky or unlucky streak:
  1. ensemble weights   - multiplier = skill ** (n / (n + prior)), bounded 0.25x..2x;
  2. recalibration      - p' = sigmoid(a * logit(p) + b), fitted with a ridge prior towards a=1, b=0
                          (fixes systematic over/under-confidence, e.g. an LLM saying 90% for 70% events);
  3. forecaster memory  - a short track-record note shown to Claude in its next forecasts.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Settings
from .models import logit, sigmoid
from .storage import Store

log = logging.getLogger(__name__)

BINS = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0001)]


@dataclass
class SourceScore:
    source: str
    markets: int
    brier: float
    log_loss: float
    shared_markets: int  # resolved markets where both this source and Polymarket had a price
    skill_vs_market: float | None  # Brier(market) / Brier(source) on shared markets; > 1 = better
    disagreements: int  # snapshots where |p - market| > 5 points
    closer_when_disagreeing: float | None  # share of those where this source was closer to the outcome
    multiplier: float
    recal_a: float = 1.0  # p' = sigmoid(a*logit(p) + b); a < 1 = was over-confident
    recal_b: float = 0.0
    reliability: list[dict] = field(default_factory=list)  # [{lo, hi, predicted, observed, n}]


class Calibrator:
    def __init__(self, s: Settings, store: Store):
        self.s = s
        self.store = store
        self.multipliers: dict[str, float] = {}
        self.recal: dict[str, tuple[float, float]] = {}
        self.scores: list[SourceScore] = []
        self.by_source: dict[str, SourceScore] = {}

    # ---- applied by the engine ----------------------------------------------------------
    def multiplier(self, source: str) -> float:
        return self.multipliers.get(source, 1.0) if self.s.calib_adaptive else 1.0

    def recalibrate(self, source: str, p: float) -> float:
        if not self.s.calib_adaptive or source not in self.recal:
            return p
        a, b = self.recal[source]
        return min(0.99, max(0.01, sigmoid(a * logit(p) + b)))

    def feedback(self, source: str = "llm") -> str | None:
        """Plain-language track record for the forecaster (None until there's enough evidence)."""
        sc = self.by_source.get(source)
        if not sc or sc.markets < self.s.calib_feedback_min_markets:
            return None
        lines = [f"{sc.markets} of your earlier forecasts on this engine have resolved. Brier score "
                 f"{sc.brier:.3f} (Polymarket's price on the same markets: "
                 f"{(sc.brier * sc.skill_vs_market):.3f})." if sc.skill_vs_market else
                 f"{sc.markets} of your earlier forecasts have resolved. Brier score {sc.brier:.3f}."]
        for b in sc.reliability:
            if b["n"] >= 3 and abs(b["observed"] - b["predicted"]) >= 0.1:
                lines.append(f"When you forecast {b['lo']:.0%}-{min(b['hi'], 1):.0%} (avg {b['predicted']:.0%}), "
                             f"it happened {b['observed']:.0%} of the time (n={b['n']:.0f}).")
        if sc.recal_a < 0.85:
            lines.append("Overall you have been over-confident: hold probabilities further from 0% and 100% "
                         "unless the evidence is decisive.")
        elif sc.recal_a > 1.15:
            lines.append("Overall you have been under-confident: commit further from 50% when evidence is clear.")
        return " ".join(lines)

    # ---- learning ------------------------------------------------------------------------
    def recompute(self, quiet: bool = False, persist: bool = True) -> list[SourceScore]:
        rows = self.store.calibration_rows()
        # source -> market -> list of (prob, outcome); snapshots share timestamps across sources
        by_src: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by_src[r["source"]][r["market_id"]].append((float(r["prob"]), float(r["yes_payout"])))

        pm = by_src.get("polymarket", {})
        pm_brier = {mid: _mean([(p - y) ** 2 for p, y in obs]) for mid, obs in pm.items()}
        pm_mean_prob = {mid: _mean([p for p, _ in obs]) for mid, obs in pm.items()}
        k = self.s.calib_prior_markets

        scores: list[SourceScore] = []
        for src, markets in sorted(by_src.items()):
            briers = {mid: _mean([(p - y) ** 2 for p, y in obs]) for mid, obs in markets.items()}
            lls = [_mean([_ll(p, y) for p, y in obs]) for obs in markets.values()]
            shared = [mid for mid in briers if mid in pm_brier]
            skill = None
            if src != "polymarket" and shared:
                b_src = _mean([briers[m] for m in shared])
                b_pm = _mean([pm_brier[m] for m in shared])
                skill = (b_pm + 1e-4) / (b_src + 1e-4)
            dis = closer = 0
            if src != "polymarket":
                for mid, obs in markets.items():
                    if mid not in pm_mean_prob:
                        continue
                    for p, y in obs:
                        q = pm_mean_prob[mid]
                        if abs(p - q) > 0.05:
                            dis += 1
                            closer += abs(p - y) < abs(q - y)
            mult = 1.0
            n = len(shared)
            if skill is not None and n >= self.s.calib_min_markets:
                # Brier ratios are extreme on tiny samples: cap before shrinking towards 1.
                capped = max(0.25, min(4.0, skill))
                mult = max(0.25, min(2.0, math.exp(math.log(capped) * n / (n + k))))
            a, b = (1.0, 0.0)
            if src != "polymarket" and len(markets) >= self.s.calib_min_markets:
                a, b = fit_recalibration(markets, prior=k)
            scores.append(SourceScore(src, len(markets), _mean(list(briers.values())), _mean(lls), n, skill, dis,
                                      (closer / dis) if dis else None, mult, a, b, reliability(markets)))
        self.scores = scores
        self.by_source = {sc.source: sc for sc in scores}
        self.multipliers = {sc.source: sc.multiplier for sc in scores if sc.source != "polymarket"}
        self.recal = {sc.source: (sc.recal_a, sc.recal_b) for sc in scores
                      if sc.source != "polymarket" and (sc.recal_a, sc.recal_b) != (1.0, 0.0)}
        if scores and persist:
            self.store.save_calib_history(datetime.now(timezone.utc).strftime("%Y-%m-%d"), scores)
            if not quiet:
                log.info("calibration: %s", ", ".join(
                    f"{sc.source} n={sc.markets} brier={sc.brier:.3f} x{sc.multiplier:.2f}" for sc in scores))
        return scores


def reliability(markets: dict[str, list[tuple[float, float]]]) -> list[dict]:
    """Forecast bucket -> observed frequency. Each market counts once (its snapshots share weight 1)."""
    acc = [[0.0, 0.0, 0.0] for _ in BINS]  # sum predicted*w, sum outcome*w, sum w
    for obs in markets.values():
        w = 1.0 / len(obs)
        for p, y in obs:
            for i, (lo, hi) in enumerate(BINS):
                if lo <= p < hi:
                    acc[i][0] += p * w
                    acc[i][1] += y * w
                    acc[i][2] += w
                    break
    return [{"lo": lo, "hi": min(hi, 1.0), "predicted": s_p / n, "observed": s_y / n, "n": n}
            for (lo, hi), (s_p, s_y, n) in zip(BINS, acc) if n > 0]


def fit_recalibration(markets: dict[str, list[tuple[float, float]]], prior: float) -> tuple[float, float]:
    """Logistic recalibration p' = sigmoid(a*logit(p) + b) by Newton's method, with a ridge prior of
    strength `prior` (in markets) pulling towards the identity a=1, b=0."""
    pts = []
    for obs in markets.values():
        w = 1.0 / len(obs)
        pts += [(logit(p), y, w) for p, y in obs]
    a, b = 1.0, 0.0
    lam = float(prior)
    for _ in range(50):
        ga, gb = lam * (a - 1.0), lam * b
        haa, hab, hbb = lam, 0.0, lam
        for x, y, w in pts:
            p = sigmoid(a * x + b)
            r, s = w * (p - y), w * p * (1 - p)
            ga += r * x
            gb += r
            haa += s * x * x
            hab += s * x
            hbb += s
        det = haa * hbb - hab * hab
        if det <= 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (haa * gb - hab * ga) / det
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-7:
            break
    return max(0.3, min(2.0, a)), max(-1.5, min(1.5, b))


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _ll(p: float, y: float) -> float:
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
