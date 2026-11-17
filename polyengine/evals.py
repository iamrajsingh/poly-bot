"""Side-by-side evaluation of the cross-venue matcher on different models.

Every model sees the identical production prompt and candidate list for the same live markets.
The first model is the reference. Scored on what the engine acts on - accepted matches
(same/inverse at confidence >= 0.7):
  * agree     both accept the candidate with the same relation
  * flipped   both accept it but disagree on same vs inverse   <- the dangerous error
  * extra     only this model accepts it (a false match feeds a wrong price into the ensemble)
  * missed    only the reference accepts it (lost coverage, not wrong trades)
Disagreements are written out in full for a human to adjudicate - the reference isn't ground truth.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from .config import Settings
from .engine import Engine

log = logging.getLogger(__name__)


async def eval_matcher(s: Settings, n_markets: int, specs: list[tuple[str, str]], out_dir: Path) -> dict:
    eng = Engine(s)
    try:
        await eng.kalshi.refresh()
        await eng.discover()
        markets = sorted(eng.universe.values(), key=lambda m: -m.volume_24h)

        # 1. Markets that actually have candidates (no LLM cost).
        sem = asyncio.Semaphore(4)
        cases = []

        async def collect(m):
            async with sem:
                if len(cases) >= n_markets:
                    return
                try:
                    c = await eng.matcher._candidates(m)
                except Exception:  # noqa: BLE001
                    return
                if c and len(cases) < n_markets:
                    cases.append((m, c))

        await asyncio.gather(*(collect(m) for m in markets))
        print(f"{len(cases)} markets with {sum(len(c) for _, c in cases)} candidates")

        # 2. Every model judges every case.
        results: dict[str, list] = {}
        costs: dict[str, float] = {}
        for model, effort in specs:
            key = f"{model}:{effort}"
            before = _spend(eng)
            t0 = time.monotonic()

            async def run(m, c, model=model, effort=effort):
                async with sem:
                    return await eng.matcher.judge(m, c, model=model, effort=effort)

            outs = await asyncio.gather(*(run(m, c) for m, c in cases))
            results[key] = outs
            costs[key] = _spend(eng) - before
            print(f"  {key}: {sum(o is not None for o in outs)}/{len(outs)} answered, "
                  f"${costs[key]:.3f}, {time.monotonic() - t0:.0f}s")

        # 3. Score against the reference.
        ref_key = f"{specs[0][0]}:{specs[0][1]}"
        accepted = {k: [_accepted(eng, o, c) for o, (_, c) in zip(v, cases)] for k, v in results.items()}
        report = {"markets": len(cases), "candidates": sum(len(c) for _, c in cases), "reference": ref_key,
                  "models": {}, "disagreements": []}
        for key in results:
            agree = flipped = extra = missed = 0
            for i, (m, c) in enumerate(cases):
                ref, mine = accepted[ref_key][i], accepted[key][i]
                for cid in set(ref) | set(mine):
                    r, x = ref.get(cid), mine.get(cid)
                    if r and x:
                        if r == x:
                            agree += 1
                        else:
                            flipped += 1
                    elif x:
                        extra += 1
                    else:
                        missed += 1
                    if key != ref_key and r != x:
                        report["disagreements"].append({
                            "model": key, "market": m.question, "rules": m.description[:600],
                            "candidate": c[cid][2][:600], "reference": r or "unrelated/low-conf",
                            "this_model": x or "unrelated/low-conf"})
            failed = sum(o is None for o in results[key])
            report["models"][key] = {"agree": agree, "flipped": flipped, "extra": extra, "missed": missed,
                                     "failed_calls": failed, "usd": round(costs[key], 4),
                                     "usd_per_market": round(costs[key] / max(1, len(cases)), 5)}
        report["raw"] = results
        report["cases"] = [{"market": m.question, "cands": c} for m, c in cases]
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"matcher-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(report, indent=2, default=str))
        report["path"] = str(path)
        return report
    finally:
        await eng.aclose()


def _accepted(eng: Engine, out: dict | None, cands: dict) -> dict[str, str]:
    """candidate id -> relation, for candidates this verdict would actually use."""
    found = eng.matcher.accept(out, cands)
    by_ref = {(f.venue, f.ref): f.relation for f in found}
    return {cid: by_ref[(v, ref)] for cid, (v, ref, _) in cands.items() if (v, ref) in by_ref}


def _spend(eng: Engine) -> float:
    return eng.store.llm_usage_today()[1]
