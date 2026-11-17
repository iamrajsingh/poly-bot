"""The learning loop: recalibration, shrunk weights, track-record feedback, learning picks."""

import random
from datetime import timedelta

import pytest

from polyengine.calibration import Calibrator, fit_recalibration, reliability
from polyengine.config import Settings
from polyengine.engine import Engine
from polyengine.models import Market, logit, sigmoid, utcnow
from polyengine.storage import Store


def settings(tmp_path, **kw):
    return Settings(_env_file=None, db_path=str(tmp_path / "t.db"), kill_switch_file=str(tmp_path / "STOP"), **kw)


def overconfident_world(n=400, seed=1):
    """True probability q; the forecaster reports sigmoid(2*logit(q)) - classic LLM over-confidence."""
    rng = random.Random(seed)
    markets = {}
    for i in range(n):
        q = rng.uniform(0.05, 0.95)
        y = 1.0 if rng.random() < q else 0.0
        markets[str(i)] = [(sigmoid(2 * logit(q)), y)]
    return markets


def test_recalibration_undoes_overconfidence():
    a, b = fit_recalibration(overconfident_world(), prior=20)
    assert 0.35 < a < 0.7 and abs(b) < 0.3  # ~0.5 undoes the 2x stretch
    # with only a handful of markets the prior keeps it near identity
    small = dict(list(overconfident_world().items())[:4])
    a2, _ = fit_recalibration(small, prior=20)
    assert 0.8 < a2 <= 1.05


def test_reliability_bins():
    bins = reliability(overconfident_world())
    top = [b for b in bins if b["lo"] == 0.9][0]
    assert top["predicted"] > 0.9 and top["observed"] < top["predicted"] - 0.1  # says 95%, happens ~80%


def test_learning_loop_end_to_end(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    for mid, obs in overconfident_world(120).items():
        p, y = obs[0]
        q = sigmoid(0.5 * logit(p))  # the market prices the true probability
        store.record_snapshot(mid, "q", None, {"polymarket": q, "llm": p})
        store.save_outcome(mid, y)
    cal = Calibrator(s, store)
    cal.recompute(quiet=True)
    llm = cal.by_source["llm"]
    assert llm.recal_a < 0.85
    assert cal.multiplier("llm") < 1.0  # less accurate than the market -> weighted down, gradually
    assert cal.recalibrate("llm", 0.95) < 0.9  # extreme forecasts pulled in
    note = cal.feedback("llm")
    assert note and "over-confident" in note and "happened" in note
    assert store.calib_history()[0]["source"] in ("llm", "polymarket")


def test_shrinkage_moves_gradually(tmp_path):
    s = settings(tmp_path, calib_min_markets=3, calib_prior_markets=20)
    store = Store(s.db_path)
    for i in range(4):  # 4 lucky resolutions for "kalshi"
        y = float(i % 2)
        store.record_snapshot(str(i), "q", None, {"polymarket": 0.5, "kalshi": 0.9 if y else 0.1})
        store.save_outcome(str(i), y)
    cal = Calibrator(s, store)
    cal.recompute(quiet=True)
    assert 1.0 < cal.multiplier("kalshi") < 1.5  # better, but only 4 markets of evidence


async def test_learning_picks_soon_resolving_unresearched(tmp_path):
    eng = Engine(settings(tmp_path, learning_forecasts_per_day=2, learning_horizon_hours=72))

    def mk(i, hours):
        return Market(str(i), "c", f"Q{i}?", "", "s", "e", "E", utcnow() + timedelta(hours=hours), "Y", "N",
                      0.5, 1e5, 1e5)

    ordered = [mk(1, 500), mk(2, 24), mk(3, 48), mk(4, 60)]
    picks = eng._learning_picks(ordered, set(), slots=5)
    assert [m.id for m in picks] == ["2", "3"]  # within 72h, capped at 2/day
    eng.store.put(f"learning:{__import__('time').strftime('%Y-%m-%d', __import__('time').gmtime())}", 2)
    assert eng._learning_picks(ordered, set(), slots=5) == []  # daily quota used
    await eng.aclose()


def test_forecaster_sees_track_record():
    from polyengine.research.forecaster import build_prompt
    m = Market("1", "c", "Will X?", "rules", "s", "e", "E", utcnow() + timedelta(days=2), "Y", "N", 0.5, 1, 1)
    p = build_prompt(m, [], feedback="When you forecast 70%-90%, it happened 55% of the time (n=9).")
    assert "<your_track_record>" in p and "55%" in p
    assert "<your_track_record>" not in build_prompt(m, [])
