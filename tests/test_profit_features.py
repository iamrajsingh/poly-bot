"""Basket arbitrage, maker repricing, calibration."""

from datetime import timedelta

import pytest

from polyengine.arbitrage import BasketScanner
from polyengine.calibration import Calibrator
from polyengine.config import Settings
from polyengine.engine import Engine
from polyengine.models import Level, Order, OrderBook, OrderStatus, Side, utcnow
from polyengine.storage import Store


def settings(tmp_path, **kw):
    base = dict(_env_file=None, db_path=str(tmp_path / "t.db"), kill_switch_file=str(tmp_path / "STOP"),
                paper_starting_cash=10_000)
    base.update(kw)
    return Settings(**base)


def ob(token, bids, asks):
    return OrderBook(token, [Level(p, s) for p, s in bids], [Level(p, s) for p, s in asks], 0.01)


def gamma_market(i, fee_type="zero_fees"):
    return {"id": str(i), "question": f"Outcome {i}?", "outcomes": '["Yes","No"]',
            "clobTokenIds": f'["y{i}","n{i}"]', "outcomePrices": '["0.3","0.7"]', "enableOrderBook": True,
            "acceptingOrders": True, "closed": False, "orderMinSize": 5, "feeType": fee_type,
            "feeSchedule": {"rate": 0, "exponent": 1}, "endDate": (utcnow() + timedelta(days=30)).isoformat()}


class FakePM:
    def __init__(self, books):
        self.books = books

    async def book(self, token):
        return self.books[token]


def event(n, augmented=False):
    return {"slug": "ev", "title": "Who wins?", "negRisk": True, "negRiskAugmented": augmented,
            "endDate": (utcnow() + timedelta(days=30)).isoformat(),
            "markets": [gamma_market(i) for i in range(n)]}


# ---------------------------------------------------------------- basket arbitrage
async def test_no_basket_found_and_sized(tmp_path):
    s = settings(tmp_path, arb_min_return=0.01, arb_min_apr=0.0, arb_max_usd=1_000)
    # 3 outcomes; NO asks 0.60 + 0.65 + 0.70 = 1.95 < payout 2 -> 2.56% locked in
    books = {"n0": ob("n0", [], [(0.60, 100), (0.70, 1000)]),
             "n1": ob("n1", [], [(0.65, 100)]),
             "n2": ob("n2", [], [(0.70, 100)])}
    sc = BasketScanner(s, FakePM(books))
    ev = event(3)
    b = await sc.evaluate(ev, sc._open_markets(ev), "NO")
    assert b is not None
    assert b.sets == 100  # limited by the depth at the good prices
    assert b.cost == pytest.approx(195)
    assert b.payout_min == pytest.approx(200)
    assert b.ret > 0.01


async def test_no_basket_rejected_when_fees_eat_it(tmp_path):
    s = settings(tmp_path, arb_min_return=0.01, arb_min_apr=0.0)
    books = {"n0": ob("n0", [], [(0.66, 100)]), "n1": ob("n1", [], [(0.66, 100)]), "n2": ob("n2", [], [(0.67, 100)])}
    sc = BasketScanner(s, FakePM(books))
    ev = event(3)
    for m in ev["markets"]:
        m["feeType"], m["feeSchedule"] = "crypto_fees_v2", {"rate": 0.07, "exponent": 1}
    assert await sc.evaluate(ev, sc._open_markets(ev), "NO") is None  # 1.99 + ~0.045 fees > 2


def test_incomplete_or_resolved_event_is_skipped(tmp_path):
    sc = BasketScanner(settings(tmp_path), FakePM({}))
    ev = event(3)
    ev["markets"][1]["acceptingOrders"] = False  # can't buy one leg -> not a hedge
    assert sc._open_markets(ev) is None
    ev = event(3)
    ev["markets"][0].update(closed=True, outcomePrices='["1","0"]')  # someone already won
    assert sc._open_markets(ev) is None


async def test_yes_basket_never_on_augmented_events(tmp_path):
    s = settings(tmp_path, arb_yes_baskets=True)
    sc = BasketScanner(s, FakePM({}))
    seen = []

    async def fake_events():
        return [event(3, augmented=True)]

    async def fake_eval(ev, ms, kind):
        seen.append(kind)

    sc.neg_risk_events, sc.evaluate = fake_events, fake_eval
    sc._prefilter = lambda kind, ms: True
    await sc.scan()
    assert seen == ["NO"]


async def test_execute_basket_unwinds_short_leg(tmp_path):
    """Last leg only fills 60 of 100 (book thinned mid-execution): the hedge shrinks to 60 and
    the 40 extra shares already bought on the other legs are sold back."""
    s = settings(tmp_path, arb_min_return=0.0, arb_min_apr=0.0, arb_max_usd=1_000)
    eng = Engine(s)
    books = {"n0": ob("n0", [(0.58, 1000)], [(0.60, 1000)]),
             "n1": ob("n1", [(0.63, 1000)], [(0.65, 1000)]),
             "n2": ob("n2", [(0.68, 1000)], [(0.70, 1000)])}
    eng.arb.pm = FakePM(books)
    eng.pm.book = FakePM(books).book
    ev = event(3)
    b = await eng.arb.evaluate(ev, eng.arb._open_markets(ev), "NO")
    b.sets = 100
    real_place = eng.broker.place

    async def thin_last_leg(order, book, market):
        if order.token_id == "n2" and order.side == Side.BUY:
            book = ob("n2", [(0.68, 1000)], [(0.70, 60)])
        return await real_place(order, book, market)

    eng.broker.place = thin_last_leg
    await eng.execute_basket(b)
    pos = {p.token_id: p.shares for p in eng.portfolio.positions()}
    assert pos == {"n0": pytest.approx(60), "n1": pytest.approx(60), "n2": pytest.approx(60)}
    assert all(p.strategy == "arb" for p in eng.portfolio.positions())
    await eng.aclose()


# ---------------------------------------------------------------- repricing
async def test_maker_reprices_up_when_outbid_but_never_past_edge(tmp_path):
    s = settings(tmp_path, min_edge=0.04)
    eng = Engine(s)
    eng.fair_by_token["Y"] = 0.50
    o = Order("x", "1", "Y", "YES", Side.BUY, 0.40, 50)
    # outbid at 0.42: step to 0.43 (still <= 0.46 cap)
    assert eng._maker_target(o, ob("Y", [(0.42, 10)], [(0.48, 10)])) == pytest.approx(0.43)
    # outbid at 0.47: cap is fair - min_edge = 0.46
    assert eng._maker_target(o, ob("Y", [(0.47, 10)], [(0.49, 10)])) == pytest.approx(0.46)
    # we're the best bid ourselves: don't chase our own order
    assert eng._maker_target(o, ob("Y", [(0.40, 50)], [(0.48, 10)])) == pytest.approx(0.40)
    # fair collapsed: edge gone
    eng.fair_by_token["Y"] = 0.04
    assert eng._maker_target(o, ob("Y", [(0.40, 50)], [(0.48, 10)])) is None
    await eng.aclose()


async def test_reprice_replaces_order(tmp_path):
    s = settings(tmp_path, min_edge=0.04)
    eng = Engine(s)
    eng.fair_by_token["Y"] = 0.50
    from polyengine.models import Market
    m = Market("1", "c", "Q?", "", "q", "ev", "E", utcnow() + timedelta(days=5), "Y", "N", 0.4, 1e5, 1e5)
    eng.universe["1"] = m
    bk = ob("Y", [(0.40, 100)], [(0.48, 100)])
    o = await eng.place(m, "Y", "YES", Side.BUY, 0.40, 50, post_only=True, purpose="entry", book=bk)
    bk2 = ob("Y", [(0.43, 100)], [(0.48, 100)])
    await eng._maybe_reprice(o, bk2)
    live = eng.store.open_orders()
    assert len(live) == 1 and live[0].price == pytest.approx(0.44) and live[0].size == 50
    await eng.aclose()


# ---------------------------------------------------------------- calibration
def test_calibration_rewards_the_more_accurate_source(tmp_path):
    s = settings(tmp_path, calib_min_markets=5)
    store = Store(s.db_path)
    for i in range(10):
        y = 1.0 if i % 2 == 0 else 0.0
        # market is lukewarm, "kalshi" is sharper, "manifold" is backwards
        store.record_snapshot(str(i), "q", None, {"polymarket": 0.5, "kalshi": 0.8 if y else 0.2,
                                                  "manifold": 0.3 if y else 0.7})
        store.save_outcome(str(i), y)
    cal = Calibrator(s, store)
    scores = {sc.source: sc for sc in cal.recompute()}
    assert scores["kalshi"].skill_vs_market > 1 and cal.multiplier("kalshi") > 1
    assert scores["manifold"].skill_vs_market < 1 and cal.multiplier("manifold") < 1
    assert scores["kalshi"].closer_when_disagreeing == 1.0
    assert cal.multiplier("llm") == 1.0  # no data yet -> unchanged


def test_calibration_waits_for_enough_markets(tmp_path):
    s = settings(tmp_path, calib_min_markets=30)
    store = Store(s.db_path)
    store.record_snapshot("1", "q", None, {"polymarket": 0.5, "llm": 0.9})
    store.save_outcome("1", 1.0)
    cal = Calibrator(s, store)
    cal.recompute()
    assert cal.multiplier("llm") == 1.0


def test_watch_list_skips_resolved(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.record_snapshot("a", "q", utcnow() - timedelta(days=1), {"polymarket": 0.5})
    store.record_snapshot("b", "q", utcnow() - timedelta(days=1), {"polymarket": 0.5})
    store.save_outcome("b", 1.0)
    assert store.markets_to_check() == ["a"]


def test_old_database_is_migrated(tmp_path):
    import sqlite3
    path = str(tmp_path / "old.db")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE orders (id TEXT PRIMARY KEY, market_id TEXT, token_id TEXT, outcome TEXT, side TEXT, "
               "price REAL, size REAL, filled REAL, avg_fill_price REAL, status TEXT, post_only INTEGER, "
               "purpose TEXT, created_at TEXT, updated_at TEXT)")
    db.commit()
    db.close()
    store = Store(path)
    store.upsert_order(Order("o1", "1", "Y", "YES", Side.BUY, 0.4, 10, fee_rate=0.05))
    assert store.open_orders()[0].fee_rate == 0.05
    assert store.open_orders()[0].status == OrderStatus.OPEN


# ---------------------------------------------------------------- research triage
async def test_research_only_where_another_venue_disagrees(tmp_path):
    from polyengine.models import VenueQuote as V
    eng = Engine(settings(tmp_path, min_agreeing_sources=2))
    pm = V("polymarket", 0.50, 3)
    assert eng.research_score([pm], 0.50) == 0  # no other venue: a forecast alone can't trade
    assert eng.research_score([pm, V("kalshi", 0.505, 2)], 0.50) == 0  # venue agrees with market
    assert eng.research_score([pm, V("kalshi", 0.58, 2)], 0.50) == pytest.approx(0.08)
    assert eng.research_score([pm, V("kalshi", 0.40, 2), V("manifold", 0.45, 1)], 0.50) == pytest.approx(0.15)
    eng.s.research_requires_venue_signal = False
    assert eng.research_score([pm], 0.50) > 0
    await eng.aclose()
