import math
from datetime import timedelta

import pytest

from polyengine.config import Settings
from polyengine.execution.paper import PaperBroker
from polyengine.models import Level, Market, Order, OrderBook, OrderStatus, Position, Side, VenueQuote, utcnow
from polyengine.orderbook import stats, sweep, taker_fee_per_share
from polyengine.portfolio import Portfolio
from polyengine.risk import RiskManager, RiskState, kelly_fraction
from polyengine.signals import ensemble, evaluate
from polyengine.storage import Store
from polyengine.venues.polymarket import parse_market


def book(bids, asks, token="Y", last=None):
    return OrderBook(token, [Level(p, s) for p, s in bids], [Level(p, s) for p, s in asks], 0.01, last)


def market(**kw):
    d = dict(id="1", condition_id="c", question="Will X happen?", description="", slug="x", event_slug="ev",
             event_title="X", end_date=utcnow() + timedelta(days=10), yes_token="Y", no_token="N",
             yes_price=0.40, liquidity=50_000, volume_24h=20_000)
    d.update(kw)
    return Market(**d)


def settings(tmp_path, **kw):
    return Settings(_env_file=None, db_path=str(tmp_path / "t.db"), kill_switch_file=str(tmp_path / "STOP"), **kw)


# ---------------------------------------------------------------- order book
def test_stats_and_sweep():
    b = book([(0.40, 100), (0.39, 200)], [(0.42, 50), (0.43, 100), (0.50, 1000)])
    st = stats(b)
    assert st.best_bid == 0.40 and st.best_ask == 0.42
    assert st.spread == pytest.approx(0.02)
    # microprice leans toward the thin side (ask has less size -> price heads up)
    assert st.mid < st.microprice < 0.42
    f = sweep(b, Side.BUY, shares=100)
    assert f.shares == 100 and f.avg_price == pytest.approx((50 * 0.42 + 50 * 0.43) / 100)
    assert f.complete
    f = sweep(b, Side.BUY, shares=500, limit=0.43)
    assert f.shares == 150 and not f.complete


def test_fee_formula():
    # Polymarket docs: 100 shares at 0.50, rate 0.07 -> $1.75
    assert 100 * taker_fee_per_share(0.5, 0.07) == pytest.approx(1.75)


def test_parse_gamma_market():
    raw = {"id": "5", "question": "Q?", "outcomes": '["Yes","No"]', "clobTokenIds": '["a","b"]',
           "outcomePrices": '["0.3","0.7"]', "enableOrderBook": True, "endDate": "2030-01-01T00:00:00Z",
           "liquidityNum": 1000, "volume24hr": 50, "events": [{"slug": "ev", "title": "E"}]}
    m = parse_market(raw)
    assert m and m.yes_token == "a" and m.no_token == "b" and m.yes_price == 0.3 and m.event_slug == "ev"
    assert parse_market({**raw, "outcomes": '["A","B","C"]'}) is None
    # fee schedule
    assert m.fee_rate is None  # unknown -> config default
    m = parse_market({**raw, "feeType": "zero_fees", "feeSchedule": {"rate": 0, "exponent": 1}})
    assert m.fee_rate == 0
    m = parse_market({**raw, "feeType": "crypto_fees_v2", "feeSchedule": {"rate": 0.07, "exponent": 1}})
    assert m.fee_rate == 0.07
    assert parse_market({**raw, "feesEnabled": False}).fee_rate == 0


def test_zero_fee_market_takes_liquidity(tmp_path):
    """Same prices: a fee-free market clears the taker bar, a 7% market does not."""
    s = settings(tmp_path, min_edge=0.04, taker_edge_premium=0.03)
    yb = book([(0.39, 500)], [(0.40, 500)])
    nb = book([(0.60, 500)], [(0.61, 500)], token="N")
    # fair ~0.478: edge at the 0.40 ask is ~0.078 gross; a 7% fee costs ~0.017 of it
    q = [VenueQuote("polymarket", 0.395, 3), VenueQuote("llm", 0.56, 1.6), VenueQuote("kalshi", 0.54, 2)]
    free, _ = evaluate(market(fee_rate=0.0), yb, nb, q, 0.9, s)
    costly, _ = evaluate(market(fee_rate=0.07), yb, nb, q, 0.9, s)
    assert free is not None and free.use_taker
    assert costly is not None and not costly.use_taker


# ---------------------------------------------------------------- signals
def test_ensemble_logodds():
    fair, disp = ensemble([VenueQuote("a", 0.5, 1), VenueQuote("b", 0.5, 1)])
    assert fair == pytest.approx(0.5) and disp == pytest.approx(0)
    fair, _ = ensemble([VenueQuote("polymarket", 0.40, 3), VenueQuote("kalshi", 0.55, 2), VenueQuote("llm", 0.60, 1.6)])
    assert 0.40 < fair < 0.60


def test_evaluate_requires_agreement(tmp_path):
    s = settings(tmp_path)
    m = market()
    yb = book([(0.39, 500)], [(0.41, 500)])
    nb = book([(0.59, 500)], [(0.61, 500)], token="N")
    # Only the LLM disagrees with the market -> no trade even with a big gap.
    q = [VenueQuote("polymarket", 0.40, 3), VenueQuote("llm", 0.70, 1.6)]
    sig, why = evaluate(m, yb, nb, q, 0.9, s)
    assert sig is None and "agree" in why
    # LLM + Kalshi both say YES is cheap -> buy YES.
    q.append(VenueQuote("kalshi", 0.62, 2.0))
    sig, why = evaluate(m, yb, nb, q, 0.9, s)
    assert sig is not None and sig.outcome == "YES" and sig.edge >= s.min_edge, why
    assert sig.price < 0.41 or sig.use_taker


def test_evaluate_buys_no_side(tmp_path):
    s = settings(tmp_path)
    m = market(yes_price=0.60)
    yb = book([(0.59, 500)], [(0.61, 500)])
    nb = book([(0.39, 500)], [(0.41, 500)], token="N")
    q = [VenueQuote("polymarket", 0.60, 3), VenueQuote("llm", 0.35, 1.6), VenueQuote("kalshi", 0.40, 2)]
    sig, _ = evaluate(m, yb, nb, q, 0.8, s)
    assert sig is not None and sig.outcome == "NO" and sig.token_id == "N"


# ---------------------------------------------------------------- risk
def test_kelly():
    assert kelly_fraction(0.6, 0.5) == pytest.approx(0.2)
    assert kelly_fraction(0.4, 0.5) == 0


def test_risk_caps_and_kill_switch(tmp_path):
    s = settings(tmp_path, max_position_usd=20)
    r = RiskManager(s)
    m = market()
    sig, _ = evaluate(m, book([(0.39, 5000)], [(0.41, 5000)]), book([(0.59, 5000)], [(0.61, 5000)], token="N"),
                      [VenueQuote("polymarket", 0.4, 3), VenueQuote("llm", 0.7, 1.6), VenueQuote("kalshi", 0.65, 2)],
                      0.9, s)
    st = RiskState(bankroll=10_000, cash=10_000, positions=[], open_orders=[], realized_today=0, unrealized=0)
    shares, _ = r.size(sig, book([(0.39, 5000)], [(0.41, 5000)]), st, "ev", 5)
    assert 0 < shares * sig.price <= 20 + 1e-6
    (tmp_path / "STOP").write_text("")
    shares, why = r.size(sig, book([(0.39, 5000)], [(0.41, 5000)]), st, "ev", 5)
    assert shares == 0 and "kill" in why


def test_daily_loss_halts(tmp_path):
    r = RiskManager(settings(tmp_path))
    st = RiskState(bankroll=1000, cash=900, positions=[], open_orders=[], realized_today=-60, unrealized=0)
    assert r.halted(st)


# ---------------------------------------------------------------- paper broker + portfolio
async def test_paper_roundtrip_pnl(tmp_path):
    s = settings(tmp_path, paper_starting_cash=1000)
    store = Store(s.db_path)
    br = PaperBroker(store, 1000)
    pf = Portfolio(s, store, br)
    m = market()
    b = book([(0.39, 500)], [(0.41, 500)], last=0.40)

    # post-only order that would cross is rejected
    o = await br.place(Order("", "1", "Y", "YES", Side.BUY, 0.41, 10), b, m)
    assert o.status == OrderStatus.REJECTED

    # maker bid rests, then fills when the ask comes down to it
    o = await br.place(Order("", "1", "Y", "YES", Side.BUY, 0.40, 100), b, m)
    assert o.status == OrderStatus.OPEN
    o = await br.sync(o, book([(0.38, 500)], [(0.40, 500)], last=0.40))
    assert o.status == OrderStatus.FILLED and o.avg_fill_price == pytest.approx(0.40)
    pf.apply_fill(o, o.filled, o.avg_fill_price, m)
    assert await br.cash() == pytest.approx(960)

    # taker exit at 0.50 -> realized = 100*(0.50-0.40) - fee
    exit_o = await br.place(Order("", "1", "Y", "YES", Side.SELL, 0.50, 100, post_only=False,
                                  fee_rate=s.taker_fee_rate),
                            book([(0.50, 500)], [(0.52, 500)]), m)
    assert exit_o.filled == 100
    pf.apply_fill(exit_o, exit_o.filled, exit_o.avg_fill_price, m)
    fee = 100 * taker_fee_per_share(0.5, s.taker_fee_rate)
    assert store.realized_pnl_total() == pytest.approx(10 - fee)
    assert await br.cash() == pytest.approx(1010 - fee)
    assert pf.positions() == []


async def test_settlement(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    br = PaperBroker(store, 100)
    pf = Portfolio(s, store, br)
    p = Position("1", "N", "NO", 50, 0.30)
    store.save_position(p)
    pf.settle(p, 1.0)
    assert store.realized_pnl_total() == pytest.approx(35)
    assert await br.cash() == pytest.approx(150)


def test_exit_rules(tmp_path):
    s = settings(tmp_path)
    pf = Portfolio(s, Store(s.db_path), None)
    p = Position("1", "Y", "YES", 100, 0.40)
    m = market()
    # hard stop
    plan = pf.exit_plan(p, book([(0.25, 500)], [(0.27, 500)]), 0.5, m)
    assert plan and plan.taker and "stop" in plan.reason
    # thesis broken: fair collapsed below the bid
    plan = pf.exit_plan(p, book([(0.40, 500)], [(0.42, 500)]), 0.30, m)
    assert plan and plan.taker and "thesis" in plan.reason
    # edge captured -> maker exit inside the spread
    plan = pf.exit_plan(p, book([(0.55, 500)], [(0.58, 500)]), 0.555, m)
    assert plan and not plan.taker and 0.55 < plan.price < 0.58
    # nothing to do
    assert pf.exit_plan(p, book([(0.41, 500)], [(0.43, 500)]), 0.55, m) is None
    assert math.isfinite(plan.price)


def test_one_venue_cannot_count_twice(tmp_path):
    from polyengine.signals import merge_by_venue
    s = settings(tmp_path)
    m = market()
    yb = book([(0.39, 500)], [(0.41, 500)])
    nb = book([(0.59, 500)], [(0.61, 500)], token="N")
    # Kalshi's "D wins" and inverted "R wins" contracts of the same race both disagree with Polymarket
    q = [VenueQuote("polymarket", 0.40, 3), VenueQuote("kalshi", 0.60, 2, ref="SENATE-D"),
         VenueQuote("kalshi", 0.58, 2, ref="SENATE-R")]
    sig, _ = evaluate(m, yb, nb, q, None, s)
    assert sig is not None  # unmerged: one venue looks like two agreeing sources (the bug)
    merged = merge_by_venue(q)
    assert [x.venue for x in merged] == ["polymarket", "kalshi"]
    assert 0.58 < merged[1].prob < 0.60 and merged[1].weight == 2
    sig, why = evaluate(m, yb, nb, merged, None, s)
    assert sig is None and "agree" in why  # one venue alone is not enough
