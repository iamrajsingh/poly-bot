"""The 24x7 loop: discover -> match -> research -> signal -> size -> execute -> monitor -> exit."""

from __future__ import annotations

import asyncio
import logging
import math
import signal as os_signal
import time
from dataclasses import asdict, replace
from datetime import timedelta

from .arbitrage import Basket, BasketScanner
from .calibration import Calibrator
from .config import Settings
from .connectivity import Connectivity
from .execution.paper import PaperBroker
from .http import Http
from .llm import LLM, BudgetExceeded
from .matching import Matcher
from .models import Forecast, Market, Order, OrderBook, OrderStatus, Side, VenueQuote, utcnow
from .orderbook import stats
from .portfolio import Portfolio
from .research.forecaster import Forecaster
from .research.news import headlines
from .risk import RiskManager, RiskState
from .signals import AGREE_EPS, ensemble, evaluate, merge_by_venue, timing_ok
from .storage import Store
from .venues.kalshi import Kalshi
from .venues.manifold import Manifold
from .venues.polymarket import Polymarket

log = logging.getLogger("polyengine")

_STAGES = [("skip: empty book", "book"), ("skip: spread", "book"), ("skip: no independent", "venues"),
           ("no trade", "edge"), ("defer", "timing"), ("blocked", "risk"), ("would-enter", "order"),
           ("enter", "order")]


def _stage(decision: str) -> str:
    """Furthest pipeline stage a market reached: book -> venues -> edge -> timing -> risk -> order."""
    return next((stage for prefix, stage in _STAGES if decision.startswith(prefix)), "book")


class Engine:
    def __init__(self, s: Settings, live: bool = False):
        self.s = s
        self.store = Store(s.db_path)
        self.net = Connectivity(on_change=lambda st: self.store.put("network", st))
        self.store.put("network", self.net.status())
        self.http = Http(s.http_timeout, net=self.net)
        self.pm = Polymarket(self.http, s.pm_gamma_host, s.pm_clob_host)
        self.kalshi = Kalshi(self.http)
        self.manifold = Manifold(self.http)
        self.llm = LLM(s, self.store)
        self.forecaster = Forecaster(s, self.llm)
        self.matcher = Matcher(s, self.llm, self.store, self.kalshi, self.manifold)
        if live:
            from .execution.live import LiveBroker
            self.broker = LiveBroker(s)
        else:
            self.broker = PaperBroker(self.store, s.paper_starting_cash)
        self.portfolio = Portfolio(s, self.store, self.broker)
        self.risk = RiskManager(s)
        self.arb = BasketScanner(s, self.pm)
        self.calib = Calibrator(s, self.store)
        self._snapshot_at: dict[str, float] = {}
        self._match_slots = 10**6  # per-cycle LLM allowances; reset by analysis_cycle
        self._research_slots = 10**6
        self.universe: dict[str, Market] = {}
        # Restored from the DB so exit rules work immediately after a restart (if not too stale).
        self.fair_by_token: dict[str, float] = self.store.load_fairs(s.research_ttl_minutes)
        self.unrealized = 0.0
        self._stop = asyncio.Event()
        self._sem = asyncio.Semaphore(3)

    # ------------------------------------------------------------------ discovery
    def passes_filters(self, m: Market) -> str | None:
        s = self.s
        if not m.accepting_orders:
            return "not accepting orders"
        if m.liquidity < s.min_liquidity_usd:
            return "low liquidity"
        if m.volume_24h < s.min_volume_24h_usd:
            return "low volume"
        hrs = m.hours_to_resolution
        if hrs is None or hrs < s.min_hours_to_resolution or hrs > s.max_days_to_resolution * 24:
            return "resolution window"
        if not (s.min_price <= m.yes_price <= s.max_price):
            return "price extreme"
        return None

    async def discover(self) -> None:
        markets = await self.pm.active_markets(self.s.scan_limit)
        kept = {m.id: m for m in markets if self.passes_filters(m) is None}
        self.universe = kept
        self.store.put("discovery", {"ts": time.time(), "scanned": len(markets), "passed": len(kept)})
        log.info("discovery: %d markets scanned, %d pass filters", len(markets), len(kept))

    async def market(self, market_id: str) -> Market | None:
        return self.universe.get(market_id) or await self.pm.market(market_id)

    # ------------------------------------------------------------------ research
    def research_fresh(self, m: Market, yes_mid: float | None = None) -> bool:
        latest = self.store.latest_research(m.id)
        if not latest:
            return False
        f, px = latest
        if utcnow() - f.ts > timedelta(minutes=self.s.research_ttl_minutes):
            return False
        ref = yes_mid if yes_mid is not None else m.yes_price
        return abs(ref - px) < self.s.research_price_move_trigger

    async def research(self, m: Market, yes_mid: float, allow_llm: bool) -> Forecast | None:
        latest = self.store.latest_research(m.id)
        if self.research_fresh(m, yes_mid) or not allow_llm or self._research_slots <= 0:
            return latest[0] if latest else None
        self._research_slots -= 1
        try:
            news = await headlines(self.http, m.question, limit=self.s.news_headlines)
            f = await self.forecaster.forecast(m, news, prior=latest[0] if latest else None,
                                               feedback=self.calib.feedback("llm"))
        except BudgetExceeded as e:
            log.warning("%s", e)
            return latest[0] if latest else None
        if f:
            self.store.save_research(m.id, f, yes_mid)
            log.info("RESEARCH %s | P(yes)=%.3f conf=%.2f (market %.3f) | %s", m.question[:70], f.prob,
                     f.confidence, yes_mid, f.rationale.replace("\n", " ")[:160])
            return f
        return latest[0] if latest else None

    def research_score(self, quotes: list[VenueQuote], yes_mid: float) -> float:
        """How much a forecast could matter here (0 = it can't lead to a trade).

        A trade needs `min_agreeing_sources` independent sources beyond the Polymarket price. The LLM
        is one; the rest must come from other venues - so research only pays off where other
        venues already disagree with Polymarket. Score = total disagreement on the best side.
        """
        need = max(0, self.s.min_agreeing_sources - 1)
        if need == 0 or not self.s.research_requires_venue_signal:
            return 1e-3 + sum(abs(q.prob - yes_mid) for q in quotes if q.venue not in ("polymarket", "llm"))
        ext = [q for q in quotes if q.venue not in ("polymarket", "llm")]
        up = [q.prob - yes_mid for q in ext if q.prob > yes_mid + AGREE_EPS]
        down = [yes_mid - q.prob for q in ext if q.prob < yes_mid - AGREE_EPS]
        return max(sum(up) if len(up) >= need else 0.0, sum(down) if len(down) >= need else 0.0)

    async def gather_quotes(self, m: Market, yes_mid: float, allow_match: bool, allow_research: bool,
                            force_research: bool = False) -> tuple[list[VenueQuote], Forecast | None, float]:
        quotes = [VenueQuote("polymarket", yes_mid, self.s.w_polymarket, ref=m.slug, title=m.question,
                             liquidity=m.liquidity)]
        matches = []
        try:
            if self.store.get_matches(m.id, self.s.match_cache_hours) is not None:
                matches = await self.matcher.matches(m, self.s.match_cache_hours)  # cached
            elif allow_match and self._match_slots > 0:
                self._match_slots -= 1
                matches = await self.matcher.matches(m, self.s.match_cache_hours)
        except BudgetExceeded:
            pass
        for mt in matches:
            if mt.venue == "kalshi":
                k = await self.kalshi.quote(mt.ref)
                if not k or k.mid is None or k.spread > 0.10:
                    continue
                p = k.mid if mt.relation == "same" else 1 - k.mid
                w = (self.s.w_kalshi * self.calib.multiplier("kalshi") * mt.confidence
                     * (1.0 if k.spread <= 0.04 else 0.5))
                quotes.append(VenueQuote("kalshi", p, w, ref=mt.ref, title=mt.title, liquidity=k.open_interest,
                                         note=mt.caveat))
            elif mt.venue == "manifold":
                mq = await self.manifold.quote(mt.ref)
                if mq is None or mq[1] < self.s.manifold_min_bettors:
                    continue
                p = mq[0] if mt.relation == "same" else 1 - mq[0]
                participation = min(1.0, mq[1] / 50)  # a crowd of 50+ bettors gets full weight
                quotes.append(VenueQuote("manifold", p, self.s.w_manifold * self.calib.multiplier("manifold")
                                         * mt.confidence * participation, ref=mt.ref, title=mt.title,
                                         note=f"{mq[1]} bettors. {mt.caveat}"))
        quotes = merge_by_venue(quotes)
        cal = [self._calibrated(q) for q in quotes]  # learned corrections; raw values kept for learning
        score = self.research_score(cal, yes_mid)
        held = any(p.market_id == m.id for p in self.portfolio.positions())
        worth_it = force_research or held or score > 0
        f = await self.research(m, yes_mid, allow_research and worth_it)
        if f:
            raw = VenueQuote("llm", f.prob, self.s.w_llm * self.calib.multiplier("llm") * (0.5 + 0.5 * f.confidence),
                             ref=f.model, note=f"conf {f.confidence:.2f}")
            quotes.append(raw)
            cal.append(self._calibrated(raw))
        self._snapshot(m, quotes)  # raw predictions: calibration must learn from what each source said
        return cal, f, score

    def _calibrated(self, q: VenueQuote) -> VenueQuote:
        if q.venue == "polymarket":
            return q
        p = self.calib.recalibrate(q.venue, q.prob)
        if abs(p - q.prob) < 1e-4:
            return q
        return replace(q, prob=p, note=f"{q.note} (raw {q.prob:.3f} -> calibrated {p:.3f})".strip())

    # ------------------------------------------------------------------ analysis
    async def books(self, m: Market) -> tuple[OrderBook, OrderBook]:
        yb, nb = await asyncio.gather(self.pm.book(m.yes_token), self.pm.book(m.no_token))
        return yb, nb

    async def risk_state(self) -> RiskState:
        cash = await self.broker.cash()
        positions = self.portfolio.positions()
        marked = sum(p.cost for p in positions) + self.unrealized
        return RiskState(bankroll=cash + marked, cash=cash, positions=positions,
                         open_orders=self.store.open_orders(),
                         realized_today=self.store.realized_pnl_today(), unrealized=self.unrealized)

    async def analyze(self, m: Market, allow_match: bool = True, allow_research: bool = False,
                      trade: bool = True, force_research: bool = False) -> dict:
        """Full pipeline for one market. Returns a summary dict (used by the CLI, the cycle and
        the dashboard, which shows how far each market got through the pipeline and why)."""
        out = await self._analyze(m, allow_match, allow_research, trade, force_research)
        out["stage"] = _stage(out.get("decision", ""))
        out.update(id=m.id, event=m.event_title, outcomes=list(m.outcomes), end_date=m.end_date,
                   hours=m.hours_to_resolution, volume_24h=m.volume_24h, liquidity=m.liquidity,
                   fee_rate=self.s.fee_for(m)[0], matches=self.store.get_matches(m.id, 24 * 7) or [])
        self.store.save_analysis(m.id, out["stage"], out)
        return out

    async def _analyze(self, m: Market, allow_match: bool, allow_research: bool, trade: bool,
                       force_research: bool) -> dict:
        yb, nb = await self.books(m)
        ys = stats(yb)
        out: dict = {"market": m.question, "url": m.url, "book": asdict(ys),
                     "levels": {"bids": [[lv.price, lv.size] for lv in yb.bids[:25]],
                                "asks": [[lv.price, lv.size] for lv in yb.asks[:25]]}}
        if ys.mid is None:
            out["decision"] = "skip: empty book"
            return out
        if ys.spread is not None and ys.spread > self.s.max_spread:
            out["decision"] = f"skip: spread {ys.spread:.3f}"
            return out

        quotes, f, score = await self.gather_quotes(m, ys.mid, allow_match, allow_research, force_research)
        out["research_score"] = score
        out["research_fresh"] = self.research_fresh(m, ys.mid)
        out["quotes"] = [asdict(q) for q in quotes]
        if f:
            out["forecast"] = asdict(f)
        if len(quotes) < 2:
            out["decision"] = "skip: no independent sources yet"
            return out

        fair_yes, disp = ensemble(quotes)
        self.fair_by_token[m.yes_token] = fair_yes
        self.fair_by_token[m.no_token] = 1 - fair_yes
        self.store.save_fairs({m.yes_token: fair_yes, m.no_token: 1 - fair_yes})
        out["fair_yes"], out["dispersion"] = fair_yes, disp

        sig, reason = evaluate(m, yb, nb, quotes, f.confidence if f else None, self.s)
        if sig is None:
            out["decision"] = f"no trade: {reason}"
            return out
        book = yb if sig.outcome == "YES" else nb
        out["signal"] = {k: v for k, v in asdict(sig).items() if k != "quotes"}

        try:
            hist = await self.pm.price_history(sig.token_id, interval="1d", fidelity=30)
        except Exception:  # noqa: BLE001
            hist = []
        ok, why = timing_ok(sig, book, hist, self.s)
        action = "defer"
        shares = 0.0
        if ok:
            shares, why = self.risk.size(sig, book, await self.risk_state(), m.event_slug, m.min_order_size)
            action = "enter" if shares > 0 and trade else ("would-enter" if shares > 0 else "blocked")
        out["decision"] = f"{action}: {sig.outcome} {shares:.2f}sh @ {sig.price:.3f} "\
                          f"({'taker' if sig.use_taker else 'maker'}) edge {sig.edge:+.3f} | {why}"
        self.store.log_signal(m.id, m.question, sig.outcome, sig.fair, sig.price, sig.edge, sig.confidence,
                              sig.agreeing_sources, sig.use_taker, action, f"{sig.reason} | {why}",
                              [asdict(q) for q in quotes])
        if action == "enter":
            await self.place(m, sig.token_id, sig.outcome, Side.BUY, sig.price, shares,
                             post_only=not sig.use_taker, purpose="entry", book=book, fair=sig.fair,
                             reason=f"fair {sig.fair:.3f}, edge {sig.edge:+.3f}, {sig.agreeing_sources} sources agree")
        log.info("SIGNAL %s | %s", m.question[:70], out["decision"])
        return out

    def _learning_picks(self, ordered: list[Market], skip: set, slots: int) -> list[Market]:
        """Unresearched markets resolving within the learning horizon, highest volume first."""
        left = self.s.learning_forecasts_per_day - self.store.get(
            f"learning:{time.strftime('%Y-%m-%d', time.gmtime())}", 0)
        n = min(left, slots)
        if n <= 0:
            return []
        picks = [m for m in ordered if m.id not in skip and m.hours_to_resolution is not None
                 and m.hours_to_resolution <= self.s.learning_horizon_hours
                 and not self.store.latest_research(m.id)]
        return picks[:n]

    def _snapshot(self, m: Market, quotes: list[VenueQuote]) -> None:
        """Record every source's probability (at most hourly per market) for calibration."""
        now = time.time()
        if now - self._snapshot_at.get(m.id, 0) < self.s.snapshot_interval_minutes * 60:
            return
        self._snapshot_at[m.id] = now
        probs: dict[str, list[float]] = {}
        for q in quotes:
            probs.setdefault(q.venue, []).append(q.prob)
        self.store.record_snapshot(m.id, m.question, m.end_date,
                                   {src: sum(ps) / len(ps) for src, ps in probs.items()})

    async def place(self, m: Market, token: str, outcome: str, side: Side, price: float, shares: float,
                    post_only: bool, purpose: str, book: OrderBook, fair: float = 0.0, reason: str = "") -> Order:
        fee_rate, fee_exp = self.s.fee_for(m)
        o = Order(id="", market_id=m.id, token_id=token, outcome=outcome, side=side, price=round(price, 4),
                  size=round(shares, 2), post_only=post_only, purpose=purpose, fee_rate=fee_rate,
                  fee_exponent=fee_exp, reason=reason[:300])
        o = await self.broker.place(o, book, m)
        if o.status == OrderStatus.REJECTED or not o.id:
            log.warning("order rejected: %s %s %.2f @ %.3f", side.value, outcome, shares, price)
            return o
        # Record the order (unfilled, open) before booking any fill: if the process dies in between,
        # the next sync sees the order and books the fill from the exchange's matched size.
        filled, avg, status = o.filled, o.avg_fill_price, o.status
        o.filled, o.avg_fill_price, o.status = 0.0, 0.0, OrderStatus.OPEN
        self.store.upsert_order(o)
        o.filled, o.avg_fill_price, o.status = filled, avg, status
        if o.filled > 0:
            self.portfolio.apply_fill(o, o.filled, o.avg_fill_price, m, fair)
        self.store.upsert_order(o)
        log.info("ORDER %s %s %s %.2f @ %.3f [%s, %s] -> %s", self.broker.name, side.value, outcome, o.size,
                 o.price, "maker" if post_only else "taker", purpose, o.status.value)
        return o

    async def analysis_cycle(self) -> None:
        if not self.universe:
            return
        st = await self.risk_state()
        if (h := self.risk.halted(st)):
            log.warning("entries halted: %s", h)
            return
        # Markets we hold are always re-evaluated (fresh fair values drive the exit rules),
        # even if they've dropped out of the discovery filters, and get research priority.
        held = {p.market_id for p in st.positions}
        mkts = dict(self.universe)
        for mid in held - mkts.keys():
            if (hm := await self.market(mid)):
                mkts[mid] = hm
        ordered = sorted(mkts.values(), key=lambda m: (m.id not in held, -m.volume_24h))
        llm_ok = self.llm.budget_ok()
        cycle_t0 = time.monotonic()
        usd_start = self.store.llm_usage_today()[1]
        self._match_slots = self.s.max_matches_per_cycle if llm_ok else 0
        self._research_slots = 0

        async def run(m: Market, **kw) -> dict | None:
            async with self._sem:
                try:
                    return await self.analyze(m, **kw)
                except Exception as e:  # noqa: BLE001 - one bad market must not stop the cycle
                    log.warning("analyze %s failed: %s", m.id, e)
                    return None

        # Phase 1 (cheap): books + other venues (+ cached research) for every market. Trades on
        # anything that already has an edge, and measures where a fresh forecast could matter.
        outs = await asyncio.gather(*(run(m, allow_match=True) for m in ordered))

        # Phase 2 (expensive): forecast only where it can change a decision, biggest first.
        cands = []
        for m, out in zip(ordered, outs):
            if not out or out.get("research_fresh") or "research_score" not in out:
                continue
            score = float("inf") if m.id in held else out["research_score"]
            if score > 0:
                cands.append((score, m))
        cands.sort(key=lambda c: -c[0])
        budget = self.s.max_markets_researched_per_cycle if llm_ok else 0
        self._research_slots = budget
        top = [m for _, m in cands[:budget]]
        await asyncio.gather(*(run(m, allow_match=False, allow_research=True) for m in top))

        # Learning forecasts: a few per day on soon-resolving markets, so outcomes arrive quickly.
        learn = self._learning_picks(ordered, {m.id for m in top}, budget - len(top)) if llm_ok else []
        if learn:
            day_key = f"learning:{time.strftime('%Y-%m-%d', time.gmtime())}"
            self.store.put(day_key, self.store.get(day_key, 0) + len(learn))
            await asyncio.gather(*(run(m, allow_match=False, allow_research=True, force_research=True)
                                   for m in learn))
        calls, usd = self.store.llm_usage_today()
        disc = self.store.get("discovery", {})
        ok_outs = [o for o in outs if o]
        self.store.put("funnel", {
            "ts": time.time(),
            "stages": [
                ["Scanned", disc.get("scanned", len(ordered))],
                ["Passed filters", disc.get("passed", len(ordered))],
                ["Tradable book", sum(1 for o in ok_outs if o["stage"] != "book")],
                ["Matched another venue", sum(1 for o in ok_outs if any(
                    q["venue"] in ("kalshi", "manifold") for q in o.get("quotes", [])))],
                ["Venue disagrees", len(cands)],
                ["Has Claude forecast", sum(1 for o in ok_outs if any(
                    q["venue"] == "llm" for q in o.get("quotes", [])))],
                ["Edge found", sum(1 for o in ok_outs if "signal" in o)],
                ["Order placed", sum(1 for o in ok_outs if o["stage"] == "order")],
            ],
            "researched_this_cycle": len(top),
        })
        self.store.record_cycle(
            duration=time.monotonic() - cycle_t0, markets=len(ordered),
            tradable=sum(1 for o in ok_outs if o["stage"] != "book"),
            matched=sum(1 for o in ok_outs if any(q["venue"] in ("kalshi", "manifold") for q in o.get("quotes", []))),
            worth=len(cands), researched=len(top), learning=len(learn),
            signals=sum(1 for o in ok_outs if "signal" in o),
            orders=sum(1 for o in ok_outs if o["stage"] == "order"), llm_usd=usd - usd_start)
        log.info("analysis cycle: %d markets, %d worth researching, %d researched, %d learning | "
                 "LLM today: %d calls ~$%.2f", len(ordered), len(cands), len(top), len(learn), calls, usd)

    # ------------------------------------------------------------------ monitoring / exits
    async def monitor_cycle(self) -> None:
        st = await self.risk_state()
        halted = self.risk.halted(st)
        book_cache: dict[str, OrderBook | None] = {}

        async def book(token: str) -> OrderBook | None:
            if token not in book_cache:
                try:
                    book_cache[token] = await self.pm.book(token)
                except Exception:  # noqa: BLE001 - resolved markets have no book
                    book_cache[token] = None
            return book_cache[token]

        # 1. Orders: sync fills, cancel stale or halted entries.
        for o in self.store.open_orders():
            bk = await book(o.token_id)
            prev_filled, prev_avg = o.filled, o.avg_fill_price
            o = await self.broker.sync(o, bk)
            self._book_delta(o, prev_filled, prev_avg)
            age = (utcnow() - o.created_at).total_seconds()
            if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL) and (
                    age > self.s.maker_order_ttl_seconds or (halted and o.purpose == "entry")):
                prev_filled, prev_avg = o.filled, o.avg_fill_price
                o = await self.broker.cancel(o)
                self._book_delta(o, prev_filled, prev_avg)
                o.status = OrderStatus.CANCELLED if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL) else o.status
                log.info("cancelled %s order %s (%s)", o.purpose, o.id, "halted" if halted else f"ttl {age:.0f}s")
            elif self.s.reprice_enabled and o.post_only and bk is not None and \
                    o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
                o = await self._maybe_reprice(o, bk)
            self.store.upsert_order(o)

        # 2. Positions: settle resolved markets, mark-to-market, run exit rules.
        #    Arb legs are only settled and marked - selling one leg would break the hedge.
        open_exits = {o.token_id: o for o in self.store.open_orders() if o.side == Side.SELL}
        unreal = 0.0
        for p in self.portfolio.positions():
            bk = await book(p.token_id)
            if bk is None or bk.best_bid is None:
                payout_yes = await self.pm.resolution(p.market_id)
                if payout_yes is not None:
                    self.portfolio.settle(p, payout_yes if p.outcome == "YES" else 1 - payout_yes)
                continue
            unreal += p.shares * (self.portfolio.mark(p, bk) - p.avg_price)
            if p.strategy == "arb":
                continue
            m = await self.market(p.market_id)
            fair = self.fair_by_token.get(p.token_id)
            plan = self.portfolio.exit_plan(p, bk, fair, m)
            if not plan or m is None:
                continue
            existing = open_exits.get(p.token_id)
            if existing:
                if existing.post_only and plan.taker:  # escalate maker exit to taker
                    await self.broker.cancel(existing)
                    existing.status = OrderStatus.CANCELLED
                    self.store.upsert_order(existing)
                else:
                    continue
            if p.shares < m.min_order_size:
                continue  # dust below exchange minimum - hold to resolution
            log.info("EXIT %s %s: %s", p.outcome, p.question[:60], plan.reason)
            await self.place(m, p.token_id, p.outcome, Side.SELL, plan.price, p.shares,
                             post_only=not plan.taker, purpose="exit", book=bk, reason=plan.reason)
        self.unrealized = unreal
        cash = await self.broker.cash()
        pos_value = sum(p.cost for p in self.portfolio.positions()) + unreal
        self.store.put("heartbeat", {"ts": time.time(), "mode": self.broker.name, "cash": cash,
                                     "equity": cash + pos_value, "halted": halted})
        if time.time() - getattr(self, "_equity_at", 0) >= 60:
            self._equity_at = time.time()
            self.store.record_equity(cash, pos_value)

    def _maker_target(self, o: Order, bk: OrderBook) -> float | None:
        """Where a resting maker order should sit now. None = edge gone, cancel it."""
        tick = bk.tick_size or 0.01
        bb, ba = bk.best_bid, bk.best_ask
        if o.side == Side.BUY:
            fair = self.fair_by_token.get(o.token_id)
            if fair is None:
                return o.price  # no fresh view: leave it alone
            cap = math.floor((fair - self.s.min_edge) / tick + 1e-9) * tick
            if cap < tick:
                return None
            if bb is not None and bb > o.price + 1e-9:  # outbid: step back to the top, within our edge
                desired = bb + tick if (ba is None or bb + tick < ba - 1e-9) else bb
                return round(min(desired, cap), 4)
            # At (or above) the best bid - possibly our own order - so never chase ourselves upward.
            return round(min(o.price, cap), 4) if o.price - cap > 1e-9 else o.price
        # SELL (maker exit): if someone undercut our ask, undercut them back.
        if ba is not None and ba < o.price - 1e-9:
            return round(ba - tick if bb is None or ba - tick > bb + 1e-9 else ba, 4)
        return o.price

    async def _maybe_reprice(self, o: Order, bk: OrderBook) -> Order:
        target = self._maker_target(o, bk)
        if target is not None and abs(target - o.price) < max(self.s.reprice_threshold, bk.tick_size) - 1e-9:
            return o
        prev_filled, prev_avg = o.filled, o.avg_fill_price
        o = await self.broker.cancel(o)
        self._book_delta(o, prev_filled, prev_avg)
        o.status = OrderStatus.CANCELLED if o.status in (OrderStatus.OPEN, OrderStatus.PARTIAL) else o.status
        m = await self.market(o.market_id)
        if target is None or m is None or o.remaining < m.min_order_size:
            log.info("cancelled %s %s order @ %.3f (edge gone)", o.purpose, o.side.value, o.price)
            return o
        log.info("REPRICE %s %s %.2f: %.3f -> %.3f", o.purpose, o.outcome, o.remaining, o.price, target)
        self.store.upsert_order(o)
        await self.place(m, o.token_id, o.outcome, o.side, target, o.remaining, post_only=True,
                         purpose=o.purpose, book=bk, fair=self.fair_by_token.get(o.token_id, 0.0),
                         reason=o.reason)
        return o

    def _book_delta(self, o: Order, prev_filled: float, prev_avg: float) -> None:
        d = o.filled - prev_filled
        if d <= 1e-9:
            return
        px = (o.filled * o.avg_fill_price - prev_filled * prev_avg) / d
        m = self.universe.get(o.market_id)
        self.portfolio.apply_fill(o, d, px, m, self.fair_by_token.get(o.token_id, 0.0))

    # ------------------------------------------------------------------ basket arbitrage
    async def arb_cycle(self, trade: bool = True) -> list[Basket]:
        if not self.s.arb_enabled:
            return []
        if self.risk.halted(await self.risk_state()):
            return []
        baskets = await self.arb.scan()
        self.store.put("arb_last", {
            "ts": time.time(), "events_scanned": self.arb.last_scanned, "near_misses": self.arb.near_misses,
            "baskets": [{"event": b.event_slug, "title": b.title, "kind": b.kind, "legs": len(b.legs),
                         "sets": b.sets, "cost": b.cost, "payout": b.payout_min, "ret": b.ret, "apr": b.apr,
                         "days": b.days} for b in baskets]})
        for b in sorted(baskets, key=lambda b: -b.apr):
            log.info("ARB %s basket %s: %d legs x %.0f sets, cost $%.2f -> min payout $%.2f "
                     "(+%.2f%%, %.0f%% APR, %.0fd)", b.kind, b.event_slug, len(b.legs), b.sets, b.cost,
                     b.payout_min, 100 * b.ret, 100 * b.apr, b.days)
            if trade:
                await self.execute_basket(b)
        return baskets

    async def execute_basket(self, b: Basket) -> None:
        st = await self.risk_state()
        if any(p.strategy == "arb" and p.event_slug == b.event_slug for p in st.positions):
            return  # one basket per event at a time
        arb_exposure = sum(p.cost for p in st.positions if p.strategy == "arb")
        room = min(self.s.arb_max_exposure_pct * st.bankroll - arb_exposure, st.cash, self.s.arb_max_usd)
        min_size = max(leg.market.min_order_size for leg in b.legs)
        sets = b.sets if b.cost <= room else math.floor(b.sets * room / b.cost)
        if sets < min_size:
            log.info("ARB %s skipped: no room (room $%.2f)", b.event_slug, room)
            return

        # Thinnest leg first: if anything under-fills it's the cheapest place to find out.
        legs = sorted(b.legs, key=lambda lg: sum(x.size for x in lg.book.asks if x.price <= lg.limit))
        target = float(sets)
        filled: list[tuple] = []
        for leg in legs:
            o = await self.place(leg.market, leg.token_id, leg.outcome, Side.BUY, leg.limit, target,
                                 post_only=False, purpose="arb", book=leg.book,
                                 reason=f"{b.kind} basket {b.event_slug}: +{100 * b.ret:.2f}% locked")
            filled.append((leg, o.filled))
            if o.filled + 1e-9 < target:
                target = math.floor(o.filled * 100) / 100
                if target < min_size:
                    target = 0.0
                    break

        # Anything bought beyond the fully hedged quantity is unhedged: sell it back.
        for leg, got in filled:
            excess = round(got - target, 2)
            if excess <= 0:
                continue
            bk = await self.pm.book(leg.token_id)
            if excess < leg.market.min_order_size or bk.best_bid is None:
                log.warning("ARB %s: %.2f unhedged %s shares left in %s (below min size / no bid)",
                            b.event_slug, excess, leg.outcome, leg.market.question[:50])
                continue
            await self.place(leg.market, leg.token_id, leg.outcome, Side.SELL,
                             max(bk.best_bid - 2 * bk.tick_size, bk.tick_size), excess, post_only=False,
                             purpose="arb", book=bk, reason="unwind unhedged excess")
        if target > 0:
            log.info("ARB %s executed: %.2f sets hedged across %d legs", b.event_slug, target, len(b.legs))
        else:
            log.warning("ARB %s aborted: a leg could not fill; unwound", b.event_slug)

    # ------------------------------------------------------------------ calibration
    async def calibration_cycle(self) -> None:
        resolved = 0
        for mid in self.store.markets_to_check():
            try:
                payout = await self.pm.resolution(mid)
            except Exception:  # noqa: BLE001
                continue
            if payout is None:
                self.store.mark_checked(mid)
            else:
                self.store.save_outcome(mid, payout)
                resolved += 1
        if resolved:
            log.info("calibration: %d newly resolved markets", resolved)
        self.calib.recompute()

    # ------------------------------------------------------------------ run loop
    async def _sleep(self, seconds: float) -> None:
        """Sleep, but wake early on stop or when the network comes back (catch-up)."""
        gen = self.net.generation
        end = time.monotonic() + seconds
        while not self._stop.is_set() and self.net.generation == gen:
            left = end - time.monotonic()
            if left <= 0:
                return
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=min(1.0, left))
            except asyncio.TimeoutError:
                pass

    async def _every(self, seconds: int, fn, name: str, delay_first: bool = False) -> None:
        if delay_first:
            await self._sleep(seconds)
        while not self._stop.is_set():
            await self.net.wait_online(self._stop)  # paused while offline
            if self._stop.is_set():
                break
            t0 = time.monotonic()
            try:
                await fn()
            except Exception as e:  # noqa: BLE001 - keep the 24x7 loop alive
                if not self.net.online:
                    log.info("%s interrupted by network loss (%s); will rerun after reconnect",
                             name, type(e).__name__)
                    continue
                log.exception("%s failed", name)
            await self._sleep(max(1.0, seconds - (time.monotonic() - t0)))

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._stop.set)
            except NotImplementedError:
                pass
        await self._resume_summary()
        watcher = asyncio.create_task(self.net.watch(self._stop))
        # Startup needs the network; if it's down, wait for it rather than exiting (and crash-looping).
        while not self._stop.is_set():
            try:
                await self.kalshi.refresh()
                await self.discover()
                orphans = await self.broker.reconcile({o.id for o in self.store.open_orders()})
                if orphans:
                    log.warning("cancelled %d exchange orders unknown to the database (crash leftovers)", orphans)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("startup waiting for network (%s: %s)", type(e).__name__, e)
                self.net.mark_offline()
                await self.net.wait_online(self._stop)
                await self._sleep(5)
        tasks = [
            watcher,
            asyncio.create_task(self._every(self.s.discovery_interval, self.discover, "discovery", True)),
            asyncio.create_task(self._every(self.s.kalshi_refresh_interval, self.kalshi.refresh, "kalshi", True)),
            asyncio.create_task(self._every(self.s.analysis_interval, self.analysis_cycle, "analysis")),
            asyncio.create_task(self._every(self.s.monitor_interval, self.monitor_cycle, "monitor")),
            asyncio.create_task(self._every(self.s.arb_interval, self.arb_cycle, "arbitrage")),
            asyncio.create_task(self._every(self.s.calibration_interval, self.calibration_cycle, "calibration")),
        ]
        await self._stop.wait()
        log.info("stop requested: letting in-flight work finish (max 20s)")
        _, pending = await asyncio.wait(tasks, timeout=20)
        for t in pending:
            t.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        log.info("shutting down: cancelling resting orders")
        for o in self.store.open_orders():
            prev_filled, prev_avg = o.filled, o.avg_fill_price
            o = await self.broker.cancel(o)
            self._book_delta(o, prev_filled, prev_avg)
            o.status = OrderStatus.CANCELLED
            self.store.upsert_order(o)
        await self.http.aclose()

    async def _resume_summary(self) -> None:
        hb = self.store.get("heartbeat") or {}
        calls, usd = self.store.llm_usage_today()
        last = f"{(time.time() - hb['ts']) / 60:.0f} min ago" if hb.get("ts") else "never"
        log.info("polyengine starting [%s] cash=%.2f | resuming: %d positions, %d open orders, %d fair values "
                 "restored, last heartbeat %s, LLM today %d calls ~$%.2f", self.broker.name,
                 await self.broker.cash(), len(self.portfolio.positions()), len(self.store.open_orders()),
                 len(self.fair_by_token), last, calls, usd)

    async def aclose(self) -> None:
        await self.http.aclose()
