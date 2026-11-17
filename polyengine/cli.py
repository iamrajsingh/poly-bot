"""Command line: scan | analyze | run | status | cancel-all."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import get_settings
from .engine import Engine
from .orderbook import stats


def setup_logging(level: str, db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    fh = RotatingFileHandler(Path(db_path).parent / "engine.log", maxBytes=20_000_000, backupCount=5)
    fh.setFormatter(fmt)
    root.handlers = [fh]
    if sys.stdout.isatty():  # under launchd stdout is a file: the rotating engine.log is enough
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.handlers.append(sh)
    for noisy in ("httpx", "httpcore", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def cmd_scan(eng: Engine, top: int) -> None:
    await eng.discover()
    mkts = sorted(eng.universe.values(), key=lambda m: m.volume_24h, reverse=True)[:top]
    books = await asyncio.gather(*(eng.pm.book(m.yes_token) for m in mkts), return_exceptions=True)
    print(f"{'YES':>6} {'bid':>6} {'ask':>6} {'sprd':>5} {'imb':>5} {'liq$':>9} {'vol24h$':>9} {'hrs':>6}  question")
    for m, b in zip(mkts, books):
        if isinstance(b, Exception):
            continue
        st = stats(b)
        f = lambda v: f"{v:.3f}" if v is not None else "  -  "  # noqa: E731
        print(f"{m.yes_price:6.3f} {f(st.best_bid):>6} {f(st.best_ask):>6} {st.spread or 0:5.3f} "
              f"{st.imbalance:+5.2f} {m.liquidity:9.0f} {m.volume_24h:9.0f} {m.hours_to_resolution or 0:6.0f}  "
              f"{m.question[:70]}")
    print(f"\n{len(eng.universe)} markets pass filters.")


async def cmd_analyze(eng: Engine, ref: str, trade: bool) -> None:
    m = await eng.market(ref)
    if not m:
        print(f"market not found: {ref}")
        return
    print(f"{m.question}\n{m.url}\nfilters: {eng.passes_filters(m) or 'pass'}\n")
    await eng.kalshi.refresh()
    out = await eng.analyze(m, allow_match=True, allow_research=True, trade=trade, force_research=True)
    print(json.dumps(out, indent=2, default=str))


async def cmd_status(eng: Engine) -> None:
    cash = await eng.broker.cash()
    pos = eng.portfolio.positions()
    print(f"mode: {eng.broker.name}   cash: ${cash:,.2f}")
    print(f"realized PnL today: {eng.store.realized_pnl_today():+.2f}   all-time: {eng.store.realized_pnl_total():+.2f}")
    calls, usd = eng.store.llm_usage_today()
    print(f"LLM today: {calls} requests, ~${usd:.2f}\n")
    if pos:
        books = await asyncio.gather(*(eng.pm.book(p.token_id) for p in pos), return_exceptions=True)
        print(f"{'out':>3} {'shares':>8} {'avg':>6} {'bid':>6} {'uPnL':>8}  question")
        for p, b in zip(pos, books):
            bid = None if isinstance(b, Exception) else b.best_bid
            u = p.shares * ((bid or p.avg_price) - p.avg_price)
            print(f"{p.outcome:>3} {p.shares:8.2f} {p.avg_price:6.3f} {bid or 0:6.3f} {u:+8.2f}  {p.question[:70]}")
    else:
        print("no open positions")
    orders = eng.store.open_orders()
    if orders:
        print("\nopen orders:")
        for o in orders:
            print(f"  {o.side.value:4} {o.outcome:3} {o.size:8.2f} @ {o.price:.3f} filled {o.filled:.2f} "
                  f"[{o.purpose}{', maker' if o.post_only else ''}] {o.id}")


async def cmd_arb(eng: Engine, trade: bool) -> None:
    baskets = await eng.arb_cycle(trade=trade)
    if not baskets:
        print("no basket arbitrage right now (normal on liquid events - it appears briefly after news)")
    for b in baskets:
        print(f"{b.kind} basket  {b.event_slug}\n  {len(b.legs)} legs x {b.sets:.0f} sets  cost ${b.cost:.2f}  "
              f"min payout ${b.payout_min:.2f}  +{100 * b.ret:.2f}%  ~{100 * b.apr:.0f}% APR  {b.days:.0f} days")
        for lg in b.legs:
            print(f"    {lg.outcome} @ <= {lg.limit:.3f}  {lg.market.question[:70]}")


async def cmd_calibration(eng: Engine) -> None:
    await eng.calibration_cycle()
    if not eng.calib.scores:
        print("no resolved markets yet - calibration fills in as researched markets resolve")
        return
    print(f"{'source':10} {'mkts':>5} {'brier':>6} {'logloss':>7} {'shared':>6} {'skill':>6} "
          f"{'disagree':>8} {'closer%':>7} {'weight x':>8}")
    for sc in eng.calib.scores:
        skill = f"{sc.skill_vs_market:.2f}" if sc.skill_vs_market is not None else "-"
        closer = f"{100 * sc.closer_when_disagreeing:.0f}" if sc.closer_when_disagreeing is not None else "-"
        print(f"{sc.source:10} {sc.markets:5d} {sc.brier:6.3f} {sc.log_loss:7.3f} {sc.shared_markets:6d} "
              f"{skill:>6} {sc.disagreements:8d} {closer:>7} {sc.multiplier:8.2f}")
    print(f"\nskill > 1 = more accurate than the Polymarket price on the same markets. Weights adapt after "
          f"{eng.s.calib_min_markets} shared resolutions (PE_CALIB_MIN_MARKETS).")


def attach_event_log(eng: Engine) -> None:
    from .dashboard import EventLogHandler
    logging.getLogger("polyengine").addHandler(EventLogHandler(eng.store))


def project_dir() -> Path:
    here = Path(__file__).resolve().parents[1]
    return here if (here / ".venv").exists() else Path.cwd()


def cmd_service(args: argparse.Namespace) -> None:
    from . import service
    action = args.action
    if action == "install":
        service.install(project_dir(), live=args.live, dashboard=not args.no_dashboard, port=args.port)
    elif action == "uninstall":
        service.uninstall()
    elif action == "stop":
        service.stop()
    elif action == "start":
        service.start()
    elif action == "logs":
        log = project_dir() / "data" / "engine.log"
        os.execvp("tail", ["tail", "-n", str(args.lines), "-F", str(log)])
    else:  # status
        st = service.status()
        print(f"service: {'installed' if st['installed'] else 'not installed'}, "
              f"{'loaded' if st['loaded'] else 'not loaded'}, state={st['state']}, pid={st['pid']}, "
              f"runs={st['runs']}, last exit={st['last_exit']}")
        from .storage import Store
        s = get_settings()
        db = Path(s.db_path)
        if not db.is_absolute():
            db = project_dir() / db
        if db.exists():
            store = Store(str(db))
            hb, net = store.get("heartbeat") or {}, store.get("network") or {}
            if hb.get("ts"):
                print(f"engine: last heartbeat {time.time() - hb['ts']:.0f}s ago, mode={hb.get('mode')}, "
                      f"equity=${hb.get('equity', 0):,.2f}")
            if net:
                state = "online" if net.get("online") else \
                    f"OFFLINE since {time.strftime('%H:%M:%S', time.localtime(net['offline_since']))}"
                print(f"network: {state}, outages this run: {net.get('outages', 0)}")


async def amain(args: argparse.Namespace) -> None:
    s = get_settings()
    live = getattr(args, "live", False)
    if live and not s.live_trading and args.cmd not in ("dashboard", "status", "calibration"):
        sys.exit("refusing live mode: set PE_LIVE_TRADING=true in .env as well as passing --live")
    if live:
        s.db_path = s.db_path.replace(".db", ".live.db")  # never mix paper and live books
    setup_logging(s.log_level, s.db_path)
    if args.cmd == "dashboard":
        from .dashboard import serve
        print(f"dashboard on http://127.0.0.1:{args.port}  (reading {s.db_path}; Ctrl-C to stop)")
        await asyncio.to_thread(serve, s, "127.0.0.1", args.port)
        return
    trades = args.cmd == "run" or (args.cmd in ("analyze", "arb") and getattr(args, "trade", False))
    if trades:
        from .service import InstanceLock
        lock = InstanceLock(s.db_path)
        if not lock.acquire():
            # non-zero exit: under launchd this retries every 30s until the other instance stops
            sys.exit(f"another polyengine (pid {lock.holder()}) is already trading on {s.db_path}; "
                     f"stop it first (e.g. `polyengine service stop`)")
    eng = Engine(s, live=live)
    if args.cmd in ("run", "analyze", "arb"):
        attach_event_log(eng)
    if args.cmd == "run" and args.dashboard:
        from .dashboard import serve
        serve(s, "127.0.0.1", args.port, background=True)
        print(f"dashboard on http://127.0.0.1:{args.port}")
    try:
        if args.cmd == "scan":
            await cmd_scan(eng, args.top)
        elif args.cmd == "analyze":
            await cmd_analyze(eng, args.market, args.trade)
        elif args.cmd == "status":
            await cmd_status(eng)
        elif args.cmd == "arb":
            await cmd_arb(eng, args.trade)
        elif args.cmd == "calibration":
            await cmd_calibration(eng)
        elif args.cmd == "eval-matcher":
            from .evals import eval_matcher
            specs = [tuple(x.split(":")) for x in args.models.split(",")]
            r = await eval_matcher(s, args.n, specs, Path(s.db_path).parent / "evals")
            print(f"\n{r['markets']} markets, {r['candidates']} candidates, reference = {r['reference']}")
            print(f"{'model':32} {'agree':>5} {'FLIP':>5} {'extra':>5} {'missed':>6} {'failed':>6} {'$/market':>9}")
            for k, v in r["models"].items():
                print(f"{k:32} {v['agree']:5d} {v['flipped']:5d} {v['extra']:5d} {v['missed']:6d} "
                      f"{v['failed_calls']:6d} {v['usd_per_market']:9.5f}")
            print(f"\ndisagreements to review: {len(r['disagreements'])} -> {r['path']}")
        elif args.cmd == "cancel-all":
            await eng.broker.cancel_all()
            print("cancel-all sent")
        elif args.cmd == "run":
            await eng.run()
            return
    finally:
        await eng.aclose()


def main() -> None:
    p = argparse.ArgumentParser(prog="polyengine", description="Polymarket research + trading engine")
    sub = p.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="list tradable markets + book stats (no LLM, no orders)")
    sc.add_argument("--top", type=int, default=30)
    an = sub.add_parser("analyze", help="full research pipeline on one market (slug or id)")
    an.add_argument("market")
    an.add_argument("--trade", action="store_true", help="actually place the order (paper unless --live)")
    an.add_argument("--live", action="store_true")
    rn = sub.add_parser("run", help="run 24x7")
    rn.add_argument("--live", action="store_true", help="real orders (also needs PE_LIVE_TRADING=true)")
    rn.add_argument("--dashboard", action="store_true", help="also serve the web dashboard")
    rn.add_argument("--port", type=int, default=8642)
    db = sub.add_parser("dashboard", help="web dashboard over the engine database (read-only)")
    db.add_argument("--port", type=int, default=8642)
    db.add_argument("--live", action="store_true", help="show the live-trading database")
    st = sub.add_parser("status", help="positions, orders, PnL, LLM spend")
    st.add_argument("--live", action="store_true")
    sv = sub.add_parser("service", help="run 24x7 on macOS via launchd (auto-start, auto-restart)")
    sv.add_argument("action", choices=["install", "uninstall", "start", "stop", "status", "logs"])
    sv.add_argument("--live", action="store_true", help="install in live-trading mode")
    sv.add_argument("--no-dashboard", action="store_true")
    sv.add_argument("--port", type=int, default=8642)
    sv.add_argument("--lines", type=int, default=50, help="for `logs`")
    ab = sub.add_parser("arb", help="scan neg-risk events for basket arbitrage")
    ab.add_argument("--trade", action="store_true", help="execute found baskets (paper unless --live)")
    ab.add_argument("--live", action="store_true")
    cb = sub.add_parser("calibration", help="accuracy of each source vs resolved outcomes")
    cb.add_argument("--live", action="store_true")
    ev = sub.add_parser("eval-matcher", help="compare models on the cross-venue matcher (costs a little)")
    ev.add_argument("--n", type=int, default=40, help="markets with candidates to test")
    ev.add_argument("--models", default="claude-opus-5-5:low,claude-haiku-5-5:low,claude-haiku-5-5:medium",
                    help="model:effort list; the first is the reference")
    ca = sub.add_parser("cancel-all", help="cancel every resting order")
    ca.add_argument("--live", action="store_true")
    args = p.parse_args()
    if args.cmd == "service":
        cmd_service(args)
        return
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
