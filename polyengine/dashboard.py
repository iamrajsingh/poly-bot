"""Local web dashboard: watch the engine analyse markets live.

Read-only view over the engine's SQLite database, served on 127.0.0.1 with the standard library
(no extra dependencies). Run it alongside the engine (`polyengine run --dashboard`) or on its
own against an existing database (`polyengine dashboard`).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from urllib.parse import parse_qs, urlparse

from .calibration import Calibrator
from .config import Settings
from .storage import Store

log = logging.getLogger(__name__)

# First word of an engine log line -> activity-feed kind.
_KINDS = {"RESEARCH", "SIGNAL", "ORDER", "FILL", "EXIT", "ARB", "REPRICE", "SETTLE", "NETWORK", "LLM"}
_PREFIXES = {"analysis cycle": "CYCLE", "discovery:": "SCAN", "calibration": "LEARN", "system woke": "WAKE",
             "polyengine starting": "START", "stop requested": "STOP", "still offline": "NETWORK"}


class EventLogHandler(logging.Handler):
    """Mirrors the engine's INFO+ log lines into the `events` table for the activity feed."""

    def __init__(self, store: Store):
        super().__init__(level=logging.INFO)
        self.store = store

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            first = msg.split(" ", 1)[0].rstrip(":")
            prefixed = next((k for p, k in _PREFIXES.items() if msg.startswith(p)), None)
            if first in _KINDS:
                kind = first
            elif prefixed:
                kind = prefixed
            elif record.levelno >= logging.ERROR:
                kind = "ERROR"
            elif record.levelno >= logging.WARNING:
                kind = "WARN"
            else:
                kind = "INFO"
            self.store.event(record.levelname, kind, msg[:600])
        except Exception:  # noqa: BLE001 - logging must never break the engine
            self.handleError(record)


def _summary(a: dict) -> dict:
    book = a.get("book") or {}
    sig = a.get("signal") or {}
    return {
        "id": a.get("market_id"), "ts": a.get("ts"), "question": a.get("market"), "url": a.get("url"),
        "event": a.get("event"), "stage": a.get("stage"), "decision": a.get("decision"),
        "mid": book.get("mid"), "spread": book.get("spread"), "fair_yes": a.get("fair_yes"),
        "edge": sig.get("edge"), "outcome": sig.get("outcome"), "hours": a.get("hours"),
        "volume_24h": a.get("volume_24h"), "research_score": a.get("research_score"),
        "quotes": [{"venue": q["venue"], "prob": q["prob"]} for q in a.get("quotes", [])],
    }


def _mark(a: dict | None, outcome: str) -> float | None:
    """Best bid for a held outcome, from the latest YES book snapshot."""
    if not a:
        return None
    lv = a.get("levels") or {}
    if outcome == "YES":
        return lv["bids"][0][0] if lv.get("bids") else None
    return round(1 - lv["asks"][0][0], 4) if lv.get("asks") else None


def build_state(store: Store, s: Settings) -> dict:
    hb = store.get("heartbeat") or {}
    analyses = store.analyses()
    by_id = {a["market_id"]: a for a in analyses}
    cash = hb.get("cash", store.get("paper_cash", s.paper_starting_cash))

    positions = []
    for p in store.positions():
        mark = _mark(by_id.get(p.market_id), p.outcome)
        d = asdict(p)
        d.update(mark=mark, upnl=(p.shares * (mark - p.avg_price)) if mark is not None else None)
        positions.append(d)

    calls, usd = store.llm_usage_today()
    cal = Calibrator(s, store)
    scores = [asdict(sc) for sc in cal.recompute(quiet=True, persist=False)]
    return {
        "now": time.time(),
        "mode": hb.get("mode", "paper"),
        "heartbeat": hb.get("ts"),
        "halted": hb.get("halted"),
        "network": store.get("network"),
        "cash": cash,
        "equity": hb.get("equity", cash),
        "starting_cash": s.paper_starting_cash,
        "realized_today": store.realized_pnl_today(),
        "realized_total": store.realized_pnl_total(),
        "equity_series": store.equity_series(),
        "positions": positions,
        "orders": [asdict(o) for o in store.open_orders()],
        "funnel": store.get("funnel"),
        "arb": store.get("arb_last"),
        "markets": [_summary(a) for a in analyses],
        "events": store.events(150),
        "llm": {"calls": calls, "usd": usd, "cap_usd": s.max_llm_usd_per_day,
                "cap_calls": s.max_llm_calls_per_day, "by_purpose": store.llm_by_purpose_today()},
        "calibration": scores,
        "config": {"min_edge": s.min_edge, "min_agreeing_sources": s.min_agreeing_sources,
                   "kelly_fraction": s.kelly_fraction, "max_position_usd": s.max_position_usd,
                   "weights": {"polymarket": s.w_polymarket, "kalshi": s.w_kalshi,
                               "manifold": s.w_manifold, "llm": s.w_llm}},
    }


def _exit_class(note: str, kind: str) -> str:
    """'SELL exit: take profit +31%' -> 'take profit'; settlements -> 'held to resolution'."""
    if kind == "settle":
        return "held to resolution"
    reason = note.split(": ", 1)[1] if ": " in note else note
    if "to resolution" in reason:
        return "near resolution"
    reason = reason.split(":")[0]
    return re.sub(r"[-+]?\d[\d.]*%?", "", reason).strip(" ,") or note


def build_analytics(store: Store, s: Settings) -> dict:
    trades = [t for t in store.closed_trades() if abs(t["realized_pnl"] or 0) > 1e-9]
    group = lambda key: defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})  # noqa: E731
    by_strategy, by_reason = group("s"), group("r")
    for t in trades:
        for bucket, k in ((by_strategy, t["strategy"] or "directional"),
                          (by_reason, _exit_class(t["note"] or "", t["kind"]))):
            bucket[k]["n"] += 1
            bucket[k]["wins"] += t["realized_pnl"] > 0
            bucket[k]["pnl"] += t["realized_pnl"]
    wins = [t["realized_pnl"] for t in trades if t["realized_pnl"] > 0]
    losses = [t["realized_pnl"] for t in trades if t["realized_pnl"] < 0]

    eq = store.equity_series(2000)
    peak, max_dd, dd_series = None, 0.0, []
    for ts, v in eq:
        peak = v if peak is None else max(peak, v)
        dd = (v - peak) / peak if peak else 0.0
        max_dd = min(max_dd, dd)
        dd_series.append((ts, dd))

    cal = Calibrator(s, store)
    scores = cal.recompute(quiet=True, persist=False)
    secrets = {"pm_private_key", "anthropic_api_key", "pm_funder"}
    return {
        "now": time.time(),
        "trading": {
            "closed": len(trades), "wins": len(wins), "losses": len(losses),
            "win_rate": len(wins) / len(trades) if trades else None,
            "avg_win": sum(wins) / len(wins) if wins else None,
            "avg_loss": sum(losses) / len(losses) if losses else None,
            "profit_factor": (sum(wins) / -sum(losses)) if losses and wins else None,
            "realized_total": store.realized_pnl_total(), "fees": sum(t["fee"] or 0 for t in store.ledger_rows(10_000)),
            "max_drawdown": max_dd, "by_strategy": dict(by_strategy), "by_reason": dict(by_reason),
            "pnl_by_day": store.pnl_by_day(), "ledger": store.ledger_rows(300),
        },
        "drawdown": dd_series[-600:],
        "cycles": store.cycles(300),
        "llm": {"daily": store.llm_daily(30), "cap_usd": s.max_llm_usd_per_day},
        "research": {"by_day": store.research_by_day(30)},
        "learning": {
            "scores": [asdict(sc) for sc in scores], "history": store.calib_history(),
            "counts": store.learning_counts(), "feedback": cal.feedback("llm"),
            "params": {"prior_markets": s.calib_prior_markets, "min_markets": s.calib_min_markets,
                       "feedback_min_markets": s.calib_feedback_min_markets,
                       "learning_per_day": s.learning_forecasts_per_day,
                       "learning_horizon_hours": s.learning_horizon_hours, "adaptive": s.calib_adaptive},
        },
        "outages": store.events_of(("NETWORK", "WAKE", "START", "STOP"), 60),
        "problems": store.events_of(("WARN", "ERROR"), 60),
        "config": {k: v for k, v in s.model_dump().items() if k not in secrets},
    }


def make_handler(store: Store, s: Settings):
    page_path = resources.files("polyengine").joinpath("web/index.html")

    db_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:  # silence per-request logging
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj) -> None:
            self._send(200, json.dumps(obj, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            # Request threads share one SQLite connection; an interleaved cursor raises "another row available".
            with db_lock:
                self._get()

        def _get(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path in ("/", "/index.html"):
                    self._send(200, page_path.read_bytes(), "text/html; charset=utf-8")
                elif url.path == "/api/state":
                    self._json(build_state(store, s))
                elif url.path == "/api/analytics":
                    self._json(build_analytics(store, s))
                elif url.path == "/api/market":
                    mid = (parse_qs(url.query).get("id") or [""])[0]
                    a = store.analysis(mid)
                    if not a:
                        self._send(404, b'{"error":"not found"}', "application/json")
                        return
                    a["history"] = {"snapshots": store.snapshots_for(mid), "research": store.research_history(mid),
                                    "signals": store.signals_for(mid), "ledger": store.ledger_rows(100, mid)}
                    self._json(a)
                else:
                    self._send(404, b"not found", "text/plain")
            except Exception as e:  # noqa: BLE001
                log.exception("dashboard request failed")
                self._send(500, json.dumps({"error": str(e)}).encode(), "application/json")

    return Handler


def serve(s: Settings, host: str = "127.0.0.1", port: int = 8642, background: bool = False) -> ThreadingHTTPServer:
    store = Store(s.db_path)
    httpd = ThreadingHTTPServer((host, port), make_handler(store, s))
    log.info("dashboard: http://%s:%d", host, port)
    if background:
        threading.Thread(target=httpd.serve_forever, name="dashboard", daemon=True).start()
    else:
        httpd.serve_forever()
    return httpd
