"""SQLite persistence: research, signals, orders, positions, PnL, LLM spend. Survives restarts."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import Forecast, Order, OrderStatus, Position, Side

SCHEMA = """
CREATE TABLE IF NOT EXISTS research (
    market_id TEXT, ts TEXT, prob REAL, confidence REAL, market_price REAL,
    rationale TEXT, key_factors TEXT, sources TEXT, resolution_risks TEXT, model TEXT
);
CREATE INDEX IF NOT EXISTS research_mkt ON research(market_id, ts);
CREATE TABLE IF NOT EXISTS matches (market_id TEXT PRIMARY KEY, ts TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS signals (
    ts TEXT, market_id TEXT, question TEXT, outcome TEXT, fair REAL, price REAL, edge REAL,
    confidence REAL, agreeing INTEGER, taker INTEGER, action TEXT, reason TEXT, quotes TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id TEXT PRIMARY KEY, market_id TEXT, token_id TEXT, outcome TEXT, side TEXT, price REAL,
    size REAL, filled REAL, avg_fill_price REAL, status TEXT, post_only INTEGER, purpose TEXT,
    created_at TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS positions (
    token_id TEXT PRIMARY KEY, market_id TEXT, outcome TEXT, shares REAL, avg_price REAL,
    event_slug TEXT, question TEXT, fair_at_entry REAL, opened_at TEXT, realized_pnl REAL
);
CREATE TABLE IF NOT EXISTS ledger (
    ts TEXT, kind TEXT, market_id TEXT, token_id TEXT, shares REAL, price REAL, fee REAL,
    cash_delta REAL, realized_pnl REAL, note TEXT
);
CREATE TABLE IF NOT EXISTS llm_calls (
    ts TEXT, purpose TEXT, model TEXT, input_tokens INTEGER, output_tokens INTEGER,
    searches INTEGER, usd REAL
);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
-- calibration: every source's probability per market over time, and how markets resolved
CREATE TABLE IF NOT EXISTS snapshots (ts TEXT, market_id TEXT, source TEXT, prob REAL);
CREATE INDEX IF NOT EXISTS snapshots_mkt ON snapshots(market_id);
CREATE TABLE IF NOT EXISTS watch (market_id TEXT PRIMARY KEY, question TEXT, end_date TEXT, checked_at TEXT);
CREATE TABLE IF NOT EXISTS outcomes (market_id TEXT PRIMARY KEY, yes_payout REAL, ts TEXT);
-- dashboard: latest analysis per market, activity feed, equity curve
CREATE TABLE IF NOT EXISTS analyses (market_id TEXT PRIMARY KEY, ts TEXT, stage TEXT, data TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, level TEXT, kind TEXT,
                                   message TEXT);
CREATE TABLE IF NOT EXISTS equity (ts TEXT, cash REAL, positions_value REAL);
-- latest ensemble fair value per outcome token: exit rules need it right after a restart
CREATE TABLE IF NOT EXISTS fair_values (token_id TEXT PRIMARY KEY, fair REAL, ts TEXT);
-- one row per analysis cycle (pipeline counts, duration, spend) and per day per source (learning)
CREATE TABLE IF NOT EXISTS cycles (ts TEXT, duration REAL, markets INTEGER, tradable INTEGER, matched INTEGER,
                                   worth INTEGER, researched INTEGER, learning INTEGER, signals INTEGER,
                                   orders INTEGER, llm_usd REAL);
CREATE TABLE IF NOT EXISTS calib_history (day TEXT, source TEXT, markets INTEGER, brier REAL, skill REAL,
                                          multiplier REAL, recal_a REAL, recal_b REAL, PRIMARY KEY (day, source));
"""

# Columns added after the first release; ALTERed into existing databases on open.
MIGRATIONS = [
    ("orders", "fee_rate", "REAL DEFAULT 0"),
    ("orders", "fee_exponent", "REAL DEFAULT 1"),
    ("positions", "strategy", "TEXT DEFAULT 'directional'"),
    ("llm_calls", "cache_read", "INTEGER DEFAULT 0"),
    ("llm_calls", "cache_write", "INTEGER DEFAULT 0"),
    ("orders", "reason", "TEXT DEFAULT ''"),
    ("ledger", "question", "TEXT DEFAULT ''"),
    ("ledger", "strategy", "TEXT DEFAULT ''"),
    ("ledger", "outcome", "TEXT DEFAULT ''"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class Store:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, col, decl in MIGRATIONS:
            cols = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if col not in cols:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
        self._lock = threading.RLock()  # re-entrant: transaction() holds it across several writes

    @contextmanager
    def transaction(self):
        """All-or-nothing group of writes, so a crash or power cut can't leave e.g. a position
        updated without its cash and ledger entries."""
        with self._lock:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self.db.execute("ROLLBACK")
                raise
            self.db.execute("COMMIT")

    def _x(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.db.execute(sql, args)

    # ---- kv ------------------------------------------------------------------
    def get(self, k: str, default: Any = None) -> Any:
        r = self._x("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return json.loads(r["v"]) if r else default

    def put(self, k: str, v: Any) -> None:
        self._x("INSERT OR REPLACE INTO kv(k, v) VALUES(?, ?)", (k, json.dumps(v)))

    # ---- research -----------------------------------------------------------
    def save_research(self, market_id: str, f: Forecast, market_price: float) -> None:
        self._x("INSERT INTO research VALUES (?,?,?,?,?,?,?,?,?,?)", (
            market_id, f.ts.isoformat(), f.prob, f.confidence, market_price, f.rationale,
            json.dumps(f.key_factors), json.dumps(f.sources), json.dumps(f.resolution_risks), f.model))

    def latest_research(self, market_id: str) -> tuple[Forecast, float] | None:
        r = self._x("SELECT * FROM research WHERE market_id=? ORDER BY ts DESC LIMIT 1", (market_id,)).fetchone()
        if not r:
            return None
        f = Forecast(prob=r["prob"], confidence=r["confidence"], rationale=r["rationale"],
                     key_factors=json.loads(r["key_factors"]), sources=json.loads(r["sources"]),
                     resolution_risks=json.loads(r["resolution_risks"]), model=r["model"],
                     ts=datetime.fromisoformat(r["ts"]))
        return f, r["market_price"]

    # ---- matches ------------------------------------------------------------
    def get_matches(self, market_id: str, max_age_hours: float) -> list[dict] | None:
        r = self._x("SELECT ts, data FROM matches WHERE market_id=?", (market_id,)).fetchone()
        if not r:
            return None
        if datetime.fromisoformat(r["ts"]) < datetime.now(timezone.utc) - timedelta(hours=max_age_hours):
            return None
        return json.loads(r["data"])

    def save_matches(self, market_id: str, data: list[dict]) -> None:
        self._x("INSERT OR REPLACE INTO matches VALUES (?,?,?)", (market_id, _now(), json.dumps(data)))

    # ---- signals ------------------------------------------------------------
    def log_signal(self, market_id: str, question: str, outcome: str, fair: float, price: float,
                   edge: float, conf: float, agreeing: int, taker: bool, action: str, reason: str,
                   quotes: list[dict]) -> None:
        self._x("INSERT INTO signals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            _now(), market_id, question, outcome, fair, price, edge, conf, agreeing, int(taker),
            action, reason, json.dumps(quotes)))

    # ---- orders -------------------------------------------------------------
    def upsert_order(self, o: Order) -> None:
        self._x("INSERT OR REPLACE INTO orders (id, market_id, token_id, outcome, side, price, size, filled, "
                "avg_fill_price, status, post_only, purpose, created_at, updated_at, fee_rate, fee_exponent, reason) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    o.id, o.market_id, o.token_id, o.outcome, o.side.value, o.price, o.size, o.filled,
                    o.avg_fill_price, o.status.value, int(o.post_only), o.purpose, o.created_at.isoformat(),
                    _now(), o.fee_rate, o.fee_exponent, o.reason))

    def open_orders(self) -> list[Order]:
        rows = self._x("SELECT * FROM orders WHERE status IN ('OPEN','PARTIAL')").fetchall()
        return [Order(id=r["id"], market_id=r["market_id"], token_id=r["token_id"], outcome=r["outcome"],
                      side=Side(r["side"]), price=r["price"], size=r["size"], filled=r["filled"],
                      avg_fill_price=r["avg_fill_price"], status=OrderStatus(r["status"]),
                      post_only=bool(r["post_only"]), purpose=r["purpose"],
                      created_at=datetime.fromisoformat(r["created_at"]),
                      fee_rate=r["fee_rate"] or 0.0, fee_exponent=r["fee_exponent"] or 1.0,
                      reason=r["reason"] or "") for r in rows]

    # ---- positions ----------------------------------------------------------
    def save_position(self, p: Position) -> None:
        if p.shares <= 1e-6:
            self._x("DELETE FROM positions WHERE token_id=?", (p.token_id,))
            return
        self._x("INSERT OR REPLACE INTO positions (token_id, market_id, outcome, shares, avg_price, event_slug, "
                "question, fair_at_entry, opened_at, realized_pnl, strategy) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
                    p.token_id, p.market_id, p.outcome, p.shares, p.avg_price, p.event_slug, p.question,
                    p.fair_at_entry, p.opened_at.isoformat(), p.realized_pnl, p.strategy))

    def positions(self) -> list[Position]:
        return [Position(market_id=r["market_id"], token_id=r["token_id"], outcome=r["outcome"],
                         shares=r["shares"], avg_price=r["avg_price"], event_slug=r["event_slug"],
                         question=r["question"], fair_at_entry=r["fair_at_entry"],
                         opened_at=datetime.fromisoformat(r["opened_at"]), realized_pnl=r["realized_pnl"],
                         strategy=r["strategy"] or "directional")
                for r in self._x("SELECT * FROM positions").fetchall()]

    # ---- ledger / pnl -------------------------------------------------------
    def ledger(self, kind: str, market_id: str, token_id: str, shares: float, price: float, fee: float,
               cash_delta: float, realized: float, note: str = "", question: str = "", strategy: str = "",
               outcome: str = "") -> None:
        self._x("INSERT INTO ledger (ts, kind, market_id, token_id, shares, price, fee, cash_delta, realized_pnl, "
                "note, question, strategy, outcome) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    _now(), kind, market_id, token_id, shares, price, fee, cash_delta, realized, note, question,
                    strategy, outcome))

    def realized_pnl_today(self) -> float:
        r = self._x("SELECT COALESCE(SUM(realized_pnl),0) s FROM ledger WHERE substr(ts,1,10)=?",
                    (_today(),)).fetchone()
        return float(r["s"])

    def realized_pnl_total(self) -> float:
        return float(self._x("SELECT COALESCE(SUM(realized_pnl),0) s FROM ledger").fetchone()["s"])

    # ---- llm ----------------------------------------------------------------
    def record_llm_call(self, purpose: str, model: str, tin: int, tout: int, searches: int, usd: float,
                        cache_read: int = 0, cache_write: int = 0) -> None:
        self._x("INSERT INTO llm_calls (ts, purpose, model, input_tokens, output_tokens, searches, usd, cache_read, "
                "cache_write) VALUES (?,?,?,?,?,?,?,?,?)",
                (_now(), purpose, model, tin, tout, searches, usd, cache_read, cache_write))

    def llm_usage_today(self) -> tuple[int, float]:
        r = self._x("SELECT COUNT(*) n, COALESCE(SUM(usd),0) usd FROM llm_calls WHERE substr(ts,1,10)=?",
                    (_today(),)).fetchone()
        return int(r["n"]), float(r["usd"])

    # ---- calibration ----------------------------------------------------------
    def record_snapshot(self, market_id: str, question: str, end_date: datetime | None,
                        probs: dict[str, float]) -> None:
        ts = _now()
        with self._lock:
            self.db.executemany("INSERT INTO snapshots VALUES (?,?,?,?)",
                                [(ts, market_id, src, p) for src, p in probs.items()])
            self.db.execute("INSERT OR IGNORE INTO watch VALUES (?,?,?,NULL)",
                            (market_id, question, end_date.isoformat() if end_date else None))

    def markets_to_check(self, limit: int = 40, recheck_hours: float = 12) -> list[str]:
        """Watched, unresolved markets that are past their end date or not checked recently."""
        now = datetime.now(timezone.utc)
        stale = (now - timedelta(hours=recheck_hours)).isoformat()
        rows = self._x(
            "SELECT w.market_id FROM watch w LEFT JOIN outcomes o ON o.market_id = w.market_id "
            "WHERE o.market_id IS NULL AND (w.end_date IS NULL OR w.end_date <= ? OR w.checked_at IS NULL "
            "OR w.checked_at <= ?) ORDER BY w.end_date LIMIT ?", (now.isoformat(), stale, limit)).fetchall()
        return [r["market_id"] for r in rows]

    def mark_checked(self, market_id: str) -> None:
        self._x("UPDATE watch SET checked_at=? WHERE market_id=?", (_now(), market_id))

    def save_outcome(self, market_id: str, yes_payout: float) -> None:
        self._x("INSERT OR REPLACE INTO outcomes VALUES (?,?,?)", (market_id, yes_payout, _now()))

    def calibration_rows(self) -> list[sqlite3.Row]:
        """(market_id, source, prob, yes_payout) for every snapshot of a resolved market."""
        return self._x("SELECT s.market_id, s.source, s.prob, o.yes_payout FROM snapshots s "
                       "JOIN outcomes o ON o.market_id = s.market_id WHERE o.yes_payout IN (0, 1)").fetchall()

    # ---- dashboard ------------------------------------------------------------
    def save_analysis(self, market_id: str, stage: str, data: dict) -> None:
        self._x("INSERT OR REPLACE INTO analyses VALUES (?,?,?,?)",
                (market_id, _now(), stage, json.dumps(data, default=str)))

    def analyses(self) -> list[dict]:
        rows = self._x("SELECT market_id, ts, stage, data FROM analyses ORDER BY ts DESC").fetchall()
        return [{"market_id": r["market_id"], "ts": r["ts"], "stage": r["stage"], **json.loads(r["data"])}
                for r in rows]

    def analysis(self, market_id: str) -> dict | None:
        r = self._x("SELECT ts, stage, data FROM analyses WHERE market_id=?", (market_id,)).fetchone()
        return {"market_id": market_id, "ts": r["ts"], "stage": r["stage"], **json.loads(r["data"])} if r else None

    def event(self, level: str, kind: str, message: str) -> None:
        cur = self._x("INSERT INTO events (ts, level, kind, message) VALUES (?,?,?,?)",
                      (_now(), level, kind, message))
        if cur.lastrowid and cur.lastrowid % 500 == 0:  # keep the feed bounded
            self._x("DELETE FROM events WHERE id < ?", (cur.lastrowid - 5000,))

    def events(self, limit: int = 200, after_id: int = 0) -> list[dict]:
        rows = self._x("SELECT * FROM events WHERE id > ? ORDER BY id DESC LIMIT ?", (after_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def record_equity(self, cash: float, positions_value: float) -> None:
        self._x("INSERT INTO equity VALUES (?,?,?)", (_now(), cash, positions_value))

    def equity_series(self, max_points: int = 600) -> list[tuple[str, float]]:
        rows = self._x("SELECT ts, cash + positions_value AS eq FROM equity ORDER BY ts").fetchall()
        step = max(1, len(rows) // max_points)
        pts = [(r["ts"], r["eq"]) for r in rows[::step]]
        if rows and (not pts or pts[-1][0] != rows[-1]["ts"]):
            pts.append((rows[-1]["ts"], rows[-1]["eq"]))
        return pts

    def llm_by_purpose_today(self) -> list[dict]:
        rows = self._x("SELECT substr(purpose, 1, instr(purpose, ':') - 1) AS kind, COUNT(*) n, SUM(usd) usd, "
                       "SUM(searches) searches FROM llm_calls WHERE substr(ts,1,10)=? GROUP BY kind",
                       (_today(),)).fetchall()
        return [dict(r) for r in rows]

    # ---- fair values (restart-safe exit inputs) -------------------------------------
    def save_fairs(self, fairs: dict[str, float]) -> None:
        ts = _now()
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO fair_values VALUES (?,?,?)",
                                [(t, f, ts) for t, f in fairs.items()])

    def load_fairs(self, max_age_minutes: float) -> dict[str, float]:
        cutoff = (datetime.now(timezone.utc) - timedelta(minutes=max_age_minutes)).isoformat()
        rows = self._x("SELECT token_id, fair FROM fair_values WHERE ts >= ?", (cutoff,)).fetchall()
        return {r["token_id"]: r["fair"] for r in rows}

    # ---- analytics ----------------------------------------------------------------------
    def record_cycle(self, duration: float, markets: int, tradable: int, matched: int, worth: int,
                     researched: int, learning: int, signals: int, orders: int, llm_usd: float) -> None:
        self._x("INSERT INTO cycles VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            _now(), duration, markets, tradable, matched, worth, researched, learning, signals, orders, llm_usd))

    def cycles(self, limit: int = 300) -> list[dict]:
        rows = self._x("SELECT * FROM cycles ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def save_calib_history(self, day: str, scores) -> None:
        with self._lock:
            self.db.executemany("INSERT OR REPLACE INTO calib_history VALUES (?,?,?,?,?,?,?,?)", [
                (day, sc.source, sc.markets, sc.brier, sc.skill_vs_market, sc.multiplier, sc.recal_a, sc.recal_b)
                for sc in scores])

    def calib_history(self) -> list[dict]:
        return [dict(r) for r in self._x("SELECT * FROM calib_history ORDER BY day").fetchall()]

    def ledger_rows(self, limit: int = 300, market_id: str | None = None) -> list[dict]:
        if market_id:
            rows = self._x("SELECT * FROM ledger WHERE market_id=? ORDER BY ts DESC LIMIT ?", (market_id, limit))
        else:
            rows = self._x("SELECT * FROM ledger ORDER BY ts DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows.fetchall()]

    def pnl_by_day(self, days: int = 60) -> list[dict]:
        rows = self._x("SELECT substr(ts,1,10) day, SUM(realized_pnl) pnl, SUM(fee) fees, COUNT(*) n FROM ledger "
                       "GROUP BY day ORDER BY day DESC LIMIT ?", (days,)).fetchall()
        return [dict(r) for r in reversed(rows)]

    def closed_trades(self) -> list[dict]:
        """Every realizing event (sell or settlement) - the unit for win rate and P&L by strategy."""
        return [dict(r) for r in self._x(
            "SELECT * FROM ledger WHERE (kind='settle' OR shares < 0) ORDER BY ts").fetchall()]

    def llm_daily(self, days: int = 30) -> list[dict]:
        rows = self._x(
            "SELECT substr(ts,1,10) day, substr(purpose,1,instr(purpose,':')-1) kind, model, COUNT(*) n, "
            "SUM(usd) usd, SUM(input_tokens) input, SUM(cache_read) cache_read, SUM(cache_write) cache_write, "
            "SUM(output_tokens) output, SUM(searches) searches FROM llm_calls "
            "WHERE ts >= date('now', ?) GROUP BY day, kind, model ORDER BY day", (f"-{days} day",)).fetchall()
        return [dict(r) for r in rows]

    def research_history(self, market_id: str) -> list[dict]:
        return [dict(r) for r in self._x(
            "SELECT ts, prob, confidence, market_price, model FROM research WHERE market_id=? ORDER BY ts",
            (market_id,)).fetchall()]

    def research_by_day(self, days: int = 30) -> list[dict]:
        return [dict(r) for r in self._x(
            "SELECT substr(ts,1,10) day, COUNT(*) n, COUNT(DISTINCT market_id) markets, "
            "AVG(ABS(prob - market_price)) avg_gap, AVG(confidence) conf FROM research "
            "WHERE ts >= date('now', ?) GROUP BY day ORDER BY day", (f"-{days} day",)).fetchall()]

    def snapshots_for(self, market_id: str) -> list[dict]:
        return [dict(r) for r in self._x(
            "SELECT ts, source, prob FROM snapshots WHERE market_id=? ORDER BY ts", (market_id,)).fetchall()]

    def signals_for(self, market_id: str, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self._x(
            "SELECT ts, outcome, fair, price, edge, confidence, agreeing, taker, action, reason FROM signals "
            "WHERE market_id=? ORDER BY ts DESC LIMIT ?", (market_id, limit)).fetchall()]

    def learning_counts(self) -> dict:
        r = self._x("SELECT (SELECT COUNT(*) FROM watch) watched, (SELECT COUNT(*) FROM outcomes) resolved, "
                    "(SELECT COUNT(*) FROM snapshots) snapshots, (SELECT COUNT(*) FROM research) forecasts, "
                    "(SELECT COUNT(DISTINCT r.market_id) FROM research r JOIN outcomes o "
                    "ON o.market_id = r.market_id) forecasts_resolved").fetchone()
        return dict(r)

    def events_of(self, kinds: tuple[str, ...], limit: int = 100) -> list[dict]:
        q = ",".join("?" * len(kinds))
        return [dict(r) for r in self._x(
            f"SELECT * FROM events WHERE kind IN ({q}) ORDER BY id DESC LIMIT ?", (*kinds, limit)).fetchall()]
