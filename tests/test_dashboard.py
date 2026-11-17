import json
import logging
import threading
import urllib.request

from polyengine.config import Settings
from polyengine.dashboard import EventLogHandler, build_state, make_handler
from polyengine.engine import _stage
from polyengine.models import Position
from polyengine.storage import Store


def settings(tmp_path):
    return Settings(_env_file=None, db_path=str(tmp_path / "t.db"), kill_switch_file=str(tmp_path / "STOP"))


def test_stage_from_decision():
    assert _stage("skip: spread 0.050") == "book"
    assert _stage("skip: no independent sources yet") == "venues"
    assert _stage("no trade: no edge") == "edge"
    assert _stage("defer: YES 0.00sh") == "timing"
    assert _stage("blocked: YES 0.00sh") == "risk"
    assert _stage("would-enter: YES 10sh") == "order"
    assert _stage("enter: YES 10sh") == "order"


def test_state_and_http(tmp_path):
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.save_analysis("m1", "edge", {"market": "Will X?", "decision": "no trade: no edge",
                                       "book": {"mid": 0.4, "spread": 0.01},
                                       "levels": {"bids": [[0.39, 100]], "asks": [[0.41, 100]]},
                                       "quotes": [{"venue": "polymarket", "prob": 0.4, "weight": 3}]})
    store.save_position(Position("m1", "N", "NO", 10, 0.55, question="Will X?"))
    store.put("heartbeat", {"ts": 1.0, "mode": "paper", "cash": 900.0, "equity": 905.0})
    store.record_equity(900, 5)
    logging.getLogger("polyengine.test").addHandler(EventLogHandler(store))
    logging.getLogger("polyengine.test").setLevel(logging.INFO)
    logging.getLogger("polyengine.test").info("ORDER paper BUY YES 10 @ 0.4")

    st = build_state(store, s)
    assert st["markets"][0]["question"] == "Will X?" and st["markets"][0]["stage"] == "edge"
    assert st["positions"][0]["mark"] == 0.59  # NO bid = 1 - YES best ask
    assert st["events"][0]["kind"] == "ORDER"
    assert st["cash"] == 900.0 and st["equity"] == 905.0

    from http.server import ThreadingHTTPServer
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store, s))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        assert b"<title>polyengine</title>" in urllib.request.urlopen(base + "/").read()
        state = json.loads(urllib.request.urlopen(base + "/api/state").read())
        assert state["mode"] == "paper"
        detail = json.loads(urllib.request.urlopen(base + "/api/market?id=m1").read())
        assert detail["levels"]["bids"][0] == [0.39, 100]
    finally:
        httpd.shutdown()


def test_analytics_payload(tmp_path):
    from polyengine.dashboard import _exit_class, build_analytics
    s = settings(tmp_path)
    store = Store(s.db_path)
    store.ledger("fill", "m1", "Y", 10, 0.40, 0, -4, 0, "BUY entry: fair 0.5", question="Will X?", strategy="directional",
                 outcome="YES")
    store.ledger("fill", "m1", "Y", -10, 0.55, 0.02, 5.48, 1.48, "SELL exit: take profit +37%", question="Will X?",
                 strategy="directional", outcome="YES")
    store.ledger("settle", "m2", "N", -5, 0.0, 0, 0, -2.0, "resolution", question="Will Z?", strategy="arb", outcome="NO")
    store.record_cycle(12.5, 150, 140, 20, 3, 2, 1, 1, 0, 0.41)
    store.record_equity(1000, 0)
    store.record_equity(990, 5)
    a = build_analytics(store, s)
    t = a["trading"]
    assert t["closed"] == 2 and t["wins"] == 1 and t["losses"] == 1 and t["win_rate"] == 0.5
    assert t["by_reason"]["take profit"]["pnl"] == 1.48 and t["by_reason"]["held to resolution"]["pnl"] == -2.0
    assert t["by_strategy"]["arb"]["n"] == 1 and t["max_drawdown"] < 0
    assert a["cycles"][0]["researched"] == 2 and "anthropic_api_key" not in a["config"]
    assert _exit_class("SELL exit: thesis broken: fair 0.300 < bid 0.400", "fill") == "thesis broken"
