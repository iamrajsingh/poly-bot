"""Network loss -> pause -> resume, restart safety, single instance, launchd plist."""

import asyncio
import plistlib
import subprocess
from pathlib import Path

import httpx
import pytest

from polyengine.config import Settings
from polyengine.connectivity import Connectivity
from polyengine.engine import Engine
from polyengine.http import Http
from polyengine.service import InstanceLock, build_plist
from polyengine.storage import Store


def settings(tmp_path, **kw):
    return Settings(_env_file=None, db_path=str(tmp_path / "t.db"), kill_switch_file=str(tmp_path / "STOP"), **kw)


# ---------------------------------------------------------------- connectivity state machine
async def test_offline_after_consecutive_failures_and_recovery():
    seen = []
    net = Connectivity(fail_threshold=3, on_change=seen.append)
    net.failure()
    net.failure()
    assert net.online
    net.success()  # any response resets the count
    net.failure()
    net.failure()
    assert net.online
    net.failure()
    assert not net.online and net.offline_since and net.outages == 1
    gen = net.generation
    net.success()
    assert net.online and net.generation == gen + 1 and net.last_outage_seconds is not None
    assert [st["online"] for st in seen] == [False, True]


async def test_watch_probes_until_back():
    answers = iter([False, False, True])

    async def probe():
        return next(answers)

    net = Connectivity(probe=probe)
    net.mark_offline()
    stop = asyncio.Event()
    import polyengine.connectivity as c
    real = c._sleep_or_stop

    async def fast(stop_, seconds):
        await real(stop_, 0.01)

    c._sleep_or_stop = fast
    try:
        task = asyncio.create_task(net.watch(stop))
        await asyncio.wait_for(net.wait_online(stop), timeout=2)
        assert net.online
        stop.set()
        await task
    finally:
        c._sleep_or_stop = real


async def test_only_exchange_failures_count_toward_offline():
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    net = Connectivity(fail_threshold=2)
    http = Http(net=net)
    http._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    for _ in range(3):
        with pytest.raises(httpx.ConnectError):
            await http.get_json("https://api.gdeltproject.org/x", retries=0)
    assert net.online  # a flaky news API must not pause trading
    for _ in range(2):
        with pytest.raises(httpx.ConnectError):
            await http.get_json("https://clob.polymarket.com/book", retries=0)
    assert not net.online
    await http.aclose()


# ---------------------------------------------------------------- engine loops pause and resume
async def test_loops_pause_offline_and_catch_up_immediately_on_reconnect(tmp_path):
    eng = Engine(settings(tmp_path))
    calls = []

    async def work():
        calls.append(eng.net.online)

    loop = asyncio.create_task(eng._every(3600, work, "test"))  # hourly: only a wake-up can rerun it
    await asyncio.sleep(0.05)
    assert calls == [True]  # first run
    eng.net.mark_offline()
    await asyncio.sleep(1.2)
    assert calls == [True]  # paused
    eng.net.success()  # reconnect -> generation bump wakes the sleeping loop
    await asyncio.sleep(1.2)
    assert calls == [True, True]  # caught up without waiting the hour
    eng._stop.set()
    await asyncio.wait_for(loop, 3)
    await eng.aclose()


async def test_cycle_interrupted_by_network_loss_reruns_after_reconnect(tmp_path):
    eng = Engine(settings(tmp_path))
    runs = []

    async def work():
        runs.append(1)
        if len(runs) == 1:
            eng.net.mark_offline()
            raise httpx.ConnectError("lost")

    loop = asyncio.create_task(eng._every(3600, work, "test"))
    await asyncio.sleep(0.1)
    assert len(runs) == 1
    eng.net.success()
    await asyncio.sleep(0.2)
    assert len(runs) == 2  # reran straight away, not an hour later
    eng._stop.set()
    await asyncio.wait_for(loop, 3)
    await eng.aclose()


# ---------------------------------------------------------------- restart safety
async def test_fair_values_survive_restart(tmp_path):
    s = settings(tmp_path, research_ttl_minutes=60)
    store = Store(s.db_path)
    store.save_fairs({"Y": 0.62, "N": 0.38})
    eng = Engine(s)
    assert eng.fair_by_token == {"Y": 0.62, "N": 0.38}
    await eng.aclose()
    assert Store(s.db_path).load_fairs(max_age_minutes=0) == {}  # too stale to drive exits


def test_transaction_is_all_or_nothing(tmp_path):
    store = Store(str(tmp_path / "t.db"))
    store.put("paper_cash", 100.0)
    with pytest.raises(RuntimeError):
        with store.transaction():
            store.put("paper_cash", 50.0)
            store.ledger("fill", "m", "t", 1, 0.5, 0, -50, 0)
            raise RuntimeError("power cut")
    assert store.get("paper_cash") == 100.0
    assert store.realized_pnl_total() == 0 and not store._x("SELECT * FROM ledger").fetchall()


# ---------------------------------------------------------------- single instance + launchd
def test_instance_lock(tmp_path):
    a, b = InstanceLock(str(tmp_path / "x.db")), InstanceLock(str(tmp_path / "x.db"))
    assert a.acquire()
    assert not b.acquire()
    assert b.holder().isdigit()
    a.release()
    assert b.acquire()
    b.release()


def test_launchd_plist(tmp_path):
    pl = build_plist(Path("/Users/me/poly"), live=False, dashboard=True, port=8642)
    assert pl["ProgramArguments"][:3] == ["/usr/bin/caffeinate", "-i", "-s"]
    assert pl["ProgramArguments"][3].endswith(".venv/bin/polyengine") and "--live" not in pl["ProgramArguments"]
    assert pl["KeepAlive"] == {"SuccessfulExit": False} and pl["RunAtLoad"] is True
    assert "--live" in build_plist(Path("/p"), live=True, dashboard=False, port=1)["ProgramArguments"]
    path = tmp_path / "a.plist"
    path.write_bytes(plistlib.dumps(pl))
    if Path("/usr/bin/plutil").exists():
        assert subprocess.run(["plutil", "-lint", str(path)], capture_output=True).returncode == 0


async def test_idle_heartbeat_detects_outage_without_traffic():
    async def probe():
        return False  # exchange unreachable

    net = Connectivity(fail_threshold=3, probe=probe)
    net.last_success -= 60  # nothing has talked to the exchange for a minute
    import polyengine.connectivity as c
    real = c._sleep_or_stop

    async def fast(stop_, seconds):
        await real(stop_, 0.01)

    c._sleep_or_stop = fast
    stop = asyncio.Event()
    try:
        task = asyncio.create_task(net.watch(stop))
        for _ in range(100):
            if not net.online:
                break
            await asyncio.sleep(0.01)
        assert not net.online
        stop.set()
        await task
    finally:
        c._sleep_or_stop = real


def test_sleep_wake_detection_triggers_catch_up():
    net = Connectivity()
    assert not net.check_wake(wall=1000.0, mono=50.0)
    assert not net.check_wake(wall=1002.0, mono=52.0)  # normal 2s tick
    gen = net.generation
    assert net.check_wake(wall=1002.0 + 3600, mono=54.0)  # an hour of wall time, 2s of monotonic
    assert net.generation == gen + 1 and net.wakes == 1
