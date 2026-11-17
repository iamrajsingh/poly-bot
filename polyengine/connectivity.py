"""Network watchdog: pause the engine when the connection drops, resume with a catch-up when it returns.

Every market-data request reports its outcome here. After a few consecutive transport failures
(no response at all - an HTTP error status still means the network is up) the engine is marked
offline: trading loops stop starting new cycles and a prober checks the exchange with backoff.
When it answers again, `generation` increments, which wakes every loop immediately so the
engine catches up (books fills, settles markets that resolved meanwhile, re-analyses).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import httpx

log = logging.getLogger(__name__)

PROBE_URL = "https://clob.polymarket.com/time"


class Connectivity:
    def __init__(self, fail_threshold: int = 3, probe: Callable[[], Awaitable[bool]] | None = None,
                 on_change: Callable[[dict], None] | None = None):
        self.fail_threshold = fail_threshold
        self.online = True
        self.offline_since: float | None = None
        self.generation = 0  # bumps on every recovery; loops compare it to cut their sleep short
        self.outages = 0
        self.last_outage_seconds: float | None = None
        self._failures = 0
        self.last_success = time.monotonic()
        self.wakes = 0
        self._last_wall: float | None = None
        self._last_mono = 0.0
        self._online = asyncio.Event()
        self._online.set()
        self._probe = probe or _http_probe
        self._on_change = on_change

    # ---- reports from request sites -------------------------------------------
    def success(self) -> None:
        self._failures = 0
        self.last_success = time.monotonic()
        if not self.online:
            self._set_online()

    def failure(self) -> None:
        self._failures += 1
        if self.online and self._failures >= self.fail_threshold:
            self._set_offline()

    def mark_offline(self) -> None:
        """Called when something definitive (e.g. startup) shows the exchange is unreachable."""
        if self.online:
            self._set_offline()

    def status(self) -> dict:
        return {"online": self.online, "offline_since": self.offline_since, "outages": self.outages,
                "last_outage_seconds": self.last_outage_seconds, "ts": time.time()}

    # ---- waiting ----------------------------------------------------------------
    async def wait_online(self, stop: asyncio.Event) -> None:
        """Return once online (or when stop is set)."""
        if self.online:
            return
        waiters = [asyncio.create_task(self._online.wait()), asyncio.create_task(stop.wait())]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for w in waiters:
                w.cancel()

    def check_wake(self, wall: float | None = None, mono: float | None = None) -> bool:
        """Detect that the Mac slept: wall-clock time jumped but the monotonic clock (which stops
        during sleep on macOS) didn't. Bumps `generation` so every loop runs a catch-up now."""
        wall = time.time() if wall is None else wall
        mono = time.monotonic() if mono is None else mono
        woke = False
        if self._last_wall is not None:
            slept = (wall - self._last_wall) - (mono - self._last_mono)
            if slept > 30:
                woke = True
                self.wakes += 1
                self.generation += 1
                self.last_success = -1e9  # verify the network right away; it's often still reconnecting
                log.warning("system woke from sleep (%s asleep) - running a catch-up cycle", _fmt(slept))
        self._last_wall, self._last_mono = wall, mono
        return woke

    async def watch(self, stop: asyncio.Event, max_backoff: float = 60.0, idle_probe: float = 15.0) -> None:
        """Heartbeat while online (probe if nothing has succeeded for `idle_probe` seconds, so an
        outage is noticed even when the engine is idle), and while offline probe with exponential
        backoff (5s -> 60s) until the exchange answers."""
        delay = 5.0
        while not stop.is_set():
            self.check_wake()
            if self.online:
                delay = 5.0
                if time.monotonic() - self.last_success >= idle_probe:
                    if await self._probe():
                        self.success()
                    else:
                        self.failure()
                        await _sleep_or_stop(stop, 5.0)
                        continue
                await _sleep_or_stop(stop, 2.0)
                continue
            if await self._probe():
                self.success()
                continue
            log.info("still offline (%s); next check in %.0fs", _fmt(time.time() - (self.offline_since or 0)), delay)
            self._notify()  # keep the status fresh so the dashboard knows the engine is alive, just paused
            await _sleep_or_stop(stop, delay)
            delay = min(max_backoff, delay * 2)

    # ---- transitions --------------------------------------------------------------
    def _set_offline(self) -> None:
        self.online = False
        self.offline_since = time.time()
        self._online.clear()
        self.outages += 1
        log.warning("NETWORK lost - pausing trading loops; will resume automatically when it returns")
        self._notify()

    def _set_online(self) -> None:
        dur = time.time() - (self.offline_since or time.time())
        self.online = True
        self.last_outage_seconds = dur
        self.offline_since = None
        self._failures = 0
        self.generation += 1
        self._online.set()
        log.warning("NETWORK restored after %s - resuming with a catch-up cycle", _fmt(dur))
        self._notify()

    def _notify(self) -> None:
        if self._on_change:
            try:
                self._on_change(self.status())
            except Exception:  # noqa: BLE001 - status reporting must never break the watchdog
                log.debug("connectivity status callback failed", exc_info=True)


async def _http_probe() -> bool:
    try:
        async with httpx.AsyncClient(timeout=5.0) as c:
            r = await c.get(PROBE_URL)
            return r.status_code < 500
    except (httpx.TransportError, httpx.TimeoutException):
        return False


async def _sleep_or_stop(stop: asyncio.Event, seconds: float) -> None:
    try:
        await asyncio.wait_for(stop.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def _fmt(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
