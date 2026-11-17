"""Shared async HTTP client with retry/backoff for public market-data APIs."""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

log = logging.getLogger(__name__)

_RETRY_STATUS = {429, 500, 502, 503, 504}


class Http:
    """`net` (a Connectivity) is told about every request: any HTTP response means the network is
    up; only transport failures (no response at all) count towards going offline."""

    def __init__(self, timeout: float = 20.0, net=None):
        self.net = net
        self._client = httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "polyengine/0.1 (+research bot)"},
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )

    async def get_json(self, url: str, params: dict[str, Any] | None = None, retries: int = 3) -> Any:
        resp = await self._get(url, params, retries)
        return resp.json()

    async def get_text(self, url: str, params: dict[str, Any] | None = None, retries: int = 2) -> str:
        resp = await self._get(url, params, retries)
        return resp.text

    async def _get(self, url: str, params: dict[str, Any] | None, retries: int) -> httpx.Response:
        for attempt in range(retries + 1):
            try:
                resp = await self._client.get(url, params=params)
                if self.net:
                    self.net.success()
                if resp.status_code in _RETRY_STATUS and attempt < retries:
                    delay = float(resp.headers.get("retry-after", 0)) or (2 ** attempt + random.random())
                    log.debug("GET %s -> %s, retrying in %.1fs", url, resp.status_code, delay)
                    await asyncio.sleep(min(delay, 30))
                    continue
                resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.TimeoutException):
                if attempt >= retries or (self.net and not self.net.online):
                    # Only the exchange decides "offline": a flaky news/venue API must not pause trading.
                    if self.net and "polymarket.com" in url:
                        self.net.failure()
                    raise
                await asyncio.sleep(2 ** attempt + random.random())
        raise RuntimeError("unreachable")

    async def aclose(self) -> None:
        await self._client.aclose()
