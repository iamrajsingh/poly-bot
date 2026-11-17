"""Live broker on the Polymarket CLOB via the official py-clob-client.

Before first use:
  * fund the wallet with USDC on Polygon and set token allowances (trading once through the
    Polymarket UI does this for proxy wallets; EOAs must approve the exchange contracts);
  * set PE_PM_PRIVATE_KEY (+ PE_PM_FUNDER and PE_PM_SIGNATURE_TYPE for proxy wallets);
  * test with tiny sizes (PE_MAX_POSITION_USD=5) before raising limits.
"""

from __future__ import annotations

import asyncio
import logging
import time

from ..config import Settings
from ..models import Market, Order, OrderBook, OrderStatus, Side

log = logging.getLogger(__name__)

_STATUS = {
    "LIVE": OrderStatus.OPEN, "OPEN": OrderStatus.OPEN, "DELAYED": OrderStatus.OPEN,
    "MATCHED": OrderStatus.FILLED, "FILLED": OrderStatus.FILLED,
    "CANCELED": OrderStatus.CANCELLED, "CANCELLED": OrderStatus.CANCELLED,
    "UNMATCHED": OrderStatus.CANCELLED, "EXPIRED": OrderStatus.CANCELLED,
}


def _tick_str(tick: float) -> str:
    for t in ("0.1", "0.01", "0.001", "0.0001"):
        if abs(float(t) - tick) < 1e-9:
            return t
    return "0.01"


class LiveBroker:
    name = "live"

    def __init__(self, s: Settings):
        if not s.pm_private_key:
            raise RuntimeError("PE_PM_PRIVATE_KEY is required for live trading")
        from py_clob_client.client import ClobClient

        self.s = s
        self.client = ClobClient(
            s.pm_clob_host, key=s.pm_private_key, chain_id=s.pm_chain_id,
            signature_type=s.pm_signature_type, funder=s.pm_funder or None,
        )
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    async def cash(self) -> float:
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

        r = await asyncio.to_thread(self.client.get_balance_allowance,
                                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL))
        return float(r.get("balance", 0)) / 1e6  # USDC has 6 decimals

    def adjust_cash(self, delta: float) -> None:
        pass  # exchange is the source of truth

    async def place(self, order: Order, book: OrderBook, market: Market) -> Order:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY, SELL

        args = OrderArgs(token_id=order.token_id, price=round(order.price, 4), size=round(order.size, 2),
                         side=BUY if order.side == Side.BUY else SELL)
        if order.post_only:
            # Dead man's switch: resting orders expire on the exchange by themselves, so if this Mac
            # loses its connection (or sleeps) they can't sit there going stale. The CLOB requires
            # expirations at least 60s out, hence the extra minute.
            args.expiration = int(time.time()) + 60 + self.s.maker_order_ttl_seconds
        opts = PartialCreateOrderOptions(tick_size=_tick_str(book.tick_size or market.tick_size),
                                         neg_risk=market.neg_risk)
        try:
            signed = await asyncio.to_thread(self.client.create_order, args, opts)
            otype = OrderType.GTD if order.post_only else OrderType.FAK
            resp = await asyncio.to_thread(self.client.post_order, signed, otype, order.post_only)
        except Exception as e:  # noqa: BLE001 - surface any exchange error as a rejection
            log.error("order rejected: %s", e)
            order.status = OrderStatus.REJECTED
            return order
        if not resp or not resp.get("success", True) or not resp.get("orderID"):
            log.error("order rejected: %s", resp)
            order.status = OrderStatus.REJECTED
            return order
        order.id = resp["orderID"]
        order.status = OrderStatus.OPEN
        return await self.sync(order, book)

    async def sync(self, order: Order, book: OrderBook | None) -> Order:
        try:
            r = await asyncio.to_thread(self.client.get_order, order.id)
        except Exception as e:  # noqa: BLE001
            log.warning("get_order %s failed: %s", order.id, e)
            return order
        if not r:
            return order
        matched = float(r.get("size_matched") or 0)
        if matched > order.filled:
            # The order endpoint reports matched size, not fill VWAP; the limit price is a
            # conservative (worst-case) estimate of the fill price.
            new = matched - order.filled
            order.avg_fill_price = (order.filled * order.avg_fill_price + new * order.price) / matched
            order.filled = matched
        status = _STATUS.get(str(r.get("status", "")).upper(), order.status)
        if status == OrderStatus.OPEN and 0 < order.filled < order.size:
            status = OrderStatus.PARTIAL
        order.status = status
        return order

    async def cancel(self, order: Order) -> Order:
        try:
            await asyncio.to_thread(self.client.cancel, order.id)
        except Exception as e:  # noqa: BLE001
            log.warning("cancel %s failed: %s", order.id, e)
        return await self.sync(order, None)

    async def reconcile(self, known_order_ids: set[str]) -> int:
        try:
            live_orders = await asyncio.to_thread(self.client.get_orders)
        except Exception as e:  # noqa: BLE001
            log.warning("reconcile: could not list exchange orders: %s", e)
            return 0
        unknown = [o["id"] for o in live_orders or [] if o.get("id") and o["id"] not in known_order_ids]
        for oid in unknown:
            try:
                await asyncio.to_thread(self.client.cancel, oid)
            except Exception as e:  # noqa: BLE001
                log.warning("reconcile: cancel %s failed: %s", oid, e)
        return len(unknown)

    async def cancel_all(self) -> None:
        await asyncio.to_thread(self.client.cancel_all)
