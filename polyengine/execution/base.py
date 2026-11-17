from __future__ import annotations

from typing import Protocol

from ..models import Market, Order, OrderBook


class Broker(Protocol):
    """Places and tracks orders. `sync` returns the order with updated `filled`/`avg_fill_price`/`status`."""

    name: str

    async def place(self, order: Order, book: OrderBook, market: Market) -> Order: ...

    async def sync(self, order: Order, book: OrderBook | None) -> Order: ...

    async def cancel(self, order: Order) -> Order: ...

    async def cancel_all(self) -> None: ...

    async def cash(self) -> float: ...

    async def reconcile(self, known_order_ids: set[str]) -> int:
        """Cancel exchange orders the database doesn't know about (left by a crash). Returns count."""

    def adjust_cash(self, delta: float) -> None:
        """Paper brokers track cash locally; live brokers read it from the exchange."""
