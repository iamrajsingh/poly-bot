"""Paper broker: simulates fills against the live Polymarket order book.

Taker (FAK) orders sweep the visible book immediately. Resting maker orders fill when the market
trades through them: the best ask drops to our bid, or a trade prints strictly below our bid
(price priority means our order would have been hit first). This is deliberately conservative -
paper results should be no better than live results.
"""

from __future__ import annotations

import uuid

from ..models import Market, Order, OrderBook, OrderStatus, Side
from ..orderbook import sweep
from ..storage import Store


class PaperBroker:
    name = "paper"

    def __init__(self, store: Store, starting_cash: float):
        self.store = store
        if self.store.get("paper_cash") is None:
            self.store.put("paper_cash", starting_cash)
        self._last_trade_at_place: dict[str, float | None] = {}

    async def cash(self) -> float:
        return float(self.store.get("paper_cash", 0.0))

    def adjust_cash(self, delta: float) -> None:
        self.store.put("paper_cash", float(self.store.get("paper_cash", 0.0)) + delta)

    async def place(self, order: Order, book: OrderBook, market: Market) -> Order:
        order.id = f"paper-{uuid.uuid4().hex[:12]}"
        bb, ba = book.best_bid, book.best_ask
        if order.post_only:
            crosses = (order.side == Side.BUY and ba is not None and order.price >= ba) or \
                      (order.side == Side.SELL and bb is not None and order.price <= bb)
            if crosses:
                order.status = OrderStatus.REJECTED  # same as the exchange's post-only rejection
                return order
            self._last_trade_at_place[order.id] = book.last_trade
            order.status = OrderStatus.OPEN
            return order

        # FAK: fill what's available up to the limit, cancel the rest.
        fill = sweep(book, order.side, shares=order.size, limit=order.price)
        order.filled = fill.shares
        order.avg_fill_price = fill.avg_price
        # FAK remainder is cancelled; any partially filled shares stay recorded in `filled`.
        order.status = OrderStatus.FILLED if fill.complete else OrderStatus.CANCELLED
        return order

    async def sync(self, order: Order, book: OrderBook | None) -> Order:
        if order.status not in (OrderStatus.OPEN, OrderStatus.PARTIAL) or book is None:
            return order
        lt = book.last_trade
        # After a restart we don't know the print at placement; use the current one as the baseline.
        lt0 = self._last_trade_at_place.setdefault(order.id, lt)
        traded_through = lt is not None and lt != lt0 and (
            (order.side == Side.BUY and lt < order.price) or (order.side == Side.SELL and lt > order.price))
        touched = (order.side == Side.BUY and book.best_ask is not None and book.best_ask <= order.price) or \
                  (order.side == Side.SELL and book.best_bid is not None and book.best_bid >= order.price)
        if traded_through or touched:
            prev_cost = order.filled * order.avg_fill_price
            order.avg_fill_price = (prev_cost + order.remaining * order.price) / order.size
            order.filled = order.size
            order.status = OrderStatus.FILLED
        return order

    async def cancel(self, order: Order) -> Order:
        if order.status in (OrderStatus.OPEN, OrderStatus.PARTIAL):
            order.status = OrderStatus.CANCELLED
        self._last_trade_at_place.pop(order.id, None)
        return order

    async def reconcile(self, known_order_ids: set[str]) -> int:
        return 0  # paper orders exist only in the database

    async def cancel_all(self) -> None:
        self._last_trade_at_place.clear()
