"""Order book analytics: liquidity, pressure, and the true cost of trading a given size."""

from __future__ import annotations

from dataclasses import dataclass

from .models import Level, OrderBook, Side


@dataclass
class BookStats:
    best_bid: float | None
    best_ask: float | None
    mid: float | None
    spread: float | None
    microprice: float | None
    bid_depth: float  # USD within `window` of the mid
    ask_depth: float
    imbalance: float  # -1 (all asks) .. +1 (all bids) near the touch
    last_trade: float | None


def taker_fee_per_share(price: float, fee_rate: float, exponent: float = 1.0) -> float:
    """Polymarket taker fee per share: rate * (p * (1 - p)) ** exponent. Makers pay nothing."""
    return fee_rate * (price * (1 - price)) ** exponent


def stats(book: OrderBook, window: float = 0.05) -> BookStats:
    bb, ba = book.best_bid, book.best_ask
    mid = (bb + ba) / 2 if bb is not None and ba is not None else None
    spread = (ba - bb) if bb is not None and ba is not None else None

    micro = None
    if bb is not None and ba is not None:
        bs, as_ = book.bids[0].size, book.asks[0].size
        # Size-weighted mid: leans toward the side with *less* resting size (where price is headed).
        micro = (bb * as_ + ba * bs) / (bs + as_) if bs + as_ > 0 else mid

    ref = mid if mid is not None else (bb or ba or 0.5)
    bid_depth = sum(l.price * l.size for l in book.bids if l.price >= ref - window)
    ask_depth = sum(l.price * l.size for l in book.asks if l.price <= ref + window)
    tot = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / tot if tot > 0 else 0.0
    return BookStats(bb, ba, mid, spread, micro, bid_depth, ask_depth, imbalance, book.last_trade)


@dataclass
class Fill:
    shares: float
    cost: float  # USD paid (BUY) or received (SELL), before fees
    avg_price: float
    worst_price: float
    slippage: float  # avg_price vs touch, always >= 0
    complete: bool


def sweep(book: OrderBook, side: Side, usd: float | None = None, shares: float | None = None,
          limit: float | None = None) -> Fill:
    """Walk the book to fill `usd` (BUY) or `shares` (SELL/BUY), never past `limit`."""
    levels = book.asks if side == Side.BUY else book.bids
    if not levels:
        return Fill(0, 0, 0, 0, 0, False)
    touch = levels[0].price
    got_sh = got_usd = 0.0
    worst = touch
    for lvl in levels:
        if limit is not None and ((side == Side.BUY and lvl.price > limit) or
                                  (side == Side.SELL and lvl.price < limit)):
            break
        take = lvl.size
        if shares is not None:
            take = min(take, shares - got_sh)
        if usd is not None:
            take = min(take, (usd - got_usd) / lvl.price)
        if take <= 0:
            break
        got_sh += take
        got_usd += take * lvl.price
        worst = lvl.price
        if (shares is not None and got_sh >= shares - 1e-9) or (usd is not None and got_usd >= usd - 1e-9):
            break
    if got_sh == 0:
        return Fill(0, 0, 0, touch, 0, False)
    avg = got_usd / got_sh
    target_hit = (shares is not None and got_sh >= shares - 1e-6) or (usd is not None and got_usd >= usd - 1e-6)
    return Fill(got_sh, got_usd, avg, worst, abs(avg - touch), target_hit)


def no_book_from_yes(book: OrderBook, no_token: str) -> OrderBook:
    """Derive the NO book from the YES book (YES bid at p == NO ask at 1-p). Useful as a fallback."""
    return OrderBook(
        token_id=no_token,
        bids=[Level(round(1 - l.price, 6), l.size) for l in book.asks],
        asks=[Level(round(1 - l.price, 6), l.size) for l in book.bids],
        tick_size=book.tick_size,
        last_trade=(1 - book.last_trade) if book.last_trade is not None else None,
    )
