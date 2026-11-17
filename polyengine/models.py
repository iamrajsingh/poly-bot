"""Plain data types shared across the engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def clamp(p: float, lo: float = 0.001, hi: float = 0.999) -> float:
    return max(lo, min(hi, p))


def logit(p: float) -> float:
    p = clamp(p)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Market:
    """A binary Polymarket market. `yes_token`/`no_token` are CLOB token ids."""

    id: str
    condition_id: str
    question: str
    description: str
    slug: str
    event_slug: str
    event_title: str
    end_date: datetime | None
    yes_token: str
    no_token: str
    yes_price: float
    liquidity: float
    volume_24h: float
    tick_size: float = 0.01
    min_order_size: float = 5.0
    neg_risk: bool = False
    category: str = ""
    accepting_orders: bool = True
    outcomes: tuple[str, str] = ("Yes", "No")  # "YES" in this codebase always means outcomes[0]
    fee_rate: float | None = None  # taker fee rate from the market's feeSchedule; None = unknown
    fee_exponent: float = 1.0

    @property
    def hours_to_resolution(self) -> float | None:
        if not self.end_date:
            return None
        return (self.end_date - utcnow()).total_seconds() / 3600

    @property
    def is_yes_no(self) -> bool:
        return [o.lower() for o in self.outcomes] == ["yes", "no"]

    @property
    def url(self) -> str:
        return f"https://polymarket.com/event/{self.event_slug or self.slug}"


@dataclass
class Level:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: list[Level]  # sorted best (highest) first
    asks: list[Level]  # sorted best (lowest) first
    tick_size: float = 0.01
    last_trade: float | None = None
    ts: datetime = field(default_factory=utcnow)

    @property
    def best_bid(self) -> float | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return self.asks[0].price if self.asks else None


@dataclass
class VenueQuote:
    """A probability for the *same proposition* as a Polymarket market, from some source."""

    venue: str  # polymarket | kalshi | manifold | llm
    prob: float  # probability of the Polymarket YES outcome (already polarity-corrected)
    weight: float
    ref: str = ""  # ticker / url / id
    title: str = ""
    liquidity: float = 0.0
    note: str = ""


@dataclass
class Forecast:
    prob: float
    confidence: float
    rationale: str
    key_factors: list[str]
    sources: list[str]
    resolution_risks: list[str]
    model: str
    ts: datetime = field(default_factory=utcnow)


@dataclass
class Signal:
    market_id: str
    token_id: str  # token we'd buy
    outcome: str  # YES | NO
    fair: float  # fair prob of the outcome we'd buy
    price: float  # price we'd pay
    edge: float  # net of fees, per $1 share
    confidence: float
    agreeing_sources: int
    quotes: list[VenueQuote]
    use_taker: bool
    reason: str = ""


@dataclass
class Order:
    id: str
    market_id: str
    token_id: str
    outcome: str
    side: Side
    price: float
    size: float  # shares
    filled: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.OPEN
    post_only: bool = True
    created_at: datetime = field(default_factory=utcnow)
    purpose: str = "entry"  # entry | exit | arb
    fee_rate: float = 0.0  # taker fee schedule of the market, captured at placement
    fee_exponent: float = 1.0
    reason: str = ""  # why it was placed (exit reason, signal summary) - kept for the trade history

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled)


@dataclass
class Position:
    market_id: str
    token_id: str
    outcome: str
    shares: float
    avg_price: float
    event_slug: str = ""
    question: str = ""
    fair_at_entry: float = 0.0
    opened_at: datetime = field(default_factory=utcnow)
    realized_pnl: float = 0.0
    strategy: str = "directional"  # directional | arb (arb legs are held to resolution)

    @property
    def cost(self) -> float:
        return self.shares * self.avg_price
