"""Plain data records passed between the agent's components."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class Tick:
    """Latest market quote. ``ts`` is the exchange time, ``received_at`` our local clock."""

    ts: float
    price: float
    bid: float | None = None
    ask: float | None = None
    received_at: float = 0.0

    @property
    def spread_bps(self) -> float:
        if self.bid and self.ask and self.ask >= self.bid > 0:
            mid = (self.ask + self.bid) / 2
            return (self.ask - self.bid) / mid * 1e4
        return 0.0

    @property
    def buy_price(self) -> float:
        return self.ask if self.ask else self.price

    @property
    def sell_price(self) -> float:
        return self.bid if self.bid else self.price


@dataclass(slots=True)
class Fill:
    side: Side
    qty: float  # BTC
    price: float  # average fill price in quote currency
    fee_usd: float
    ts: float
    order_id: str = ""

    @property
    def notional(self) -> float:
        return self.qty * self.price


@dataclass(slots=True)
class Position:
    """BTC the agent itself bought. The agent never sells coins it didn't buy."""

    qty: float
    entry_price: float
    cost_usd: float  # cash spent including the entry fee
    opened_at: float
    peak_price: float

    def market_value(self, price: float) -> float:
        return self.qty * price

    def unrealized_pnl(self, price: float, fee_rate: float) -> float:
        """P&L if closed now at ``price``, net of the estimated exit fee."""
        return self.qty * price * (1 - fee_rate) - self.cost_usd

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class NewsItem:
    id: str
    title: str
    url: str
    source: str
    published_at: float
    summary: str = ""
    sentiment: float = 0.0  # -1 (very bearish) .. +1 (very bullish) for BTC
    relevance: float = 0.0  # 0 .. 1, how much the story is about bitcoin / crypto markets
    impact: float = 0.0  # 0 .. 1, how market-moving the story is likely to be
    scorer: str = "none"  # "lexicon" | "claude"
    rationale: str = ""

    @property
    def contribution(self) -> float:
        """Signed strength of this story's push on the price."""
        return self.sentiment * self.relevance * self.impact

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class Decision:
    ts: float
    action: Literal["buy", "sell", "hold"]
    reason: str
    details: dict = field(default_factory=dict)
