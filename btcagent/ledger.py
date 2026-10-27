"""The agent's own books: the capital it manages, its open position, realised P&L and fees.

Paper and live trading share this ledger, so risk limits mean the same thing in both. In live
mode it also enforces the budget: the agent can only spend its own cash and only sell the BTC
that it bought itself — never coins that were already in your account.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from btcagent.domain import Fill, Position


@dataclass
class TradeRecord:
    ts: float
    side: str
    qty: float
    price: float
    fee_usd: float
    reason: str
    pnl_usd: float | None = None  # realised P&L, set on sells

    def to_dict(self) -> dict:
        return asdict(self)


class Ledger:
    def __init__(self, capital_usd: float):
        self.capital = capital_usd
        self.cash = capital_usd
        self.position: Position | None = None
        self.realized_pnl = 0.0
        self.fees_paid = 0.0
        self.round_trips = 0
        self.wins = 0

    def equity(self, price: float) -> float:
        return self.cash + (self.position.market_value(price) if self.position else 0.0)

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.round_trips if self.round_trips else None

    def apply_buy(self, fill: Fill, reason: str = "") -> TradeRecord:
        cost = fill.qty * fill.price + fill.fee_usd
        self.cash -= cost
        self.fees_paid += fill.fee_usd
        if self.position is None:
            self.position = Position(fill.qty, fill.price, cost, fill.ts, fill.price)
        else:  # scale in: volume-weighted entry price
            pos = self.position
            total_qty = pos.qty + fill.qty
            pos.entry_price = (pos.entry_price * pos.qty + fill.price * fill.qty) / total_qty
            pos.qty, pos.cost_usd = total_qty, pos.cost_usd + cost
        return TradeRecord(fill.ts, "buy", fill.qty, fill.price, fill.fee_usd, reason)

    def apply_sell(self, fill: Fill, reason: str = "") -> TradeRecord:
        pos = self.position
        if pos is None or fill.qty <= 0:
            raise ValueError("sell without an open position")
        qty = min(fill.qty, pos.qty)
        proceeds = qty * fill.price - fill.fee_usd
        cost_share = pos.cost_usd * (qty / pos.qty)
        pnl = proceeds - cost_share
        self.cash += proceeds
        self.fees_paid += fill.fee_usd
        self.realized_pnl += pnl
        remaining = pos.qty - qty
        if remaining <= pos.qty * 1e-9:
            self.position = None
            self.round_trips += 1
            self.wins += pnl > 0
        else:
            pos.qty, pos.cost_usd = remaining, pos.cost_usd - cost_share
        return TradeRecord(fill.ts, "sell", qty, fill.price, fill.fee_usd, reason, pnl_usd=pnl)

    def to_dict(self) -> dict:
        return {
            "capital": self.capital,
            "cash": self.cash,
            "position": self.position.to_dict() if self.position else None,
            "realized_pnl": self.realized_pnl,
            "fees_paid": self.fees_paid,
            "round_trips": self.round_trips,
            "wins": self.wins,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Ledger:
        ledger = cls(data["capital"])
        ledger.cash = data["cash"]
        ledger.position = Position(**data["position"]) if data.get("position") else None
        ledger.realized_pnl = data.get("realized_pnl", 0.0)
        ledger.fees_paid = data.get("fees_paid", 0.0)
        ledger.round_trips = data.get("round_trips", 0)
        ledger.wins = data.get("wins", 0)
        return ledger
