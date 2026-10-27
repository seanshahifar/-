"""Broker interface: turns trade decisions into fills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from btcagent.domain import Fill, Tick


class OrderError(RuntimeError):
    """An order could not be placed or did not fill; the ledger is left unchanged."""


class Broker(ABC):
    name: str = "broker"
    live: bool = False

    async def prepare(self) -> None:
        """Connect / load markets before trading starts."""

    @abstractmethod
    async def buy(self, usd: float, tick: Tick) -> Fill:
        """Spend about ``usd`` (fees included) on BTC at market."""

    @abstractmethod
    async def sell(self, qty: float, tick: Tick) -> Fill:
        """Sell ``qty`` BTC at market."""

    async def close(self) -> None:
        return None


def simulate_buy(usd: float, reference_price: float, fee_rate: float, slippage_bps: float, ts: float) -> Fill:
    price = reference_price * (1 + slippage_bps / 1e4)
    fee = usd * fee_rate
    return Fill("buy", (usd - fee) / price, price, fee, ts)


def simulate_sell(qty: float, reference_price: float, fee_rate: float, slippage_bps: float, ts: float) -> Fill:
    price = reference_price * (1 - slippage_bps / 1e4)
    return Fill("sell", qty, price, qty * price * fee_rate, ts)
