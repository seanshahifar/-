"""Broker interface: turns trade decisions into fills."""

from __future__ import annotations

from abc import ABC, abstractmethod

from btcagent.domain import Fill, Tick


class OrderError(RuntimeError):
    """An order was rejected or did not fill; nothing changed on the exchange."""


class OrderStatusUnknown(RuntimeError):
    """An order may have executed but its outcome couldn't be determined. Trading must halt."""


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

    def is_sellable(self, qty: float, price: float) -> bool:
        """Whether the exchange would accept a sell of ``qty`` (False for dust below its minimums)."""
        return qty > 1e-12

    async def close(self) -> None:
        return None


def simulate_buy(usd: float, reference_price: float, fee_rate: float, slippage_bps: float, ts: float) -> Fill:
    price = reference_price * (1 + slippage_bps / 1e4)
    fee = usd * fee_rate
    return Fill("buy", (usd - fee) / price, price, fee, ts)


def simulate_sell(qty: float, reference_price: float, fee_rate: float, slippage_bps: float, ts: float) -> Fill:
    price = reference_price * (1 - slippage_bps / 1e4)
    return Fill("sell", qty, price, qty * price * fee_rate, ts)
