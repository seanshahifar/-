"""Live trading through ccxt: real market orders on your exchange account.

Safety properties:
  * buys never exceed the free quote balance on the exchange,
  * sells never exceed the free BTC balance or the amount the agent bought,
  * amounts are rounded *down* to the exchange's precision and checked against its minimums,
  * the actual filled amount, average price and fee reported by the exchange drive the ledger.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

from btcagent.broker.base import Broker, OrderError
from btcagent.domain import Fill, Tick

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"closed", "canceled", "cancelled", "rejected", "expired"}


class LiveBroker(Broker):
    name = "live"
    live = True

    def __init__(
        self,
        exchange,
        symbol: str,
        fee_rate: float,
        clock: Callable[[], float] = time.time,
        fill_timeout_seconds: float = 30.0,
        poll_seconds: float = 0.5,
    ):
        self.exchange = exchange
        self.symbol = symbol
        self.fee_rate = fee_rate  # only used to estimate fees an exchange doesn't report
        self.clock = clock
        self.fill_timeout = fill_timeout_seconds
        self.poll_seconds = poll_seconds
        self.market: dict = {}

    async def prepare(self) -> None:
        await self.exchange.load_markets()
        self.market = self.exchange.market(self.symbol)
        quote_free, base_free = await self.free_balances()
        log.warning(
            "LIVE trading on %s %s. Free balance: %.2f %s, %.8f %s",
            self.exchange.id, self.symbol, quote_free, self.market["quote"], base_free, self.market["base"],
        )

    async def free_balances(self) -> tuple[float, float]:
        balance = await self.exchange.fetch_balance()
        free = balance.get("free", {}) or {}
        return float(free.get(self.market["quote"]) or 0.0), float(free.get(self.market["base"]) or 0.0)

    def _limit(self, kind: str) -> float:
        value = ((self.market.get("limits") or {}).get(kind) or {}).get("min")
        return float(value) if value else 0.0

    async def buy(self, usd: float, tick: Tick) -> Fill:
        quote_free, _ = await self.free_balances()
        if usd > quote_free:
            raise OrderError(f"insufficient {self.market['quote']}: need {usd:.2f}, free {quote_free:.2f}")
        if usd < self._limit("cost"):
            raise OrderError(f"order {usd:.2f} below exchange minimum {self._limit('cost')}")
        if self.exchange.has.get("createMarketBuyOrderWithCost"):
            cost = float(self.exchange.cost_to_precision(self.symbol, usd))
            order = await self.exchange.create_market_buy_order_with_cost(self.symbol, cost)
        else:
            amount = float(self.exchange.amount_to_precision(self.symbol, usd * (1 - self.fee_rate) / tick.buy_price))
            if amount <= 0 or amount < self._limit("amount"):
                raise OrderError(f"buy amount {amount} below exchange minimum {self._limit('amount')}")
            order = await self.exchange.create_order(self.symbol, "market", "buy", amount)
        return self._to_fill("buy", await self._wait_for_fill(order), tick)

    async def sell(self, qty: float, tick: Tick) -> Fill:
        _, base_free = await self.free_balances()
        amount = float(self.exchange.amount_to_precision(self.symbol, min(qty, base_free)))
        if amount <= 0 or amount < self._limit("amount"):
            raise OrderError(f"sell amount {amount} below exchange minimum {self._limit('amount')} (dust)")
        order = await self.exchange.create_order(self.symbol, "market", "sell", amount)
        return self._to_fill("sell", await self._wait_for_fill(order), tick)

    async def _wait_for_fill(self, order: dict) -> dict:
        deadline = self.clock() + self.fill_timeout
        while order.get("status") not in TERMINAL_STATUSES and self.clock() < deadline:
            await asyncio.sleep(self.poll_seconds)
            order = await self.exchange.fetch_order(order["id"], self.symbol)
        if not order.get("filled"):
            raise OrderError(f"order {order.get('id')} not filled (status {order.get('status')})")
        if order.get("status") not in TERMINAL_STATUSES:
            log.warning("Order %s still %s after %.0fs; booking the partial fill", order["id"], order.get("status"),
                        self.fill_timeout)
        return order

    def _to_fill(self, side: str, order: dict, tick: Tick) -> Fill:
        filled = float(order["filled"])
        if order.get("average"):
            price = float(order["average"])
        elif order.get("cost"):
            price = float(order["cost"]) / filled
        else:
            price = tick.price
        fee_usd = fee_base = 0.0
        fees = order.get("fees") or ([order["fee"]] if order.get("fee") else [])
        reported = False
        for fee in fees:
            if not fee or fee.get("cost") is None:
                continue
            reported = True
            cost, currency = float(fee["cost"]), fee.get("currency")
            if currency == self.market["base"]:
                fee_base += cost
                fee_usd += cost * price
            elif currency == self.market["quote"] or currency is None:
                fee_usd += cost
            else:  # paid in another coin (e.g. BNB): estimate its USD value
                fee_usd += filled * price * self.fee_rate
        if not reported:
            fee_usd = filled * price * self.fee_rate
        qty = filled - fee_base if side == "buy" else filled
        log.warning("LIVE %s %.8f %s @ %.2f (fee %.4f USD) order %s", side.upper(), qty, self.market["base"], price,
                    fee_usd, order.get("id"))
        return Fill(side, qty, price, fee_usd, self.clock(), order_id=str(order.get("id", "")))  # type: ignore[arg-type]
