"""Live trading through ccxt: real market orders on your exchange account.

Safety properties:
  * buys never exceed the free quote balance on the exchange,
  * sells never exceed the free BTC balance or the amount the agent bought,
  * amounts are rounded *down* to the exchange's precision and checked against its minimums,
  * the filled amount, average price and fee reported by the exchange drive the ledger,
  * an order is never forgotten: errors while waiting for it are retried, an order that doesn't
    finish in time is cancelled, and if its outcome still can't be read from the order it is
    inferred from the change in the BTC balance. If even that fails, ``OrderStatusUnknown`` is
    raised and the agent halts trading until a human checks the account.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable

import ccxt

from btcagent.broker.base import Broker, OrderError, OrderStatusUnknown
from btcagent.domain import Fill, Tick

log = logging.getLogger(__name__)

TERMINAL_STATUSES = {"closed", "canceled", "cancelled", "rejected", "expired"}
BALANCE_CHECKS = 4  # reads of the balance when reconciling an unclear order


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
        quote_free, base_free, _ = await self.balances()
        log.warning(
            "LIVE trading on %s %s. Free balance: %.2f %s, %.8f %s",
            self.exchange.id, self.symbol, quote_free, self.market["quote"], base_free, self.market["base"],
        )

    async def balances(self) -> tuple[float, float, float]:
        """(free quote, free BTC, total BTC including any locked in open orders)."""
        balance = await self.exchange.fetch_balance()
        free, total = balance.get("free") or {}, balance.get("total") or {}
        base_free = float(free.get(self.market["base"]) or 0.0)
        return float(free.get(self.market["quote"]) or 0.0), base_free, float(total.get(self.market["base"]) or base_free)

    def _limit(self, kind: str) -> float:
        value = ((self.market.get("limits") or {}).get(kind) or {}).get("min")
        return float(value) if value else 0.0

    def order_amount(self, qty: float) -> float:
        """``qty`` rounded down to the exchange's step size; 0 if it's below the minimum amount."""
        try:
            amount = float(self.exchange.amount_to_precision(self.symbol, qty))
        except ccxt.InvalidOrder:  # rounds to zero
            return 0.0
        return amount if amount >= self._limit("amount") else 0.0

    def is_sellable(self, qty: float, price: float) -> bool:
        amount = self.order_amount(qty)
        return amount > 0 and amount * price >= self._limit("cost")

    # ------------------------------------------------------------------ orders
    async def buy(self, usd: float, tick: Tick) -> Fill:
        quote_free, _, base_before = await self.balances()
        if usd > quote_free:
            raise OrderError(f"insufficient {self.market['quote']}: need {usd:.2f}, free {quote_free:.2f}")
        if usd < self._limit("cost"):
            raise OrderError(f"order {usd:.2f} below exchange minimum {self._limit('cost')}")
        if self.exchange.has.get("createMarketBuyOrderWithCost"):
            cost = float(self.exchange.cost_to_precision(self.symbol, usd))
            submit = self.exchange.create_market_buy_order_with_cost(self.symbol, cost)
        else:
            amount = self.order_amount(usd * (1 - self.fee_rate) / tick.buy_price)
            if amount <= 0:
                raise OrderError(f"buy of {usd:.2f} is below the exchange's minimum amount")
            submit = self.exchange.create_order(self.symbol, "market", "buy", amount)
        return await self._execute("buy", submit, base_before, tick)

    async def sell(self, qty: float, tick: Tick) -> Fill:
        _, base_free, base_before = await self.balances()
        amount = self.order_amount(min(qty, base_free))
        if amount <= 0 or amount * tick.sell_price < self._limit("cost"):
            raise OrderError(f"sell of {min(qty, base_free):.8f} is below the exchange minimum (dust)")
        submit = self.exchange.create_order(self.symbol, "market", "sell", amount)
        return await self._execute("sell", submit, base_before, tick)

    async def _execute(self, side: str, submit, base_before: float, tick: Tick) -> Fill:
        try:
            order = await submit
        except ccxt.NetworkError as exc:  # the request may have reached the exchange anyway
            return await self._reconcile(side, None, base_before, tick, exc)
        except ccxt.BaseError as exc:  # the exchange answered with a rejection: nothing happened
            raise OrderError(f"{side} rejected: {exc}") from exc
        except Exception as exc:
            return await self._reconcile(side, None, base_before, tick, exc)
        try:
            final = await self._wait_for_fill(order)
        except OrderError:
            raise
        except Exception as exc:
            return await self._reconcile(side, order.get("id"), base_before, tick, exc)
        return self._to_fill(side, final, tick)

    async def _wait_for_fill(self, order: dict) -> dict:
        order_id = order["id"]
        deadline = self.clock() + self.fill_timeout
        while not (order.get("status") in TERMINAL_STATUSES and order.get("filled") is not None):
            if self.clock() >= deadline:
                order = await self._cancel_and_fetch(order_id)
                break
            await asyncio.sleep(self.poll_seconds)
            try:
                order = await self.exchange.fetch_order(order_id, self.symbol)
            except (ccxt.NetworkError, ccxt.OrderNotFound) as exc:  # transient, or not visible yet: keep asking
                log.warning("Still waiting for order %s: %s", order_id, exc)
        if order.get("filled") is None:
            raise OrderStatusUnknown(f"order {order_id}: the exchange didn't report a filled amount")
        if not order["filled"]:
            raise OrderError(f"order {order_id} not filled (status {order.get('status')})")
        if order.get("status") not in TERMINAL_STATUSES:
            log.warning("Order %s still %s after cancelling; booking its partial fill", order_id, order.get("status"))
        return order

    async def _cancel_and_fetch(self, order_id: str) -> dict:
        """An order that didn't finish in time is cancelled, so a late fill can't go unrecorded."""
        try:
            await self.exchange.cancel_order(order_id, self.symbol)
        except Exception as exc:  # e.g. it filled in the meantime
            log.warning("Cancelling order %s failed: %s", order_id, exc)
        return await self.exchange.fetch_order(order_id, self.symbol)

    async def _reconcile(self, side: str, order_id: str | None, base_before: float, tick: Tick, cause) -> Fill:
        """The order's outcome is unclear: infer what executed from the change in the BTC balance."""
        log.warning("Outcome of %s order %s unclear (%s); checking the balance", side, order_id or "?", cause)
        if order_id:
            with contextlib.suppress(Exception):
                await self.exchange.cancel_order(order_id, self.symbol)
        read_once = False
        for attempt in range(BALANCE_CHECKS):
            await asyncio.sleep(self.poll_seconds * (attempt + 1))
            try:
                _, _, base_after = await self.balances()
            except Exception as exc:
                log.warning("Balance check failed: %s", exc)
                continue
            read_once = True
            moved = base_after - base_before if side == "buy" else base_before - base_after
            if moved > 1e-12:
                price = tick.buy_price if side == "buy" else tick.sell_price
                log.warning("Booked %s of %.8f BTC from the balance change (order %s)", side, moved, order_id or "?")
                return Fill(side, moved, price, moved * price * self.fee_rate, self.clock(),  # type: ignore[arg-type]
                            order_id=str(order_id or "reconciled"))
        if not read_once:
            raise OrderStatusUnknown(f"{side} order {order_id or '?'} may have executed but the balance is "
                                     f"unreadable ({cause})")
        raise OrderError(f"{side} order did not execute ({cause}); balance unchanged")

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
