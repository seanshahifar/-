import asyncio
import math

import ccxt
import pytest

from btcagent.broker.base import OrderError, OrderStatusUnknown
from btcagent.broker.live import LiveBroker
from btcagent.broker.paper import PaperBroker
from btcagent.domain import Tick

TICK = Tick(ts=0, price=50_000, bid=49_990, ask=50_010, received_at=0)
PRICE = 50_000.0


def test_paper_broker_charges_fees_and_slippage():
    broker = PaperBroker(fee_rate=0.001, slippage_bps=10, clock=lambda: 1.0)
    buy = asyncio.run(broker.buy(1_000, TICK))
    assert buy.price == pytest.approx(50_010 * 1.001)
    assert buy.fee_usd == pytest.approx(1.0)
    assert buy.qty * buy.price + buy.fee_usd == pytest.approx(1_000)
    sell = asyncio.run(broker.sell(buy.qty, TICK))
    assert sell.price == pytest.approx(49_990 * 0.999)
    assert sell.fee_usd == pytest.approx(sell.qty * sell.price * 0.001)


class FakeExchange:
    """A ccxt-like exchange whose balances move when orders fill, with injectable failures."""

    id = "fakex"

    def __init__(self, cost_orders=True, usd=1_000.0, btc=0.05, fee=None, statuses=("closed",), fill_ratio=1.0,
                 step=1e-8, fetch_errors=(), create_error=None, create_executes=True, cancel_error=None,
                 report_filled=True):
        self.has = {"createMarketBuyOrderWithCost": cost_orders}
        self.usd, self.btc = usd, btc
        self.fee = fee
        self.statuses = list(statuses)
        self.fill_ratio = fill_ratio
        self.step = step
        self.fetch_errors = list(fetch_errors)
        self.create_error = create_error
        self.create_executes = create_executes
        self.cancel_error = cancel_error
        self.report_filled = report_filled
        self.balance_reads_allowed = math.inf
        self.calls, self.cancels, self.orders = [], [], {}

    async def load_markets(self):
        return {}

    def market(self, symbol):
        return {"base": "BTC", "quote": "USD", "limits": {"cost": {"min": 5.0}, "amount": {"min": 1e-5}}}

    async def fetch_balance(self):
        if self.balance_reads_allowed <= 0:
            raise ccxt.NetworkError("balance endpoint down")
        self.balance_reads_allowed -= 1
        return {"free": {"USD": self.usd, "BTC": self.btc}, "total": {"USD": self.usd, "BTC": self.btc}}

    def cost_to_precision(self, symbol, cost):
        return f"{math.floor(cost * 100) / 100:.2f}"

    def amount_to_precision(self, symbol, amount):
        steps = math.floor(amount / self.step + 1e-9)
        if steps == 0:
            raise ccxt.InvalidOrder("amount must be greater than minimum amount precision")
        return f"{steps * self.step:.8f}"

    def _place(self, side, amount=None, cost=None):
        filled = (amount if amount is not None else cost / PRICE) * self.fill_ratio
        fee_btc = self.fee["cost"] if self.fee and self.fee.get("currency") == "BTC" and side == "buy" else 0.0
        if side == "buy":
            self.btc += filled - fee_btc
            self.usd -= filled * PRICE
        else:
            self.btc -= filled
            self.usd += filled * PRICE
        order_id = str(len(self.orders) + 1)
        self.orders[order_id] = {"id": order_id, "filled": filled, "average": PRICE, "cost": filled * PRICE,
                                 "fee": self.fee}
        if self.create_error is not None:
            if not self.create_executes:  # undo: the request never reached the matching engine
                self.btc, self.usd = (self.btc - filled + fee_btc, self.usd + filled * PRICE) if side == "buy" else \
                    (self.btc + filled, self.usd - filled * PRICE)
            raise self.create_error
        if not self.report_filled:  # some exchanges answer "closed" before reporting the fill
            return {"id": order_id, "status": "closed", "filled": None}
        return {"id": order_id, "status": None, "filled": None}

    async def create_market_buy_order_with_cost(self, symbol, cost):
        self.calls.append(("buy_cost", cost))
        return self._place("buy", cost=cost)

    async def create_order(self, symbol, type_, side, amount):
        self.calls.append((side, amount))
        return self._place(side, amount=amount)

    async def fetch_order(self, order_id, symbol):
        if self.fetch_errors:
            raise self.fetch_errors.pop(0)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {**self.orders[order_id], "status": status}

    async def cancel_order(self, order_id, symbol):
        self.cancels.append(order_id)
        if self.cancel_error is not None:
            raise self.cancel_error
        self.statuses = ["canceled"]
        return {"id": order_id}


def broker(exchange, timeout=0.2):
    b = LiveBroker(exchange, "BTC/USD", fee_rate=0.004, poll_seconds=0, fill_timeout_seconds=timeout)
    asyncio.run(b.prepare())
    return b


def test_live_buy_by_cost_and_fee_in_quote():
    ex = FakeExchange(fee={"cost": 0.4, "currency": "USD"}, statuses=["open", "closed"])
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert ex.calls == [("buy_cost", 100.0)]
    assert fill.qty == pytest.approx(100 / PRICE) and fill.fee_usd == pytest.approx(0.4)


def test_live_buy_by_amount_when_cost_orders_unsupported():
    ex = FakeExchange(cost_orders=False)
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    side, amount = ex.calls[0]
    assert side == "buy" and amount <= 100 / TICK.ask
    assert fill.fee_usd == pytest.approx(fill.qty * fill.price * 0.004)  # estimated: exchange reported none


def test_fee_charged_in_btc_reduces_the_coins_received():
    ex = FakeExchange(fee={"cost": 0.00001, "currency": "BTC"})
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert fill.qty == pytest.approx(100 / PRICE - 0.00001)
    assert fill.fee_usd == pytest.approx(0.00001 * PRICE)


def test_live_buy_refuses_more_than_free_balance_or_below_minimum():
    with pytest.raises(OrderError, match="insufficient"):
        asyncio.run(broker(FakeExchange(usd=50)).buy(100.0, TICK))
    with pytest.raises(OrderError, match="minimum"):
        asyncio.run(broker(FakeExchange()).buy(1.0, TICK))


def test_live_sell_never_exceeds_free_btc_and_rounds_down():
    ex = FakeExchange(btc=0.0123456789)
    fill = asyncio.run(broker(ex).sell(0.5, TICK))
    assert ex.calls[0] == ("sell", 0.01234567)
    assert fill.qty == pytest.approx(0.01234567)


def test_dust_is_not_sellable_and_raises():
    b = broker(FakeExchange(btc=1e-7))
    assert not b.is_sellable(1e-7, PRICE)  # rounds to zero on this exchange: ccxt raises InvalidOrder
    assert not b.is_sellable(0.00004, PRICE)  # $2 is under the $5 minimum cost
    assert b.is_sellable(0.001, PRICE)
    with pytest.raises(OrderError, match="dust"):
        asyncio.run(b.sell(1.0, TICK))


def test_transient_errors_while_waiting_are_retried():
    ex = FakeExchange(fetch_errors=[ccxt.RequestTimeout("slow"), ccxt.OrderNotFound("not yet visible")])
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert fill.qty == pytest.approx(100 / PRICE)


def test_an_order_that_never_finishes_is_cancelled_and_its_partial_fill_booked():
    ex = FakeExchange(statuses=["open"], fill_ratio=0.5)
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert ex.cancels == ["1"]
    assert fill.qty == pytest.approx(50 / PRICE)


def test_unfilled_order_is_cancelled_and_reported():
    ex = FakeExchange(statuses=["open"], fill_ratio=0.0)
    with pytest.raises(OrderError, match="not filled"):
        asyncio.run(broker(ex).buy(100.0, TICK))
    assert ex.cancels == ["1"]


def test_timeout_on_submit_is_reconciled_from_the_balance():
    ex = FakeExchange(create_error=ccxt.RequestTimeout("timed out"), create_executes=True)
    fill = asyncio.run(broker(ex).buy(100.0, TICK))  # the order did go through
    assert fill.qty == pytest.approx(100 / PRICE) and fill.order_id == "reconciled"

    ex = FakeExchange(create_error=ccxt.RequestTimeout("timed out"), create_executes=False)
    with pytest.raises(OrderError, match="did not execute"):
        asyncio.run(broker(ex).buy(100.0, TICK))


def test_order_lost_after_execution_is_reconciled():
    lost = [ccxt.OrderNotFound("gone")] * 50  # the exchange never shows the order again
    ex = FakeExchange(btc=0.6, fetch_errors=lost, cancel_error=ccxt.OrderNotFound("gone"))
    fill = asyncio.run(broker(ex, timeout=0.05).sell(0.1, TICK))
    assert fill.qty == pytest.approx(0.1)  # booked from the BTC balance drop, never re-sold


def test_rejections_are_definitive():
    ex = FakeExchange(create_error=ccxt.InsufficientFunds("no money"), create_executes=False)
    with pytest.raises(OrderError, match="rejected"):
        asyncio.run(broker(ex).buy(100.0, TICK))


def test_unknown_outcome_when_even_the_balance_is_unreadable():
    ex = FakeExchange(create_error=ccxt.RequestTimeout("timed out"))
    b = broker(ex)
    ex.balance_reads_allowed = 1  # the pre-trade check works, then the exchange goes dark
    with pytest.raises(OrderStatusUnknown):
        asyncio.run(b.buy(100.0, TICK))


def test_closed_status_without_filled_amount_is_fetched():
    ex = FakeExchange(report_filled=False)
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert fill.qty == pytest.approx(100 / PRICE)
