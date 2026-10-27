import asyncio
import math

import pytest

from btcagent.broker.base import OrderError
from btcagent.broker.live import LiveBroker
from btcagent.broker.paper import PaperBroker
from btcagent.domain import Tick

TICK = Tick(ts=0, price=50_000, bid=49_990, ask=50_010, received_at=0)


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
    id = "fakex"

    def __init__(self, cost_orders=True, quote_free=1_000.0, base_free=0.05, fee=None, statuses=("closed",),
                 fill_ratio=1.0):
        self.has = {"createMarketBuyOrderWithCost": cost_orders}
        self.quote_free, self.base_free = quote_free, base_free
        self.fee = fee
        self.statuses = list(statuses)
        self.fill_ratio = fill_ratio
        self.calls = []
        self.orders = {}

    async def load_markets(self):
        return {}

    def market(self, symbol):
        return {"base": "BTC", "quote": "USD", "limits": {"cost": {"min": 5.0}, "amount": {"min": 1e-5}}}

    async def fetch_balance(self):
        return {"free": {"USD": self.quote_free, "BTC": self.base_free}}

    def cost_to_precision(self, symbol, cost):
        return f"{math.floor(cost * 100) / 100:.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{math.floor(amount * 1e8) / 1e8:.8f}"

    def _order(self, side, amount=None, cost=None):
        price = 50_000.0
        filled = (amount if amount is not None else cost / price) * self.fill_ratio
        order = {"id": str(len(self.orders) + 1), "status": "open", "filled": 0.0, "side": side}
        self.orders[order["id"]] = {"final": {"id": order["id"], "filled": filled, "average": price,
                                              "cost": filled * price, "fee": self.fee}}
        return order

    async def create_market_buy_order_with_cost(self, symbol, cost):
        self.calls.append(("buy_cost", cost))
        return self._order("buy", cost=cost)

    async def create_order(self, symbol, type_, side, amount):
        self.calls.append((side, amount))
        return self._order(side, amount=amount)

    async def fetch_order(self, order_id, symbol):
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        return {**self.orders[order_id]["final"], "status": status}


def broker(exchange, **kwargs):
    b = LiveBroker(exchange, "BTC/USD", fee_rate=0.004, poll_seconds=0, **kwargs)
    asyncio.run(b.prepare())
    return b


def test_live_buy_by_cost_and_fee_in_quote():
    ex = FakeExchange(fee={"cost": 0.4, "currency": "USD"}, statuses=["open", "closed"])
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert ex.calls == [("buy_cost", 100.0)]
    assert fill.qty == pytest.approx(100 / 50_000)
    assert fill.fee_usd == pytest.approx(0.4)


def test_live_buy_by_amount_when_cost_orders_unsupported():
    ex = FakeExchange(cost_orders=False)
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    side, amount = ex.calls[0]
    assert side == "buy" and amount <= 100 / TICK.ask
    assert fill.fee_usd == pytest.approx(fill.qty * fill.price * 0.004)  # estimated: exchange reported none


def test_fee_charged_in_btc_reduces_the_coins_received():
    ex = FakeExchange(fee={"cost": 0.00001, "currency": "BTC"})
    fill = asyncio.run(broker(ex).buy(100.0, TICK))
    assert fill.qty == pytest.approx(100 / 50_000 - 0.00001)
    assert fill.fee_usd == pytest.approx(0.00001 * 50_000)


def test_live_buy_refuses_more_than_free_balance_or_below_minimum():
    with pytest.raises(OrderError, match="insufficient"):
        asyncio.run(broker(FakeExchange(quote_free=50)).buy(100.0, TICK))
    with pytest.raises(OrderError, match="minimum"):
        asyncio.run(broker(FakeExchange()).buy(1.0, TICK))


def test_live_sell_never_exceeds_free_btc_and_rounds_down():
    ex = FakeExchange(base_free=0.0123456789)
    fill = asyncio.run(broker(ex).sell(0.5, TICK))
    side, amount = ex.calls[0]
    assert side == "sell" and amount == 0.01234567
    assert fill.qty == pytest.approx(0.01234567)


def test_dust_and_unfilled_orders_raise():
    with pytest.raises(OrderError, match="dust"):
        asyncio.run(broker(FakeExchange(base_free=1e-7)).sell(1.0, TICK))
    ex = FakeExchange(fill_ratio=0.0, statuses=["canceled"])
    with pytest.raises(OrderError, match="not filled"):
        asyncio.run(broker(ex).buy(100.0, TICK))
