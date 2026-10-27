"""Paper trading: real (or simulated) prices, pretend money, realistic fees and slippage."""

from __future__ import annotations

import time
from collections.abc import Callable

from btcagent.broker.base import Broker, simulate_buy, simulate_sell
from btcagent.domain import Fill, Tick


class PaperBroker(Broker):
    name = "paper"
    live = False

    def __init__(self, fee_rate: float, slippage_bps: float, clock: Callable[[], float] = time.time):
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.clock = clock

    async def buy(self, usd: float, tick: Tick) -> Fill:
        return simulate_buy(usd, tick.buy_price, self.fee_rate, self.slippage_bps, self.clock())

    async def sell(self, qty: float, tick: Tick) -> Fill:
        return simulate_sell(qty, tick.sell_price, self.fee_rate, self.slippage_bps, self.clock())
