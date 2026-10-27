"""Synthetic BTC-like market for offline development and automated tests.

Nothing here is real market data and nothing measured on it says anything about real
profitability. It exists so the whole pipeline (features, training, backtest, live loop,
dashboard) can be exercised without network access or an exchange account.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from btcagent.domain import Tick
from btcagent.market.exchange import OHLCV_COLUMNS
from btcagent.market.feed import MarketFeed

DEFAULT_MOMENTUM = 0.08  # lag-1 autocorrelation of bar returns: a small, learnable edge
DEFAULT_VOL_PER_MINUTE = 0.0009  # ~0.09% per minute, roughly BTC's typical 1m volatility
SPREAD_BPS = 1.0


def generate_bars(
    n_bars: int,
    timeframe_seconds: int = 60,
    end: pd.Timestamp | None = None,
    start_price: float = 65_000.0,
    seed: int | None = 0,
    momentum: float = DEFAULT_MOMENTUM,
    vol_per_minute: float = DEFAULT_VOL_PER_MINUTE,
) -> pd.DataFrame:
    """Simulate ``n_bars`` OHLCV bars ending at ``end`` (bar open time; default: last closed bar)."""
    rng = np.random.default_rng(seed)
    if end is None:
        now = time.time()
        end = pd.Timestamp((math.floor(now / timeframe_seconds) - 1) * timeframe_seconds, unit="s", tz="UTC")
    index = pd.date_range(end=end, periods=n_bars, freq=pd.Timedelta(seconds=timeframe_seconds), name="ts", unit="ms")

    # Volatility clustering: log-volatility follows a slow AR(1) process.
    log_vol = lfilter([1.0], [1.0, -0.995], rng.normal(0, 0.06, n_bars))
    sigma = vol_per_minute * math.sqrt(timeframe_seconds / 60) * np.exp(log_vol)
    shocks = sigma * rng.standard_t(df=4, size=n_bars) / math.sqrt(2.0)  # fat tails, unit variance
    jumps = (rng.random(n_bars) < 0.002) * rng.normal(0, 8, n_bars) * sigma
    returns = lfilter([1.0], [1.0, -momentum], shocks + jumps)

    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.6, (2, n_bars))) * sigma
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    activity = 1 + 25 * np.abs(returns) / np.maximum(sigma, 1e-12) / 10
    volume = rng.lognormal(mean=1.5, sigma=0.5, size=n_bars) * activity * (timeframe_seconds / 60)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index
    )[OHLCV_COLUMNS]


class SimulatedFeed(MarketFeed):
    """Real-time synthetic ticks that continue the same process as :func:`generate_bars`."""

    name = "simulated"
    simulated = True
    source = "simulator"

    def __init__(
        self,
        timeframe_seconds: int = 60,
        history_bars: int = 2_000,
        seed: int | None = None,
        tick_seconds: float = 1.0,
        clock: Callable[[], float] = time.time,
        momentum: float = DEFAULT_MOMENTUM,
        vol_per_minute: float = DEFAULT_VOL_PER_MINUTE,
    ):
        self.tf = timeframe_seconds
        self.tick_seconds = tick_seconds
        self.clock = clock
        self.momentum = momentum
        self.vol_per_bar = vol_per_minute * math.sqrt(timeframe_seconds / 60)
        self.rng = np.random.default_rng(seed)
        now = clock()
        last_closed = pd.Timestamp((math.floor(now / self.tf) - 1) * self.tf, unit="s", tz="UTC")
        self.bars = generate_bars(
            history_bars, timeframe_seconds, end=last_closed, seed=seed, momentum=momentum, vol_per_minute=vol_per_minute
        )
        self.price = float(self.bars["close"].iloc[-1])
        self.last_bar_return = float(np.log(self.bars["close"].iloc[-1] / self.bars["close"].iloc[-2]))
        self._bar_start = last_closed + pd.Timedelta(seconds=self.tf)
        self._current = self._new_bar(self.price)
        self._last_tick_at = now

    def _new_bar(self, price: float) -> dict[str, float]:
        return {"open": price, "high": price, "low": price, "close": price, "volume": 0.0}

    def step(self, now: float) -> Tick:
        """Advance the simulated market to ``now`` and return the new quote."""
        while now >= (self._bar_start + pd.Timedelta(seconds=self.tf)).timestamp():
            self._close_bar()
        dt = max(now - self._last_tick_at, 1e-3)
        self._last_tick_at = now
        frac = min(dt / self.tf, 1.0)
        drift = self.momentum * self.last_bar_return * frac
        noise = self.vol_per_bar * math.sqrt(frac) * self.rng.standard_normal()
        self.price *= math.exp(drift + noise)
        bar = self._current
        bar["high"] = max(bar["high"], self.price)
        bar["low"] = min(bar["low"], self.price)
        bar["close"] = self.price
        bar["volume"] += float(self.rng.lognormal(0.0, 0.5)) * frac * 4
        half_spread = self.price * SPREAD_BPS / 2e4
        return Tick(ts=now, price=self.price, bid=self.price - half_spread, ask=self.price + half_spread, received_at=now)

    def _close_bar(self) -> None:
        bar = self._current
        row = pd.DataFrame([bar], index=pd.DatetimeIndex([self._bar_start], name="ts").as_unit("ms"))
        self.bars = pd.concat([self.bars, row[OHLCV_COLUMNS]]).iloc[-20_000:]
        self.last_bar_return = math.log(bar["close"] / bar["open"]) if bar["open"] > 0 else 0.0
        self._bar_start += pd.Timedelta(seconds=self.tf)
        self._current = self._new_bar(bar["close"])

    async def ticks(self) -> AsyncIterator[Tick]:
        while True:
            await asyncio.sleep(self.tick_seconds)
            yield self.step(self.clock())

    async def fetch_bars(self, count: int) -> pd.DataFrame:
        current = pd.DataFrame([self._current], index=pd.DatetimeIndex([self._bar_start], name="ts").as_unit("ms"))
        return pd.concat([self.bars, current[OHLCV_COLUMNS]]).tail(count)

    async def fetch_history(self, days: float) -> pd.DataFrame:
        return self.bars.tail(int(days * 86_400 / self.tf))
