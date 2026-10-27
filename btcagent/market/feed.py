"""Live price feeds. The agent consumes ``Tick``s and periodically re-syncs OHLCV bars."""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable

import pandas as pd

from btcagent.config import ExchangeConfig
from btcagent.domain import Tick
from btcagent.market.exchange import create_exchange, fetch_ohlcv_history, fetch_recent_bars

log = logging.getLogger(__name__)

WEBSOCKET_RETRY_SECONDS = 60.0


class MarketFeed(ABC):
    name: str = "feed"
    simulated: bool = False
    source: str = "-"  # how prices currently arrive, shown on the dashboard
    last_error: str | None = None  # why prices aren't arriving, if they aren't

    @abstractmethod
    def ticks(self) -> AsyncIterator[Tick]:
        """Yield quotes forever (until cancelled)."""

    @abstractmethod
    async def fetch_bars(self, count: int) -> pd.DataFrame:
        """Return the most recent ``count`` bars; the last one may still be in progress."""

    @abstractmethod
    async def fetch_history(self, days: float) -> pd.DataFrame:
        """Return roughly the last ``days`` of bars (used for background retraining)."""

    async def close(self) -> None:
        return None


class CcxtFeed(MarketFeed):
    """Real exchange prices: websocket stream when available, REST polling as automatic fallback."""

    def __init__(self, cfg: ExchangeConfig, timeframe: str, clock: Callable[[], float] = time.time):
        self.cfg = cfg
        self.timeframe = timeframe
        self.clock = clock
        self.name = cfg.id
        self.rest = create_exchange(cfg)
        self.stream = create_exchange(cfg, streaming=True) if cfg.use_websocket else None
        self._ws_retry_at = 0.0

    async def ticks(self) -> AsyncIterator[Tick]:
        backoff = 1.0
        while True:
            if self.stream is not None and self.clock() >= self._ws_retry_at:
                try:
                    ticker = await self.stream.watch_ticker(self.cfg.symbol)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.warning("Websocket feed error (%s); using REST polling for %ds", exc, WEBSOCKET_RETRY_SECONDS)
                    self._ws_retry_at = self.clock() + WEBSOCKET_RETRY_SECONDS
                else:
                    self.source, self.last_error = "websocket", None
                    if tick := self._to_tick(ticker):
                        yield tick
                    continue
            try:
                ticker = await self.rest.fetch_ticker(self.cfg.symbol)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("REST ticker error: %s (retrying in %.0fs)", exc, backoff)
                self.last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0
            self.source, self.last_error = "polling", None
            if tick := self._to_tick(ticker):
                yield tick
            await asyncio.sleep(self.cfg.poll_interval_seconds)

    def _to_tick(self, ticker: dict) -> Tick | None:
        price = ticker.get("last") or ticker.get("close")
        if not price:
            return None
        now = self.clock()
        ts = ticker.get("timestamp")
        return Tick(
            ts=ts / 1000 if ts else now,
            price=float(price),
            bid=float(ticker["bid"]) if ticker.get("bid") else None,
            ask=float(ticker["ask"]) if ticker.get("ask") else None,
            received_at=now,
        )

    async def fetch_bars(self, count: int) -> pd.DataFrame:
        return await fetch_recent_bars(self.rest, self.cfg.symbol, self.timeframe, count)

    async def fetch_history(self, days: float) -> pd.DataFrame:
        since = self.rest.milliseconds() - int(days * 86_400_000)
        return await fetch_ohlcv_history(self.rest, self.cfg.symbol, self.timeframe, since)

    async def close(self) -> None:
        await self.rest.close()
        if self.stream is not None:
            await self.stream.close()
