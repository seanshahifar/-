"""Exchange access through ccxt (one API for 100+ exchanges) and OHLCV history handling."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from pathlib import Path

import ccxt
import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro
import numpy as np
import pandas as pd

from btcagent.config import TIMEFRAME_SECONDS, ConfigError, ExchangeConfig, Secrets

log = logging.getLogger(__name__)

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
DEFAULT_OHLCV_LIMIT = 300
RETRY_DELAYS = (1.0, 2.0, 5.0, 10.0, 20.0)  # pauses before each retry of a failed page in a long download
LIVE_RETRY_DELAYS = (1.0,)  # the per-minute live sync retries once, then lets the next minute try again
PAGE_INTERVAL_SECONDS = 0.15  # public market data is limited to ~10 requests/s per IP (e.g. Coinbase): stay well under
_now = time.monotonic


def create_exchange(cfg: ExchangeConfig, secrets: Secrets | None = None, *, streaming: bool = False):
    """Build an async ccxt client. ``streaming=True`` returns the websocket-capable (ccxt.pro) class."""
    module = ccxt_pro if streaming else ccxt_async
    exchange_cls = getattr(module, cfg.id, None)
    if exchange_cls is None:
        kind = "websocket" if streaming else "REST"
        raise ConfigError(f"Unknown exchange id {cfg.id!r} (no {kind} support in ccxt)")
    options: dict = {"enableRateLimit": True, "aiohttp_trust_env": True}
    if secrets is not None and secrets.has_exchange_keys:
        options.update(apiKey=secrets.api_key, secret=secrets.api_secret)
        if secrets.api_password:
            options["password"] = secrets.api_password
    exchange = exchange_cls(options)
    if cfg.sandbox:
        exchange.set_sandbox_mode(True)
    return exchange


def epoch_ms(index: pd.DatetimeIndex) -> np.ndarray:
    """Epoch milliseconds of a DatetimeIndex, whatever its internal resolution (pandas 3 defaults to us)."""
    return index.as_unit("ms").asi8


def rows_to_frame(rows: list[list[float]]) -> pd.DataFrame:
    """Convert ccxt OHLCV rows ``[ms, open, high, low, close, volume]`` to a UTC-indexed DataFrame."""
    if not rows:
        return empty_bars()
    frame = pd.DataFrame(rows, columns=["ts", *OHLCV_COLUMNS])
    frame["ts"] = pd.to_datetime(frame["ts"].astype("int64"), unit="ms", utc=True).dt.as_unit("ms")
    frame = frame.set_index("ts").astype("float64")
    return frame[~frame.index.duplicated(keep="last")].sort_index()


def empty_bars() -> pd.DataFrame:
    index = pd.DatetimeIndex([], tz="UTC", name="ts").as_unit("ms")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in OHLCV_COLUMNS}, index=index)


def merge_bars(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old.empty:
        return new.sort_index()
    if new.empty:
        return old.sort_index()
    merged = pd.concat([old, new])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


def drop_incomplete(bars: pd.DataFrame, timeframe_seconds: int, now: float) -> pd.DataFrame:
    """Keep only bars whose period has fully ended by ``now`` (exchanges also return the live bar)."""
    if bars.empty:
        return bars
    close_ms = epoch_ms(bars.index) + timeframe_seconds * 1000
    return bars[close_ms <= now * 1000]


def fill_gaps(bars: pd.DataFrame, timeframe_seconds: int) -> pd.DataFrame:
    """Insert flat zero-volume bars for minutes with no trades so every row is exactly one period apart."""
    if len(bars) < 2:
        return bars
    step = pd.Timedelta(seconds=timeframe_seconds)
    full = pd.date_range(bars.index[0], bars.index[-1], freq=step, name="ts", unit="ms")
    if len(full) == len(bars):
        return bars
    filled = bars.reindex(full)
    missing = filled["close"].isna()
    filled["close"] = filled["close"].ffill()
    for column in ("open", "high", "low"):
        filled.loc[missing, column] = filled.loc[missing, "close"]
    filled["volume"] = filled["volume"].fillna(0.0)
    return filled


def clean_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Drop malformed rows (non-positive or non-finite prices, high < low)."""
    if bars.empty:
        return bars
    prices = bars[["open", "high", "low", "close"]]
    ok = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) & (bars["high"] >= bars["low"])
    ok &= np.isfinite(bars["volume"]) & (bars["volume"] >= 0)
    return bars[ok]


def ohlcv_limit(exchange) -> int:
    features = getattr(exchange, "features", None) or {}
    limit = features.get("spot", {}).get("fetchOHLCV", {}).get("limit")
    return int(limit) if limit else DEFAULT_OHLCV_LIMIT


async def fetch_page(
    exchange, symbol: str, timeframe: str, since: int, limit: int, retry_delays: tuple[float, ...] = RETRY_DELAYS
) -> list[list[float]]:
    """One page of bars, retrying timeouts, rate limits and temporary outages after the given pauses."""
    for delay in retry_delays:
        try:
            return await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except ccxt.NetworkError as exc:
            log.warning("Price history request failed (%s); retrying in %.0fs", type(exc).__name__, delay)
            await asyncio.sleep(delay)
    return await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)


async def fetch_recent_bars(exchange, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    """Most recent ``count`` bars (the last one may still be in progress)."""
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    since = exchange.milliseconds() - count * tf_ms
    return await fetch_ohlcv_history(exchange, symbol, timeframe, since, retry_delays=LIVE_RETRY_DELAYS)


async def fetch_ohlcv_history(
    exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int | None = None,
    progress: Callable[[int, int], None] | None = None,
    retry_delays: tuple[float, ...] = RETRY_DELAYS,
) -> pd.DataFrame:
    """Download OHLCV bars in pages from ``since_ms`` up to ``until_ms`` (default: now)."""
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000
    until_ms = until_ms or exchange.milliseconds()
    limit = ohlcv_limit(exchange)
    cursor = since_ms
    rows: list[list[float]] = []
    last_request: float | None = None
    while cursor < until_ms:
        if PAGE_INTERVAL_SECONDS > 0 and last_request is not None:
            wait = PAGE_INTERVAL_SECONDS - (_now() - last_request)
            if wait > 0:
                await asyncio.sleep(wait)
        last_request = _now()
        batch = await fetch_page(exchange, symbol, timeframe, cursor, limit, retry_delays)
        if not batch:
            cursor += limit * tf_ms  # no trades in this window (or exchange downtime): skip ahead
            continue
        rows.extend(batch)
        last = int(batch[-1][0])
        if last < cursor:  # exchange ignored `since`; nothing more to page through
            break
        cursor = last + tf_ms
        if progress:
            progress(min(cursor, until_ms) - since_ms, until_ms - since_ms)
    frame = rows_to_frame(rows)
    return frame[epoch_ms(frame.index) < until_ms]


def history_path(data_dir: str | Path, exchange_id: str, symbol: str, timeframe: str) -> Path:
    return Path(data_dir) / f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.csv"


def save_bars(bars: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = bars.copy()
    out.index = epoch_ms(out.index)
    out.index.name = "ts"
    out.to_csv(path)


def load_bars(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    rows = frame[["ts", *OHLCV_COLUMNS]].to_numpy().tolist()
    return rows_to_frame(rows)
