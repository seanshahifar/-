import asyncio

import numpy as np
import pandas as pd

from btcagent.market.exchange import (clean_bars, drop_incomplete, epoch_ms, fetch_ohlcv_history, fill_gaps,
                                      load_bars, merge_bars, rows_to_frame, save_bars)
from btcagent.market.simulation import SimulatedFeed, generate_bars

T0 = 1_700_000_000_000  # ms, a whole minute


def rows(n, start=T0, price=100.0):
    return [[start + i * 60_000, price, price + 1, price - 1, price + 0.5, 1.0] for i in range(n)]


def test_rows_to_frame_and_csv_roundtrip(tmp_path):
    frame = rows_to_frame(rows(5))
    assert list(epoch_ms(frame.index)) == [T0 + i * 60_000 for i in range(5)]
    path = tmp_path / "bars.csv"
    save_bars(frame, path)
    loaded = load_bars(path)
    assert (epoch_ms(loaded.index) == epoch_ms(frame.index)).all()
    assert np.allclose(loaded.to_numpy(), frame.to_numpy())


def test_fill_gaps_inserts_flat_zero_volume_bars():
    data = rows(6)
    del data[2:4]
    filled = fill_gaps(rows_to_frame(data), 60)
    assert len(filled) == 6
    gap = filled.iloc[2]
    assert gap["volume"] == 0 and gap["open"] == gap["close"] == filled.iloc[1]["close"]


def test_drop_incomplete_removes_the_live_bar():
    frame = rows_to_frame(rows(3))
    last_open = epoch_ms(frame.index)[-1] / 1000
    assert len(drop_incomplete(frame, 60, last_open + 59)) == 2
    assert len(drop_incomplete(frame, 60, last_open + 60)) == 3


def test_clean_bars_drops_malformed_rows():
    data = rows(4)
    data[1][4] = -5.0  # negative close
    data[2][2] = 50.0  # high below low
    assert len(clean_bars(rows_to_frame(data))) == 2


def test_merge_bars_prefers_newer_rows():
    old = rows_to_frame(rows(3, price=100))
    new = rows_to_frame(rows(2, start=T0 + 120_000, price=200))
    merged = merge_bars(old, new)
    assert len(merged) == 4 and merged["open"].iloc[2] == 200


class PagingExchange:
    """Serves at most `limit` bars per call, like a real exchange, with a gap of missing data."""

    features = {"spot": {"fetchOHLCV": {"limit": 100}}}

    def __init__(self, data):
        self.data = data
        self.calls = 0

    def milliseconds(self):
        return self.data[-1][0] + 60_000

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        return [r for r in self.data if r[0] >= since][:limit]


def test_fetch_history_pages_through_all_bars():
    data = rows(450)
    del data[200:330]  # exchange outage longer than one page
    exchange = PagingExchange(data)
    frame = asyncio.run(fetch_ohlcv_history(exchange, "BTC/USD", "1m", T0))
    assert len(frame) == 320
    assert exchange.calls >= 4


def test_generate_bars_is_valid_ohlcv():
    bars = generate_bars(2_000, 60, seed=3)
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
    assert (bars["volume"] > 0).all()
    assert bars.index.is_monotonic_increasing and bars.index.tz is not None


def test_simulated_feed_closes_bars_in_real_time():
    clock = [1_800_000_030.0]
    feed = SimulatedFeed(60, history_bars=300, seed=4, clock=lambda: clock[0])
    before = len(feed.bars)
    for _ in range(150):  # from 30s into a minute to 3 minutes later: bars close at +60s, +120s, +180s
        clock[0] += 1
        tick = feed.step(clock[0])
    assert len(feed.bars) == before + 3
    assert tick.bid < tick.price < tick.ask
    latest = asyncio.run(feed.fetch_bars(5))
    assert len(latest) == 5 and latest["close"].iloc[-1] == tick.price
    assert isinstance(latest.index, pd.DatetimeIndex)


class FailingRest:
    async def fetch_ticker(self, symbol):
        raise ConnectionError("exchange unreachable")


def test_ccxt_feed_reports_why_prices_stopped(monkeypatch):
    from btcagent.config import ExchangeConfig
    from btcagent.market.feed import CcxtFeed

    feed = CcxtFeed(ExchangeConfig(use_websocket=False), "1m")
    asyncio.run(feed.rest.close())
    feed.rest = FailingRest()

    async def stop_after_first_retry(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", stop_after_first_retry)

    async def consume():
        async for _ in feed.ticks():
            pass

    try:
        asyncio.run(consume())
    except asyncio.CancelledError:
        pass
    assert feed.last_error == "ConnectionError: exchange unreachable"


def test_stalled_websocket_falls_back_to_rest(monkeypatch):
    import btcagent.market.feed as feed_module
    from btcagent.config import ExchangeConfig
    from btcagent.market.feed import CcxtFeed

    class SilentStream:
        async def watch_ticker(self, symbol):
            await asyncio.Event().wait()  # connected, but never sends anything

        async def close(self):
            pass

    class Rest:
        def __init__(self):
            self.calls = 0

        async def fetch_ticker(self, symbol):
            self.calls += 1
            return {"last": 63_000.0, "bid": 62_999.0, "ask": 63_001.0, "timestamp": None}

        async def close(self):
            pass

    monkeypatch.setattr(feed_module, "WEBSOCKET_STALL_SECONDS", 0.05)
    feed = CcxtFeed(ExchangeConfig(), "1m")
    asyncio.run(feed.rest.close())
    asyncio.run(feed.stream.close())
    feed.stream, feed.rest = SilentStream(), Rest()

    async def first_tick():
        async for tick in feed.ticks():
            return tick

    tick = asyncio.run(asyncio.wait_for(first_tick(), timeout=5))
    assert tick.price == 63_000.0 and feed.source == "polling"
