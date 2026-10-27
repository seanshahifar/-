import numpy as np
import pandas as pd
import pytest

from btcagent.backtest import fee_sensitivity, run_backtest
from btcagent.market.exchange import rows_to_frame
from btcagent.model import train_model

from conftest import FakeModel, make_settings

PERMISSIVE = ("trading.fee_rate=0", "trading.slippage_bps=0.5", "strategy.min_edge_multiple=0",
              "strategy.entry_threshold=0.02", "strategy.exit_threshold=-0.01", "risk.max_trades_per_hour=1000",
              "risk.min_seconds_between_trades=0", "risk.max_daily_loss_pct=0.5", "risk.max_drawdown_pct=0.5",
              "model.horizon_bars=1")


@pytest.fixture(scope="module")
def trained(learnable_bars):
    test_start = learnable_bars.index[-1] - pd.Timedelta(days=2)
    model = train_model(learnable_bars[learnable_bars.index < test_start], "1m", horizon=1, n_splits=4)
    return model, test_start


def test_backtest_books_balance(learnable_bars, trained):
    model, test_start = trained
    result = run_backtest(learnable_bars, model, make_settings(*PERMISSIVE), start=test_start)
    m = result.metrics
    sells = [t for t in result.trades if t.side == "sell"]
    assert m["round_trips"] == len(sells) > 10
    assert m["final_equity"] == pytest.approx(10_000 + sum(t.pnl_usd for t in sells))
    assert m["fees_paid"] == pytest.approx(sum(t.fee_usd for t in result.trades))
    assert result.equity.index[0] >= test_start
    assert 0 <= m["exposure"] <= 1 and m["max_drawdown"] <= 0


def test_realistic_fees_stop_the_agent_from_churning(learnable_bars, trained):
    model, test_start = trained
    rows = fee_sensitivity(learnable_bars, model, make_settings(*PERMISSIVE[1:], "strategy.min_edge_multiple=1.25"),
                           start=test_start, fee_rates=[0.0, 0.006])
    by_fee = {row["fee_rate"]: row for row in rows}
    assert by_fee[0.006]["round_trips"] == 0  # a 1-minute edge can't pay a 1.2% round trip
    assert by_fee[0.0]["round_trips"] > 0


def gap_down_bars():
    rng = np.random.default_rng(0)
    t0 = 1_800_000_000_000
    price, rows = 100.0, []
    for i in range(300):
        close = price * np.exp(rng.normal(0, 0.0005))
        rows.append([t0 + i * 60_000, price, max(price, close) * 1.0002, min(price, close) * 0.9998, close, 5.0])
        price = close
    gap_open = price * 0.95  # overnight-style gap far through the 1% stop
    rows.append([t0 + 300 * 60_000, gap_open, gap_open * 1.001, gap_open * 0.999, gap_open, 5.0])
    return rows_to_frame(rows), gap_open


def test_stops_fill_at_the_gap_open_not_the_stop_price():
    bars, gap_open = gap_down_bars()
    settings = make_settings(*PERMISSIVE, "risk.stop_loss_pct=0.01", "risk.take_profit_pct=0", "risk.max_hold_minutes=0")
    result = run_backtest(bars, FakeModel(p_up=0.9, horizon_bars=1), settings, start=bars.index[290])
    stop = next(t for t in result.trades if t.reason == "stop (gap)")
    assert stop.price < gap_open  # filled at the open (minus spread/slippage), well below the stop level
    assert stop.pnl_usd < 0


def test_a_gap_above_the_target_takes_profit_at_the_open():
    bars, gap_open = gap_down_bars()
    up = bars.copy()
    last = up.index[-1]
    up.loc[last, ["open", "high", "close"]] = gap_open / 0.95 * 1.05  # opens 5% higher instead
    up.loc[last, "low"] = up.loc[last, "open"] * 0.97  # then dips through the stop within the bar
    settings = make_settings(*PERMISSIVE, "risk.stop_loss_pct=0.01", "risk.take_profit_pct=0.02",
                             "risk.max_hold_minutes=0")
    result = run_backtest(up, FakeModel(p_up=0.9, horizon_bars=1), settings, start=up.index[290])
    assert any(t.reason == "take profit (gap)" and t.pnl_usd > 0 for t in result.trades)
