"""Bar-by-bar backtest reusing the live agent's strategy, risk rules, ledger and fill model.

Conservative by construction: decisions happen at bar close, stops are checked against each
bar's low before its high (a bar that touches both counts as a stop-out), gaps fill at the open,
and every fill pays the configured fee, slippage and half the spread. Historical news is not
available, so the news signal is neutral in backtests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from btcagent.broker.base import simulate_buy, simulate_sell
from btcagent.config import TIMEFRAME_SECONDS, Settings
from btcagent.features import WARMUP_BARS, compute_features
from btcagent.ledger import Ledger, TradeRecord
from btcagent.model import TrainedModel
from btcagent.risk import RiskManager
from btcagent.strategy import Strategy

FEE_SENSITIVITY = [0.0, 0.0005, 0.001, 0.0025, 0.004, 0.006]


@dataclass
class BacktestResult:
    metrics: dict
    equity: pd.Series
    trades: list[TradeRecord]


def run_backtest(
    bars: pd.DataFrame,
    model: TrainedModel,
    settings: Settings,
    start: pd.Timestamp | None = None,
    fee_rate: float | None = None,
    spread_bps: float = 1.0,
) -> BacktestResult:
    """Simulate trading ``bars`` from ``start`` (earlier bars only warm up the features)."""
    trading = settings.trading if fee_rate is None else replace(settings.trading, fee_rate=fee_rate)
    strategy = Strategy(settings.strategy, trading)
    risk = RiskManager(settings.risk)
    ledger = Ledger(trading.capital_usd)
    tf = TIMEFRAME_SECONDS[model.timeframe]

    features = compute_features(bars)
    probs = model.predict_proba(features)
    moves = features["vol_60"].to_numpy() * math.sqrt(model.horizon_bars) * math.sqrt(2 / math.pi)
    opens, highs, lows, closes = (bars[c].to_numpy() for c in ("open", "high", "low", "close"))
    starts = bars.index.as_unit("s").asi8.astype(float)
    first = max(WARMUP_BARS, int(bars.index.searchsorted(start)) if start is not None else 0)
    if first >= len(bars):
        raise ValueError("No bars to backtest after warm-up")

    half_spread = spread_bps / 2e4
    trades: list[TradeRecord] = []
    equity = np.empty(len(bars) - first)
    in_market = np.zeros(len(bars) - first, dtype=bool)

    def sell(price: float, ts: float, reason: str) -> None:
        fill = simulate_sell(ledger.position.qty, price * (1 - half_spread), trading.fee_rate, trading.slippage_bps, ts)
        trades.append(ledger.apply_sell(fill, reason))
        risk.record_trade(ts)

    for k, i in enumerate(range(first, len(bars))):
        bar_open, bar_close = starts[i], starts[i] + tf

        # Intrabar: a position held into this bar can hit its stop (checked first) or its target.
        position = ledger.position
        if position is not None:
            stop, target = risk.stop_price(position), risk.take_profit_price(position)
            if lows[i] <= stop:
                sell(min(opens[i], stop), bar_open, "stop (intrabar)")
            elif target is not None and highs[i] >= target:
                sell(max(opens[i], target), bar_open, "take profit (intrabar)")
            else:
                position.peak_price = max(position.peak_price, highs[i])

        # Bar close: the same decision the live agent makes every few seconds.
        price = closes[i]
        forced = risk.on_equity(ledger.equity(price), bar_close)
        move = moves[i] if np.isfinite(moves[i]) else None
        signal = strategy.signal(float(probs[i]), 0.0, move, spread_bps)
        if ledger.position is not None:
            reason = forced or risk.exit_reason(ledger.position, price, bar_close)
            if reason is None:
                exit_now, reason = strategy.should_exit(signal)
                reason = reason if exit_now else None
            if reason:
                sell(price, bar_close, reason)
        elif forced is None and risk.entry_blocked(bar_close, 0.0, spread_bps) is None:
            enter, reason = strategy.should_enter(signal)
            size = risk.position_size_usd(ledger.equity(price), ledger.cash, strategy.conviction(signal)) if enter else 0
            if size > 0:
                fill = simulate_buy(size, price * (1 + half_spread), trading.fee_rate, trading.slippage_bps, bar_close)
                trades.append(ledger.apply_buy(fill, reason))
                risk.record_trade(bar_close)

        equity[k] = ledger.equity(price)
        in_market[k] = ledger.position is not None

    if ledger.position is not None:  # close at the end so every trade is counted
        sell(closes[-1], starts[-1] + tf, "end of backtest")
        equity[-1] = ledger.equity(closes[-1])

    curve = pd.Series(equity, index=bars.index[first:], name="equity")
    metrics = _metrics(curve, trades, ledger, closes[first - 1], closes[-1], in_market, tf, trading.fee_rate)
    return BacktestResult(metrics=metrics, equity=curve, trades=trades)


def _metrics(curve, trades, ledger, first_price, last_price, in_market, tf, fee_rate) -> dict:
    returns = curve.pct_change().dropna()
    bars_per_year = 365 * 24 * 3600 / tf
    sharpe = float(returns.mean() / returns.std() * math.sqrt(bars_per_year)) if returns.std() > 0 else 0.0
    pnls = [t.pnl_usd for t in trades if t.side == "sell" and t.pnl_usd is not None]
    gains, losses = sum(p for p in pnls if p > 0), -sum(p for p in pnls if p < 0)
    return {
        "start": curve.index[0].isoformat(),
        "end": curve.index[-1].isoformat(),
        "fee_rate": fee_rate,
        "final_equity": float(curve.iloc[-1]),
        "total_return": float(curve.iloc[-1] / ledger.capital - 1),
        "buy_and_hold_return": float(last_price / first_price - 1),
        "max_drawdown": float((curve / curve.cummax() - 1).min()),
        "sharpe": sharpe,
        "round_trips": len(pnls),
        "win_rate": (sum(p > 0 for p in pnls) / len(pnls)) if pnls else None,
        "avg_trade_pnl": (sum(pnls) / len(pnls)) if pnls else None,
        "profit_factor": (gains / losses) if losses > 0 else None,
        "fees_paid": ledger.fees_paid,
        "exposure": float(in_market.mean()),
    }


def fee_sensitivity(bars, model, settings, start=None, fee_rates=None) -> list[dict]:
    rates = sorted(set(fee_rates or FEE_SENSITIVITY) | {settings.trading.fee_rate})
    return [run_backtest(bars, model, settings, start=start, fee_rate=rate).metrics for rate in rates]
