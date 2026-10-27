"""Feature engineering shared by training, backtesting and live trading.

Every feature at bar ``t`` uses only bars up to and including ``t`` (no look-ahead), and the
same function runs everywhere, so the model sees identical inputs in research and in production.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1", "ret_3", "ret_5", "ret_15", "ret_30", "ret_60",
    "vol_15", "vol_60", "vol_ratio",
    "rsi_14", "macd_hist", "bb_pos", "dist_sma_60",
    "range_pct", "close_pos", "vol_z",
    "hour_sin", "hour_cos",
]
WARMUP_BARS = 120  # bars needed before every feature is defined
LIVE_LOOKBACK_BARS = 500  # bars fetched for live inference (EMAs fully converged)


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low, volume = bars["close"], bars["high"], bars["low"], bars["volume"]
    log_close = np.log(close)
    r1 = log_close.diff()
    f = pd.DataFrame(index=bars.index)

    for n in (1, 3, 5, 15, 30, 60):
        f[f"ret_{n}"] = log_close.diff(n)
    f["vol_15"] = r1.rolling(15).std()
    f["vol_60"] = r1.rolling(60).std()
    f["vol_ratio"] = f["vol_15"] / f["vol_60"]

    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    f["rsi_14"] = (rsi - 50) / 50

    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False, min_periods=9).mean()) / close

    sma20, std20 = close.rolling(20).mean(), close.rolling(20).std()
    f["bb_pos"] = (close - sma20) / (2 * std20)
    f["dist_sma_60"] = close / close.rolling(60).mean() - 1

    bar_range = high - low
    f["range_pct"] = bar_range / close
    f["close_pos"] = ((close - low) / bar_range.where(bar_range > 0)).fillna(0.5)
    log_volume = np.log1p(volume)
    f["vol_z"] = (log_volume - log_volume.rolling(60).mean()) / log_volume.rolling(60).std()

    hours = bars.index.hour + bars.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    return f.replace([np.inf, -np.inf], np.nan)[FEATURE_COLUMNS]


def forward_returns(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Log return from the close of bar ``t`` to the close of bar ``t + horizon`` (the training label)."""
    close = bars["close"]
    return np.log(close.shift(-horizon) / close)


def expected_abs_move(features_row: pd.Series, horizon: int) -> float | None:
    """Typical absolute move over the horizon implied by current volatility (fraction, e.g. 0.002)."""
    vol = features_row.get("vol_60")
    if vol is None or not np.isfinite(vol) or vol <= 0:
        return None
    return float(vol * np.sqrt(horizon) * np.sqrt(2 / np.pi))  # E|N(0, s^2)| = s * sqrt(2/pi)
