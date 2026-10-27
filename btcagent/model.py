"""The price model: probability that BTC closes higher ``horizon`` bars from now.

Training uses purged walk-forward validation: every score is measured on data that comes
strictly *after* the data the model was fitted on, with a gap so labels can't leak.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from btcagent.features import FEATURE_COLUMNS, WARMUP_BARS, compute_features, forward_returns

log = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 2_000
CONFIDENT = 0.55  # probability used to report "when the model is confident" statistics
MODEL_FORMAT_VERSION = 1


def make_estimator(n_samples: int, seed: int = 0) -> HistGradientBoostingClassifier:
    # Deliberately small, shallow and heavily regularised: market data is mostly noise.
    return HistGradientBoostingClassifier(
        learning_rate=0.03,
        max_iter=120,
        max_depth=3,
        max_leaf_nodes=8,
        min_samples_leaf=int(np.clip(n_samples * 0.01, 200, 1000)),
        l2_regularization=10.0,
        early_stopping=False,
        random_state=seed,
    )


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_calibration(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Platt scaling on out-of-fold predictions: p = sigmoid(a * logit(raw) + b).

    A model with no out-of-sample skill gets ``a ~ 0`` so its probabilities collapse to the base
    rate and the agent never trades on it. ``a`` is floored at 0 so we never invert a model.
    """
    calibrator = LogisticRegression(C=1.0).fit(_logit(raw).reshape(-1, 1), y)
    slope = max(0.0, float(calibrator.coef_[0][0]))
    intercept = float(calibrator.intercept_[0]) if slope > 0 else float(_logit(np.array([y.mean()]))[0])
    return slope, intercept


def apply_calibration(raw: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(slope * _logit(raw) + intercept)))


@dataclass
class TrainedModel:
    estimator: HistGradientBoostingClassifier
    timeframe: str
    horizon_bars: int
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    trained_at: str = ""
    data_start: str = ""
    data_end: str = ""
    n_samples: int = 0
    metrics: dict = field(default_factory=dict)
    calibration: tuple[float, float] = (1.0, 0.0)
    format_version: int = MODEL_FORMAT_VERSION

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Calibrated P(up) for each row."""
        raw = self.estimator.predict_proba(features[self.feature_columns].to_numpy(dtype=float))[:, 1]
        return apply_calibration(raw, *self.calibration)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> TrainedModel:
        # joblib files can execute code when loaded: only load models you trained yourself.
        model = joblib.load(path)
        if not isinstance(model, cls) or model.format_version != MODEL_FORMAT_VERSION:
            raise ValueError(f"{path} is not a compatible btcagent model; retrain with `btcagent train`")
        return model

    def summary(self) -> dict:
        return {
            "timeframe": self.timeframe,
            "horizon_bars": self.horizon_bars,
            "trained_at": self.trained_at,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "n_samples": self.n_samples,
            "metrics": self.metrics,
        }


def build_dataset(bars: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    features = compute_features(bars)
    forward = forward_returns(bars, horizon)
    data = features.assign(_forward=forward).iloc[WARMUP_BARS:]
    data = data[data["_forward"].notna()]
    labels = (data["_forward"] > 0).astype(int)
    return data[FEATURE_COLUMNS], labels, data["_forward"]


def _fold_metrics(y: pd.Series, p: np.ndarray, forward: pd.Series, base_rate: float) -> dict:
    confident_long = p >= CONFIDENT
    confident_short = p <= 1 - CONFIDENT
    clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan"),
        "accuracy": float(((p > 0.5) == y.to_numpy()).mean()),
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "baseline_log_loss": float(log_loss(y, np.full(len(y), base_rate), labels=[0, 1])),
        "long_coverage": float(confident_long.mean()),
        "long_avg_move_bps": float(forward[confident_long].mean() * 1e4) if confident_long.any() else 0.0,
        "short_coverage": float(confident_short.mean()),
        "short_avg_move_bps": float(forward[confident_short].mean() * 1e4) if confident_short.any() else 0.0,
        "n": int(len(y)),
    }


def train_model(
    bars: pd.DataFrame,
    timeframe: str,
    horizon: int,
    n_splits: int = 5,
    seed: int = 0,
) -> TrainedModel:
    """Fit on all of ``bars`` after reporting out-of-sample walk-forward scores in ``model.metrics``."""
    X, y, forward = build_dataset(bars, horizon)
    if len(X) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"Need at least {MIN_TRAINING_SAMPLES} usable bars to train, got {len(X)}. The exchange may only serve "
            "a short price history; use one with deeper history (coinbase, bitstamp, binanceus)."
        )
    values = X.to_numpy(dtype=float)

    splits = list(TimeSeriesSplit(n_splits=n_splits, gap=horizon).split(values))
    out_of_fold = []
    for train_idx, test_idx in splits:
        estimator = make_estimator(len(train_idx), seed).fit(values[train_idx], y.iloc[train_idx])
        out_of_fold.append(estimator.predict_proba(values[test_idx])[:, 1])
    test_rows = np.concatenate([test_idx for _, test_idx in splits])
    slope, intercept = fit_calibration(np.concatenate(out_of_fold), y.iloc[test_rows].to_numpy())

    folds = []
    for (train_idx, test_idx), raw in zip(splits, out_of_fold):
        p = apply_calibration(raw, slope, intercept)
        folds.append(_fold_metrics(y.iloc[test_idx], p, forward.iloc[test_idx], float(y.iloc[train_idx].mean())))
        log.debug("fold %d: auc=%.4f", len(folds), folds[-1]["auc"])

    def mean(key: str) -> float:
        return float(np.nanmean([fold[key] for fold in folds]))

    metrics = {
        "auc": mean("auc"),
        "auc_min": float(np.nanmin([fold["auc"] for fold in folds])),
        "accuracy": mean("accuracy"),
        "log_loss": mean("log_loss"),
        "baseline_log_loss": mean("baseline_log_loss"),
        "long_coverage": mean("long_coverage"),
        "long_avg_move_bps": mean("long_avg_move_bps"),
        "short_coverage": mean("short_coverage"),
        "short_avg_move_bps": mean("short_avg_move_bps"),
        "typical_move_bps": float(forward.abs().mean() * 1e4),
        "up_rate": float(y.mean()),
        "calibration_slope": slope,
        "folds": folds,
    }
    final = make_estimator(len(values), seed).fit(values, y)
    return TrainedModel(
        estimator=final,
        timeframe=timeframe,
        horizon_bars=horizon,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        data_start=X.index[0].isoformat(),
        data_end=X.index[-1].isoformat(),
        n_samples=int(len(X)),
        metrics=metrics,
        calibration=(slope, intercept),
    )


def describe_metrics(metrics: dict) -> list[str]:
    """Human-readable lines (plus honest warnings) for the CLI."""
    lines = [
        f"Out-of-sample AUC        {metrics['auc']:.4f}  (0.5 = coin flip; worst fold {metrics['auc_min']:.4f})",
        f"Directional accuracy     {metrics['accuracy']:.2%}  (bars that went up: {metrics['up_rate']:.2%})",
        f"Log loss vs. baseline    {metrics['log_loss']:.5f} vs {metrics['baseline_log_loss']:.5f}",
        f"When P(up) >= {CONFIDENT:.2f}       avg move {metrics['long_avg_move_bps']:+.2f} bps "
        f"on {metrics['long_coverage']:.1%} of bars",
        f"When P(up) <= {1 - CONFIDENT:.2f}       avg move {metrics['short_avg_move_bps']:+.2f} bps "
        f"on {metrics['short_coverage']:.1%} of bars",
        f"Typical move over horizon {metrics['typical_move_bps']:.1f} bps",
    ]
    lines.append(f"Calibration slope        {metrics['calibration_slope']:.3f}  (0 = no usable skill, ~1 = trusted as-is)")
    if metrics["auc"] < 0.52:
        lines.append("WARNING: the model is barely better than a coin flip on unseen data.")
    if metrics["log_loss"] >= metrics["baseline_log_loss"]:
        lines.append("WARNING: probabilities are no better calibrated than always predicting the base rate.")
    return lines
