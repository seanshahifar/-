import numpy as np
import pandas as pd
import pytest

from btcagent.features import FEATURE_COLUMNS, WARMUP_BARS, compute_features, expected_abs_move, forward_returns
from btcagent.model import MIN_TRAINING_SAMPLES, TrainedModel, apply_calibration, fit_calibration, train_model


def test_features_have_no_lookahead(learnable_bars):
    full = compute_features(learnable_bars)
    cut = 5_000
    truncated = compute_features(learnable_bars.iloc[:cut])
    # Appending future bars must not change any past feature value.
    pd.testing.assert_frame_equal(full.iloc[:cut], truncated, check_exact=False, rtol=1e-9, atol=1e-12)


def test_features_are_complete_after_warmup(learnable_bars):
    features = compute_features(learnable_bars)
    assert list(features.columns) == FEATURE_COLUMNS
    after = features.iloc[WARMUP_BARS:]
    assert np.isfinite(after.to_numpy()).all()


def test_forward_returns_look_exactly_horizon_bars_ahead(learnable_bars):
    fwd = forward_returns(learnable_bars, 3)
    close = learnable_bars["close"]
    assert fwd.iloc[10] == pytest.approx(np.log(close.iloc[13] / close.iloc[10]))
    assert fwd.iloc[-3:].isna().all()


def test_expected_abs_move_scales_with_horizon():
    row = pd.Series({"vol_60": 0.001})
    assert expected_abs_move(row, 4) == pytest.approx(2 * expected_abs_move(row, 1))
    assert expected_abs_move(pd.Series({"vol_60": np.nan}), 4) is None


def test_model_finds_a_real_pattern(learnable_bars):
    model = train_model(learnable_bars, "1m", horizon=1, n_splits=4)
    assert model.metrics["auc"] > 0.55
    assert model.metrics["calibration_slope"] > 0.5
    assert model.metrics["long_avg_move_bps"] > 0 > model.metrics["short_avg_move_bps"]


def test_model_without_skill_is_calibrated_to_a_coin_flip(noise_bars):
    model = train_model(noise_bars, "1m", horizon=5, n_splits=4)
    p = model.predict_proba(compute_features(noise_bars).iloc[WARMUP_BARS:])
    # Calibration on out-of-sample folds must stop it from claiming confidence it doesn't have.
    assert np.mean(np.abs(p - 0.5) > 0.05) < 0.02
    assert model.metrics["long_coverage"] < 0.05


def test_calibration_never_inverts_a_model():
    rng = np.random.default_rng(0)
    raw = rng.uniform(0.3, 0.7, 5_000)
    y = (rng.uniform(size=5_000) > raw).astype(int)  # anti-predictive
    slope, intercept = fit_calibration(raw, y)
    assert slope == 0.0
    assert np.ptp(apply_calibration(raw, slope, intercept)) < 1e-9


def test_model_roundtrip_and_rejects_foreign_files(tmp_path, learnable_bars):
    model = train_model(learnable_bars.iloc[:4_000], "1m", horizon=1, n_splits=3)
    path = tmp_path / "m.joblib"
    model.save(path)
    loaded = TrainedModel.load(path)
    features = compute_features(learnable_bars.iloc[:4_000]).iloc[-5:]
    assert np.allclose(loaded.predict_proba(features), model.predict_proba(features))

    import joblib
    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        TrainedModel.load(tmp_path / "bad.joblib")


def test_training_requires_enough_data(learnable_bars):
    with pytest.raises(ValueError, match=str(MIN_TRAINING_SAMPLES)):
        train_model(learnable_bars.iloc[:1_000], "1m", horizon=1)
