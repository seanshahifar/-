from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btcagent.config import Settings
from btcagent.market.simulation import generate_bars
from btcagent.storage import Journal


class FakeClock:
    def __init__(self, now: float = 1_800_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeModel:
    """Stands in for TrainedModel with a fixed probability."""

    def __init__(self, p_up: float = 0.8, timeframe: str = "1m", horizon_bars: int = 15):
        self.p_up = p_up
        self.timeframe = timeframe
        self.horizon_bars = horizon_bars
        self.metrics = {"auc": 0.6, "calibration_slope": 1.0}

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.p_up)

    def summary(self) -> dict:
        return {"timeframe": self.timeframe, "horizon_bars": self.horizon_bars, "trained_at": "", "data_start": "",
                "data_end": "", "n_samples": 0, "metrics": self.metrics}


def make_settings(*overrides: str) -> Settings:
    return Settings.load(None, overrides=list(overrides))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture(scope="session")
def learnable_bars() -> pd.DataFrame:
    """Synthetic bars with strong lag-1 momentum: a pattern a model *should* find."""
    return generate_bars(12_000, 60, seed=21, momentum=0.2, end=pd.Timestamp("2026-01-10", tz="UTC"))


@pytest.fixture(scope="session")
def noise_bars() -> pd.DataFrame:
    """Synthetic bars with no autocorrelation at all: nothing to learn."""
    return generate_bars(12_000, 60, seed=22, momentum=0.0, end=pd.Timestamp("2026-01-10", tz="UTC"))


@pytest.fixture
def journal_factory():
    """Journals created by a test, closed when it ends."""
    created = []

    def make(path=":memory:") -> Journal:
        journal = Journal(path)
        created.append(journal)
        return journal

    yield make
    for journal in created:
        journal.close()
