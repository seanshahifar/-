import argparse

import pytest

from btcagent.cli import build_parser, load_model
from btcagent.config import ConfigError
from btcagent.model import train_model

from conftest import make_settings


@pytest.fixture(scope="module")
def saved_model(tmp_path_factory, learnable_bars):
    path = tmp_path_factory.mktemp("models") / "m.joblib"
    train_model(learnable_bars.iloc[:4_000], "1m", horizon=1, n_splits=3).save(path)
    return path


def settings_for(path, *overrides):
    return make_settings(f"model.path={path}", *overrides)


def test_a_missing_model_is_not_an_error(tmp_path):
    for auto in ("true", "false"):
        assert load_model(settings_for(tmp_path / "none.joblib", f"model.auto_train={auto}")) is None


def test_a_matching_saved_model_is_loaded(saved_model):
    model = load_model(settings_for(saved_model, "model.horizon_bars=1"))
    assert model is not None and model.horizon_bars == 1


def test_a_model_for_different_settings_is_retrained_when_auto_train_is_on(saved_model):
    assert load_model(settings_for(saved_model, "model.horizon_bars=15", "model.auto_train=true")) is None
    with pytest.raises(ConfigError, match="Retrain"):
        load_model(settings_for(saved_model, "model.horizon_bars=15", "model.auto_train=false"))


def test_an_unreadable_model_file_is_replaced_only_when_auto_train_is_on(tmp_path):
    broken = tmp_path / "broken.joblib"
    broken.write_bytes(b"this is not a model")
    assert load_model(settings_for(broken, "model.auto_train=true")) is None
    with pytest.raises(Exception):
        load_model(settings_for(broken, "model.auto_train=false"))


def test_run_accepts_host_and_port_for_hosting():
    args = build_parser().parse_args(["run", "--host", "0.0.0.0", "--port", "10000"])
    assert isinstance(args, argparse.Namespace) and args.host == "0.0.0.0" and args.port == 10000


def test_config_overrides_can_come_from_the_environment(monkeypatch):
    from btcagent.cli import env_overrides
    from btcagent.config import Settings

    monkeypatch.delenv("BTCAGENT_SET", raising=False)
    assert env_overrides() == []
    monkeypatch.setenv("BTCAGENT_SET", "trading.fee_rate=0.001  risk.max_position_usd=250")
    overrides = env_overrides()
    assert overrides == ["trading.fee_rate=0.001", "risk.max_position_usd=250"]
    settings = Settings.load(None, overrides)
    assert settings.trading.fee_rate == 0.001 and settings.risk.max_position_usd == 250


def test_a_hosted_start_never_logs_the_dashboard_token(monkeypatch, caplog):
    import asyncio
    import socket

    from btcagent.cli import run_agent
    from btcagent.config import Settings

    token = "t0ps3cret-" + "x" * 24
    monkeypatch.setenv("DASHBOARD_TOKEN", token)
    settings = Settings.load(None, ["news.enabled=false"])
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    args = argparse.Namespace(sim=True, fresh=False, flatten_on_exit=False)
    caplog.set_level("INFO")
    with pytest.raises(TimeoutError):  # the agent runs until it is stopped; stop it after the startup logs
        asyncio.run(asyncio.wait_for(run_agent(settings, args, True, "0.0.0.0", port), timeout=8))
    messages = " ".join(record.getMessage() for record in caplog.records)
    assert "Dashboard listening on 0.0.0.0" in messages
    assert token not in messages
