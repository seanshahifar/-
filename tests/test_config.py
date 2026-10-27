import pytest

from btcagent.config import LIVE_CONFIRM_PHRASE, ConfigError, Secrets, Settings


def test_defaults_without_a_file_are_paper_mode():
    settings = Settings.load(None)
    assert settings.trading.mode == "paper"
    assert settings.timeframe_seconds == 60
    assert not settings.is_live


def test_repo_config_file_loads():
    settings = Settings.load("config.toml")
    assert settings.exchange.symbol == "BTC/USD"
    assert settings.risk.stop_loss_pct > 0


def test_overrides_parse_toml_literals():
    s = Settings.load(None, overrides=["trading.fee_rate=0.001", "exchange.id=kraken", "risk.max_trades_per_hour=5",
                                       "news.enabled=false"])
    assert s.trading.fee_rate == 0.001
    assert s.exchange.id == "kraken"
    assert s.risk.max_trades_per_hour == 5
    assert s.news.enabled is False


@pytest.mark.parametrize("override", [
    "risk.stop_lose_pct=0.1",  # typo must not silently disable a limit
    "trading.mode=yolo",
    "risk.stop_loss_pct=0",  # a stop loss is mandatory
    "trading.fee_rate='lots'",
    "strategy.exit_threshold=0.5",  # must sit below the entry threshold
    "nosuchsection.key=1",
])
def test_invalid_config_is_rejected(override):
    with pytest.raises(ConfigError):
        Settings.load(None, overrides=[override])


def test_live_mode_is_locked_without_keys_and_confirmation(monkeypatch):
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("BTCAGENT_LIVE_CONFIRM", raising=False)
    settings = Settings.load(None, overrides=["trading.mode=live"])
    with pytest.raises(ConfigError, match="locked"):
        settings.require_live_ready()


def test_live_mode_needs_the_exact_phrase(monkeypatch):
    monkeypatch.setenv("EXCHANGE_API_KEY", "k")
    monkeypatch.setenv("EXCHANGE_API_SECRET", "s")
    monkeypatch.setenv("BTCAGENT_LIVE_CONFIRM", "yes")
    with pytest.raises(ConfigError):
        Settings.load(None, overrides=["trading.mode=live"]).require_live_ready()
    monkeypatch.setenv("BTCAGENT_LIVE_CONFIRM", LIVE_CONFIRM_PHRASE)
    Settings.load(None, overrides=["trading.mode=live"]).require_live_ready()


def test_secrets_never_appear_in_repr():
    secrets = Secrets(api_key="AKIA-very-secret", api_secret="shh")
    assert "very-secret" not in repr(secrets) and "shh" not in repr(secrets)
