"""Settings: non-secret options from ``config.toml``, secrets from environment variables.

Every option has a safe default, so the agent runs (in paper mode) with no config
file at all. Unknown keys are rejected so a typo can't silently disable a risk limit.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LIVE_CONFIRM_PHRASE = "I_ACCEPT_THE_RISK"

TIMEFRAME_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}

DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://cointelegraph.com/rss",
    "https://decrypt.co/feed",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://news.google.com/rss/search?q=bitcoin+OR+BTC&hl=en-US&gl=US&ceid=US:en",
]


class ConfigError(ValueError):
    """Raised for invalid or unsafe configuration."""


@dataclass
class ExchangeConfig:
    id: str = "coinbase"
    symbol: str = "BTC/USD"
    sandbox: bool = False
    use_websocket: bool = True
    poll_interval_seconds: float = 2.0


@dataclass
class TradingConfig:
    mode: str = "paper"
    capital_usd: float = 10_000.0
    fee_rate: float = 0.006
    slippage_bps: float = 2.0


@dataclass
class ModelConfig:
    path: str = "models/btc_model.joblib"
    timeframe: str = "1m"
    horizon_bars: int = 15
    training_days: int = 60
    retrain_every_hours: float = 0.0
    auto_train: bool = True


@dataclass
class StrategyConfig:
    decision_interval_seconds: float = 5.0
    model_weight: float = 0.75
    news_weight: float = 0.25
    entry_threshold: float = 0.10
    exit_threshold: float = -0.05
    min_edge_multiple: float = 1.25
    require_model: bool = True


@dataclass
class RiskConfig:
    max_position_usd: float = 1_000.0
    max_position_pct: float = 0.5
    min_order_usd: float = 10.0
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    trailing_stop_pct: float = 0.0
    max_hold_minutes: float = 240.0
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.10
    max_trades_per_hour: int = 12
    min_seconds_between_trades: float = 30.0
    max_data_age_seconds: float = 15.0
    max_spread_bps: float = 20.0
    news_shock_threshold: float = -0.5
    news_shock_pause_minutes: float = 30.0


@dataclass
class NewsConfig:
    enabled: bool = True
    poll_seconds: float = 60.0
    feeds: list[str] = field(default_factory=lambda: list(DEFAULT_FEEDS))
    half_life_minutes: float = 90.0
    max_age_hours: float = 12.0
    claude: str = "auto"
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    claude_batch_size: int = 12


@dataclass
class DashboardConfig:
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class StorageConfig:
    data_dir: str = "data"
    journal_path: str = "data/journal.db"


@dataclass
class Secrets:
    """Credentials read from the environment. Never stored in config.toml."""

    api_key: str = ""
    api_secret: str = ""
    api_password: str = ""
    live_confirm: str = ""
    dashboard_token: str = ""

    @classmethod
    def from_env(cls) -> Secrets:
        return cls(
            api_key=os.environ.get("EXCHANGE_API_KEY", "").strip(),
            api_secret=os.environ.get("EXCHANGE_API_SECRET", "").strip(),
            api_password=os.environ.get("EXCHANGE_API_PASSWORD", "").strip(),
            live_confirm=os.environ.get("BTCAGENT_LIVE_CONFIRM", "").strip(),
            dashboard_token=os.environ.get("DASHBOARD_TOKEN", "").strip(),
        )

    @property
    def has_exchange_keys(self) -> bool:
        return bool(self.api_key and self.api_secret)

    def __repr__(self) -> str:  # never leak secrets into logs
        return f"Secrets(exchange_keys={'set' if self.has_exchange_keys else 'unset'})"


SECTIONS: dict[str, type] = {
    "exchange": ExchangeConfig,
    "trading": TradingConfig,
    "model": ModelConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "news": NewsConfig,
    "dashboard": DashboardConfig,
    "storage": StorageConfig,
}


@dataclass
class Settings:
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    news: NewsConfig = field(default_factory=NewsConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    secrets: Secrets = field(default_factory=Secrets)

    @classmethod
    def load(cls, path: str | Path | None = "config.toml", overrides: list[str] | None = None) -> Settings:
        """Load settings from a TOML file (missing file = defaults) plus ``section.key=value`` overrides."""
        raw: dict[str, Any] = {}
        if path is not None and Path(path).exists():
            with open(path, "rb") as fh:
                raw = tomllib.load(fh)
        for override in overrides or []:
            _apply_override(raw, override)

        unknown = set(raw) - set(SECTIONS)
        if unknown:
            raise ConfigError(f"Unknown config section(s): {', '.join(sorted(unknown))}")
        sections = {name: _build_section(cls_, raw.get(name, {}), name) for name, cls_ in SECTIONS.items()}
        settings = cls(**sections, secrets=Secrets.from_env())
        settings.validate()
        return settings

    @property
    def timeframe_seconds(self) -> int:
        return TIMEFRAME_SECONDS[self.model.timeframe]

    @property
    def is_live(self) -> bool:
        return self.trading.mode == "live"

    def validate(self) -> None:
        errors: list[str] = []

        def check(ok: bool, message: str) -> None:
            if not ok:
                errors.append(message)

        t, s, r, m, n = self.trading, self.strategy, self.risk, self.model, self.news
        check(t.mode in ("paper", "live"), "trading.mode must be 'paper' or 'live'")
        check(0 <= t.fee_rate < 0.05, "trading.fee_rate must be a fraction, e.g. 0.006 for 0.6%")
        check(t.slippage_bps >= 0, "trading.slippage_bps must be >= 0")
        check(t.capital_usd > 0, "trading.capital_usd must be > 0")
        check(m.timeframe in TIMEFRAME_SECONDS, f"model.timeframe must be one of {list(TIMEFRAME_SECONDS)}")
        check(m.horizon_bars >= 1, "model.horizon_bars must be >= 1")
        check(m.training_days >= 2, "model.training_days must be >= 2")
        check(m.retrain_every_hours >= 0, "model.retrain_every_hours must be >= 0")
        check(s.decision_interval_seconds >= 1, "strategy.decision_interval_seconds must be >= 1")
        check(s.model_weight >= 0 and s.news_weight >= 0, "strategy weights must be >= 0")
        check(s.model_weight + s.news_weight > 0, "strategy weights must not both be 0")
        check(0 < s.entry_threshold <= 1, "strategy.entry_threshold must be in (0, 1]")
        check(-1 <= s.exit_threshold < s.entry_threshold, "strategy.exit_threshold must be in [-1, entry_threshold)")
        check(s.min_edge_multiple >= 0, "strategy.min_edge_multiple must be >= 0")
        check(r.max_position_usd > 0, "risk.max_position_usd must be > 0")
        check(0 < r.max_position_pct <= 1, "risk.max_position_pct must be in (0, 1]")
        check(r.min_order_usd > 0, "risk.min_order_usd must be > 0")
        check(0 < r.stop_loss_pct < 0.5, "risk.stop_loss_pct must be in (0, 0.5) — a stop loss is mandatory")
        check(r.take_profit_pct >= 0, "risk.take_profit_pct must be >= 0 (0 disables)")
        check(0 <= r.trailing_stop_pct < 0.5, "risk.trailing_stop_pct must be in [0, 0.5)")
        check(r.max_hold_minutes >= 0, "risk.max_hold_minutes must be >= 0 (0 disables)")
        check(0 < r.max_daily_loss_pct < 1, "risk.max_daily_loss_pct must be in (0, 1)")
        check(0 < r.max_drawdown_pct < 1, "risk.max_drawdown_pct must be in (0, 1)")
        check(r.max_trades_per_hour >= 1, "risk.max_trades_per_hour must be >= 1")
        check(r.min_seconds_between_trades >= 0, "risk.min_seconds_between_trades must be >= 0")
        check(r.max_data_age_seconds > 0, "risk.max_data_age_seconds must be > 0")
        check(r.max_spread_bps > 0, "risk.max_spread_bps must be > 0")
        check(-1 <= r.news_shock_threshold < 0, "risk.news_shock_threshold must be in [-1, 0)")
        check(n.poll_seconds >= 10, "news.poll_seconds must be >= 10 (be polite to news sites)")
        check(n.half_life_minutes > 0, "news.half_life_minutes must be > 0")
        check(n.claude in ("auto", "on", "off"), "news.claude must be 'auto', 'on' or 'off'")
        check(
            n.claude_effort in ("low", "medium", "high", "xhigh", "max"),
            "news.claude_effort must be low|medium|high|xhigh|max",
        )
        check(1 <= n.claude_batch_size <= 50, "news.claude_batch_size must be in [1, 50]")
        check("/" in self.exchange.symbol, "exchange.symbol must look like 'BTC/USD'")
        check(self.exchange.poll_interval_seconds >= 0.5, "exchange.poll_interval_seconds must be >= 0.5")
        check(1 <= self.dashboard.port <= 65535, "dashboard.port must be a valid TCP port")
        if errors:
            raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(errors))

    def require_live_ready(self) -> None:
        """Refuse to trade real money unless every explicit opt-in is present."""
        if not self.is_live:
            return
        problems = []
        if not self.secrets.has_exchange_keys:
            problems.append("EXCHANGE_API_KEY / EXCHANGE_API_SECRET are not set")
        if self.secrets.live_confirm != LIVE_CONFIRM_PHRASE:
            problems.append(f"BTCAGENT_LIVE_CONFIRM must be set to {LIVE_CONFIRM_PHRASE}")
        if self.exchange.sandbox:
            problems.append("exchange.sandbox is true — sandbox keys can't trade live; use paper mode instead")
        if problems:
            raise ConfigError("Live trading is locked:\n  - " + "\n  - ".join(problems))


def _build_section(cls: type, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{section}] must be a table")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ConfigError(f"Unknown key(s) in [{section}]: {', '.join(sorted(unknown))}")
    kwargs = {key: _coerce(value, hints[key], f"{section}.{key}") for key, value in data.items()}
    return cls(**kwargs)


def _coerce(value: Any, expected: Any, name: str) -> Any:
    if expected is bool:
        if isinstance(value, bool):
            return value
    elif expected is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif expected is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif expected is str:
        if isinstance(value, str):
            return value
    elif typing.get_origin(expected) is list:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return list(value)
    raise ConfigError(f"{name}: expected {getattr(expected, '__name__', expected)}, got {value!r}")


def _apply_override(raw: dict[str, Any], override: str) -> None:
    """Apply ``section.key=value``; the value is parsed as a TOML literal (bare words become strings)."""
    if "=" not in override or "." not in override.split("=", 1)[0]:
        raise ConfigError(f"Override must look like section.key=value, got {override!r}")
    dotted, literal = override.split("=", 1)
    section, key = dotted.strip().split(".", 1)
    try:
        value = tomllib.loads(f"v = {literal.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = literal.strip()
    raw.setdefault(section, {})[key] = value
