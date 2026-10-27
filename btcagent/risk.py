"""Risk manager: the rules that can always say "no".

Entries can be blocked for many reasons; exits that reduce risk are never blocked by rate
limits. Hard limits (daily loss, max drawdown, kill switch) also force an immediate exit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone

from btcagent.config import RiskConfig
from btcagent.domain import Position


def utc_day(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")


def next_utc_midnight(ts: float) -> float:
    day = datetime.fromtimestamp(ts, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return (day + timedelta(days=1)).timestamp()


@dataclass
class RiskState:
    day: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted_until: float = 0.0
    halt_reason: str = ""
    killed: bool = False
    kill_reason: str = ""
    trade_times: list[float] = field(default_factory=list)
    last_trade_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class RiskManager:
    def __init__(self, cfg: RiskConfig, state: RiskState | None = None):
        self.cfg = cfg
        self.state = state or RiskState()

    # --- account-level limits -------------------------------------------------
    def on_equity(self, equity: float, now: float) -> str | None:
        """Track daily and peak equity. Returns a reason if a hard limit says: flatten now."""
        s = self.state
        day = utc_day(now)
        if s.day != day:
            s.day, s.day_start_equity = day, equity
            if s.halt_reason.startswith("daily loss"):
                s.halted_until, s.halt_reason = 0.0, ""
        s.peak_equity = max(s.peak_equity, equity)
        if s.killed:
            return s.kill_reason
        if s.peak_equity > 0 and equity <= s.peak_equity * (1 - self.cfg.max_drawdown_pct):
            drop = 1 - equity / s.peak_equity
            self.kill(f"max drawdown hit ({drop:.1%} below peak equity)")
            return s.kill_reason
        if s.day_start_equity > 0 and equity <= s.day_start_equity * (1 - self.cfg.max_daily_loss_pct):
            if now >= s.halted_until or not s.halt_reason.startswith("daily loss"):
                loss = 1 - equity / s.day_start_equity
                s.halted_until = next_utc_midnight(now)
                s.halt_reason = f"daily loss limit hit ({loss:.1%} today)"
            return s.halt_reason
        return None

    def kill(self, reason: str) -> None:
        self.state.killed, self.state.kill_reason = True, reason

    def reset(self, equity: float, now: float) -> None:
        """Manual reset after a kill: limits start fresh from the current equity."""
        s = self.state
        s.killed, s.kill_reason = False, ""
        s.halted_until, s.halt_reason = 0.0, ""
        s.peak_equity = equity
        s.day, s.day_start_equity = utc_day(now), equity

    def rebase(self, delta: float) -> None:
        """Capital was added or withdrawn: shift the equity baselines so it isn't mistaken for P&L."""
        s = self.state
        s.peak_equity = max(0.0, s.peak_equity + delta)
        s.day_start_equity = max(0.0, s.day_start_equity + delta)

    def news_shock(self, news_score: float, now: float) -> bool:
        """Very negative news pauses new buys for a while (and the engine sells)."""
        if news_score > self.cfg.news_shock_threshold:
            return False
        if now >= self.state.halted_until:
            self.state.halted_until = now + self.cfg.news_shock_pause_minutes * 60
            self.state.halt_reason = f"negative news shock (news score {news_score:+.2f})"
        return True

    # --- trade-level rules ----------------------------------------------------
    def entry_blocked(self, now: float, data_age: float, spread_bps: float) -> str | None:
        s = self.state
        if s.killed:
            return f"KILLED: {s.kill_reason}. Reset required"
        if now < s.halted_until:
            until = datetime.fromtimestamp(s.halted_until, timezone.utc).strftime("%H:%M UTC")
            return f"paused: {s.halt_reason} until {until}"
        if data_age > self.cfg.max_data_age_seconds:
            return f"price data is stale ({data_age:.0f}s old)"
        if spread_bps > self.cfg.max_spread_bps:
            return f"spread too wide ({spread_bps:.1f} bps > {self.cfg.max_spread_bps:.0f})"
        s.trade_times = [t for t in s.trade_times if now - t < 3600]
        if len(s.trade_times) >= self.cfg.max_trades_per_hour:
            return f"hourly trade limit reached ({self.cfg.max_trades_per_hour})"
        if now - s.last_trade_at < self.cfg.min_seconds_between_trades:
            return f"cooling down ({self.cfg.min_seconds_between_trades - (now - s.last_trade_at):.0f}s)"
        return None

    def position_size_usd(self, equity: float, cash: float, conviction: float) -> float:
        cap = min(self.cfg.max_position_usd, equity * self.cfg.max_position_pct, cash * 0.995)
        size = cap * (0.5 + 0.5 * max(0.0, min(1.0, conviction)))
        return size if size >= self.cfg.min_order_usd else 0.0

    def stop_price(self, position: Position) -> float:
        stop = position.entry_price * (1 - self.cfg.stop_loss_pct)
        if self.cfg.trailing_stop_pct > 0 and position.peak_price > position.entry_price:
            stop = max(stop, position.peak_price * (1 - self.cfg.trailing_stop_pct))
        return stop

    def take_profit_price(self, position: Position) -> float | None:
        return position.entry_price * (1 + self.cfg.take_profit_pct) if self.cfg.take_profit_pct > 0 else None

    def exit_reason(self, position: Position, price: float, now: float) -> str | None:
        position.peak_price = max(position.peak_price, price)
        if price <= self.stop_price(position):
            trailing = self.cfg.trailing_stop_pct > 0 and self.stop_price(position) > position.entry_price * (
                1 - self.cfg.stop_loss_pct
            )
            return "trailing stop" if trailing else f"stop loss ({self.cfg.stop_loss_pct:.2%})"
        target = self.take_profit_price(position)
        if target is not None and price >= target:
            return f"take profit ({self.cfg.take_profit_pct:.2%})"
        if self.cfg.max_hold_minutes > 0 and now - position.opened_at >= self.cfg.max_hold_minutes * 60:
            return f"max holding time ({self.cfg.max_hold_minutes:.0f} min)"
        return None

    def record_trade(self, now: float) -> None:
        self.state.trade_times.append(now)
        self.state.last_trade_at = now
