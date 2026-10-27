import pytest

from btcagent.domain import Fill, Position
from btcagent.ledger import Ledger
from btcagent.risk import RiskManager, next_utc_midnight
from btcagent.strategy import Strategy

from conftest import make_settings

NOW = 1_800_000_000.0  # 2027-01-15 08:00 UTC


def risk(*overrides):
    return RiskManager(make_settings(*overrides).risk)


def position(entry=100.0, opened=NOW):
    return Position(qty=1.0, entry_price=entry, cost_usd=entry, opened_at=opened, peak_price=entry)


# ----------------------------------------------------------------- exits
def test_stop_loss_and_take_profit():
    r = risk("risk.stop_loss_pct=0.01", "risk.take_profit_pct=0.02")
    assert r.exit_reason(position(), 99.5, NOW) is None
    assert "stop loss" in r.exit_reason(position(), 98.99, NOW)
    assert "take profit" in r.exit_reason(position(), 102.0, NOW)


def test_trailing_stop_ratchets_up_with_the_peak():
    r = risk("risk.stop_loss_pct=0.02", "risk.take_profit_pct=0", "risk.trailing_stop_pct=0.01")
    pos = position()
    assert r.exit_reason(pos, 105.0, NOW) is None  # new peak
    assert r.stop_price(pos) == pytest.approx(103.95)
    assert r.exit_reason(pos, 103.9, NOW) == "trailing stop"


def test_max_holding_time():
    r = risk("risk.max_hold_minutes=30")
    assert r.exit_reason(position(opened=NOW - 29 * 60), 100.0, NOW) is None
    assert "max holding" in r.exit_reason(position(opened=NOW - 30 * 60), 100.0, NOW)


# ----------------------------------------------------------------- account limits
def test_daily_loss_limit_halts_until_midnight_then_clears():
    r = risk("risk.max_daily_loss_pct=0.03", "risk.max_drawdown_pct=0.5")
    assert r.on_equity(10_000, NOW) is None
    assert "daily loss" in r.on_equity(9_690, NOW + 60)
    assert "daily loss" in r.entry_blocked(NOW + 120, 0, 0)
    tomorrow = next_utc_midnight(NOW) + 1
    assert r.on_equity(9_690, tomorrow) is None  # a new day starts from the current equity
    assert r.entry_blocked(tomorrow, 0, 0) is None


def test_drawdown_kill_switch_needs_manual_reset():
    r = risk("risk.max_drawdown_pct=0.10", "risk.max_daily_loss_pct=0.5")
    r.on_equity(10_000, NOW)
    r.on_equity(12_000, NOW + 60)  # new peak
    assert "max drawdown" in r.on_equity(10_790, NOW + 120)
    assert r.state.killed
    assert "KILLED" in r.entry_blocked(NOW + 10 * 86_400, 0, 0)  # never expires on its own
    r.reset(10_790, NOW + 200)
    assert r.on_equity(10_790, NOW + 260) is None
    assert r.entry_blocked(NOW + 260, 0, 0) is None


def test_entry_guards():
    r = risk("risk.max_data_age_seconds=10", "risk.max_spread_bps=5", "risk.max_trades_per_hour=2",
             "risk.min_seconds_between_trades=30")
    assert "stale" in r.entry_blocked(NOW, 11, 1)
    assert "spread" in r.entry_blocked(NOW, 1, 6)
    r.record_trade(NOW)
    assert "cooling down" in r.entry_blocked(NOW + 10, 1, 1)
    r.record_trade(NOW + 40)
    assert "hourly trade limit" in r.entry_blocked(NOW + 100, 1, 1)
    assert r.entry_blocked(NOW + 3_700, 1, 1) is None


def test_news_shock_pauses_buying():
    r = risk("risk.news_shock_threshold=-0.5", "risk.news_shock_pause_minutes=30")
    assert not r.news_shock(-0.4, NOW)
    assert r.news_shock(-0.6, NOW)
    assert "news shock" in r.entry_blocked(NOW + 60, 0, 0)
    assert r.entry_blocked(NOW + 31 * 60, 0, 0) is None


def test_position_sizing_respects_every_cap():
    r = risk("risk.max_position_usd=1000", "risk.max_position_pct=0.5", "risk.min_order_usd=10")
    assert r.position_size_usd(equity=100_000, cash=100_000, conviction=1) == 1000  # absolute cap
    assert r.position_size_usd(equity=1_000, cash=1_000, conviction=1) == 500  # % of equity cap
    assert r.position_size_usd(equity=1_000, cash=200, conviction=1) == pytest.approx(199)  # cash cap
    assert r.position_size_usd(equity=1_000, cash=1_000, conviction=0) == 250  # half size at the threshold
    assert r.position_size_usd(equity=15, cash=15, conviction=1) == 0  # below exchange minimum


# ----------------------------------------------------------------- ledger
def test_ledger_round_trip_accounting():
    ledger = Ledger(1_000)
    ledger.apply_buy(Fill("buy", qty=0.01, price=50_000, fee_usd=3.0, ts=NOW))
    assert ledger.cash == pytest.approx(497)
    assert ledger.equity(50_000) == pytest.approx(997)
    trade = ledger.apply_sell(Fill("sell", qty=0.01, price=51_000, fee_usd=3.06, ts=NOW + 60))
    assert trade.pnl_usd == pytest.approx(510 - 3.06 - 503)
    assert ledger.position is None and ledger.round_trips == 1 and ledger.wins == 1
    assert ledger.cash == pytest.approx(1_000 + trade.pnl_usd)
    assert ledger.fees_paid == pytest.approx(6.06)


def test_ledger_partial_sell_and_persistence():
    ledger = Ledger(1_000)
    ledger.apply_buy(Fill("buy", 0.02, 25_000, 1.0, NOW))
    ledger.apply_sell(Fill("sell", 0.01, 25_000, 0.5, NOW))
    assert ledger.position.qty == pytest.approx(0.01)
    assert ledger.position.cost_usd == pytest.approx(250.5)
    restored = Ledger.from_dict(ledger.to_dict())
    assert restored.to_dict() == ledger.to_dict()
    with pytest.raises(ValueError):
        Ledger(10).apply_sell(Fill("sell", 1, 1, 0, NOW))


# ----------------------------------------------------------------- strategy
def strategy(*overrides):
    settings = make_settings(*overrides)
    return Strategy(settings.strategy, settings.trading)


def test_round_trip_cost_counts_fees_twice():
    s = strategy("trading.fee_rate=0.006", "trading.slippage_bps=2")
    assert s.round_trip_cost_bps(spread_bps=1) == pytest.approx(125)


def test_fees_block_trades_the_edge_cannot_pay_for():
    s = strategy("trading.fee_rate=0.006", "strategy.min_edge_multiple=1.25", "strategy.entry_threshold=0.1")
    strong = s.signal(p_up=0.75, news_score=0.5, expected_move=0.003, spread_bps=1)
    assert strong.combined > 0.4
    enter, why = s.should_enter(strong)
    assert not enter and "required" in why

    cheap = strategy("trading.fee_rate=0.0", "trading.slippage_bps=1", "strategy.min_edge_multiple=1.25")
    enter, why = cheap.should_enter(cheap.signal(p_up=0.75, news_score=0.5, expected_move=0.003, spread_bps=1))
    assert enter, why


def test_no_model_means_no_entries_and_negative_signal_means_exit():
    s = strategy("trading.fee_rate=0", "strategy.min_edge_multiple=0")
    enter, why = s.should_enter(s.signal(None, news_score=1.0, expected_move=0.01, spread_bps=0))
    assert not enter and "no trained model" in why
    exit_now, why = s.should_exit(s.signal(0.3, news_score=-0.2, expected_move=0.01, spread_bps=0))
    assert exit_now and "signal turned" in why


def test_conviction_scales_from_threshold_to_max():
    s = strategy("strategy.entry_threshold=0.2")
    assert s.conviction(s.signal(0.6, 0.2, 0.01, 0)) == 0  # combined exactly 0.2
    assert s.conviction(s.signal(1.0, 1.0, 0.01, 0)) == 1
