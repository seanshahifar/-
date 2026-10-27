import asyncio
import json

import pytest

from btcagent.broker.paper import PaperBroker
from btcagent.domain import Tick
from btcagent.engine import TradingAgent
from btcagent.market.simulation import SimulatedFeed

from conftest import FakeClock, FakeModel, make_settings

FAST = ("trading.fee_rate=0", "trading.slippage_bps=0", "strategy.min_edge_multiple=0",
        "strategy.entry_threshold=0.05", "risk.stop_loss_pct=0.01", "risk.take_profit_pct=0.05",
        "risk.max_daily_loss_pct=0.5", "risk.max_drawdown_pct=0.5", "news.enabled=false")


class FakeNews:
    def __init__(self, score=0.0):
        self.value = score
        self.on_item = None

    def score(self, now=None):
        return self.value

    def snapshot(self, limit=25):
        return {"score": self.value, "scorer": "fake", "claude_error": None, "last_poll_at": None,
                "feed_errors": 0, "feeds": 0, "items": []}


def make_agent(clock, journal, p_up=0.8, news=None, fresh=False):
    settings = make_settings(*FAST)
    feed = SimulatedFeed(60, history_bars=400, seed=7, clock=clock)
    agent = TradingAgent(settings, feed, PaperBroker(0, 0, clock), journal,
                         model=FakeModel(p_up), news=news, mode="sim", clock=clock, fresh=fresh)
    asyncio.run(agent.sync_bars())
    return agent, feed


def tick_at(price, clock):
    return Tick(ts=clock(), price=price, bid=price - 0.5, ask=price + 0.5, received_at=clock())


def test_buys_on_a_strong_signal_then_stops_out(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    decision = asyncio.run(agent.decide())
    assert decision.action == "buy", decision.reason
    entry = agent.ledger.position.entry_price

    clock.advance(5)
    agent.on_tick(tick_at(entry * 0.985, clock))
    assert agent._wake.is_set()  # a crossed stop wakes the decision loop immediately
    decision = asyncio.run(agent.decide())
    assert decision.action == "sell" and "stop loss" in decision.reason
    assert agent.ledger.position is None and agent.ledger.round_trips == 1
    assert len(agent.trades) == 2


def test_weak_signal_or_stale_data_means_hold(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory(), p_up=0.51)
    agent.on_tick(feed.step(clock()))
    assert asyncio.run(agent.decide()).action == "hold"

    agent, feed = make_agent(clock, journal_factory(), p_up=0.9)
    agent.on_tick(feed.step(clock()))
    clock.advance(60)  # no ticks for a minute
    decision = asyncio.run(agent.decide())
    assert decision.action == "hold" and "stale" in decision.reason


def test_negative_news_shock_sells_and_blocks_buying(clock, journal_factory):
    news = FakeNews(0.0)
    agent, feed = make_agent(clock, journal_factory(), news=news)
    agent.on_tick(feed.step(clock()))
    assert asyncio.run(agent.decide()).action == "buy"
    news.value = -0.9
    clock.advance(5)
    agent.on_tick(feed.step(clock()))
    decision = asyncio.run(agent.decide())
    assert decision.action == "sell" and "news shock" in decision.reason
    clock.advance(60)
    agent.on_tick(feed.step(clock()))
    assert asyncio.run(agent.decide()).action == "hold"


def test_kill_switch_flattens_and_blocks_until_reset(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    assert agent.ledger.position is not None
    asyncio.run(agent.kill())
    assert agent.ledger.position is None
    clock.advance(60)
    agent.on_tick(feed.step(clock()))
    decision = asyncio.run(agent.decide())
    assert decision.action == "hold" and "KILLED" in decision.reason
    agent.reset()
    clock.advance(60)
    agent.on_tick(feed.step(clock()))
    assert asyncio.run(agent.decide()).action == "buy"


def test_pause_blocks_entries_but_not_exits(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.pause()
    clock.advance(5)
    agent.on_tick(tick_at(agent.ledger.position.entry_price * 0.98, clock))
    assert asyncio.run(agent.decide()).action == "sell"
    clock.advance(60)
    agent.on_tick(feed.step(clock()))
    decision = asyncio.run(agent.decide())
    assert decision.action == "hold" and "paused" in decision.reason


def test_state_survives_a_restart(clock, tmp_path, journal_factory):
    journal = journal_factory(tmp_path / "journal.db")
    agent, feed = make_agent(clock, journal)
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    held = agent.ledger.position.qty
    agent.persist()

    restarted, _ = make_agent(clock, journal)
    assert restarted.resumed
    assert restarted.ledger.position.qty == pytest.approx(held)
    assert [t.side for t in restarted.trades] == ["buy"]
    fresh, _ = make_agent(clock, journal, fresh=True)
    assert not fresh.resumed and fresh.ledger.position is None and fresh.ledger.cash == 10_000


def test_snapshot_is_strict_json(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory(), news=FakeNews(0.1))
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.model.metrics["auc"] = float("nan")  # must not break the dashboard
    snapshot = agent.snapshot()
    json.dumps(snapshot, allow_nan=False)
    assert snapshot["mode"] == "sim" and snapshot["simulated"] is True
    assert snapshot["position"]["stop"] < snapshot["position"]["entry_price"]
    assert snapshot["model"]["metrics"]["auc"] is None


class MinimumOrderBroker(PaperBroker):
    """Paper broker with an exchange-style $10 minimum order and a configurable sell outcome."""

    def __init__(self, clock, sell_error=None, sell_fraction=1.0):
        super().__init__(0, 0, clock)
        self.sell_error = sell_error
        self.sell_fraction = sell_fraction

    def is_sellable(self, qty, price):
        return qty * price >= 10

    async def sell(self, qty, tick):
        if self.sell_error is not None:
            raise self.sell_error
        return await super().sell(qty * self.sell_fraction, tick)


def test_unsellable_dust_is_written_off_without_an_order(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.ledger.position.qty = 1e-7  # a fraction of a cent
    agent.broker = MinimumOrderBroker(clock, sell_error=AssertionError("must not try to sell dust"))
    asyncio.run(agent.kill())
    assert agent.ledger.position is None
    assert any("dust" in e["message"] for e in agent.events)


def test_a_failed_sell_never_writes_the_position_off(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.broker = MinimumOrderBroker(clock, sell_error=RuntimeError("exchange timeout"))
    asyncio.run(agent.kill())
    assert agent.ledger.position is not None  # still tracked, still protected, retried next cycle


def test_a_dust_remainder_after_a_sell_closes_the_round_trip(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.broker = MinimumOrderBroker(clock, sell_fraction=0.9999)  # leaves ~5 cents behind
    clock.advance(5)
    agent.on_tick(tick_at(agent.ledger.position.entry_price * 0.98, clock))
    assert asyncio.run(agent.decide()).action == "sell"
    assert agent.ledger.position is None and agent.ledger.round_trips == 1


def test_unknown_order_outcome_halts_trading(clock, journal_factory):
    from btcagent.broker.base import OrderStatusUnknown

    class LostBroker(PaperBroker):
        async def buy(self, usd, tick):
            raise OrderStatusUnknown("exchange stopped answering")

    agent, feed = make_agent(clock, journal_factory())
    agent.broker = LostBroker(0, 0, clock)
    agent.on_tick(feed.step(clock()))
    decision = asyncio.run(agent.decide())
    assert decision.action == "hold" and "outcome unknown" in decision.reason
    assert agent.risk.state.killed


def test_an_order_in_flight_is_booked_even_during_shutdown(clock, journal_factory):
    release = asyncio.Event()

    class SlowBroker(PaperBroker):
        async def buy(self, usd, tick):
            await release.wait()  # the exchange is still working on it...
            return await super().buy(usd, tick)

    agent, feed = make_agent(clock, journal_factory())
    agent.broker = SlowBroker(0, 0, clock)
    agent.on_tick(feed.step(clock()))

    async def scenario():
        decision = asyncio.create_task(agent.decide())
        await asyncio.sleep(0.01)
        decision.cancel()  # Ctrl-C / SIGTERM arrives mid-order
        await asyncio.sleep(0.01)
        release.set()  # ...and the order completes anyway
        with pytest.raises(asyncio.CancelledError):
            await decision

    asyncio.run(scenario())
    assert agent.ledger.position is not None  # the fill was booked, not lost


def test_pause_survives_a_restart(clock, journal_factory):
    journal = journal_factory()
    agent, _ = make_agent(clock, journal)
    agent.pause()
    restarted, _ = make_agent(clock, journal)
    assert restarted.paused


def test_reset_needs_a_price_first(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    with pytest.raises(ValueError, match="first price"):
        agent.reset()
    agent.on_tick(feed.step(clock()))
    agent.reset()


def test_a_changed_capital_setting_is_applied_on_resume(clock, journal_factory):
    journal = journal_factory()
    agent, feed = make_agent(clock, journal)
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())  # sets the risk baselines at $10,000
    agent.persist()

    settings = make_settings(*FAST, "trading.capital_usd=1000")
    feed2 = SimulatedFeed(60, history_bars=400, seed=7, clock=clock)
    smaller = TradingAgent(settings, feed2, PaperBroker(0, 0, clock), journal, model=FakeModel(0.8), mode="sim",
                           clock=clock)
    assert smaller.ledger.capital == 1000
    assert smaller.ledger.cash == pytest.approx(agent.ledger.cash - 9000)
    assert smaller.risk.state.peak_equity == pytest.approx(agent.risk.state.peak_equity - 9000)
    asyncio.run(smaller.sync_bars())
    smaller.on_tick(feed2.step(clock()))
    decision = asyncio.run(smaller.decide())
    assert "drawdown" not in decision.reason  # a withdrawal isn't a loss


def test_a_failing_model_save_does_not_stop_the_agent(clock, journal_factory, monkeypatch):
    import btcagent.engine as engine_module

    class Candidate(FakeModel):
        def save(self, path):
            raise PermissionError("models/ is read-only")

    agent, feed = make_agent(clock, journal_factory())
    original_model = agent.model
    agent.feed.simulated = False  # pretend it's a real feed so the model gets saved
    monkeypatch.setattr(engine_module, "train_model", lambda *a, **k: Candidate(0.7))
    sleeps = {"n": 0}

    async def sleep(_seconds):
        sleeps["n"] += 1
        if sleeps["n"] > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent._retrain_loop())
    assert agent.model is original_model
    assert any("Could not swap in" in e["message"] for e in agent.events)


def test_feed_errors_are_survived(clock, journal_factory, monkeypatch):
    agent, feed = make_agent(clock, journal_factory())
    calls = {"n": 0}

    async def flaky_ticks():
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("socket closed")
        yield feed.step(clock())
        raise asyncio.CancelledError  # end the test after one good tick

    async def no_wait(_seconds):
        return None

    monkeypatch.setattr(agent.feed, "ticks", flaky_ticks)
    monkeypatch.setattr(asyncio, "sleep", no_wait)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(agent._tick_loop())
    assert agent.tick is not None and calls["n"] == 2
    assert any("reconnecting" in e["message"] for e in agent.events)


def test_flatten_on_exit_sells_the_position(clock, journal_factory):
    journal = journal_factory()
    agent, feed = make_agent(clock, journal)
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    assert agent.ledger.position is not None
    asyncio.run(agent.shutdown(flatten=True))
    assert agent.ledger.position is None
    assert agent.trades[-1].reason == "shutdown (flatten on exit)"
    assert journal.load_state("sim:agent")["ledger"]["position"] is None


# ----------------------------------------------------------------------------- first-boot training
def make_agent_without_model(clock, journal, *overrides):
    settings = make_settings(*FAST, "model.horizon_bars=1", *overrides)
    feed = SimulatedFeed(60, history_bars=3_000, seed=7, clock=clock)
    agent = TradingAgent(settings, feed, PaperBroker(0, 0, clock), journal, model=None, mode="sim", clock=clock)
    asyncio.run(agent.sync_bars())
    return agent, feed


def test_the_first_model_is_trained_in_the_background_and_installed(clock, journal_factory):
    agent, _ = make_agent_without_model(clock, journal_factory())
    assert agent.snapshot()["model_status"] == "missing"
    assert asyncio.run(agent._train_and_install(first=True)) is True
    assert agent.model is not None and agent.prediction is not None
    snapshot = agent.snapshot()
    assert snapshot["model_status"] == "ready" and snapshot["model_error"] is None and not agent.training
    assert any("Model trained and ready" in e["message"] for e in agent.events)


def test_training_failure_is_reported_and_never_stops_the_agent(clock, journal_factory, monkeypatch):
    agent, feed = make_agent_without_model(clock, journal_factory())

    async def unreachable(days):
        raise ConnectionError("exchange unreachable")

    monkeypatch.setattr(feed, "fetch_history", unreachable)
    assert asyncio.run(agent._train_and_install(first=True)) is False
    snapshot = agent.snapshot()
    assert snapshot["model_status"] == "missing" and "ConnectionError" in snapshot["model_error"]
    assert not agent.training


def test_the_model_loop_retries_every_15_minutes_until_it_succeeds(clock, journal_factory, monkeypatch):
    agent, feed = make_agent_without_model(clock, journal_factory())
    real_fetch, attempts, pauses = feed.fetch_history, {"n": 0}, []

    async def flaky(days):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("down")
        return await real_fetch(days)

    async def record_pause(seconds):
        pauses.append(seconds)

    monkeypatch.setattr(feed, "fetch_history", flaky)
    monkeypatch.setattr(asyncio, "sleep", record_pause)
    asyncio.run(agent._model_loop())
    assert attempts["n"] == 3 and pauses == [900.0, 900.0]
    assert agent.model is not None and agent.snapshot()["model_error"] is None


def test_auto_train_can_be_switched_off(clock, journal_factory):
    agent, _ = make_agent_without_model(clock, journal_factory(), "model.auto_train=false")
    asyncio.run(agent._model_loop())  # nothing to do: returns immediately
    assert agent.model is None


def test_a_model_that_cannot_be_saved_is_still_used(clock, journal_factory, monkeypatch):
    import btcagent.engine as engine_module

    class Candidate(FakeModel):
        def save(self, path):
            raise PermissionError("read-only disk")

    agent, feed = make_agent_without_model(clock, journal_factory())
    feed.simulated = False  # pretend it's a real exchange feed, so the model is saved to disk
    monkeypatch.setattr(engine_module, "train_model", lambda *args, **kwargs: Candidate(0.7))
    assert asyncio.run(agent._train_and_install(first=True)) is True
    assert isinstance(agent.model, Candidate)
    assert any("in memory only" in e["message"] for e in agent.events)


def test_a_worse_retrained_model_is_rejected(clock, journal_factory, monkeypatch):
    import btcagent.engine as engine_module

    class Weak(FakeModel):
        def __init__(self):
            super().__init__(0.5)
            self.metrics = {"auc": 0.5, "calibration_slope": 0.0}

    agent, _ = make_agent(clock, journal_factory())
    before = agent.model
    monkeypatch.setattr(engine_module, "train_model", lambda *args, **kwargs: Weak())
    assert asyncio.run(agent._train_and_install(first=False)) is False
    assert agent.model is before


def test_a_slow_exchange_does_not_hold_up_startup(clock, journal_factory, monkeypatch):
    import contextlib

    agent, feed = make_agent_without_model(clock, journal_factory())
    feed.tick_seconds = 0.02

    async def hang(count):
        await asyncio.Event().wait()  # the exchange never answers the first bar request

    monkeypatch.setattr(feed, "fetch_bars", hang)

    async def scenario():
        task = asyncio.create_task(agent.run())
        await asyncio.sleep(0.4)
        started = (agent.tick is not None,
                   any("Training the prediction model" in e["message"] for e in agent.events))
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, BaseExceptionGroup):
            await task
        return started

    assert asyncio.run(scenario()) == (True, True)  # prices flow and model training began, despite the hang
