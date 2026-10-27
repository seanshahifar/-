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


class BrokenSellBroker(PaperBroker):
    async def sell(self, qty, tick):
        raise RuntimeError("amount below exchange minimum")


def test_unsellable_dust_is_written_off_not_retried_forever(clock, journal_factory):
    agent, feed = make_agent(clock, journal_factory())
    agent.on_tick(feed.step(clock()))
    asyncio.run(agent.decide())
    agent.ledger.position.qty = 1e-7  # a few cents' worth
    agent.broker = BrokenSellBroker(0, 0, clock)
    asyncio.run(agent.kill())
    assert agent.ledger.position is None
    assert any("dust" in e["message"] for e in agent.events)


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
