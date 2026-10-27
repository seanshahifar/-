"""The trading agent: perceives (prices + news), predicts (model), decides (strategy + risk), acts (broker).

Concurrent loops:
  * ticks     - streams quotes; wakes the decision loop at once if a stop or target is crossed
  * bars      - re-syncs closed OHLCV bars from the exchange right after each bar closes
  * decisions - re-evaluates every ``decision_interval_seconds`` (default 5s)
  * news      - polls feeds, scores headlines (lexicon instantly, Claude when enabled)
  * snapshots - records equity and persists state so a restart resumes where it left off
  * model     - trains a first model in the background if none exists; optionally retrains to keep it fresh
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from btcagent.broker.base import Broker, OrderStatusUnknown
from btcagent.config import Settings
from btcagent.domain import Decision, Tick
from btcagent.features import LIVE_LOOKBACK_BARS, WARMUP_BARS, compute_features, expected_abs_move
from btcagent.ledger import Ledger, TradeRecord
from btcagent.market.exchange import clean_bars, drop_incomplete, empty_bars, epoch_ms, fill_gaps
from btcagent.market.feed import MarketFeed
from btcagent.model import TrainedModel, train_model
from btcagent.news.service import NewsService
from btcagent.risk import RiskManager, RiskState
from btcagent.storage import Journal
from btcagent.strategy import Signal, Strategy

log = logging.getLogger(__name__)

SNAPSHOT_SECONDS = 15.0
PERSIST_SECONDS = 60.0
BAR_SETTLE_SECONDS = 2.0  # wait this long after a bar closes before fetching it
MODEL_RETRY_SECONDS = 900.0  # while there is still no model, try training again every 15 minutes


@dataclass(frozen=True, slots=True)
class Prediction:
    bar_close: float  # close time (epoch s) of the last bar the prediction used
    p_up: float
    expected_move: float | None


class TradingAgent:
    def __init__(
        self,
        settings: Settings,
        feed: MarketFeed,
        broker: Broker,
        journal: Journal,
        model: TrainedModel | None = None,
        news: NewsService | None = None,
        mode: str | None = None,
        clock: Callable[[], float] = time.time,
        fresh: bool = False,
    ):
        self.settings = settings
        self.feed = feed
        self.broker = broker
        self.journal = journal
        self.model = model
        self.news = news
        self.mode = mode or settings.trading.mode
        self.clock = clock
        self.tf = settings.timeframe_seconds
        self.strategy = Strategy(settings.strategy, settings.trading)

        self.events: deque[dict] = deque(maxlen=200)
        saved = None if fresh else journal.load_state(self._state_key)
        self.paused = False
        self.training = False  # a model is being trained in the background right now
        self.train_error: str | None = None  # why the last training attempt failed
        if saved:
            self.ledger = Ledger.from_dict(saved["ledger"])
            self.risk = RiskManager(settings.risk, RiskState(**saved["risk"]))
            self.paused = bool(saved.get("paused", False))
            old_capital = self.ledger.capital
            delta = self.ledger.rebase_capital(settings.trading.capital_usd)
            if abs(delta) > 1e-9:  # trading.capital_usd changed since the last run: a deposit / withdrawal
                self.risk.rebase(delta)
                self.event("info", f"trading.capital_usd changed from ${old_capital:,.2f} to "
                                   f"${self.ledger.capital:,.2f}; cash adjusted by ${delta:+,.2f}")
        else:
            self.ledger = Ledger(settings.trading.capital_usd)
            self.risk = RiskManager(settings.risk)
        self.resumed = bool(saved)

        self.started_at = clock()
        self.tick: Tick | None = None
        self.bars = empty_bars()
        self.prediction: Prediction | None = None
        self.signal: Signal | None = None
        self.decision: Decision | None = None
        self.prices: deque[tuple[float, float]] = deque(maxlen=3600)
        self.equity_points: deque[tuple[float, float]] = deque(
            journal.equity_history(self.mode, clock() - 86_400)[-2000:], maxlen=2000
        )
        self.trades: deque[TradeRecord] = deque(journal.recent_trades(self.mode, 100), maxlen=100)
        self._wake = asyncio.Event()
        self._lock = asyncio.Lock()

        if news is not None and news.on_item is None:
            news.on_item = journal.record_news

    @property
    def _state_key(self) -> str:
        return f"{self.mode}:agent"

    # ------------------------------------------------------------------ lifecycle
    async def run(self) -> None:
        await self.broker.prepare()
        if self.resumed:
            self.event("info", f"Resumed saved {self.mode} state: equity books at ${self.ledger.cash:,.2f} cash")
        self.event("info", f"Agent started in {self.mode.upper()} mode on {self.feed.name} {self.settings.exchange.symbol}")
        if self.model is None:
            if self.settings.model.auto_train:
                self.event("info", "No trained model yet: training one in the background. The agent watches "
                                   "but won't buy until it's ready.")
            else:
                self.event("warn", "No trained model loaded: the agent will watch but not buy. "
                                   "Run `btcagent train`.")
        loops = [self._tick_loop(), self._bar_loop(), self._decision_loop(), self._snapshot_loop()]
        if self.news is not None:
            loops.append(self.news.run())
        if (self.model is None and self.settings.model.auto_train) or self.settings.model.retrain_every_hours > 0:
            loops.append(self._model_loop())
        try:
            async with asyncio.TaskGroup() as group:  # one loop failing cancels the rest
                for loop in loops:
                    group.create_task(loop)
        finally:
            self.persist()

    async def shutdown(self, flatten: bool = False) -> None:
        if flatten and self.ledger.position is not None and self.tick is not None:
            async with self._lock:
                await self._exit(self.clock(), self.tick, "shutdown (flatten on exit)")
        self.persist()
        await self.feed.close()
        await self.broker.close()

    def persist(self) -> None:
        self.journal.save_state(self._state_key, {
            "ledger": self.ledger.to_dict(), "risk": self.risk.state.to_dict(), "paused": self.paused,
        })

    # ------------------------------------------------------------------ perception
    async def _tick_loop(self) -> None:
        while True:
            try:
                async for tick in self.feed.ticks():
                    self.on_tick(tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.event("warn", f"Price feed error ({type(exc).__name__}: {exc}); reconnecting in 5s")
                await asyncio.sleep(5)

    def on_tick(self, tick: Tick) -> None:
        self.tick = tick
        self.prices.append((tick.ts, tick.price))
        position = self.ledger.position
        if position is not None:
            target = self.risk.take_profit_price(position)
            if tick.sell_price <= self.risk.stop_price(position) or (target is not None and tick.sell_price >= target):
                self._wake.set()  # act now, don't wait for the next scheduled decision

    async def _bar_loop(self) -> None:
        await self._sync_bars_safely()  # the first sync starts at once, without holding up the other loops
        while True:
            now = self.clock()
            next_close = (math.floor(now / self.tf) + 1) * self.tf
            await asyncio.sleep(max(0.5, next_close - now + BAR_SETTLE_SECONDS))
            await self._sync_bars_safely()

    async def _sync_bars_safely(self) -> None:
        try:
            await self.sync_bars()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.event("warn", f"Could not refresh price bars: {type(exc).__name__}: {exc}")

    async def sync_bars(self) -> None:
        raw = await self.feed.fetch_bars(LIVE_LOOKBACK_BARS + 5)
        bars = fill_gaps(clean_bars(drop_incomplete(raw, self.tf, self.clock())), self.tf)
        if bars.empty:
            return
        self.bars = bars
        self._update_prediction()

    def _update_prediction(self) -> None:
        if self.model is None or len(self.bars) < WARMUP_BARS:
            self.prediction = None
            return
        latest = compute_features(self.bars).iloc[[-1]]
        if latest.isna().all(axis=1).iloc[0]:
            self.prediction = None
            return
        self.prediction = Prediction(
            bar_close=float(epoch_ms(self.bars.index)[-1] / 1000 + self.tf),
            p_up=float(self.model.predict_proba(latest)[0]),
            expected_move=expected_abs_move(latest.iloc[0], self.model.horizon_bars),
        )

    def current_prediction(self, now: float) -> Prediction | None:
        """The model's view, or None if bars stopped updating (never trade on a stale forecast)."""
        p = self.prediction
        if p is None or now - p.bar_close > 3 * self.tf + BAR_SETTLE_SECONDS:
            return None
        return p

    # ------------------------------------------------------------------ decisions
    async def _decision_loop(self) -> None:
        interval = self.settings.strategy.decision_interval_seconds
        while True:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            try:
                await self.decide()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.exception("Decision cycle failed")
                self.event("error", f"Decision cycle failed: {type(exc).__name__}: {exc}")

    async def decide(self, now: float | None = None) -> Decision:
        async with self._lock:
            return await self._decide(self.clock() if now is None else now)

    async def _decide(self, now: float) -> Decision:
        tick = self.tick
        if tick is None:
            return self._hold(now, "waiting for the first price")
        data_age = now - tick.received_at
        equity = self.ledger.equity(tick.price)
        forced = self.risk.on_equity(equity, now)
        news_score = self.news.score(now) if self.news is not None else 0.0
        prediction = self.current_prediction(now)
        self.signal = signal = self.strategy.signal(
            prediction.p_up if prediction else None,
            news_score,
            prediction.expected_move if prediction else None,
            tick.spread_bps,
        )

        position = self.ledger.position
        if position is not None:
            if forced:
                return await self._exit(now, tick, forced)
            if data_age > self.settings.risk.max_data_age_seconds:
                return self._hold(now, f"price data is stale ({data_age:.0f}s old); holding position")
            reason = self.risk.exit_reason(position, tick.sell_price, now)
            if reason is None and self.risk.news_shock(news_score, now):
                reason = f"negative news shock (news score {news_score:+.2f})"
            if reason is None:
                exit_now, why = self.strategy.should_exit(signal)
                reason = why if exit_now else None
            if reason:
                return await self._exit(now, tick, reason)
            change = tick.sell_price / position.entry_price - 1
            return self._hold(now, f"holding ({change:+.2%}); stop {self.risk.stop_price(position):,.2f}")

        if forced:  # explain the limit and when it lifts, e.g. "paused: daily loss limit ... until 00:00 UTC"
            return self._hold(now, self.risk.entry_blocked(now, data_age, tick.spread_bps) or forced)
        if self.paused:
            return self._hold(now, "paused: not opening new positions")
        if self.risk.news_shock(news_score, now):
            return self._hold(now, f"negative news shock (news score {news_score:+.2f}); not buying")
        blocked = self.risk.entry_blocked(now, data_age, tick.spread_bps)
        if blocked:
            return self._hold(now, blocked)
        enter, why = self.strategy.should_enter(signal)
        if not enter:
            return self._hold(now, why)
        size = self.risk.position_size_usd(equity, self.ledger.cash, self.strategy.conviction(signal))
        if size <= 0:
            return self._hold(now, "not enough free cash for the minimum order")
        return await self._enter(now, tick, size, why)

    async def _enter(self, now: float, tick: Tick, usd: float, reason: str) -> Decision:
        return await _finish_even_if_cancelled(self._place_buy(now, tick, usd, reason))

    async def _exit(self, now: float, tick: Tick, reason: str) -> Decision:
        return await _finish_even_if_cancelled(self._place_sell(now, tick, reason))

    async def _place_buy(self, now: float, tick: Tick, usd: float, reason: str) -> Decision:
        try:
            fill = await self.broker.buy(usd, tick)
        except OrderStatusUnknown as exc:
            return self._order_status_unknown(now, "BUY", exc)
        except Exception as exc:
            self.event("error", f"BUY failed: {exc}")
            self.risk.state.last_trade_at = now  # cool down before retrying
            return self._hold(now, f"buy failed: {exc}")
        record = self.ledger.apply_buy(fill, reason)
        self._book(now, record, fill.order_id)
        self.event("trade", f"BUY {fill.qty:.6f} BTC @ {fill.price:,.2f} (${usd:,.2f}) — {reason}")
        return self._record_decision(now, "buy", reason)

    async def _place_sell(self, now: float, tick: Tick, reason: str) -> Decision:
        position = self.ledger.position
        if position is None:
            return self._hold(now, reason)
        if not self.broker.is_sellable(position.qty, tick.price):
            # Below the exchange's minimum order: it can never be sold, so stop tracking it.
            loss = self.ledger.write_off_position()
            self.persist()
            self.event("warn", f"Wrote off {position.qty:.8f} BTC of dust (below the exchange's minimum order, "
                               f"P&L ${loss:+,.2f}); it stays in your exchange account")
            return self._hold(now, "position too small to sell (dust written off)")
        try:
            fill = await self.broker.sell(position.qty, tick)
        except OrderStatusUnknown as exc:
            return self._order_status_unknown(now, "SELL", exc)
        except Exception as exc:
            self.event("error", f"SELL failed: {exc}")
            return self._hold(now, f"sell failed: {exc}")
        dust_left = not self.broker.is_sellable(position.qty - fill.qty, tick.price)
        record = self.ledger.apply_sell(fill, reason, close_remainder=dust_left)
        self._book(now, record, fill.order_id)
        self.event("trade", f"SELL {fill.qty:.6f} BTC @ {fill.price:,.2f}, P&L ${record.pnl_usd:+,.2f} — {reason}")
        return self._record_decision(now, "sell", reason)

    def _order_status_unknown(self, now: float, side: str, exc: Exception) -> Decision:
        reason = f"{side} order outcome unknown ({exc}). Check your exchange account, then press Reset"
        self.risk.kill(reason)
        self.persist()
        self.event("error", reason)
        return self._hold(now, reason)

    def _book(self, now: float, record: TradeRecord, order_id: str) -> None:
        self.risk.record_trade(now)
        self.trades.append(record)
        self.journal.record_trade(self.mode, record, order_id)
        self.persist()

    def _hold(self, now: float, reason: str) -> Decision:
        return self._record_decision(now, "hold", reason)

    def _record_decision(self, now: float, action: str, reason: str) -> Decision:
        details = self.signal.to_dict() if self.signal else {}
        self.decision = Decision(ts=now, action=action, reason=reason, details=details)  # type: ignore[arg-type]
        return self.decision

    # ------------------------------------------------------------------ controls
    def pause(self) -> None:
        self.paused = True
        self.persist()
        self.event("control", "Paused: no new positions will be opened (exits still active)")

    def resume(self) -> None:
        self.paused = False
        self.persist()
        self.event("control", "Resumed")

    async def kill(self) -> None:
        async with self._lock:
            self.risk.kill("manual kill switch")
            self.event("control", "KILL SWITCH: closing any position and halting until reset")
            if self.ledger.position is not None and self.tick is not None:
                await self._exit(self.clock(), self.tick, "manual kill switch")
            self.persist()

    def reset(self) -> None:
        if self.tick is None:  # the new loss baselines need a real valuation of any open position
            raise ValueError("No price yet: wait for the first price before resetting the limits")
        self.risk.reset(self.ledger.equity(self.tick.price), self.clock())
        self.event("control", "Risk limits reset; trading allowed again")
        self.persist()

    # ------------------------------------------------------------------ housekeeping
    async def _snapshot_loop(self) -> None:
        last_persist = 0.0
        while True:
            await asyncio.sleep(SNAPSHOT_SECONDS)
            now = self.clock()
            try:
                if self.tick is not None:
                    price = self.tick.price
                    equity = self.ledger.equity(price)
                    self.equity_points.append((now, equity))
                    btc = self.ledger.position.qty if self.ledger.position else 0.0
                    self.journal.record_equity(self.mode, now, equity, self.ledger.cash, btc, price)
                if now - last_persist >= PERSIST_SECONDS:
                    self.persist()
                    last_persist = now
            except Exception as exc:  # e.g. disk full: keep trading, but say so loudly
                self.event("error", f"Could not write the journal: {type(exc).__name__}: {exc}")

    async def _model_loop(self) -> None:
        """Train a first model if there is none, then keep it fresh if retraining is enabled."""
        cfg = self.settings.model
        while self.model is None and cfg.auto_train:
            if await self._train_and_install(first=True):
                break
            await asyncio.sleep(MODEL_RETRY_SECONDS)
        if cfg.retrain_every_hours > 0:
            await self._retrain_loop()

    async def _retrain_loop(self) -> None:
        period = self.settings.model.retrain_every_hours * 3600
        while True:
            await asyncio.sleep(period)
            await self._train_and_install(first=False)

    async def _train_and_install(self, first: bool) -> bool:
        """Download history, train a candidate and swap it in; returns whether a model was installed.

        The first model is always installed (calibration keeps a weak one from trading); a retrained
        candidate replaces the current model only if it validates at least as well.
        """
        cfg = self.settings.model
        self.training = True
        try:
            if first:
                self.event("info", f"Training the prediction model: downloading {cfg.training_days} days of "
                                   "price history first…")
            history = await self.feed.fetch_history(cfg.training_days)
            history = fill_gaps(clean_bars(drop_incomplete(history, self.tf, self.clock())), self.tf)
            candidate = await asyncio.to_thread(train_model, history, cfg.timeframe, cfg.horizon_bars)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.train_error = f"{type(exc).__name__}: {exc}"
            retry = f"; trying again in {MODEL_RETRY_SECONDS / 60:.0f} min" if first else ""
            self.event("warn", f"Model training failed: {self.train_error}{retry}")
            return False
        finally:
            self.training = False

        current_auc = self.model.metrics.get("auc", 0.5) if self.model else 0.5
        new_auc = candidate.metrics["auc"]
        better = new_auc >= max(0.5, current_auc - 0.005) and candidate.metrics["calibration_slope"] > 0
        if not (first or better):
            self.event("info", f"Retrained model rejected (AUC {new_auc:.4f} vs current {current_auc:.4f})")
            return False

        previous = self.model
        try:
            if not self.feed.simulated:
                try:
                    candidate.save(cfg.path)
                except Exception as exc:
                    if not first:
                        raise
                    self.event("warn", f"Could not save the model to {cfg.path} ({exc}); using it in memory only, "
                                       "so it will be trained again after a restart")
            self.model = candidate
            self._update_prediction()
        except Exception as exc:  # never let a retrain take the agent (and its stops) down
            self.model = previous
            self.train_error = f"{type(exc).__name__}: {exc}"
            self.event("error", f"Could not swap in the retrained model ({exc}); keeping the current one")
            return False
        self.train_error = None
        if first:
            note = "; barely better than chance, so it will rarely trade" if new_auc < 0.52 else ""
            self.event("info", f"Model trained and ready (out-of-sample AUC {new_auc:.4f}{note})")
        else:
            self.event("info", f"Model retrained and swapped in (AUC {current_auc:.4f} -> {new_auc:.4f})")
        return True

    def event(self, kind: str, message: str) -> None:
        self.events.append({"ts": self.clock(), "kind": kind, "message": message})
        level = {"error": logging.ERROR, "warn": logging.WARNING}.get(kind, logging.INFO)
        log.log(level, message)

    # ------------------------------------------------------------------ dashboard
    def snapshot(self) -> dict:
        now = self.clock()
        tick = self.tick
        price = tick.price if tick else None
        position = self.ledger.position
        equity = self.ledger.equity(price) if price else self.ledger.cash
        risk = self.risk.state
        if risk.killed:
            status = "killed"
        elif now < risk.halted_until:
            status = "halted"
        elif self.paused:
            status = "paused"
        else:
            status = "running"
        position_view = None
        if position is not None and price:
            position_view = {
                **position.to_dict(),
                "value": position.market_value(price),
                "unrealized_pnl": position.unrealized_pnl(price, self.settings.trading.fee_rate),
                "change": price / position.entry_price - 1,
                "stop": self.risk.stop_price(position),
                "take_profit": self.risk.take_profit_price(position),
            }
        step = max(1, len(self.prices) // 900)
        return json_safe({
            "now": now,
            "mode": self.mode,
            "simulated": self.feed.simulated,
            "exchange": self.feed.name,
            "feed_source": self.feed.source,
            "feed_error": self.feed.last_error,
            "symbol": self.settings.exchange.symbol,
            "status": status,
            "status_reason": risk.kill_reason or (risk.halt_reason if now < risk.halted_until else ""),
            "uptime": now - self.started_at,
            "price": None if tick is None else {
                "last": tick.price, "bid": tick.bid, "ask": tick.ask, "spread_bps": tick.spread_bps,
                "age": now - tick.received_at,
            },
            "signal": self.signal.to_dict() if self.signal else None,
            "decision": None if self.decision is None else {
                "ts": self.decision.ts, "action": self.decision.action, "reason": self.decision.reason,
            },
            "position": position_view,
            "account": {
                "capital": self.ledger.capital,
                "equity": equity,
                "cash": self.ledger.cash,
                "pnl": equity - self.ledger.capital,
                "pnl_today": equity - risk.day_start_equity if risk.day_start_equity else 0.0,
                "realized_pnl": self.ledger.realized_pnl,
                "fees_paid": self.ledger.fees_paid,
                "round_trips": self.ledger.round_trips,
                "win_rate": self.ledger.win_rate,
                "drawdown": (equity / risk.peak_equity - 1) if risk.peak_equity else 0.0,
            },
            "strategy": {
                "entry_threshold": self.settings.strategy.entry_threshold,
                "exit_threshold": self.settings.strategy.exit_threshold,
                "decision_interval_seconds": self.settings.strategy.decision_interval_seconds,
            },
            "risk": {
                "max_position_usd": self.settings.risk.max_position_usd,
                "stop_loss_pct": self.settings.risk.stop_loss_pct,
                "take_profit_pct": self.settings.risk.take_profit_pct,
                "max_daily_loss_pct": self.settings.risk.max_daily_loss_pct,
                "max_drawdown_pct": self.settings.risk.max_drawdown_pct,
                "trades_last_hour": len([t for t in risk.trade_times if now - t < 3600]),
                "max_trades_per_hour": self.settings.risk.max_trades_per_hour,
                "fee_rate": self.settings.trading.fee_rate,
            },
            "model": None if self.model is None else self.model.summary() | {"metrics": {
                k: v for k, v in self.model.metrics.items() if k != "folds"
            }},
            "model_status": "ready" if self.model is not None else ("training" if self.training else "missing"),
            "model_error": None if self.model is not None else self.train_error,
            "prices": [list(p) for p in list(self.prices)[::step]],
            "equity_curve": [list(p) for p in self.equity_points],
            "trades": [t.to_dict() for t in list(self.trades)[-50:]][::-1],
            "news": self.news.snapshot(25) if self.news is not None else None,
            "events": list(self.events)[-40:][::-1],
        })


async def _finish_even_if_cancelled(coro):
    """Run an order and its booking to completion even if the agent is shutting down meanwhile,
    so an order the exchange accepted is never left out of the books."""
    task = asyncio.ensure_future(coro)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


def json_safe(value):
    """Replace NaN/inf (invalid in JSON) with None and unwrap numpy scalars, recursively."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

