"""Blend the price model and the news signal, and only trade when the expected move pays for itself."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from btcagent.config import StrategyConfig, TradingConfig


@dataclass(frozen=True, slots=True)
class Signal:
    p_up: float | None  # calibrated model probability that price is higher after the horizon
    model_score: float  # 2 * p_up - 1, in [-1, 1]
    news_score: float  # time-decayed news sentiment, in [-1, 1]
    combined: float  # weighted blend, in [-1, 1]
    expected_move_bps: float  # typical absolute move over the horizon at current volatility
    edge_bps: float  # combined * expected move: the gain we expect before costs
    cost_bps: float  # round trip: 2 x (fee + slippage) + spread
    required_edge_bps: float  # cost x min_edge_multiple

    def to_dict(self) -> dict:
        return asdict(self)


class Strategy:
    def __init__(self, cfg: StrategyConfig, trading: TradingConfig):
        self.cfg = cfg
        self.trading = trading

    def round_trip_cost_bps(self, spread_bps: float) -> float:
        return 2 * (self.trading.fee_rate * 1e4 + self.trading.slippage_bps) + spread_bps

    def signal(self, p_up: float | None, news_score: float, expected_move: float | None, spread_bps: float) -> Signal:
        model_score = 2 * p_up - 1 if p_up is not None else 0.0
        total_weight = self.cfg.model_weight + self.cfg.news_weight
        combined = (self.cfg.model_weight * model_score + self.cfg.news_weight * news_score) / total_weight
        combined = max(-1.0, min(1.0, combined))
        move_bps = (expected_move or 0.0) * 1e4
        cost = self.round_trip_cost_bps(spread_bps)
        return Signal(
            p_up=p_up,
            model_score=model_score,
            news_score=news_score,
            combined=combined,
            expected_move_bps=move_bps,
            edge_bps=combined * move_bps,
            cost_bps=cost,
            required_edge_bps=cost * self.cfg.min_edge_multiple,
        )

    def conviction(self, signal: Signal) -> float:
        """0 at the entry threshold, 1 at a maximal signal; used for position sizing."""
        span = 1 - self.cfg.entry_threshold
        return max(0.0, min(1.0, (signal.combined - self.cfg.entry_threshold) / span)) if span > 0 else 1.0

    def should_enter(self, signal: Signal) -> tuple[bool, str]:
        if self.cfg.require_model and signal.p_up is None:
            return False, "no trained model loaded (run `btcagent train`)"
        if signal.combined < self.cfg.entry_threshold:
            return False, f"signal {signal.combined:+.3f} below entry threshold {self.cfg.entry_threshold:+.2f}"
        if signal.edge_bps < signal.required_edge_bps:
            return False, (
                f"expected edge {signal.edge_bps:.1f} bps < required {signal.required_edge_bps:.1f} bps "
                f"(round-trip cost {signal.cost_bps:.1f} bps)"
            )
        return True, f"signal {signal.combined:+.3f}, edge {signal.edge_bps:.1f} bps >= {signal.required_edge_bps:.1f} bps"

    def should_exit(self, signal: Signal) -> tuple[bool, str]:
        if signal.combined <= self.cfg.exit_threshold:
            return True, f"signal turned {signal.combined:+.3f} (exit at {self.cfg.exit_threshold:+.2f})"
        return False, ""
