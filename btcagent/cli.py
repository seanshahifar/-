"""Command line: btcagent download | train | backtest | run | news."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time
from dataclasses import replace
from pathlib import Path

import ccxt
import pandas as pd
from dotenv import load_dotenv

from btcagent.config import TIMEFRAME_SECONDS, ConfigError, Settings

log = logging.getLogger("btcagent")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btcagent",
        description="Autonomous Bitcoin trading agent (paper trading by default).",
    )
    parser.add_argument("--config", default="config.toml", help="config file (default: config.toml)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set trading.fee_rate=0.001 (repeatable)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    def data_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--exchange", help="exchange to take price history from (default: exchange.id)")
        p.add_argument("--symbol", help="market symbol for that exchange, e.g. BTC/USDT")
        p.add_argument("--data", type=Path, help="explicit path to a bars CSV")

    p = sub.add_parser("download", help="download historical price bars for training")
    data_source_args(p)
    p.add_argument("--days", type=float, help="days of history (default: model.training_days)")

    p = sub.add_parser("train", help="train the prediction model on downloaded bars")
    data_source_args(p)
    p.add_argument("--days", type=float, help="train on the most recent N days (default: model.training_days)")

    p = sub.add_parser("backtest", help="out-of-sample backtest, with a fee-sensitivity table")
    data_source_args(p)
    p.add_argument("--test-days", type=float, default=7.0, help="most recent days held out for testing (default 7)")
    p.add_argument("--use-saved-model", action="store_true", help="test the saved model instead of retraining")

    p = sub.add_parser("run", help="run the trading agent with its live dashboard")
    p.add_argument("--sim", action="store_true", help="simulated market (offline testing, synthetic prices)")
    p.add_argument("--fresh", action="store_true", help="start new paper books instead of resuming saved state")
    p.add_argument("--no-news", action="store_true", help="disable news ingestion")
    p.add_argument("--no-dashboard", action="store_true", help="run without the web dashboard")
    p.add_argument("--host", help="dashboard host (default: dashboard.host)")
    p.add_argument("--port", type=int, help="dashboard port (default: dashboard.port)")
    p.add_argument("--flatten-on-exit", action="store_true", help="sell any open position when the agent stops")

    sub.add_parser("news", help="fetch and score the latest headlines once")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("ccxt", "httpx", "httpx2", "anthropic", "urllib3", "asyncio", "uvicorn.error"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = Settings.load(args.config, args.set)
        handler = {"download": cmd_download, "train": cmd_train, "backtest": cmd_backtest,
                   "run": cmd_run, "news": cmd_news}[args.command]
        return handler(settings, args) or 0
    except ConfigError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    except ccxt.BaseError as exc:
        print(f"\nExchange error ({type(exc).__name__}): {exc}\nCheck your internet connection and that "
              "exchange.id and exchange.symbol are right for your exchange.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


# ---------------------------------------------------------------------- data
def source_settings(settings: Settings, args: argparse.Namespace):
    return replace(settings.exchange, id=args.exchange or settings.exchange.id,
                   symbol=args.symbol or settings.exchange.symbol)


def bars_path(settings: Settings, args: argparse.Namespace) -> Path:
    from btcagent.market.exchange import history_path

    if args.data:
        return args.data
    source = source_settings(settings, args)
    return history_path(settings.storage.data_dir, source.id, source.symbol, settings.model.timeframe)


def load_history(settings: Settings, args: argparse.Namespace) -> pd.DataFrame:
    from btcagent.market.exchange import clean_bars, fill_gaps, load_bars

    path = bars_path(settings, args)
    if not path.exists():
        raise FileNotFoundError(f"No price history at {path}. Run `btcagent download` first.")
    return fill_gaps(clean_bars(load_bars(path)), settings.timeframe_seconds)


def cmd_download(settings: Settings, args: argparse.Namespace) -> int:
    from btcagent.market.exchange import (create_exchange, empty_bars, epoch_ms, fetch_ohlcv_history, load_bars,
                                          merge_bars, save_bars)

    source = source_settings(settings, args)
    timeframe = settings.model.timeframe
    days = args.days or settings.model.training_days
    path = bars_path(settings, args)
    tf_ms = TIMEFRAME_SECONDS[timeframe] * 1000

    async def download() -> None:
        exchange = create_exchange(source)
        try:
            await exchange.load_markets()
            if source.symbol not in exchange.markets:
                raise ConfigError(f"{source.id} has no market {source.symbol!r}; try --symbol BTC/USDT or BTC/USD")
            existing = load_bars(path) if path.exists() else empty_bars()
            since = exchange.milliseconds() - int(days * 86_400_000)
            if not existing.empty and epoch_ms(existing.index)[0] <= since:
                since = int(epoch_ms(existing.index)[-1]) + tf_ms  # only fetch what's missing
            print(f"Downloading {source.symbol} {timeframe} bars from {source.id} ({days:g} days)…")

            def progress(done: int, total: int) -> None:
                print(f"\r  {min(100.0, 100 * done / max(total, 1)):5.1f}%", end="", flush=True)

            new = await fetch_ohlcv_history(exchange, source.symbol, timeframe, since, progress=progress)
            print()
            merged = merge_bars(existing, new)
            if merged.empty:
                raise ConfigError("The exchange returned no data.")
            save_bars(merged, path)
            print(f"Saved {len(merged):,} bars ({merged.index[0]} -> {merged.index[-1]}) to {path}")
            if source.id == "kraken":
                print("Note: Kraken's OHLC endpoint only serves the most recent 720 bars. For training data "
                      "use an exchange with deep history, e.g. --exchange coinbase or --exchange bitstamp.")
        finally:
            await exchange.close()

    asyncio.run(download())
    return 0


# ---------------------------------------------------------------------- model
def print_cost_context(settings: Settings, typical_move_bps: float) -> None:
    cost = 2 * (settings.trading.fee_rate * 1e4 + settings.trading.slippage_bps)
    print(f"\nCosts: at fee_rate {settings.trading.fee_rate:.2%} a round trip costs ~{cost:.0f} bps; "
          f"the typical {settings.model.horizon_bars}-bar move is {typical_move_bps:.0f} bps.")
    if typical_move_bps < cost * settings.strategy.min_edge_multiple:
        print("  The agent buys only when the expected edge covers the costs, so with these fees it will trade\n"
              "  rarely or never. Lower fees (maker orders, a cheaper exchange or tier) or a longer horizon\n"
              "  (model.timeframe / model.horizon_bars) are the levers. See README: 'Fees decide everything'.")


def cmd_train(settings: Settings, args: argparse.Namespace) -> int:
    from btcagent.model import describe_metrics, train_model

    bars = load_history(settings, args)
    days = args.days or settings.model.training_days
    bars = bars[bars.index >= bars.index[-1] - pd.Timedelta(days=days)]
    print(f"Training on {len(bars):,} {settings.model.timeframe} bars ({bars.index[0]} -> {bars.index[-1]}), "
          f"predicting {settings.model.horizon_bars} bars ahead…")
    model = train_model(bars, settings.model.timeframe, settings.model.horizon_bars)
    print("\nWalk-forward validation (every score is on data the model never trained on):")
    for line in describe_metrics(model.metrics):
        print("  " + line)
    model.save(settings.model.path)
    print(f"\nSaved model to {settings.model.path}")
    print_cost_context(settings, model.metrics["typical_move_bps"])
    return 0


def cmd_backtest(settings: Settings, args: argparse.Namespace) -> int:
    from btcagent.backtest import fee_sensitivity, run_backtest
    from btcagent.model import TrainedModel, describe_metrics, train_model

    bars = load_history(settings, args)
    test_start = bars.index[-1] - pd.Timedelta(days=args.test_days)
    if args.use_saved_model:
        model = TrainedModel.load(settings.model.path)
        if pd.Timestamp(model.data_end) >= test_start:
            print("WARNING: the saved model was trained on part of the test period; results will be optimistic.")
    else:
        gap = pd.Timedelta(seconds=settings.timeframe_seconds * settings.model.horizon_bars)
        train_bars = bars[bars.index < test_start - gap]
        train_bars = train_bars[train_bars.index >= train_bars.index[-1] - pd.Timedelta(days=settings.model.training_days)]
        print(f"Training on {len(train_bars):,} bars before {test_start} (the test period is never seen)…")
        model = train_model(train_bars, settings.model.timeframe, settings.model.horizon_bars)
        for line in describe_metrics(model.metrics):
            print("  " + line)

    result = run_backtest(bars, model, settings, start=test_start)
    m = result.metrics
    print(f"\nBacktest {m['start']} -> {m['end']}  (fee {m['fee_rate']:.2%}/trade, news signal neutral)")
    rows = [
        ("Strategy return", f"{m['total_return']:+.2%}"),
        ("Buy & hold return", f"{m['buy_and_hold_return']:+.2%}"),
        ("Max drawdown", f"{m['max_drawdown']:.2%}"),
        ("Sharpe (annualised)", f"{m['sharpe']:.2f}"),
        ("Round trips", f"{m['round_trips']}"),
        ("Win rate", "-" if m["win_rate"] is None else f"{m['win_rate']:.1%}"),
        ("Avg P&L per trade", "-" if m["avg_trade_pnl"] is None else f"${m['avg_trade_pnl']:+.2f}"),
        ("Fees paid", f"${m['fees_paid']:.2f}"),
        ("Time in market", f"{m['exposure']:.1%}"),
    ]
    for label, value in rows:
        print(f"  {label:<22}{value:>12}")

    print("\nFee sensitivity (same model and period):")
    print(f"  {'fee/trade':>10} {'return':>9} {'trades':>7} {'win rate':>9} {'fees':>10}")
    for row in fee_sensitivity(bars, model, settings, start=test_start):
        win = "-" if row["win_rate"] is None else f"{row['win_rate']:.0%}"
        fees = f"${row['fees_paid']:,.2f}"
        print(f"  {row['fee_rate']:>10.2%} {row['total_return']:>+9.2%} {row['round_trips']:>7} {win:>9} {fees:>10}")
    print("\nA backtest is a sanity check, not a promise: it can't model news reactions, outages or your\n"
          "exchange's exact fills, and past patterns stop working. Paper trade before risking money.")
    print_cost_context(settings, model.metrics["typical_move_bps"])
    return 0


# ---------------------------------------------------------------------- run
def cmd_run(settings: Settings, args: argparse.Namespace) -> int:
    if args.sim and settings.is_live:
        raise ConfigError("--sim can't be combined with live trading")
    settings.require_live_ready()
    if args.no_news:
        settings.news.enabled = False
    host = args.host or settings.dashboard.host
    port = args.port or settings.dashboard.port
    dashboard = settings.dashboard.enabled and not args.no_dashboard
    try:
        asyncio.run(run_agent(settings, args, dashboard, host, port))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    print("Agent stopped.")
    return 0


def load_model(settings: Settings):
    from btcagent.model import TrainedModel

    path = Path(settings.model.path)
    if not path.exists():
        log.warning("No trained model at %s: the agent will watch but not buy. Run `btcagent download` "
                    "then `btcagent train`.", path)
        return None
    model = TrainedModel.load(path)
    if model.timeframe != settings.model.timeframe or model.horizon_bars != settings.model.horizon_bars:
        raise ConfigError(
            f"The saved model predicts {model.horizon_bars} x {model.timeframe} bars, but config.toml says "
            f"{settings.model.horizon_bars} x {settings.model.timeframe}. Retrain with `btcagent train`."
        )
    return model


def make_server(config):
    import uvicorn

    class Server(uvicorn.Server):
        @contextlib.contextmanager
        def capture_signals(self):  # leave Ctrl-C to asyncio so the agent saves state and stops cleanly
            yield

    return Server(config)


async def run_agent(settings: Settings, args: argparse.Namespace, dashboard: bool, host: str, port: int) -> None:
    from btcagent.broker.live import LiveBroker
    from btcagent.broker.paper import PaperBroker
    from btcagent.engine import TradingAgent
    from btcagent.market.exchange import create_exchange
    from btcagent.market.feed import CcxtFeed
    from btcagent.market.simulation import SimulatedFeed
    from btcagent.model import describe_metrics, train_model
    from btcagent.news.service import NewsService
    from btcagent.storage import Journal

    tf = settings.timeframe_seconds
    if args.sim:
        log.warning("SIMULATED MARKET: synthetic prices for offline testing. Nothing here reflects real trading.")
        feed = SimulatedFeed(tf, history_bars=6_000)
        log.info("Training a throwaway model on the simulated history…")
        model = await asyncio.to_thread(train_model, feed.bars, settings.model.timeframe, settings.model.horizon_bars)
        for line in describe_metrics(model.metrics):
            log.info("  %s", line)
        broker = PaperBroker(settings.trading.fee_rate, settings.trading.slippage_bps)
        journal = Journal(":memory:")
        mode = "sim"
    else:
        model = load_model(settings)
        feed = CcxtFeed(settings.exchange, settings.model.timeframe)
        journal = Journal(settings.storage.journal_path)
        mode = settings.trading.mode
        if settings.is_live:
            exchange = create_exchange(settings.exchange, settings.secrets)
            broker = LiveBroker(exchange, settings.exchange.symbol, settings.trading.fee_rate)
        else:
            broker = PaperBroker(settings.trading.fee_rate, settings.trading.slippage_bps)
        if args.fresh and settings.is_live:
            saved = journal.load_state(f"{mode}:agent")
            if saved and saved["ledger"].get("position"):
                raise ConfigError("Refusing --fresh in live mode while the agent holds a position; close it first.")

    news = NewsService(settings.news) if settings.news.enabled else None
    agent = TradingAgent(settings, feed, broker, journal, model=model, news=news, mode=mode, fresh=args.fresh)

    server = None
    if dashboard:
        import uvicorn

        from btcagent.web.server import create_app

        app = create_app(agent, host=host, token=settings.secrets.dashboard_token)
        server = make_server(uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False))
        query = f"?token={settings.secrets.dashboard_token}" if settings.secrets.dashboard_token else ""
        log.info("Dashboard: http://%s:%d/%s", "localhost" if host in ("127.0.0.1", "::1") else host, port, query)

    stop = asyncio.Event()  # SIGTERM (e.g. `systemctl stop`) shuts down as gracefully as Ctrl-C
    with contextlib.suppress(NotImplementedError, RuntimeError):
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop.set)

    tasks = [asyncio.create_task(agent.run(), name="agent")]
    if server is not None:
        tasks.append(asyncio.create_task(server.serve(), name="dashboard"))
    stopper = asyncio.create_task(stop.wait(), name="stop")
    try:
        done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            if task is not stopper:
                task.result()
    finally:
        stopper.cancel()
        if server is not None:
            server.should_exit = True
        tasks[0].cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await agent.shutdown(flatten=args.flatten_on_exit)
        journal.close()


# ---------------------------------------------------------------------- news
def cmd_news(settings: Settings, args: argparse.Namespace) -> int:
    import aiohttp

    from btcagent.news.service import NewsService

    async def fetch() -> NewsService:
        service = NewsService(settings.news)
        async with aiohttp.ClientSession(trust_env=True) as session:
            await service.poll_once(session)
        return service

    service = asyncio.run(fetch())
    now = time.time()
    items = service.signal.recent(40)
    print(f"Scored by: {service.scorer_name}.  Feeds failing: {len(service.source.errors)}/{len(service.source.feeds)}")
    for url, error in service.source.errors.items():
        print(f"  ! {url}: {error}")
    print(f"\n{'sent':>6} {'rel':>4} {'imp':>4} {'age':>6}  headline")
    for item in items:
        age_min = (now - item.published_at) / 60
        print(f"{item.sentiment:>+6.2f} {item.relevance:>4.2f} {item.impact:>4.2f} {age_min:>5.0f}m  "
              f"{item.title[:100]}  [{item.source}]")
    print(f"\nAggregate news score now: {service.score(now):+.3f}  (-1 very bearish .. +1 very bullish)")
    return 0
