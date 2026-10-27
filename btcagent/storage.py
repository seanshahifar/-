"""SQLite journal: every trade, equity snapshots, scored news, and the agent's resumable state."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from btcagent.domain import NewsItem
from btcagent.ledger import TradeRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY, ts REAL, mode TEXT, side TEXT, qty REAL, price REAL,
    fee_usd REAL, pnl_usd REAL, reason TEXT, order_id TEXT);
CREATE INDEX IF NOT EXISTS trades_mode_ts ON trades (mode, ts);
CREATE TABLE IF NOT EXISTS equity (
    ts REAL, mode TEXT, equity REAL, cash REAL, btc REAL, price REAL, PRIMARY KEY (mode, ts));
CREATE TABLE IF NOT EXISTS news (
    id TEXT PRIMARY KEY, published_at REAL, source TEXT, title TEXT, url TEXT,
    sentiment REAL, relevance REAL, impact REAL, scorer TEXT, rationale TEXT);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


class Journal:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        if str(path) != ":memory:":
            self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)

    def record_trade(self, mode: str, trade: TradeRecord, order_id: str = "") -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO trades (ts, mode, side, qty, price, fee_usd, pnl_usd, reason, order_id)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (trade.ts, mode, trade.side, trade.qty, trade.price, trade.fee_usd, trade.pnl_usd, trade.reason, order_id),
            )

    def recent_trades(self, mode: str, limit: int = 100) -> list[TradeRecord]:
        rows = self.conn.execute(
            "SELECT ts, side, qty, price, fee_usd, reason, pnl_usd FROM trades WHERE mode = ? ORDER BY ts DESC LIMIT ?",
            (mode, limit),
        ).fetchall()
        return [TradeRecord(**dict(row)) for row in reversed(rows)]

    def record_equity(self, mode: str, ts: float, equity: float, cash: float, btc: float, price: float) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO equity (ts, mode, equity, cash, btc, price) VALUES (?, ?, ?, ?, ?, ?)",
                (ts, mode, equity, cash, btc, price),
            )

    def equity_history(self, mode: str, since: float) -> list[tuple[float, float]]:
        rows = self.conn.execute(
            "SELECT ts, equity FROM equity WHERE mode = ? AND ts >= ? ORDER BY ts", (mode, since)
        ).fetchall()
        return [(row["ts"], row["equity"]) for row in rows]

    def record_news(self, item: NewsItem) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO news (id, published_at, source, title, url, sentiment, relevance, impact,"
                " scorer, rationale) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (item.id, item.published_at, item.source, item.title, item.url, item.sentiment, item.relevance,
                 item.impact, item.scorer, item.rationale),
            )

    def save_state(self, key: str, value: dict) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def load_state(self, key: str) -> dict | None:
        row = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row["value"]) if row else None

    def delete_state(self, key: str) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM state WHERE key = ?", (key,))

    def close(self) -> None:
        self.conn.close()
