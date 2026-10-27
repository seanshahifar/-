"""News pipeline: poll feeds -> instant lexicon score -> Claude refinement -> decayed signal."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable

import aiohttp

from btcagent.config import NewsConfig
from btcagent.domain import NewsItem
from btcagent.news.claude import ClaudeNewsScorer
from btcagent.news.lexicon import score_headline
from btcagent.news.signal import NewsSignal
from btcagent.news.sources import RssNewsSource

log = logging.getLogger(__name__)

# Stories this old (in half-lives) barely move the signal; don't spend Claude calls on them.
CLAUDE_MAX_AGE_HALF_LIVES = 3.0


def claude_requested(cfg: NewsConfig) -> bool:
    if cfg.claude == "on":
        return True
    if cfg.claude == "off":
        return False
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


class NewsService:
    def __init__(
        self,
        cfg: NewsConfig,
        source: RssNewsSource | None = None,
        claude: ClaudeNewsScorer | None = None,
        on_item: Callable[[NewsItem], None] | None = None,
        clock: Callable[[], float] = time.time,
    ):
        self.cfg = cfg
        self.clock = clock
        self.source = source or RssNewsSource(cfg.feeds)
        if claude is None and claude_requested(cfg):
            claude = ClaudeNewsScorer(cfg.claude_model, cfg.claude_effort, cfg.claude_batch_size)
        self.claude = claude
        self.signal = NewsSignal(cfg.half_life_minutes, cfg.max_age_hours)
        self.on_item = on_item
        self.seen: set[str] = set()
        self.last_poll_at: float | None = None
        self.polls = 0

    @property
    def scorer_name(self) -> str:
        if self.claude is not None and self.claude.active:
            return f"claude ({self.claude.model})"
        return "lexicon"

    def score(self, now: float | None = None) -> float:
        return self.signal.score(self.clock() if now is None else now)

    async def poll_once(self, session: aiohttp.ClientSession) -> list[NewsItem]:
        now = self.clock()
        fetched = await self.source.fetch(session, now)
        fresh: list[NewsItem] = []
        for item in fetched:
            if item.id in self.seen or now - item.published_at > self.signal.max_age:
                continue
            self.seen.add(item.id)
            lexical = score_headline(item.title, item.summary)
            item.sentiment, item.relevance, item.impact = lexical.sentiment, lexical.relevance, lexical.impact
            item.scorer = "lexicon"
            self.signal.upsert(item)
            self._emit(item)
            fresh.append(item)

        if self.claude is not None and self.claude.active:
            horizon = CLAUDE_MAX_AGE_HALF_LIVES * self.signal.half_life
            worth_scoring = [item for item in fresh if now - item.published_at <= horizon]
            for item_id, score in (await self.claude.score(worth_scoring)).items():
                item = self.signal.items.get(item_id)
                if item is None:
                    continue
                item.sentiment, item.relevance, item.impact = score.sentiment, score.relevance, score.impact
                item.rationale, item.scorer = score.rationale, "claude"
                self._emit(item)

        self.signal.prune(now)
        self.last_poll_at = now
        self.polls += 1
        if fresh:
            log.info("News: %d new headline(s); news score now %+.2f", len(fresh), self.score(now))
        return fresh

    def _emit(self, item: NewsItem) -> None:
        if self.on_item is not None:
            self.on_item(item)

    async def run(self) -> None:
        async with aiohttp.ClientSession(trust_env=True) as session:
            while True:
                try:
                    await self.poll_once(session)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("News poll failed")
                await asyncio.sleep(self.cfg.poll_seconds)

    def snapshot(self, limit: int = 30) -> dict:
        now = self.clock()
        return {
            "score": round(self.score(now), 4),
            "scorer": self.scorer_name,
            "claude_error": self.claude.last_error if self.claude else None,
            "last_poll_at": self.last_poll_at,
            "feed_errors": len(self.source.errors),
            "feeds": len(self.source.feeds),
            "items": [
                {**item.to_dict(), "weight": round(self.signal.weight(item, now), 4)}
                for item in self.signal.recent(limit)
            ],
        }
