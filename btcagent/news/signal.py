"""Turn a stream of scored headlines into one time-decayed news score in [-1, 1]."""

from __future__ import annotations

import math

from btcagent.domain import NewsItem


class NewsSignal:
    def __init__(self, half_life_minutes: float = 90.0, max_age_hours: float = 12.0):
        self.half_life = half_life_minutes * 60.0
        self.max_age = max_age_hours * 3600.0
        self.items: dict[str, NewsItem] = {}

    def upsert(self, item: NewsItem) -> None:
        self.items[item.id] = item

    def weight(self, item: NewsItem, now: float) -> float:
        age = max(0.0, now - item.published_at)
        return 0.5 ** (age / self.half_life)

    def score(self, now: float) -> float:
        """tanh of the decayed sum: one strong story moves it a lot, several aligned stories saturate near +/-1."""
        total = sum(item.contribution * self.weight(item, now) for item in self.items.values()
                    if now - item.published_at <= self.max_age)
        return math.tanh(total)

    def prune(self, now: float) -> None:
        self.items = {k: v for k, v in self.items.items() if now - v.published_at <= self.max_age}

    def recent(self, limit: int = 30) -> list[NewsItem]:
        return sorted(self.items.values(), key=lambda item: item.published_at, reverse=True)[:limit]
