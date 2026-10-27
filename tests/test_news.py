import asyncio
import time

import pytest

from btcagent.config import NewsConfig
from btcagent.domain import NewsItem
from btcagent.news.claude import ClaudeScore
from btcagent.news.lexicon import score_headline
from btcagent.news.service import NewsService, claude_requested
from btcagent.news.signal import NewsSignal
from btcagent.news.sources import parse_feed, story_id

NOW = 1_800_000_000.0


@pytest.mark.parametrize("headline", [
    "Bitcoin surges to all-time high as ETF inflows hit record",
    "SEC approves spot Bitcoin ETFs in historic decision",
    "MicroStrategy buys bitcoin worth $500 million",
    "Bitcoin shrugs off hawkish Fed, climbs to $70K",
    "Bitcoin rises despite fears of Fed rate hike",
])
def test_bullish_headlines_score_positive(headline):
    assert score_headline(headline).sentiment > 0.05


@pytest.mark.parametrize("headline", [
    "Bitcoin plunges 8% as liquidations top $1 billion",
    "Exchange hacked, $200 million in bitcoin stolen",
    "SEC does not approve spot bitcoin ETF, delays decision",
    "Bitcoin falls below $60K as risk-off mood grips markets",
    "Fed rules out rate cut this year",
])
def test_bearish_headlines_score_negative(headline):
    assert score_headline(headline).sentiment < -0.05


def test_negation_flips_and_questions_are_dampened():
    assert score_headline("SEC approves bitcoin ETF").sentiment > 0 > score_headline("SEC does not approve bitcoin ETF").sentiment
    statement = score_headline("Bitcoin will crash")
    question = score_headline("Will bitcoin crash? Analysts weigh in")
    assert abs(question.sentiment) < abs(statement.sentiment)
    assert question.impact < statement.impact


def test_relevance_tiers():
    assert score_headline("Bitcoin hits record").relevance == 1.0
    assert score_headline("Crypto markets rally").relevance == 0.6
    assert score_headline("Fed raises interest rates").relevance == 0.45
    assert score_headline("Solana network upgrade goes live").relevance == 0.25
    assert score_headline("Nvidia earnings beat expectations").relevance == 0.1


def item(title="Bitcoin surges", sentiment=0.8, relevance=1.0, impact=0.8, age=0.0, item_id=None):
    return NewsItem(id=item_id or title, title=title, url="https://example.com", source="test",
                    published_at=NOW - age, sentiment=sentiment, relevance=relevance, impact=impact)


def test_news_signal_decays_by_half_life():
    signal = NewsSignal(half_life_minutes=60)
    signal.upsert(item(age=0))
    fresh = signal.score(NOW)
    later = NewsSignal(half_life_minutes=60)
    later.upsert(item(age=3600))
    import math
    assert math.atanh(later.score(NOW)) == pytest.approx(math.atanh(fresh) / 2)


def test_news_signal_is_bounded_and_prunes_old_items():
    signal = NewsSignal(half_life_minutes=60, max_age_hours=2)
    for i in range(20):
        signal.upsert(item(title=f"story {i}", sentiment=-1.0, impact=1.0))
    assert -1.0 <= signal.score(NOW) < -0.99
    signal.upsert(item(title="ancient", age=3 * 3600))
    signal.prune(NOW)
    assert "ancient" not in signal.items


def rss(entries):
    body = "".join(
        f"<item><title>{t}</title><link>https://ex.com/{i}</link>"
        f"<pubDate>{time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.gmtime(ts))}</pubDate>"
        f"<description>&lt;p&gt;Some &lt;b&gt;html&lt;/b&gt;&lt;/p&gt;</description></item>"
        for i, (t, ts) in enumerate(entries)
    )
    return f'<?xml version="1.0"?><rss version="2.0"><channel><title>Feed</title>{body}</channel></rss>'.encode()


def test_parse_feed_cleans_html_and_clamps_future_dates():
    items = parse_feed(rss([("Bitcoin up", NOW - 600), ("From the future", NOW + 9999)]), "https://ex.com/rss", NOW)
    assert items[0].summary == "Some html"
    assert items[0].published_at == pytest.approx(NOW - 600)
    assert items[1].published_at == NOW


def test_story_id_merges_syndicated_copies():
    assert story_id("Bitcoin tops $100K - CoinDesk") == story_id("Bitcoin tops $100K")
    assert story_id("Bitcoin tops $100K") != story_id("Bitcoin drops below $90K")


class FakeSource:
    def __init__(self, batches):
        self.batches = list(batches)
        self.feeds = ["https://ex.com/rss"]
        self.errors = {}

    async def fetch(self, session, now):
        return self.batches.pop(0) if self.batches else []


class FakeClaude:
    model = "fake"
    active = True
    last_error = None

    def __init__(self):
        self.seen = []

    async def score(self, items):
        self.seen.extend(items)
        return {i.id: ClaudeScore(-0.9, 1.0, 0.9, "context says bearish") for i in items}


def test_service_scores_instantly_then_refines_with_claude():
    fresh = NewsItem(id="a", title="Bitcoin surges to record", url="", source="s", published_at=NOW - 60)
    stale = NewsItem(id="b", title="Bitcoin surges again", url="", source="s", published_at=NOW - 20 * 3600)
    claude = FakeClaude()
    emitted = []
    service = NewsService(NewsConfig(), source=FakeSource([[fresh, stale], [fresh]]), claude=claude,
                          on_item=emitted.append, clock=lambda: NOW)
    new = asyncio.run(service.poll_once(session=None))
    assert [i.id for i in new] == ["a"]  # 20h-old story ignored
    assert service.signal.items["a"].scorer == "claude"
    assert service.signal.items["a"].sentiment == -0.9  # Claude overrode the naive lexicon reading
    assert len(emitted) == 2  # once for the lexicon score, once for the refinement
    assert asyncio.run(service.poll_once(session=None)) == []  # duplicates are ignored
    assert service.score() < 0


def test_claude_auto_mode_follows_the_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert not claude_requested(NewsConfig(claude="auto"))
    assert claude_requested(NewsConfig(claude="on"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert claude_requested(NewsConfig(claude="auto"))
    assert not claude_requested(NewsConfig(claude="off"))
