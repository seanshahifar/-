"""Fetch crypto headlines from RSS / Atom feeds (no API keys needed)."""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import html
import logging
import re
from urllib.parse import urlparse

import aiohttp
import feedparser

from btcagent.domain import NewsItem

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; btcagent/0.1; news reader)"
_TAG_RE = re.compile(r"<[^>]+>")
_PUBLISHER_SUFFIX_RE = re.compile(r"\s+[-|–—]\s+[^-|–—]{2,40}$")
_NON_WORD_RE = re.compile(r"[^a-z0-9 ]+")


def clean_text(text: str | None, limit: int) -> str:
    text = html.unescape(_TAG_RE.sub(" ", text or ""))
    return " ".join(text.split())[:limit]


def story_id(title: str) -> str:
    """Stable id from the normalised headline, so one story syndicated by several feeds is counted once."""
    normalised = _PUBLISHER_SUFFIX_RE.sub("", title.lower())
    normalised = " ".join(_NON_WORD_RE.sub(" ", normalised).split())
    return hashlib.sha1(normalised.encode()).hexdigest()[:16]


def parse_feed(content: bytes, feed_url: str, now: float) -> list[NewsItem]:
    parsed = feedparser.parse(content)
    default_source = clean_text(parsed.feed.get("title"), 60) or urlparse(feed_url).netloc
    items = []
    for entry in parsed.entries:
        title = clean_text(entry.get("title"), 300)
        if not title:
            continue
        stamp = entry.get("published_parsed") or entry.get("updated_parsed")
        published = float(calendar.timegm(stamp)) if stamp else now
        source = entry.get("source")
        publisher = clean_text(source.get("title"), 60) if isinstance(source, dict) else ""
        items.append(
            NewsItem(
                id=story_id(title),
                title=title,
                url=entry.get("link", ""),
                source=publisher or default_source,
                published_at=min(published, now),  # some feeds carry future timestamps
                summary=clean_text(entry.get("summary"), 500),
            )
        )
    return items


class RssNewsSource:
    """Polls a list of feeds concurrently, using conditional GETs to avoid re-downloading."""

    def __init__(self, feeds: list[str], timeout_seconds: float = 15.0):
        self.feeds = list(feeds)
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.errors: dict[str, str] = {}
        self._validators: dict[str, dict[str, str]] = {}

    async def fetch(self, session: aiohttp.ClientSession, now: float) -> list[NewsItem]:
        results = await asyncio.gather(*(self._fetch_one(session, url, now) for url in self.feeds), return_exceptions=True)
        items: list[NewsItem] = []
        for url, result in zip(self.feeds, results):
            if isinstance(result, BaseException):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                message = f"{type(result).__name__}: {result}"
                if self.errors.get(url) != message:
                    log.warning("News feed %s failed: %s", urlparse(url).netloc, message)
                self.errors[url] = message
            else:
                self.errors.pop(url, None)
                items.extend(result)
        return items

    async def _fetch_one(self, session: aiohttp.ClientSession, url: str, now: float) -> list[NewsItem]:
        headers = {"User-Agent": USER_AGENT, **self._validators.get(url, {})}
        async with session.get(url, headers=headers, timeout=self.timeout) as response:
            if response.status == 304:
                return []
            response.raise_for_status()
            content = await response.read()
            validators = {}
            if etag := response.headers.get("ETag"):
                validators["If-None-Match"] = etag
            if modified := response.headers.get("Last-Modified"):
                validators["If-Modified-Since"] = modified
            self._validators[url] = validators
        return await asyncio.to_thread(parse_feed, content, url, now)
