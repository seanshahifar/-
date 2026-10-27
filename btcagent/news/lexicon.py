"""Fast, offline headline scoring with a crypto-specific lexicon.

Used instantly for every new headline (and as the fallback when Claude is unavailable).
It understands multi-word phrases ("all-time high", "etf outflows"), negation
("SEC does not approve"), intensifiers ("sharply") and relevance to bitcoin.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

PHRASES: dict[str, float] = {
    "all-time high": 0.9, "all time high": 0.9, "record high": 0.8, "new high": 0.6, "fresh high": 0.6,
    "highest level": 0.5, "breaks above": 0.6, "reclaims": 0.5,
    "record low": -0.8, "multi-month low": -0.6, "multi-year low": -0.7, "new low": -0.6, "lowest level": -0.6,
    "falls below": -0.6, "drops below": -0.6, "slips below": -0.5, "breaks below": -0.6, "under pressure": -0.5,
    "rate cut": 0.5, "rate cuts": 0.5, "cuts rates": 0.5, "rate hike": -0.5, "rate hikes": -0.5, "raises rates": -0.5,
    "etf approval": 0.9, "approves etf": 0.9, "etf inflows": 0.7, "net inflows": 0.6,
    "etf outflows": -0.7, "net outflows": -0.6,
    "short squeeze": 0.6, "long squeeze": -0.6, "sell-off": -0.7, "sell off": -0.7, "flash crash": -0.9,
    "legal tender": 0.7, "strategic reserve": 0.8, "bitcoin reserve": 0.7,
    "buys bitcoin": 0.6, "bought bitcoin": 0.6, "adds bitcoin": 0.6, "buy bitcoin": 0.4,
    "sells bitcoin": -0.6, "sold bitcoin": -0.6, "dumps bitcoin": -0.7,
    "golden cross": 0.5, "death cross": -0.5, "risk-off": -0.5, "risk off": -0.5, "risk-on": 0.5,
    "green light": 0.6, "files for bankruptcy": -0.9, "chapter 11": -0.8, "rug pull": -0.8, "exit scam": -0.9,
    "whales accumulate": 0.5, "whale accumulation": 0.5, "whale selling": -0.5, "mt. gox": -0.3, "mt gox": -0.3,
}

WORDS: dict[str, float] = {
    # bullish
    "surge": 0.8, "surges": 0.8, "surged": 0.8, "surging": 0.8, "soar": 0.9, "soars": 0.9, "soared": 0.9,
    "skyrockets": 1.0, "skyrocketed": 1.0, "rocket": 0.7, "rockets": 0.7, "moon": 0.6, "moons": 0.6,
    "rally": 0.7, "rallies": 0.7, "rallied": 0.7, "rallying": 0.7, "jump": 0.6, "jumps": 0.6, "jumped": 0.6,
    "climb": 0.5, "climbs": 0.5, "climbed": 0.5, "gain": 0.4, "gains": 0.4, "gained": 0.4,
    "rise": 0.4, "rises": 0.4, "rose": 0.4, "rising": 0.4, "rebound": 0.5, "rebounds": 0.5, "rebounded": 0.5,
    "recover": 0.4, "recovers": 0.4, "recovery": 0.4, "breakout": 0.6, "bullish": 0.7, "bulls": 0.4,
    "inflows": 0.5, "inflow": 0.5, "approve": 0.6, "approves": 0.7, "approved": 0.7, "approval": 0.6,
    "adoption": 0.5, "adopts": 0.5, "adopt": 0.4, "accumulate": 0.4, "accumulates": 0.4, "accumulating": 0.4,
    "buying": 0.3, "upgrade": 0.4, "upgrades": 0.4, "outperforms": 0.4, "optimism": 0.4, "optimistic": 0.4,
    "boost": 0.5, "boosts": 0.5, "boosted": 0.5, "milestone": 0.4, "partnership": 0.3, "embraces": 0.5,
    "dovish": 0.5, "tops": 0.4, "wins": 0.4, "victory": 0.5, "greenlights": 0.6, "record": 0.2, "demand": 0.2,
    # bearish
    "crash": -1.0, "crashes": -1.0, "crashed": -1.0, "plunge": -0.9, "plunges": -0.9, "plunged": -0.9,
    "tumble": -0.7, "tumbles": -0.7, "tumbled": -0.7, "slump": -0.7, "slumps": -0.7, "slumped": -0.7,
    "slide": -0.5, "slides": -0.5, "slid": -0.5, "drop": -0.5, "drops": -0.5, "dropped": -0.5,
    "fall": -0.5, "falls": -0.5, "fell": -0.5, "falling": -0.5, "sink": -0.6, "sinks": -0.6, "sank": -0.6,
    "decline": -0.4, "declines": -0.4, "declined": -0.4, "dump": -0.7, "dumps": -0.7, "dumped": -0.7,
    "selloff": -0.7, "bearish": -0.7, "bears": -0.4, "liquidation": -0.5, "liquidations": -0.6,
    "liquidated": -0.6, "outflows": -0.5, "outflow": -0.5, "hack": -0.9, "hacked": -0.9, "hackers": -0.8,
    "exploit": -0.8, "exploited": -0.8, "breach": -0.7, "stolen": -0.8, "theft": -0.8, "scam": -0.7,
    "fraud": -0.8, "ponzi": -0.8, "lawsuit": -0.6, "sues": -0.6, "sued": -0.6, "charges": -0.5,
    "charged": -0.6, "indicted": -0.7, "indictment": -0.7, "investigation": -0.5, "probe": -0.5,
    "ban": -0.8, "bans": -0.8, "banned": -0.8, "crackdown": -0.7, "reject": -0.7, "rejects": -0.7,
    "rejected": -0.7, "rejection": -0.7, "delay": -0.4, "delays": -0.4, "delayed": -0.4,
    "bankruptcy": -0.9, "bankrupt": -0.9, "insolvent": -0.9, "insolvency": -0.9, "collapse": -0.9,
    "collapses": -0.9, "collapsed": -0.9, "fear": -0.4, "fears": -0.4, "panic": -0.7, "warns": -0.4,
    "warning": -0.4, "concerns": -0.3, "losses": -0.4, "loses": -0.4, "weakness": -0.4, "weakens": -0.4,
    "selling": -0.3, "hawkish": -0.5, "tariff": -0.3, "tariffs": -0.3, "recession": -0.5,
    "halts": -0.6, "halted": -0.6, "suspends": -0.6, "suspended": -0.6, "frozen": -0.6, "freezes": -0.6,
    "delists": -0.6, "delisted": -0.6, "fined": -0.5, "penalty": -0.4, "manipulation": -0.5,
    "capitulation": -0.7, "bloodbath": -0.9, "correction": -0.4, "downturn": -0.5, "sanctions": -0.4,
}

NEGATORS = {"not", "no", "never", "without", "denies", "denied", "deny", "fails", "failed", "won't",
            "isn't", "didn't", "doesn't", "unlikely", "__neg__"}
DISCOUNTERS = {"despite", "__disc__"}  # sentiment in the rest of the clause is mostly ignored
CLAUSE_BREAKS = {",", ";", ":", "?", "!", "but", "while", "as", "after", "though", "although"}
MULTIWORD_MARKERS = {"rules out": "__neg__", "shrugs off": "__disc__", "brushes off": "__disc__"}
INTENSIFIERS = {"massive": 1.4, "huge": 1.3, "sharply": 1.3, "sharp": 1.2, "biggest": 1.3, "major": 1.2,
                "historic": 1.3, "extreme": 1.3, "heavy": 1.2, "big": 1.1, "slightly": 0.6, "modest": 0.7,
                "modestly": 0.7, "minor": 0.7, "mild": 0.7}

BTC_TERMS = {"bitcoin", "bitcoins", "btc", "satoshi", "sats", "ibit", "fbtc", "gbtc", "microstrategy",
             "mstr", "saylor", "halving", "lightning network"}
CRYPTO_TERMS = {"crypto", "cryptocurrency", "cryptocurrencies", "digital asset", "digital assets", "stablecoin",
                "stablecoins", "tether", "usdt", "usdc", "blockchain", "coinbase", "binance", "kraken", "defi",
                "web3", "altcoin", "altcoins", "miners", "mining", "token", "tokens"}
MACRO_TERMS = {"fed", "fomc", "federal reserve", "powell", "interest rate", "interest rates", "inflation", "cpi",
               "ppi", "jobs report", "payrolls", "treasury", "yields", "dollar", "dxy", "nasdaq", "s&p",
               "stocks", "equities", "recession", "tariff", "tariffs", "liquidity", "risk assets"}
ALT_TERMS = {"ethereum", "eth", "ether", "solana", "sol", "xrp", "ripple", "dogecoin", "doge", "cardano",
             "bnb", "memecoin", "memecoins", "shiba", "pepe", "avalanche", "polkadot", "litecoin", "tron"}

HIGH_IMPACT = {"sec", "etf", "etfs", "fed", "fomc", "federal reserve", "powell", "cpi", "inflation", "blackrock",
               "fidelity", "microstrategy", "saylor", "treasury", "congress", "senate", "white house", "president",
               "law", "bill", "legislation", "regulation", "regulators", "ban", "hack", "hacked", "exploit",
               "bankruptcy", "collapse", "liquidations", "billion", "all-time high", "reserve", "government",
               "china", "tariffs", "lawsuit", "court", "approval", "approves", "rejects", "halving", "mt. gox",
               "tether", "doj", "cftc", "central bank", "emergency", "war", "sanctions", "rate cut", "rate hike"}
LOW_IMPACT = {"opinion", "analysis", "analyst", "analysts", "prediction", "predicts", "forecast", "explained",
              "explainer", "guide", "how to", "what is", "podcast", "weekly", "recap", "review", "interview",
              "op-ed", "newsletter", "technical analysis", "price analysis", "chart", "price today",
              "could", "might", "here's why", "here is why", "watch"}

_TOKEN_RE = re.compile(r"__\w+__|[a-z0-9$][a-z0-9$&'.\-]*|[,;:?!]")
_PHRASES_BY_LENGTH = sorted(PHRASES.items(), key=lambda kv: -len(kv[0]))


@dataclass(frozen=True, slots=True)
class LexiconScore:
    sentiment: float
    relevance: float
    impact: float


def _term_re(term: str) -> re.Pattern:
    return re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")


def _contains(text: str, term: str) -> bool:
    return _term_re(term).search(text) is not None


def _any(text: str, terms: set[str]) -> bool:
    return any(_contains(text, term) for term in terms)


def headline_sentiment(headline: str) -> float:
    """Signed sentiment in (-1, 1) from one pass over the tokens, honouring clause-level modifiers."""
    working = headline.lower()
    for marker, token in MULTIWORD_MARKERS.items():
        working = _term_re(marker).sub(f" {token} ", working)
    phrase_weights: dict[str, float] = {}
    for n, (phrase, weight) in enumerate(_PHRASES_BY_LENGTH):
        pattern = _term_re(phrase)
        if pattern.search(working):
            token = f"__p{n}__"
            phrase_weights[token] = weight
            working = pattern.sub(f" {token} ", working)

    total = 0.0
    clause: list[str] = []
    discounted = False
    for token in _TOKEN_RE.findall(working):
        if token in CLAUSE_BREAKS:
            clause, discounted = [], False
            continue
        weight = phrase_weights.get(token)
        if weight is None:
            weight = WORDS.get(token.strip(".'"))
        if weight is not None:
            if any(t in NEGATORS for t in clause[-3:]):
                weight *= -0.6
            if discounted:
                weight *= 0.3
            for t in clause[-2:]:
                weight *= INTENSIFIERS.get(t, 1.0)
            total += weight
        if token in DISCOUNTERS:
            discounted = True
        clause.append(token)
    return math.tanh(total / 1.2)


def score_headline(title: str, summary: str = "") -> LexiconScore:
    headline = title.lower()
    text = f"{headline} {summary.lower()[:300]}"
    question = "?" in headline
    sentiment = headline_sentiment(headline) * (0.5 if question else 1.0)

    if _any(text, BTC_TERMS):
        relevance = 1.0
    elif _any(text, CRYPTO_TERMS):
        relevance = 0.6
    elif _any(text, MACRO_TERMS):
        relevance = 0.45
    elif _any(text, ALT_TERMS):
        relevance = 0.25
    else:
        relevance = 0.1

    hits = sum(1 for term in HIGH_IMPACT if _contains(text, term))
    impact = min(1.0, 0.25 + 0.12 * hits)
    if question or _any(headline, LOW_IMPACT):
        impact *= 0.6
    return LexiconScore(sentiment=round(sentiment, 4), relevance=relevance, impact=round(impact, 4))
