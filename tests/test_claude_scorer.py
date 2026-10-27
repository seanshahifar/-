"""Exercise ClaudeNewsScorer through the real anthropic SDK with a mocked HTTP transport.

This checks the exact request that goes over the wire (structured output schema, effort,
server-side refusal fallback) and every failure path, without needing an API key.
"""

import asyncio
import json

import anthropic
import httpx2
import pytest

from btcagent.domain import NewsItem
from btcagent.news.claude import FALLBACK_BETA, ClaudeNewsScorer

NOW = 1_800_000_000.0


def items(n=2):
    return [NewsItem(id=f"id{i}", title=f"Bitcoin headline {i}", url="", source="src", published_at=NOW - 60)
            for i in range(n)]


def message(text=None, stop_reason="end_turn"):
    content = [{"type": "text", "text": text}] if text is not None else []
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 60},
    }


def make_scorer(handler, model="claude-opus-5-5"):
    seen = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        base_url="https://api.anthropic.test",
        max_retries=0,
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(record)),
    )
    return ClaudeNewsScorer(model=model, effort="low", batch_size=10, client=client), seen


def ok_handler(request):
    body = json.loads(request.content)
    payload = json.loads(body["messages"][0]["content"].split("\n", 1)[1])
    results = [{"id": p["id"], "sentiment": 3.0 if p["id"] == "n0" else -0.4, "relevance": 0.9, "impact": -1,
                "rationale": "test"} for p in payload]
    return httpx2.Response(200, json=message(json.dumps({"results": results})))


def test_request_shape_and_parsing():
    scorer, seen = make_scorer(ok_handler)
    scores = asyncio.run(scorer.score(items(2)))

    request = seen[0]
    body = json.loads(request.content)
    assert body["model"] == "claude-opus-5-5"
    assert body["fallbacks"] == "default"
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["output_config"]["effort"] == "low"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["required"] == ["results"]
    assert "temperature" not in body and "top_p" not in body
    assert "untrusted" in body["system"]

    assert scores["id0"].sentiment == 1.0  # clamped from 3.0
    assert scores["id0"].impact == 0.0  # clamped from -1
    assert scores["id1"].sentiment == -0.4
    assert scorer.stats["scored"] == 2 and scorer.stats["input_tokens"] == 120


def test_models_without_server_fallback_do_not_send_it():
    scorer, seen = make_scorer(ok_handler, model="claude-haiku-5-5")
    asyncio.run(scorer.score(items(1)))
    body = json.loads(seen[0].content)
    assert "fallbacks" not in body
    assert FALLBACK_BETA not in seen[0].headers.get("anthropic-beta", "")


def test_items_are_batched():
    scorer, seen = make_scorer(ok_handler)
    scores = asyncio.run(scorer.score(items(25)))
    assert len(seen) == 3 and len(scores) == 25


@pytest.mark.parametrize("response, still_active", [
    (lambda r: httpx2.Response(200, json=message(stop_reason="refusal")), True),
    (lambda r: httpx2.Response(200, json=message('{"results": [', stop_reason="max_tokens")), True),
    (lambda r: httpx2.Response(200, json=message("not json")), True),
    (lambda r: httpx2.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}), True),
    (lambda r: httpx2.Response(529, json={"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}), True),
    (lambda r: httpx2.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "bad"}}), False),
    (lambda r: httpx2.Response(404, json={"type": "error", "error": {"type": "not_found_error", "message": "model"}}), False),
])
def test_failures_fall_back_to_lexicon_scores(response, still_active):
    scorer, _ = make_scorer(response)
    assert asyncio.run(scorer.score(items(2))) == {}
    assert scorer.active is still_active
    assert scorer.last_error


def test_connection_errors_are_survivable():
    def boom(request):
        raise httpx2.ConnectError("no route", request=request)

    scorer, _ = make_scorer(boom)
    assert asyncio.run(scorer.score(items(1))) == {}
    assert scorer.active
