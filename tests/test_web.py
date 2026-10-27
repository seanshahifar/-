import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from btcagent.broker.paper import PaperBroker
from btcagent.config import ConfigError
from btcagent.engine import TradingAgent
from btcagent.market.simulation import SimulatedFeed
from btcagent.web.server import CONTROL_HEADER, create_app

from conftest import FakeModel, make_settings


@pytest.fixture
def agent(clock, journal_factory):
    settings = make_settings("news.enabled=false")
    feed = SimulatedFeed(60, history_bars=300, seed=1, clock=clock)
    agent = TradingAgent(settings, feed, PaperBroker(0.001, 1, clock), journal_factory(), model=FakeModel(0.5),
                         mode="sim", clock=clock)
    asyncio.run(agent.sync_bars())
    agent.on_tick(feed.step(clock()))
    return agent


def client(app):
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_dashboard_page_and_state(agent):
    c = client(create_app(agent))
    page = c.get("/")
    assert page.status_code == 200 and "BTC Agent" in page.text
    state = c.get("/api/state").json()
    assert state["status"] == "running" and state["price"]["last"] > 0


def test_controls_require_the_custom_header(agent):
    c = client(create_app(agent))
    assert c.post("/api/control/pause").status_code == 403  # a cross-site form post can't add headers
    assert c.post("/api/control/pause", headers={CONTROL_HEADER: "1"}).json()["status"] == "paused"
    assert c.post("/api/control/resume", headers={CONTROL_HEADER: "1"}).json()["status"] == "running"
    assert c.post("/api/control/kill", headers={CONTROL_HEADER: "1"}).json()["status"] == "killed"
    assert c.post("/api/control/explode", headers={CONTROL_HEADER: "1"}).status_code == 404


def test_token_protects_everything_when_set(agent):
    c = client(create_app(agent, token="s3cret"))
    assert c.get("/api/state").status_code == 401
    assert c.get("/api/state", headers={"x-dashboard-token": "s3cret"}).status_code == 200
    assert c.post("/api/control/pause", headers={CONTROL_HEADER: "1"}).status_code == 401
    assert c.post("/api/control/pause", headers={CONTROL_HEADER: "1", "x-dashboard-token": "s3cret"}).status_code == 200


def test_foreign_host_header_is_rejected(agent):
    c = TestClient(create_app(agent), base_url="http://attacker.example")  # DNS-rebinding shape
    assert c.get("/api/state").status_code == 400


LONG_TOKEN = "k3y-" + "x" * 28


def test_refuses_to_expose_without_a_strong_token(agent):
    with pytest.raises(ConfigError, match="DASHBOARD_TOKEN"):
        create_app(agent, host="0.0.0.0")
    with pytest.raises(ConfigError, match="too short"):
        create_app(agent, host="0.0.0.0", token="short")
    create_app(agent, host="0.0.0.0", token=LONG_TOKEN)
    create_app(agent, token="short")  # a dashboard only reachable from this machine may use any token


def test_healthz_is_public_and_reveals_nothing(agent):
    c = client(create_app(agent, host="0.0.0.0", token=LONG_TOKEN))
    response = c.get("/healthz")
    assert response.status_code == 200 and response.json() == {"ok": True}


def test_security_headers_on_every_response(agent):
    c = client(create_app(agent))
    for path in ("/", "/healthz", "/api/state"):
        headers = c.get(path).headers
        assert headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert "noindex" in headers["x-robots-tag"]
    assert c.get("/api/state").headers["cache-control"] == "no-store"
    assert "strict-transport-security" not in c.get("/").headers
    assert "max-age" in c.get("/", headers={"x-forwarded-proto": "https"}).headers["strict-transport-security"]


def test_a_non_ascii_token_is_rejected_not_a_server_error(agent):
    c = client(create_app(agent, token="s3cret"))
    assert c.get("/api/state", headers={"x-dashboard-token": "pässword".encode()}).status_code == 401


def test_websocket_needs_the_token_as_its_first_message(agent):
    c = client(create_app(agent, token=LONG_TOKEN))
    url = "ws://127.0.0.1:8000/ws"
    with c.websocket_connect(url) as ws:
        ws.send_json({"token": LONG_TOKEN})
        assert ws.receive_json()["mode"] == "sim"
    for wrong in ({"token": "wrong"}, {"token": 123}, {"unrelated": 1}, {"token": ""}):
        with c.websocket_connect(url) as ws:
            ws.send_json(wrong)
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
        assert closed.value.code == 1008
    with c.websocket_connect(url) as ws:
        ws.send_text("this is not json")
        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()


def test_websocket_that_never_authenticates_is_dropped(agent, monkeypatch):
    import btcagent.web.server as server

    monkeypatch.setattr(server, "AUTH_TIMEOUT_SECONDS", 0.2)
    c = client(create_app(agent, token=LONG_TOKEN))
    with c.websocket_connect("ws://127.0.0.1:8000/ws") as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_json()
    assert closed.value.code == 1008


def test_websocket_streams_and_checks_origin(agent):
    c = client(create_app(agent))
    url = "ws://127.0.0.1:8000/ws"  # TestClient would otherwise send Host: testserver
    with c.websocket_connect(url, headers={"origin": "http://127.0.0.1:8000"}) as ws:
        assert ws.receive_json()["mode"] == "sim"
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(url, headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
