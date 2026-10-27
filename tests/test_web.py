import asyncio

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from btcagent.broker.paper import PaperBroker
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


def test_refuses_to_expose_without_a_token(agent):
    with pytest.raises(ValueError):
        create_app(agent, host="0.0.0.0")
    create_app(agent, host="0.0.0.0", token="t")


def test_websocket_streams_and_checks_origin(agent):
    c = client(create_app(agent))
    url = "ws://127.0.0.1:8000/ws"  # TestClient would otherwise send Host: testserver
    with c.websocket_connect(url, headers={"origin": "http://127.0.0.1:8000"}) as ws:
        assert ws.receive_json()["mode"] == "sim"
    with pytest.raises(WebSocketDisconnect):
        with c.websocket_connect(url, headers={"origin": "https://evil.example"}) as ws:
            ws.receive_json()
