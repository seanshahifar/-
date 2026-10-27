"""Dashboard server: live state over WebSocket plus pause / resume / kill / reset controls.

Hardened for a local tool that can move money: the Host header is validated (blocks DNS
rebinding), control requests need a custom header (blocks cross-site form posts), WebSocket
origins are checked, and a token is required whenever the server is reachable off-machine.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from btcagent.engine import TradingAgent

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CONTROL_HEADER = "x-btcagent-control"
PUSH_INTERVAL_SECONDS = 1.0


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def create_app(agent: TradingAgent, host: str = "127.0.0.1", token: str = "") -> FastAPI:
    if not is_loopback(host) and not token:
        raise ValueError("Refusing to expose the dashboard beyond localhost without DASHBOARD_TOKEN set")

    app = FastAPI(title="btcagent dashboard", docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = sorted(LOOPBACK_HOSTS | {host}) if host not in ("0.0.0.0", "::") else ["*"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    def token_ok(supplied: str | None) -> bool:
        return not token or (supplied is not None and hmac.compare_digest(supplied, token))

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    async def state(request: Request) -> JSONResponse:
        if not token_ok(request.headers.get("x-dashboard-token") or request.query_params.get("token")):
            raise HTTPException(status_code=401, detail="dashboard token required")
        return JSONResponse(agent.snapshot())

    @app.post("/api/control/{action}")
    async def control(action: str, request: Request) -> JSONResponse:
        if request.headers.get(CONTROL_HEADER) != "1":
            raise HTTPException(status_code=403, detail="missing control header")
        if not token_ok(request.headers.get("x-dashboard-token")):
            raise HTTPException(status_code=401, detail="dashboard token required")
        if action == "pause":
            agent.pause()
        elif action == "resume":
            agent.resume()
        elif action == "kill":
            await agent.kill()
        elif action == "reset":
            try:
                agent.reset()
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
        else:
            raise HTTPException(status_code=404, detail=f"unknown action {action!r}")
        return JSONResponse({"ok": True, "status": agent.snapshot()["status"]})

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        request_host = (websocket.headers.get("host") or "").split(":")[0]
        if origin and urlparse(origin).hostname not in {request_host, *LOOPBACK_HOSTS}:
            await websocket.close(code=1008)
            return
        if not token_ok(websocket.query_params.get("token")):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(agent.snapshot())
                await asyncio.sleep(PUSH_INTERVAL_SECONDS)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
