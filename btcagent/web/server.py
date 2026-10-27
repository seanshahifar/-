"""Dashboard server: live state over WebSocket plus pause / resume / kill / reset controls.

Hardened for a tool that can move money. Locally: the Host header is validated (blocks DNS
rebinding), control requests need a custom header (blocks cross-site form posts) and WebSocket
origins are checked. When the dashboard is reachable from other machines (hosted online) a long
token is mandatory; it travels in a header / the first WebSocket message, never in a URL we serve.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from btcagent.config import ConfigError
from btcagent.engine import TradingAgent

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
CONTROL_HEADER = "x-btcagent-control"
TOKEN_HEADER = "x-dashboard-token"
PUSH_INTERVAL_SECONDS = 1.0
AUTH_TIMEOUT_SECONDS = 5.0
MIN_PUBLIC_TOKEN_LENGTH = 16
WS_POLICY_VIOLATION = 1008  # the page treats this close code as "wrong or missing token"

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",  # nobody may embed the page (and its kill switch) in a frame
    "Referrer-Policy": "no-referrer",
    "X-Robots-Tag": "noindex, nofollow",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; connect-src 'self' ws: wss:; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'none'; object-src 'none'"
    ),
}


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def create_app(agent: TradingAgent, host: str = "127.0.0.1", token: str = "") -> FastAPI:
    if not is_loopback(host):
        if not token:
            raise ConfigError(
                "The dashboard is reachable from other machines, so it needs a password: set the "
                "DASHBOARD_TOKEN environment variable to a long random value (at least "
                f"{MIN_PUBLIC_TOKEN_LENGTH} characters)."
            )
        if len(token) < MIN_PUBLIC_TOKEN_LENGTH:
            raise ConfigError(
                f"DASHBOARD_TOKEN is too short ({len(token)} characters). It protects the kill switch and "
                f"the controls on a public address, so use at least {MIN_PUBLIC_TOKEN_LENGTH} random characters."
            )

    app = FastAPI(title="btcagent dashboard", docs_url=None, redoc_url=None, openapi_url=None)
    allowed_hosts = sorted(LOOPBACK_HOSTS | {host}) if host not in ("0.0.0.0", "::") else ["*"]
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        if request.headers.get("x-forwarded-proto") == "https":  # behind a TLS-terminating host
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        if request.url.path.startswith("/api/") or request.url.path == "/healthz":
            response.headers["Cache-Control"] = "no-store"
        return response

    def token_ok(supplied: str | None) -> bool:
        if not token:
            return True
        # Compare bytes: compare_digest rejects non-ASCII str, which would turn a bad token into a 500.
        return supplied is not None and hmac.compare_digest(supplied.encode(), token.encode())

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        """For the hosting platform's health check: no authentication and no trading data."""
        return JSONResponse({"ok": True})

    @app.get("/api/state")
    async def state(request: Request) -> JSONResponse:
        if not token_ok(request.headers.get(TOKEN_HEADER)):
            raise HTTPException(status_code=401, detail="dashboard token required")
        return JSONResponse(agent.snapshot())

    @app.post("/api/control/{action}")
    async def control(action: str, request: Request) -> JSONResponse:
        if request.headers.get(CONTROL_HEADER) != "1":
            raise HTTPException(status_code=403, detail="missing control header")
        if not token_ok(request.headers.get(TOKEN_HEADER)):
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

    async def authenticated(websocket: WebSocket) -> bool:
        """The first message must be {"token": "..."}; nothing is sent to an unauthenticated client."""
        if not token:
            return True
        try:
            first = await asyncio.wait_for(websocket.receive_text(), AUTH_TIMEOUT_SECONDS)
            supplied = json.loads(first).get("token")
        except (asyncio.TimeoutError, WebSocketDisconnect, ValueError, AttributeError):
            return False
        return isinstance(supplied, str) and token_ok(supplied)

    async def reject(websocket: WebSocket) -> None:
        with contextlib.suppress(RuntimeError):  # the client may already be gone
            await websocket.close(code=WS_POLICY_VIOLATION)

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        request_host = (websocket.headers.get("host") or "").split(":")[0]
        if origin and urlparse(origin).hostname not in {request_host, *LOOPBACK_HOSTS}:
            await reject(websocket)
            return
        await websocket.accept()
        if not await authenticated(websocket):
            await reject(websocket)
            return
        try:
            while True:
                await websocket.send_json(agent.snapshot())
                await asyncio.sleep(PUSH_INTERVAL_SECONDS)
        except (WebSocketDisconnect, RuntimeError):
            return

    return app
