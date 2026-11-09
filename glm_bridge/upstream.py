"""Client for the glm-ai.chat WordPress REST endpoints.

Endpoints (reverse-engineered, unofficial; may change without notice):
    GET  /wp-json/glm-chat/v1/session  -> {"nonce": "...", ...}
    POST /wp-json/glm-chat/v1/message  -> text/event-stream of JSON events
    POST /wp-json/glm-chat/v1/image    -> image generation

Session handling deliberately keeps the cookie jar across nonce refreshes: the
bridge behaves like one long-lived browser tab, it does not mint fresh identities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import Settings
from .errors import BridgeError
from .sse import decode_json_event, iter_sse

log = logging.getLogger("glm_bridge.upstream")

SESSION_PATH = "/wp-json/glm-chat/v1/session"
MESSAGE_PATH = "/wp-json/glm-chat/v1/message"
IMAGE_PATH = "/wp-json/glm-chat/v1/image"

TERMINAL_TYPES = frozenset({"error", "limit", "notice", "offline"})
REASONING_TYPES = frozenset({"reasoning", "thinking", "reasoning_delta", "think"})


@dataclass(slots=True)
class UpstreamEvent:
    type: str
    data: dict[str, Any]

    @property
    def text(self) -> str:
        value = self.data.get("text")
        if value is None:
            value = self.data.get("content", "")
        return value if isinstance(value, str) else ""

    @property
    def message(self) -> str:
        value = self.data.get("message") or self.data.get("text") or ""
        return value if isinstance(value, str) else str(value)


class NonceRejected(Exception):
    """Upstream answered 401/403: the nonce expired or the session is blocked."""


class UpstreamHTTPError(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"HTTP {status}: {body[:200]}")
        self.status = status
        self.body = body


def event_from_json(obj: dict[str, Any], sse_event: str = "message") -> UpstreamEvent:
    etype = obj.get("type")
    if not isinstance(etype, str) or not etype:
        if sse_event and sse_event != "message":
            etype = sse_event
        elif "code" in obj and "message" in obj:  # WordPress REST error shape
            etype = "error"
        else:
            etype = "unknown"
    return UpstreamEvent(type=etype.lower(), data=obj)


async def _snippet(resp: httpx.Response, limit: int = 500) -> str:
    try:
        raw = await resp.aread()
    except httpx.HTTPError:
        return ""
    return raw[:limit].decode("utf-8", "replace")


def describe_upstream_body(body: str) -> str:
    """Pull a human-readable message out of a WordPress/JSON error body."""
    try:
        obj = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        obj = None
    if isinstance(obj, dict):
        for key in ("message", "error", "detail"):
            value = obj.get(key)
            if isinstance(value, str) and value:
                return value
            if isinstance(value, dict) and isinstance(value.get("message"), str):
                return value["message"]
    text = body.strip()
    if text.startswith("<"):
        return "non-JSON (HTML) response - possibly bot protection or a maintenance page"
    return text[:200] or "empty response"


@dataclass
class UpstreamState:
    nonce_fetched_at: float | None = None
    nonce_refreshes: int = 0
    session_info: dict[str, Any] = field(default_factory=dict)
    last_paid_left: Any = None
    last_error: str | None = None
    last_error_at: float | None = None
    requests: int = 0
    in_flight: int = 0


class GLMChatClient:
    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        self.base_url = settings.upstream_base_url
        kwargs: dict[str, Any] = {}
        if settings.upstream_proxy and transport is None:
            kwargs["proxy"] = settings.upstream_proxy
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"User-Agent": settings.upstream_user_agent},
            timeout=httpx.Timeout(settings.upstream_timeout, connect=settings.upstream_connect_timeout),
            follow_redirects=True,
            default_encoding="utf-8",
            transport=transport,
            **kwargs,
        )
        self._nonce: str | None = None
        self._nonce_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max(1, settings.upstream_concurrency))
        self.state = UpstreamState()

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- headers ----------------------------------------------------------
    def _headers(self, accept: str, *, nonce: str | None = None, json_body: bool = False) -> dict[str, str]:
        headers = {
            "Accept": accept,
            "Referer": self.base_url + "/",
            "X-Requested-With": self.settings.upstream_requested_with,
        }
        if json_body:
            headers["Content-Type"] = "application/json"
            headers["Origin"] = self.base_url
        if nonce:
            headers["X-GLM-Nonce"] = nonce
        return headers

    # -- session / nonce --------------------------------------------------
    async def _fetch_session(self) -> dict[str, Any]:
        try:
            resp = await self._client.get(SESSION_PATH, headers=self._headers("application/json"))
        except httpx.HTTPError as exc:
            self._record_error(f"session: {exc!r}")
            raise BridgeError(
                f"Could not reach upstream session endpoint: {exc.__class__.__name__}",
                status=502, code="upstream_unreachable", category="unavailable",
            ) from exc
        body = resp.text
        if resp.status_code != 200:
            self._record_error(f"session HTTP {resp.status_code}")
            raise BridgeError(
                f"Upstream session endpoint returned HTTP {resp.status_code}: {describe_upstream_body(body)}",
                status=502, code="upstream_session_failed",
                category="unavailable" if resp.status_code >= 500 else "auth",
            )
        try:
            data = resp.json()
        except (json.JSONDecodeError, ValueError):
            data = None
        if not isinstance(data, dict) or not isinstance(data.get("nonce"), str) or not data["nonce"]:
            self._record_error("session: no nonce")
            raise BridgeError(
                f"Upstream session response has no nonce: {describe_upstream_body(body)}",
                status=502, code="upstream_session_failed", category="unavailable",
            )
        return data

    async def nonce(self, *, refresh_from: str | None = None) -> str:
        """Return a valid nonce.

        ``refresh_from`` is the nonce that was just rejected: it is only replaced if
        nobody else replaced it already, so N concurrent 403s cause one refresh.
        """
        async with self._nonce_lock:
            now = time.monotonic()
            fresh = (
                self._nonce is not None
                and self.state.nonce_fetched_at is not None
                and now - self.state.nonce_fetched_at < self.settings.nonce_ttl
            )
            if refresh_from is None and fresh:
                return self._nonce  # type: ignore[return-value]
            if refresh_from is not None and self._nonce is not None and self._nonce != refresh_from:
                return self._nonce
            data = await self._fetch_session()
            if self._nonce is not None:
                self.state.nonce_refreshes += 1
            self._nonce = data["nonce"]
            self.state.nonce_fetched_at = time.monotonic()
            self.state.session_info = {k: v for k, v in data.items() if k != "nonce"}
            log.info("upstream session ready (refreshes=%d)", self.state.nonce_refreshes)
            return self._nonce

    # -- concurrency --------------------------------------------------------
    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(self._sem.acquire(), timeout=self.settings.upstream_queue_timeout)
        except asyncio.TimeoutError as exc:
            raise BridgeError(
                "Bridge is at its upstream concurrency limit; try again shortly",
                status=503, code="upstream_busy", category="busy", headers={"Retry-After": "5"},
            ) from exc
        self.state.in_flight += 1
        try:
            yield
        finally:
            self.state.in_flight -= 1
            self._sem.release()

    # -- chat ------------------------------------------------------------
    async def stream_message(self, body: dict[str, Any], nonce: str) -> AsyncIterator[UpstreamEvent]:
        """One attempt at the message endpoint. Raises NonceRejected / UpstreamHTTPError /
        httpx errors; yields parsed events. Holds a concurrency slot while open."""
        async with self.slot():
            self.state.requests += 1
            async with self._client.stream(
                "POST", MESSAGE_PATH, json=body,
                headers=self._headers("text/event-stream", nonce=nonce, json_body=True),
            ) as resp:
                if resp.status_code in (401, 403):
                    raise NonceRejected(await _snippet(resp))
                if resp.status_code != 200:
                    raise UpstreamHTTPError(resp.status_code, await _snippet(resp))
                async for sse in iter_sse(resp.aiter_lines()):
                    obj = decode_json_event(sse)
                    if obj is None:
                        continue
                    ev = event_from_json(obj, sse.event)
                    if ev.type == "done" and "paidLeft" in ev.data:
                        self.state.last_paid_left = ev.data.get("paidLeft")
                    yield ev

    # -- images ------------------------------------------------------------
    async def post_image(self, body: dict[str, Any], nonce: str) -> httpx.Response:
        async with self.slot():
            self.state.requests += 1
            resp = await self._client.post(
                IMAGE_PATH, json=body, headers=self._headers("*/*", nonce=nonce, json_body=True)
            )
            await resp.aread()
            if resp.status_code in (401, 403):
                raise NonceRejected(resp.text[:500])
            return resp

    # -- status ------------------------------------------------------------
    def _record_error(self, message: str) -> None:
        self.state.last_error = message
        self.state.last_error_at = time.time()

    def record_error(self, message: str) -> None:
        self._record_error(message)

    def status(self) -> dict[str, Any]:
        s = self.state
        age = None if s.nonce_fetched_at is None else round(time.monotonic() - s.nonce_fetched_at, 1)
        return {
            "base_url": self.base_url,
            "session_ready": self._nonce is not None,
            "nonce_age_seconds": age,
            "nonce_refreshes": s.nonce_refreshes,
            "session_info": s.session_info,
            "paid_left": s.last_paid_left,
            "requests": s.requests,
            "in_flight": s.in_flight,
            "concurrency": self.settings.upstream_concurrency,
            "last_error": s.last_error,
            "last_error_at": s.last_error_at,
        }
