from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from glm_bridge.app import create_app
from glm_bridge.config import Settings
from glm_bridge.images import ImageFetcher

KEY = "sk-test-key-0123456789abcdef"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def sse_body(events: list[dict[str, Any]], *, done: bool = True) -> bytes:
    out = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    if done:
        out += "data: [DONE]\n\n"
    return out.encode()


def reply(*texts: str, paid_left: int = 3, meta: dict | None = None) -> Callable[[dict], httpx.Response]:
    events: list[dict[str, Any]] = []
    if meta:
        events.append({"type": "meta", **meta})
    events += [{"type": "delta", "text": t} for t in texts]
    events.append({"type": "done", "paidLeft": paid_left})
    return lambda body: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_body(events))


def event(etype: str, **data: Any) -> Callable[[dict], httpx.Response]:
    return lambda body: httpx.Response(200, headers={"content-type": "text/event-stream"},
                                       content=sse_body([{"type": etype, **data}]))


def status(code: int, text: str = "") -> Callable[[dict], httpx.Response]:
    return lambda body: httpx.Response(code, text=text)


class ChunkStream(httpx.AsyncByteStream):
    """Yields chunks with optional delays; an Exception instance in the list is raised."""

    def __init__(self, items: list[Any]) -> None:
        self.items = items

    async def __aiter__(self):
        for item in self.items:
            if isinstance(item, (int, float)):
                await asyncio.sleep(item)
            elif isinstance(item, Exception):
                raise item
            else:
                yield item if isinstance(item, bytes) else item.encode()

    async def aclose(self) -> None:
        pass


def streamed(items: list[Any]) -> Callable[[dict], httpx.Response]:
    return lambda body: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChunkStream(items))


class FakeUpstream:
    def __init__(self) -> None:
        self.session_calls = 0
        self.nonce_counter = 0
        self.valid_nonce: str | None = None
        self.message_bodies: list[dict[str, Any]] = []
        self.message_headers: list[dict[str, str]] = []
        self.script: list[Callable[[dict], httpx.Response]] = []
        self.image_script: list[Callable[[dict], httpx.Response]] = []
        self.image_bodies: list[dict[str, Any]] = []
        self.remote_images: dict[str, httpx.Response] = {}
        self.fallback_bodies: list[dict[str, Any]] = []
        self.fallback_response: Callable[[dict], httpx.Response] | None = None
        self.session_response: Callable[[], httpx.Response] | None = None

    def expire_nonce(self) -> None:
        self.valid_nonce = "rotated-by-server"

    # upstream: glm-ai.chat
    def upstream(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/session"):
            self.session_calls += 1
            if self.session_response:
                return self.session_response()
            self.nonce_counter += 1
            self.valid_nonce = f"nonce-{self.nonce_counter}"
            return httpx.Response(200, json={"nonce": self.valid_nonce, "paidLeft": 5, "suggest": "glm-4-5-flash"})
        nonce_ok = request.headers.get("x-glm-nonce") == self.valid_nonce
        body = json.loads(request.content)
        if path.endswith("/message"):
            self.message_bodies.append(body)
            self.message_headers.append(dict(request.headers))
            if not nonce_ok:
                return httpx.Response(403, json={"code": "rest_cookie_invalid_nonce", "message": "Cookie check failed"})
            responder = self.script.pop(0) if self.script else reply("Hello", " world")
            return responder(body)
        if path.endswith("/image"):
            self.image_bodies.append(body)
            if not nonce_ok:
                return httpx.Response(403, text="forbidden")
            responder = self.image_script.pop(0) if self.image_script else (
                lambda b: httpx.Response(200, json={"url": "https://cdn.example.com/out.png"}))
            return responder(body)
        return httpx.Response(404)

    # remote image hosts (used by the image fetcher)
    def images(self, request: httpx.Request) -> httpx.Response:
        return self.remote_images.get(str(request.url), httpx.Response(404))

    # fallback OpenAI-compatible provider
    def fallback(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.fallback_bodies.append({"body": body, "auth": request.headers.get("authorization")})
        if self.fallback_response:
            return self.fallback_response(body)
        if body.get("stream"):
            chunk = {"id": "fb", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "from fallback"}}]}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  content=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
        return httpx.Response(200, json={"id": "fb", "object": "chat.completion", "model": body["model"],
                                         "choices": [{"index": 0, "message": {"role": "assistant", "content": "from fallback"},
                                                      "finish_reason": "stop"}]})


@pytest.fixture
def fake() -> FakeUpstream:
    return FakeUpstream()


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    table = {"cdn.example.com": ["93.184.216.34"], "img.example.com": ["93.184.216.34"],
             "internal.example.com": ["10.0.0.5"], "rebind.example.com": ["93.184.216.34", "127.0.0.1"]}

    async def resolve(self, host: str) -> list[str]:
        if host not in table:
            raise OSError("unknown host")
        return table[host]

    monkeypatch.setattr(ImageFetcher, "resolve", resolve)


def make_settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"api_keys": KEY, "upstream_max_retries": 1, "preflight_timeout": 5.0,
                            "upstream_min_interval": 0,
                            "keepalive_interval": 5.0}
    base.update(overrides)
    return Settings(_env_file=None, **base)


@pytest.fixture
async def make_client(fake: FakeUpstream):
    clients: list[httpx.AsyncClient] = []

    async def _make(**overrides: Any) -> httpx.AsyncClient:
        app = create_app(
            make_settings(**overrides),
            upstream_transport=httpx.MockTransport(fake.upstream),
            image_transport=httpx.MockTransport(fake.images),
            fallback_transport=httpx.MockTransport(fake.fallback),
        )
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://bridge",
                                   headers={"Authorization": f"Bearer {KEY}"}, timeout=30)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        await c.aclose()


def parse_sse(text: str) -> tuple[list[dict[str, Any]], bool]:
    chunks, done = [], False
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    done = True
                else:
                    chunks.append(json.loads(data))
    return chunks, done


def chat(content: str = "Hi", **extra: Any) -> dict[str, Any]:
    return {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": content}], **extra}
