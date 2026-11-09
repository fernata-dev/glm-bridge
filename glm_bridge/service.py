"""Chat orchestration: pre-flight, retries, nonce refresh, model switch, fallback,
and conversion of upstream events into OpenAI responses / SSE chunks.

Key idea - *pre-flight*: before answering the client we read the upstream until
the first real token (or a terminal event). Anything that fails before that point
can be retried, refreshed, switched or handed to the fallback provider, and the
client gets a proper HTTP status (429/502/503...) instead of a 200 stream that
dies. Streaming responses commit after PREFLIGHT_TIMEOUT so slow "thinking"
models are not blocked; from then on keep-alive comments are sent while idle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Settings
from .convert import Prepared, PromptBuilder
from .errors import BridgeError
from .output import OutputProcessor
from .schemas import ChatRequest
from .util import estimate_tokens
from .upstream import (
    REASONING_TYPES,
    TERMINAL_TYPES,
    GLMChatClient,
    NonceRejected,
    UpstreamEvent,
    UpstreamHTTPError,
    describe_upstream_body,
)

log = logging.getLogger("glm_bridge.service")

_END = object()
PING = object()
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$", re.S | re.I)


class UpstreamEmpty(Exception):
    pass


@dataclass(slots=True)
class _Failure:
    exc: BaseException


class _Failed(Exception):
    def __init__(self, exc: BaseException) -> None:
        super().__init__(repr(exc))
        self.exc = exc


class _Terminal(Exception):
    def __init__(self, event: UpstreamEvent) -> None:
        super().__init__(event.type)
        self.event = event


def strip_json_fence(text: str) -> str:
    m = _JSON_FENCE.match(text)
    return m.group(1).strip() if m else text.strip()


def sse(obj: Any) -> bytes:
    return b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"


SSE_DONE = b"data: [DONE]\n\n"


def terminal_error(ev: UpstreamEvent) -> BridgeError:
    msg = ev.message or f"upstream sent a '{ev.type}' event"
    extra: dict[str, Any] = {"upstream_event": ev.type}
    if isinstance(ev.data.get("switch"), str):
        extra["upstream_switch"] = ev.data["switch"]
    if ev.type == "limit":
        return BridgeError(f"Upstream usage limit reached: {msg}", status=429, type="rate_limit_error",
                           code="upstream_limit_reached", category="limit", extra=extra)
    if ev.type == "offline":
        return BridgeError(f"Upstream is offline: {msg}", status=503, code="upstream_offline",
                           category="offline", extra=extra, headers={"Retry-After": "30"})
    if ev.type == "notice":
        return BridgeError(f"Upstream notice: {msg}", status=502, code="upstream_notice", category="error", extra=extra)
    return BridgeError(f"Upstream error: {msg}", status=502, code="upstream_error", category="error", extra=extra)


def classify(exc: BaseException) -> tuple[str, BridgeError]:
    """-> (kind, error) where kind is nonce | transient | fatal."""
    if isinstance(exc, NonceRejected):
        return "nonce", BridgeError(
            "Upstream rejected the session (HTTP 403) even after refreshing the nonce. The site may be blocking "
            "this server's IP or may have changed its API.",
            status=502, code="upstream_auth_failed", category="auth")
    if isinstance(exc, UpstreamHTTPError):
        msg = describe_upstream_body(exc.body)
        if exc.status == 429:
            return "transient", BridgeError(f"Upstream rate-limited the bridge: {msg}", status=429,
                                            type="rate_limit_error", code="upstream_rate_limited", category="limit")
        if exc.status >= 500:
            return "transient", BridgeError(f"Upstream HTTP {exc.status}: {msg}", status=502,
                                            code="upstream_http_error", category="unavailable")
        if exc.status in (400, 413, 422):
            return "fatal", BridgeError(f"Upstream rejected the request: {msg}", status=400,
                                        type="invalid_request_error", code="upstream_rejected", category="rejected")
        return "fatal", BridgeError(f"Upstream HTTP {exc.status}: {msg}", status=502,
                                    code="upstream_http_error", category="error")
    if isinstance(exc, UpstreamEmpty):
        return "transient", BridgeError("Upstream returned an empty response", status=502,
                                        code="upstream_empty", category="unavailable")
    if isinstance(exc, httpx.TimeoutException):
        return "transient", BridgeError("Upstream timed out", status=504, code="upstream_timeout", category="unavailable")
    if isinstance(exc, httpx.HTTPError):
        return "transient", BridgeError(f"Upstream connection failed: {exc.__class__.__name__}", status=502,
                                        code="upstream_unreachable", category="unavailable")
    if isinstance(exc, BridgeError):
        return "fatal", exc
    return "fatal", BridgeError(f"Unexpected upstream failure: {exc.__class__.__name__}", status=502,
                                code="upstream_failure", category="error")


class EventPump:
    """Reads an async iterator in a background task so consumers can wait with a
    timeout (for pre-flight and keep-alives) without cancelling the iterator."""

    def __init__(self, source: AsyncIterator[UpstreamEvent]) -> None:
        self._source = source
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        try:
            async for item in self._source:
                self._queue.put_nowait(item)
            self._queue.put_nowait(_END)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # delivered to the consumer
            self._queue.put_nowait(_Failure(exc))
        finally:
            aclose = getattr(self._source, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # pragma: no cover - best effort cleanup
                    pass

    async def get(self, timeout: float | None) -> Any:
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout)

    async def aclose(self) -> None:
        if not self._task.done():
            self._task.cancel()
        await asyncio.wait({self._task})


@dataclass
class RunMeta:
    upstream_model: str
    provider: str = "glmchat"
    upstream_fallback: bool = False
    upstream_label: str | None = None
    paid_left: Any = None
    switched_to: str | None = None
    retries: int = 0
    nonce_refreshed: bool = False

    def observe(self, ev: UpstreamEvent) -> None:
        if ev.type == "meta":
            if ev.data.get("fallback"):
                self.upstream_fallback = True
            label = ev.data.get("label")
            if isinstance(label, str):
                self.upstream_label = label
        elif ev.type == "done" and "paidLeft" in ev.data:
            self.paid_left = ev.data.get("paidLeft")

    def to_dict(self, notes: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider, "upstream_model": self.upstream_model}
        if self.upstream_fallback:
            out["upstream_fallback"] = True
        if self.upstream_label:
            out["upstream_label"] = self.upstream_label
        if self.paid_left is not None:
            out["paid_left"] = self.paid_left
        if self.switched_to:
            out["switched_to"] = self.switched_to
        if self.retries:
            out["retries"] = self.retries
        if self.nonce_refreshed:
            out["nonce_refreshed"] = True
        out.update(notes)
        return out

    def headers(self) -> dict[str, str]:
        h = {"X-Bridge-Provider": self.provider, "X-Bridge-Upstream-Model": self.upstream_model}
        if self.upstream_fallback:
            h["X-Bridge-Upstream-Fallback"] = "1"
        if self.paid_left is not None:
            h["X-Bridge-Paid-Left"] = str(self.paid_left)
        return h


class FallbackProvider:
    """Forwards the original request to any OpenAI-compatible API (for example the
    official Zhipu/Z.ai endpoint) when glm-ai.chat fails before answering."""

    def __init__(self, settings: Settings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.settings = settings
        headers = {"Authorization": f"Bearer {settings.fallback_api_key}"} if settings.fallback_api_key else {}
        self._client = httpx.AsyncClient(
            base_url=settings.fallback_base_url or "",
            headers=headers,
            timeout=httpx.Timeout(settings.upstream_timeout, connect=settings.upstream_connect_timeout),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def applies(self, err: BridgeError) -> bool:
        return err.category in self.settings.fallback_on_set

    def _body(self, raw: dict[str, Any], stream: bool) -> dict[str, Any]:
        body = dict(raw)
        if self.settings.fallback_model:
            body["model"] = self.settings.fallback_model
        body["stream"] = stream
        return body

    @staticmethod
    def _failed(cause: BridgeError, detail: str) -> BridgeError:
        return BridgeError(f"{cause.message} (fallback provider also failed: {detail})", status=cause.status,
                           type=cause.type, code=cause.code, category=cause.category, headers=cause.headers)

    def _headers(self, cause: BridgeError) -> dict[str, str]:
        return {"X-Bridge-Provider": "fallback", "X-Bridge-Fallback-Reason": cause.code or "unknown"}

    async def complete(self, raw: dict[str, Any], cause: BridgeError) -> tuple[dict[str, Any], dict[str, str]]:
        log.warning("glm-ai.chat failed (%s); using fallback provider", cause.code)
        try:
            resp = await self._client.post("/chat/completions", json=self._body(raw, False))
        except httpx.HTTPError as exc:
            raise self._failed(cause, exc.__class__.__name__) from exc
        if resp.status_code != 200:
            raise self._failed(cause, f"HTTP {resp.status_code}: {describe_upstream_body(resp.text)}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise self._failed(cause, "non-JSON response") from exc
        if isinstance(data, dict):
            data["glm_bridge"] = {"provider": "fallback", "fallback_reason": cause.code}
        return data, self._headers(cause)

    async def stream(self, raw: dict[str, Any], cause: BridgeError) -> tuple[AsyncIterator[bytes], dict[str, str]]:
        log.warning("glm-ai.chat failed (%s); streaming from fallback provider", cause.code)
        gen = self._stream(self._body(raw, True), cause)
        await gen.__anext__()  # prime: raises before any byte is sent to the client
        return gen, self._headers(cause)

    async def _stream(self, body: dict[str, Any], cause: BridgeError) -> AsyncIterator[bytes]:
        started = False
        try:
            async with self._client.stream("POST", "/chat/completions", json=body) as resp:
                if resp.status_code != 200:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    raise self._failed(cause, f"HTTP {resp.status_code}: {describe_upstream_body(text)}")
                started = True
                yield b""
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.HTTPError as exc:
            if not started:
                raise self._failed(cause, exc.__class__.__name__) from exc
            yield sse(BridgeError(f"Fallback stream interrupted: {exc.__class__.__name__}", status=502,
                                  code="fallback_interrupted").to_dict())
            yield SSE_DONE


class ChatService:
    def __init__(self, settings: Settings, upstream: GLMChatClient, builder: PromptBuilder,
                 fallback: FallbackProvider | None = None) -> None:
        self.s = settings
        self.upstream = upstream
        self.builder = builder
        self.fallback = fallback

    # -- opening a stream with pre-flight -------------------------------------
    async def open(self, prep: Prepared, meta: RunMeta, *, stream: bool) -> tuple[EventPump, list[Any]]:
        body = dict(prep.body)
        retries = 0
        refreshed = False
        switched = False
        while True:
            try:
                nonce = await self.upstream.nonce()
            except BridgeError as err:
                if err.category == "unavailable" and retries < self.s.upstream_max_retries:
                    retries += 1
                    meta.retries = retries
                    await asyncio.sleep(self._backoff(retries))
                    continue
                raise
            pump = EventPump(self.upstream.stream_message(body, nonce))
            try:
                prefix = await self._preflight(pump, meta, stream)
                return pump, prefix
            except _Terminal as t:
                await pump.aclose()
                switch = t.event.data.get("switch")
                if (self.s.follow_switch and not switched and isinstance(switch, str)
                        and switch and switch != body["model"]):
                    log.info("upstream suggested switching %s -> %s", body["model"], switch)
                    switched = True
                    body["model"] = meta.upstream_model = meta.switched_to = switch
                    continue
                err = terminal_error(t.event)
                self.upstream.record_error(err.message)
                raise err from None
            except _Failed as f:
                await pump.aclose()
                kind, err = classify(f.exc)
                if kind == "nonce" and not refreshed:
                    refreshed = meta.nonce_refreshed = True
                    await self.upstream.nonce(refresh_from=nonce)
                    continue
                if kind == "transient" and retries < self.s.upstream_max_retries:
                    retries += 1
                    meta.retries = retries
                    delay = self._backoff(retries)
                    log.warning("upstream attempt failed (%s); retry %d in %.1fs", err.code, retries, delay)
                    await asyncio.sleep(delay)
                    continue
                self.upstream.record_error(err.message)
                raise err from None
            except BaseException:
                await pump.aclose()
                raise

    @staticmethod
    def _backoff(attempt: int) -> float:
        return min(0.5 * 2 ** (attempt - 1), 8.0) * (0.75 + random.random() / 2)

    async def _preflight(self, pump: EventPump, meta: RunMeta, stream: bool) -> list[Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.s.preflight_timeout if stream else None
        prefix: list[Any] = []
        while True:
            timeout = None if deadline is None else deadline - loop.time()
            if timeout is not None and timeout <= 0:
                return prefix
            try:
                item = await pump.get(timeout)
            except asyncio.TimeoutError:
                return prefix  # commit: slow first token, stream with keep-alives
            if item is _END:
                raise _Failed(UpstreamEmpty())
            if isinstance(item, _Failure):
                raise _Failed(item.exc)
            if item.type in TERMINAL_TYPES:
                raise _Terminal(item)
            meta.observe(item)
            prefix.append(item)
            if item.type == "delta" or item.type == "done" or item.type in REASONING_TYPES:
                return prefix

    @staticmethod
    async def _events(pump: EventPump, prefix: list[Any], keepalive: float | None) -> AsyncIterator[Any]:
        for item in prefix:
            yield item
        while True:
            try:
                item = await pump.get(keepalive)
            except asyncio.TimeoutError:
                yield PING
                continue
            if item is _END:
                return
            if isinstance(item, _Failure):
                raise _Failed(item.exc)
            yield item

    # -- non-streaming ---------------------------------------------------------
    async def complete(self, req: ChatRequest, raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        prep = await self.builder.build(req)
        meta = RunMeta(upstream_model=prep.body["model"])
        try:
            pump, prefix = await self.open(prep, meta, stream=False)
        except BridgeError as err:
            if self.fallback and self.fallback.applies(err):
                return await self.fallback.complete(raw, err)
            raise
        proc = OutputProcessor(prep.stop, prep.tool_names)
        parts: list[str] = []
        reasoning: list[str] = []
        error: BridgeError | None = None
        try:
            async for ev in self._events(pump, prefix, None):
                meta.observe(ev)
                if ev.type == "delta":
                    parts.append(proc.feed(ev.text))
                    if proc.stopped:
                        break
                elif ev.type in REASONING_TYPES:
                    reasoning.append(ev.text)
                elif ev.type == "done":
                    break
                elif ev.type in TERMINAL_TYPES:
                    error = terminal_error(ev)
                    break
        except _Failed as f:
            error = classify(f.exc)[1]
        finally:
            await pump.aclose()
        if error is not None:
            # Nothing has been sent to the client yet, so a partial answer can still be replaced.
            self.upstream.record_error(error.message)
            if self.fallback and self.fallback.applies(error):
                return await self.fallback.complete(raw, error)
            raise error

        final = proc.finish()
        content = "".join(parts) + final.text
        if prep.json_mode and not final.tool_calls:
            content = strip_json_fence(content)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        finish = final.finish_reason
        if final.tool_calls:
            message["content"] = content.strip() or None
            if prep.legacy_functions:
                message["function_call"] = final.tool_calls[0]["function"]
                finish = "function_call"
            else:
                message["tool_calls"] = final.tool_calls
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        completion_tokens = estimate_tokens(content + "".join(reasoning)) + sum(
            estimate_tokens(c["function"]["arguments"]) for c in final.tool_calls)
        body = {
            "id": "chatcmpl-" + secrets.token_hex(12),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": prep.model.id,
            "choices": [{"index": 0, "message": message, "logprobs": None, "finish_reason": finish}],
            "usage": {"prompt_tokens": prep.prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prep.prompt_tokens + completion_tokens},
            "system_fingerprint": None,
            "glm_bridge": meta.to_dict(prep.notes),
        }
        return body, meta.headers()

    # -- streaming --------------------------------------------------------------
    async def stream(self, req: ChatRequest, raw: dict[str, Any]) -> tuple[AsyncIterator[bytes], dict[str, str]]:
        prep = await self.builder.build(req)
        meta = RunMeta(upstream_model=prep.body["model"])
        try:
            pump, prefix = await self.open(prep, meta, stream=True)
        except BridgeError as err:
            if self.fallback and self.fallback.applies(err):
                return await self.fallback.stream(raw, err)
            raise
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        return self._sse(prep, meta, pump, prefix, include_usage), meta.headers()

    async def _sse(self, prep: Prepared, meta: RunMeta, pump: EventPump, prefix: list[Any],
                   include_usage: bool) -> AsyncIterator[bytes]:
        cid = "chatcmpl-" + secrets.token_hex(12)
        created = int(time.time())

        def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> bytes:
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": prep.model.id,
                   "system_fingerprint": None,
                   "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}]}
            obj.update(extra)
            return sse(obj)

        proc = OutputProcessor(prep.stop, prep.tool_names)
        out_text: list[str] = []
        error: BridgeError | None = None
        try:
            yield chunk({"role": "assistant", "content": ""})
            try:
                async for ev in self._events(pump, prefix, self.s.keepalive_interval):
                    if ev is PING:
                        yield b": keep-alive\n\n"
                        continue
                    meta.observe(ev)
                    if ev.type == "delta":
                        text = proc.feed(ev.text)
                        if text:
                            out_text.append(text)
                            yield chunk({"content": text})
                        if proc.stopped:
                            break
                    elif ev.type in REASONING_TYPES:
                        if ev.text:
                            out_text.append(ev.text)
                            yield chunk({"reasoning_content": ev.text})
                    elif ev.type == "done":
                        break
                    elif ev.type in TERMINAL_TYPES:
                        error = terminal_error(ev)
                        break
            except _Failed as f:
                error = classify(f.exc)[1]
            if error is not None:
                self.upstream.record_error(error.message)
                yield sse(error.to_dict())
                yield SSE_DONE
                return

            final = proc.finish()
            if final.text:
                out_text.append(final.text)
                yield chunk({"content": final.text})
            finish = final.finish_reason
            if final.tool_calls:
                if prep.legacy_functions:
                    yield chunk({"function_call": final.tool_calls[0]["function"]})
                    finish = "function_call"
                else:
                    yield chunk({"tool_calls": [dict(call, index=i) for i, call in enumerate(final.tool_calls)]})
                out_text.extend(c["function"]["arguments"] for c in final.tool_calls)
            yield chunk({}, finish, glm_bridge=meta.to_dict(prep.notes))
            if include_usage:
                completion = estimate_tokens("".join(out_text))
                yield sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": prep.model.id,
                           "choices": [], "usage": {"prompt_tokens": prep.prompt_tokens,
                                                    "completion_tokens": completion,
                                                    "total_tokens": prep.prompt_tokens + completion}})
            yield SSE_DONE
        finally:
            await pump.aclose()
