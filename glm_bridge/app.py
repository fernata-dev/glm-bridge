"""FastAPI application: OpenAI-compatible routes on /v1 (and mirrored without the
prefix for tools whose base URL is set without /v1)."""

from __future__ import annotations

import json
import logging
import secrets
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import APIRouter, Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .auth import Authenticator
from .config import Settings
from .convert import PromptBuilder
from .errors import BridgeError, bad_request
from .imagegen import ImageService
from .images import ImageFetcher
from .models import ModelRegistry
from .schemas import ChatRequest, ImageRequest
from .service import ChatService, FallbackProvider
from .upstream import GLMChatClient

log = logging.getLogger("glm_bridge")
access_log = logging.getLogger("glm_bridge.access")


class BodyLimitMiddleware:
    """Rejects request bodies above MAX_BODY_BYTES, including chunked uploads."""

    def __init__(self, app: Any, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                return await _send_too_large(send, self.max_bytes)
        seen = 0

        async def limited_receive() -> dict:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise BridgeError(f"Request body exceeds {self.max_bytes} bytes", status=413,
                                      type="invalid_request_error", code="request_too_large")
            return message

        await self.app(scope, limited_receive, send)


async def _send_too_large(send: Any, limit: int) -> None:
    body = json.dumps(BridgeError(f"Request body exceeds {limit} bytes", status=413, type="invalid_request_error",
                                  code="request_too_large").to_dict()).encode()
    await send({"type": "http.response.start", "status": 413,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})


def _error_response(err: BridgeError, request: Request | None = None) -> JSONResponse:
    headers = dict(err.headers)
    if request is not None and getattr(request.state, "request_id", None):
        headers["X-Request-ID"] = request.state.request_id
    return JSONResponse(err.to_dict(), status_code=err.status, headers=headers)


def create_app(
    settings: Settings | None = None,
    *,
    upstream_transport: httpx.AsyncBaseTransport | None = None,
    image_transport: httpx.AsyncBaseTransport | None = None,
    fallback_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    settings = settings or Settings()
    auth = Authenticator(settings)
    registry = ModelRegistry(settings)
    upstream = GLMChatClient(settings, transport=upstream_transport)
    registry.suggest = upstream.suggested_model
    fetcher = ImageFetcher(settings, transport=image_transport)
    builder = PromptBuilder(settings, registry, fetcher)
    fallback = FallbackProvider(settings, transport=fallback_transport) if settings.fallback_enabled else None
    chat = ChatService(settings, upstream, builder, fallback)
    images = ImageService(settings, upstream, fetcher)
    started = time.time()
    counters = {"requests": 0, "errors": 0}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if settings.auth_disabled:
            log.warning("AUTH_DISABLED=true: the API is open to anyone who can reach it")
        log.info("glm-bridge %s ready; upstream=%s default_model=%s", __version__,
                 settings.upstream_base_url, settings.default_model)
        yield
        await upstream.aclose()
        await fetcher.aclose()
        if fallback:
            await fallback.aclose()

    app = FastAPI(title="glm-bridge", version=__version__, lifespan=lifespan,
                  description="OpenAI-compatible bridge for glm-ai.chat")
    app.state.settings = settings
    app.state.upstream = upstream
    app.state.images = images

    if settings.cors_origin_list:
        app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["*"],
                           allow_headers=["*"], expose_headers=["X-Request-ID", "X-Bridge-Provider"])
    app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or secrets.token_hex(8)
        request.state.request_id = rid[:64]
        t0 = time.perf_counter()
        counters["requests"] += 1
        try:
            response = await call_next(request)
        except BridgeError as err:  # raised inside middleware (e.g. body limit)
            response = _error_response(err, request)
        if response.status_code >= 400:
            counters["errors"] += 1
        response.headers["X-Request-ID"] = request.state.request_id
        for k, v in (getattr(request.state, "rate_headers", None) or {}).items():
            response.headers[k] = v
        access_log.info("%s %s %s %.0fms key=%s rid=%s", request.method, request.url.path, response.status_code,
                        (time.perf_counter() - t0) * 1000, getattr(request.state, "key_name", "-"),
                        request.state.request_id)
        return response

    @app.exception_handler(BridgeError)
    async def _bridge_error(request: Request, err: BridgeError):
        if err.status >= 500:
            log.warning("request failed: %s (%s)", err.message, err.code)
        return _error_response(err, request)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        msg = f"Invalid request: {first.get('msg', 'validation error')}" + (f" ({loc})" if loc else "")
        return _error_response(bad_request(msg, param=loc or None), request)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error_response(BridgeError(str(exc.detail), status=exc.status_code, type="invalid_request_error",
                                           code=code), request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        log.exception("unhandled error")
        return _error_response(BridgeError("Internal bridge error", status=500, code="internal_error"), request)

    # -- public ---------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    async def root():
        return {"name": "glm-bridge", "version": __version__, "openai_base_url": "/v1", "docs": "/docs"}

    @app.get("/health", tags=["meta"])
    async def health():
        st = upstream.status()
        return {"status": "ok", "version": __version__, "uptime_seconds": round(time.time() - started),
                "upstream_session_ready": st["session_ready"]}

    @app.get("/v1/images/files/{token}", include_in_schema=False)
    async def image_file(token: str):
        item = images.store.get(token)
        if item is None:
            raise BridgeError("Image not found or expired", status=404, type="invalid_request_error", code="not_found")
        mime, raw = item
        return Response(raw, media_type=mime, headers={"Cache-Control": "private, max-age=3600"})

    # -- authenticated OpenAI-compatible API ------------------------------------
    def build_router(include_in_schema: bool) -> APIRouter:
        router = APIRouter(dependencies=[Depends(auth)], include_in_schema=include_in_schema)

        @router.get("/models", tags=["models"])
        async def list_models():
            return {"object": "list", "data": registry.list()}

        @router.get("/models/{model_id:path}", tags=["models"])
        async def get_model(model_id: str):
            return registry.get(model_id)

        @router.post("/chat/completions", tags=["chat"])
        async def chat_completions(request: Request):
            raw = await _json_body(request)
            req = _validate(ChatRequest, raw)
            if settings.log_prompts:
                log.info("prompt rid=%s %s", request.state.request_id, json.dumps(raw, ensure_ascii=False)[:4000])
            if req.stream:
                body_iter, headers = await chat.stream(req, raw)
                headers.update({"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
                return StreamingResponse(body_iter, media_type="text/event-stream", headers=headers)
            body, headers = await chat.complete(req, raw)
            return JSONResponse(body, headers=headers)

        @router.post("/images/generations", tags=["images"])
        async def image_generations(request: Request):
            req = _validate(ImageRequest, await _json_body(request))
            return await images.generate(req.prompt, n=req.n, response_format=req.response_format,
                                         reference=req.image, base_url=_public_base(request))

        @router.post("/images/edits", tags=["images"])
        async def image_edits(
            request: Request,
            prompt: str = Form(...),
            image: UploadFile = File(...),
            n: int = Form(1),
            response_format: str | None = Form(None),
        ):
            raw = await image.read(settings.max_image_bytes + 1)
            reference = images.fetcher.from_bytes(raw)
            return await images.generate(prompt, n=n, response_format=response_format, reference=reference,
                                         base_url=_public_base(request))

        @router.get("/bridge/status", tags=["meta"])
        async def bridge_status():
            return {
                "version": __version__,
                "uptime_seconds": round(time.time() - started),
                "requests": counters["requests"],
                "errors": counters["errors"],
                "upstream": upstream.status(),
                "config": {
                    "default_model": settings.default_model,
                    "system_mode": settings.system_mode,
                    "history_mode": settings.history_mode,
                    "overflow_policy": settings.overflow_policy,
                    "max_input_chars": settings.max_input_chars,
                    "tool_emulation": settings.tool_emulation,
                    "fallback_enabled": settings.fallback_enabled,
                },
            }

        return router

    async def _json_body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise bad_request("Request body must be valid JSON") from exc
        if not isinstance(data, dict):
            raise bad_request("Request body must be a JSON object")
        return data

    def _validate(model: type[BaseModel], raw: dict[str, Any]) -> Any:
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            first = exc.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            raise bad_request(f"Invalid request: {first.get('msg')}" + (f" ({loc})" if loc else ""),
                              param=loc or None) from None

    def _public_base(request: Request) -> str:
        return settings.public_base_url or str(request.base_url).rstrip("/")

    app.include_router(build_router(True), prefix="/v1")
    app.include_router(build_router(False))
    return app
