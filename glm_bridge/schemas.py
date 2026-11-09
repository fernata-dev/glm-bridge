"""Request models. Permissive on purpose: unknown OpenAI parameters are accepted
and ignored rather than rejected, because automation tools send many of them."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any = None
    response_format: dict[str, Any] | None = None
    stop: str | list[str] | None = None
    n: int | None = None
    # Accepted for compatibility; the upstream web chat does not expose them.
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    user: str | None = None


class ImageRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    prompt: str = Field(min_length=1, max_length=8000)
    model: str | None = None
    n: int = Field(default=1, ge=1)
    size: str | None = None
    response_format: str | None = None
    # Extension: optional reference image (data URL or http(s) URL).
    image: str | None = None
