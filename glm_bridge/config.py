"""Runtime configuration, read from environment variables (and an optional .env file)."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_UA = (
    "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/156.0.0.0 Safari/537.36"
)


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _json_object(value: str, name: str) -> dict[str, str]:
    if not value.strip():
        return {}
    try:
        data = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be a JSON object: {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        raise ValueError(f"{name} must be a JSON object of string -> string")
    return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # --- server -----------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    log_json: bool = False
    log_prompts: bool = False
    cors_origins: str = ""
    max_body_bytes: int = 16 * 1024 * 1024
    public_base_url: str | None = None

    # --- auth -------------------------------------------------------------
    api_keys: str = ""
    api_keys_file: str | None = None
    auth_disabled: bool = False
    default_rpm: int = 60

    # --- upstream (glm-ai.chat) -------------------------------------------
    upstream_base_url: str = "https://glm-ai.chat"
    upstream_user_agent: str = DEFAULT_UA
    upstream_requested_with: str = "com.duckduckgo.mobile.android"
    upstream_proxy: str | None = None
    upstream_timeout: float = 180.0
    upstream_connect_timeout: float = 15.0
    upstream_concurrency: int = 4
    upstream_queue_timeout: float = 60.0
    upstream_max_retries: int = 2
    nonce_ttl: float = 3600.0
    preflight_timeout: float = 20.0
    keepalive_interval: float = 15.0

    # --- prompt shaping ---------------------------------------------------
    max_input_chars: int = 8000
    max_context_chars: int = 48000
    overflow_policy: Literal["error", "truncate", "split"] = "error"
    system_mode: Literal["auto", "merge", "primer", "role"] = "auto"
    history_mode: Literal["native", "flatten"] = "native"
    tool_emulation: bool = True

    # --- models -----------------------------------------------------------
    default_model: str = "glm-5.3-flash"
    vision_model: str = "glm-5.3-flash"
    vision_models: str = "glm-5.3-flash"
    vision_routing: Literal["auto", "strict", "passthrough"] = "auto"
    model_map: str = ""
    model_aliases: str = ""
    allow_unknown_models: bool = False
    follow_switch: bool = False

    # --- images -----------------------------------------------------------
    max_image_bytes: int = 4 * 1024 * 1024
    allow_remote_images: bool = True
    image_fetch_timeout: float = 20.0
    image_max_n: int = 4
    image_cache_ttl: float = 3600.0
    image_cache_max: int = 64

    # --- optional fallback provider (any OpenAI-compatible API) -----------
    fallback_base_url: str | None = None
    fallback_api_key: str | None = None
    fallback_model: str | None = None
    fallback_on: str = "offline,limit,error,unavailable"

    @field_validator("upstream_base_url", "fallback_base_url", "public_base_url")
    @classmethod
    def _strip_slash(cls, v: str | None) -> str | None:
        return v.rstrip("/") if v else v

    @field_validator("model_map", "model_aliases")
    @classmethod
    def _valid_json(cls, v: str, info) -> str:
        _json_object(v, info.field_name.upper())
        return v

    @field_validator("upstream_concurrency", "upstream_max_retries", "max_input_chars", "image_max_n")
    @classmethod
    def _non_negative(cls, v: int, info) -> int:
        if v < 0:
            raise ValueError(f"{info.field_name.upper()} must be >= 0")
        return v

    # --- parsed views -----------------------------------------------------
    @property
    def cors_origin_list(self) -> list[str]:
        return _csv(self.cors_origins)

    @property
    def api_key_list(self) -> list[str]:
        return _csv(self.api_keys)

    @property
    def vision_model_list(self) -> list[str]:
        return [m.lower() for m in _csv(self.vision_models)]

    @property
    def model_map_dict(self) -> dict[str, str]:
        return _json_object(self.model_map, "MODEL_MAP")

    @property
    def model_alias_dict(self) -> dict[str, str]:
        return _json_object(self.model_aliases, "MODEL_ALIASES")

    @property
    def fallback_enabled(self) -> bool:
        return bool(self.fallback_base_url)

    @property
    def fallback_on_set(self) -> set[str]:
        return set(_csv(self.fallback_on))
