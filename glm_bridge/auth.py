"""API-key authentication and per-key rate limiting.

Keys are compared as SHA-256 digests with hmac.compare_digest, so keys can be
configured either in plain text ("sk-...") or pre-hashed ("sha256:<hex>") and the
server never needs the plain key on disk. Generate keys with
``python -m glm_bridge keygen``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

from starlette.requests import Request

from .config import Settings
from .errors import BridgeError


@dataclass(frozen=True, slots=True)
class KeyEntry:
    name: str
    digest: bytes
    rpm: int  # 0 = unlimited


def generate_key() -> str:
    return "sk-glm-" + secrets.token_urlsafe(32)


def hash_key(key: str) -> str:
    return "sha256:" + hashlib.sha256(key.encode()).hexdigest()


def _digest(spec: str) -> bytes:
    spec = spec.strip()
    if spec.lower().startswith("sha256:"):
        hexpart = spec[7:].strip().lower()
        try:
            raw = bytes.fromhex(hexpart)
        except ValueError as exc:
            raise ValueError("Invalid sha256: key hash") from exc
        if len(raw) != 32:
            raise ValueError("sha256: key hash must be 64 hex characters")
        return raw
    if len(spec) < 16:
        raise ValueError("API keys must be at least 16 characters long")
    return hashlib.sha256(spec.encode()).digest()


class KeyStore:
    def __init__(self, entries: list[KeyEntry]) -> None:
        self.entries = entries

    @classmethod
    def from_settings(cls, settings: Settings) -> KeyStore:
        entries: list[KeyEntry] = []
        for i, spec in enumerate(settings.api_key_list, start=1):
            entries.append(KeyEntry(name=f"key{i}", digest=_digest(spec), rpm=settings.default_rpm))
        if settings.api_keys_file:
            data = json.loads(Path(settings.api_keys_file).read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("API_KEYS_FILE must contain a JSON array")
            for i, item in enumerate(data, start=1):
                if not isinstance(item, dict) or item.get("enabled", True) is False:
                    continue
                spec = item.get("key_sha256") and f"sha256:{item['key_sha256']}" or item.get("key")
                if not isinstance(spec, str):
                    raise ValueError(f"API_KEYS_FILE entry {i} needs 'key' or 'key_sha256'")
                rpm = item.get("rpm", settings.default_rpm)
                entries.append(KeyEntry(name=str(item.get("name") or f"file{i}"), digest=_digest(spec), rpm=int(rpm)))
        names = [e.name for e in entries]
        if len(set(names)) != len(names):
            raise ValueError("API key names must be unique")
        return cls(entries)

    def verify(self, token: str) -> KeyEntry | None:
        digest = hashlib.sha256(token.encode()).digest()
        match = None
        for entry in self.entries:  # no early exit: constant work per request
            if hmac.compare_digest(entry.digest, digest):
                match = entry
        return match


class RateLimiter:
    """Token bucket per key: capacity = rpm, refill = rpm per 60 s."""

    def __init__(self) -> None:
        self._buckets: dict[str, tuple[float, float]] = {}

    def hit(self, name: str, rpm: int) -> tuple[bool, int, float]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        if rpm <= 0:
            return True, -1, 0.0
        now = time.monotonic()
        tokens, last = self._buckets.get(name, (float(rpm), now))
        tokens = min(float(rpm), tokens + (now - last) * rpm / 60.0)
        if tokens < 1.0:
            self._buckets[name] = (tokens, now)
            return False, 0, (1.0 - tokens) * 60.0 / rpm
        tokens -= 1.0
        self._buckets[name] = (tokens, now)
        return True, int(tokens), 0.0


def extract_token(request: Request) -> str | None:
    auth = request.headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    for header in ("x-api-key", "api-key"):
        value = request.headers.get(header)
        if value:
            return value.strip()
    return None


class Authenticator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.keys = KeyStore.from_settings(settings)
        self.limiter = RateLimiter()
        if not settings.auth_disabled and not self.keys.entries:
            raise ValueError(
                "No API keys configured. Set API_KEYS (generate one with `python -m glm_bridge keygen`) "
                "or explicitly set AUTH_DISABLED=true for local-only use."
            )

    def __call__(self, request: Request) -> KeyEntry | None:
        if self.settings.auth_disabled:
            return None
        token = extract_token(request)
        if not token:
            raise BridgeError(
                "Missing API key. Send 'Authorization: Bearer <key>' or 'x-api-key: <key>'.",
                status=401, type="invalid_request_error", code="missing_api_key",
            )
        entry = self.keys.verify(token)
        if entry is None:
            raise BridgeError("Invalid API key.", status=401, type="invalid_request_error", code="invalid_api_key")
        allowed, remaining, retry_after = self.limiter.hit(entry.name, entry.rpm)
        request.state.key_name = entry.name
        if not allowed:
            raise BridgeError(
                f"Rate limit of {entry.rpm} requests/minute exceeded for this API key.",
                status=429, type="rate_limit_error", code="rate_limit_exceeded",
                headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
            )
        if remaining >= 0:
            request.state.rate_headers = {
                "x-ratelimit-limit-requests": str(entry.rpm),
                "x-ratelimit-remaining-requests": str(remaining),
            }
        return entry
