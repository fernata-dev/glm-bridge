"""OpenAI-shaped errors."""

from __future__ import annotations

from typing import Any


class BridgeError(Exception):
    """An error that is returned to the API client in OpenAI's error format.

    ``category`` classifies upstream failures so the fallback provider can decide
    whether to take over: offline, limit, error, unavailable, auth, rejected.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int = 500,
        type: str = "server_error",
        code: str | None = None,
        param: str | None = None,
        category: str | None = None,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type
        self.code = code
        self.param = param
        self.category = category
        self.headers = headers or {}
        self.extra = extra or {}

    def to_dict(self) -> dict[str, Any]:
        err: dict[str, Any] = {
            "message": self.message,
            "type": self.type,
            "param": self.param,
            "code": self.code,
        }
        err.update(self.extra)
        return {"error": err}


def bad_request(message: str, *, code: str = "invalid_request", param: str | None = None) -> BridgeError:
    return BridgeError(message, status=400, type="invalid_request_error", code=code, param=param)
