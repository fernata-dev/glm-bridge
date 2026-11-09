"""Model registry: public (OpenAI-facing) ids -> upstream ids, aliases, capabilities."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .config import Settings
from .errors import BridgeError

# Public ids follow Zhipu's official dotted naming (so the same ids work against the
# official API used as a fallback). Upstream ids use the dashed form that the
# original script sends ("glm-5-3-flash"). Only that one id is confirmed; the rest
# follow the same pattern and can be corrected with MODEL_MAP (run `probe --models`).
DEFAULT_MODELS: dict[str, str] = {
    "glm-5.3": "glm-5-3",
    "glm-5.3-flash": "glm-5-3-flash",
    "glm-5.2": "glm-5-2",
    "glm-5.1": "glm-5-1",
    "glm-4.7": "glm-4-7",
    "glm-4.7-flash": "glm-4-7-flash",
    "glm-4.7-flashx": "glm-4-7-flashx",
    "glm-4.6": "glm-4-6",
    "glm-4.5": "glm-4-5",
    "glm-4.5-flash": "glm-4-5-flash",
}

IMAGE_MODEL = "glm-image"
_CREATED = int(time.time())


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    upstream: str
    vision: bool = False
    known: bool = True


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        mapping = dict(DEFAULT_MODELS)
        mapping.update({k.strip().lower(): v.strip() for k, v in settings.model_map_dict.items()})
        vision = set(settings.vision_model_list)
        self.models: dict[str, ModelInfo] = {
            pid: ModelInfo(id=pid, upstream=up, vision=pid in vision or up.lower() in vision)
            for pid, up in mapping.items()
        }
        self.aliases: dict[str, str] = {
            k.strip().lower(): v.strip().lower() for k, v in settings.model_alias_dict.items()
        }
        self._lookup: dict[str, str] = {}
        for pid, info in self.models.items():
            self._lookup[pid] = pid
            self._lookup.setdefault(info.upstream.lower(), pid)
            self._lookup.setdefault(pid.replace(".", "-"), pid)
        for name in (settings.default_model, settings.vision_model):
            if self._find(name) is None and not settings.allow_unknown_models:
                raise ValueError(f"Configured model {name!r} is not in the registry; add it via MODEL_MAP")

    def _find(self, name: str) -> ModelInfo | None:
        key = name.strip().lower()
        for _ in range(3):  # alias chains, bounded
            if key in self.aliases:
                key = self.aliases[key]
        for candidate in (key, key.replace(".", "-")):
            pid = self._lookup.get(candidate)
            if pid is not None:
                return self.models[pid]
        return None

    def resolve(self, name: str | None) -> ModelInfo:
        if not name or name.strip().lower() in {"default", "auto"}:
            name = self.settings.default_model
        info = self._find(name)
        if info is not None:
            return info
        if self.settings.allow_unknown_models:
            return ModelInfo(id=name, upstream=name, vision=False, known=False)
        raise BridgeError(
            f"The model '{name}' does not exist. GET /v1/models lists the available ids.",
            status=404, type="invalid_request_error", code="model_not_found", param="model",
        )

    def vision_target(self) -> ModelInfo:
        return self.resolve(self.settings.vision_model)

    def list(self) -> list[dict]:
        out = [
            {"id": m.id, "object": "model", "created": _CREATED, "owned_by": "glm-bridge",
             "capabilities": {"vision": m.vision, "tools": "emulated" if self.settings.tool_emulation else False}}
            for m in self.models.values()
        ]
        for alias, target in self.aliases.items():
            out.append({"id": alias, "object": "model", "created": _CREATED, "owned_by": "glm-bridge",
                        "alias_of": target})
        out.append({"id": IMAGE_MODEL, "object": "model", "created": _CREATED, "owned_by": "glm-bridge",
                    "capabilities": {"image_generation": True}})
        return out

    def get(self, model_id: str) -> dict:
        if model_id.strip().lower() == IMAGE_MODEL:
            return next(m for m in self.list() if m["id"] == IMAGE_MODEL)
        info = self.resolve(model_id)
        return {"id": info.id, "object": "model", "created": _CREATED, "owned_by": "glm-bridge",
                "capabilities": {"vision": info.vision}}
