"""OpenAI chat request -> upstream message payload.

The upstream accepts {"model", "messages": [{role, content}], "image"?} with a
character limit on the last message (8000 in the site's own client) and a single
image per request. Everything an OpenAI client can send has to be squeezed into
that shape, explicitly and predictably:

* system/developer messages, emulated-tool instructions and JSON-mode
  instructions become one "instructions" block, placed per SYSTEM_MODE;
* tool results become user turns, assistant tool calls become text;
* consecutive same-role turns are merged;
* the most recent image in the conversation is attached;
* the last message's length is enforced per OVERFLOW_POLICY;
* old turns are dropped to respect MAX_CONTEXT_CHARS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .errors import BridgeError, bad_request
from .images import ImageFetcher
from .models import ModelInfo, ModelRegistry
from .schemas import ChatRequest
from .tools import (
    ToolSpec,
    build_tool_prompt,
    normalize_tools,
    render_assistant_tool_calls,
    render_tool_result,
    tool_choice_mode,
)
from .util import estimate_tokens

INSTR_OPEN = "[System instructions]"
INSTR_CLOSE = "[/System instructions]"


@dataclass
class Turn:
    role: str
    text: str
    images: list[str] = field(default_factory=list)


@dataclass
class Prepared:
    model: ModelInfo
    requested_model: str
    body: dict[str, Any]
    tool_names: set[str] | None
    legacy_functions: bool
    stop: list[str]
    json_mode: bool
    prompt_tokens: int
    notes: dict[str, Any] = field(default_factory=dict)


# -- content helpers -------------------------------------------------------

def _image_from_part(part: dict[str, Any]) -> str | None:
    value = part.get("image_url", part.get("url"))
    if isinstance(value, dict):
        value = value.get("url")
    if isinstance(value, str) and value:
        return value
    source = part.get("source")  # Anthropic-style {"type":"image","source":{...}}
    if isinstance(source, dict):
        if source.get("type") == "base64" and source.get("data"):
            return f"data:{source.get('media_type', 'image/png')};base64,{source['data']}"
        if source.get("type") == "url" and source.get("url"):
            return str(source["url"])
    return None


def content_to_text(content: Any, images: list[str]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    if not isinstance(content, list):
        raise bad_request("Message content must be a string or an array of parts", param="messages")
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "text")
        if ptype in ("text", "input_text", "output_text"):
            parts.append(str(part.get("text") or ""))
        elif ptype in ("image_url", "input_image", "image"):
            url = _image_from_part(part)
            if url:
                images.append(url)
        elif ptype == "refusal":
            parts.append(str(part.get("refusal") or ""))
        else:
            raise bad_request(f"Content part type '{ptype}' is not supported by this bridge", param="messages")
    return "\n".join(p for p in parts if p)


def to_turns(messages: list[dict[str, Any]]) -> tuple[list[str], list[Turn]]:
    system: list[str] = []
    turns: list[Turn] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise bad_request(f"messages[{i}] must be an object", param="messages")
        role = msg.get("role")
        images: list[str] = []
        text = content_to_text(msg.get("content"), images)
        if role in ("system", "developer"):
            if text.strip():
                system.append(text.strip())
        elif role == "user":
            turns.append(Turn("user", text, images))
        elif role == "assistant":
            pieces = [text] if text.strip() else []
            if msg.get("tool_calls"):
                pieces.append(render_assistant_tool_calls(msg["tool_calls"]))
            elif isinstance(msg.get("function_call"), dict):
                pieces.append(render_assistant_tool_calls([{"function": msg["function_call"]}]))
            if pieces:
                turns.append(Turn("assistant", "\n".join(pieces)))
        elif role in ("tool", "function"):
            turns.append(Turn("user", render_tool_result(text, msg.get("name"), msg.get("tool_call_id"))))
        else:
            raise bad_request(f"messages[{i}] has unsupported role '{role}'", param="messages")
    return system, turns


def merge_same_role(turns: list[Turn]) -> list[Turn]:
    merged: list[Turn] = []
    for t in turns:
        if merged and merged[-1].role == t.role:
            prev = merged[-1]
            prev.text = f"{prev.text}\n\n{t.text}" if prev.text and t.text else prev.text or t.text
            prev.images.extend(t.images)
        else:
            merged.append(Turn(t.role, t.text, list(t.images)))
    return merged


def json_mode_prompt(response_format: dict[str, Any] | None) -> str | None:
    if not response_format:
        return None
    rtype = response_format.get("type")
    if rtype == "json_object":
        return "Respond with one valid JSON object only. No markdown code fences and no text before or after it."
    if rtype == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema", spec)
        return (
            "Respond with one valid JSON value only, matching this JSON Schema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
            + ". No markdown code fences and no text before or after it."
        )
    return None


# -- the builder -------------------------------------------------------------

class PromptBuilder:
    def __init__(self, settings: Settings, registry: ModelRegistry, fetcher: ImageFetcher) -> None:
        self.s = settings
        self.registry = registry
        self.fetcher = fetcher

    async def build(self, req: ChatRequest) -> Prepared:
        s = self.s
        if req.n is not None and req.n > 1:
            raise bad_request("n > 1 is not supported by this bridge", param="n")
        model = self.registry.resolve(req.model)
        notes: dict[str, Any] = {}

        # Tools
        mode, forced = tool_choice_mode(req.tool_choice, req.function_call)
        specs: list[ToolSpec] = normalize_tools(req.tools, req.functions)
        tool_prompt = None
        tool_names: set[str] | None = None
        if specs and mode != "none":
            if not s.tool_emulation:
                raise bad_request("Tool calling is disabled on this bridge (TOOL_EMULATION=false)", param="tools")
            if forced and forced not in {sp.name for sp in specs}:
                raise bad_request(f"tool_choice names unknown tool '{forced}'", param="tool_choice")
            tool_prompt = build_tool_prompt(specs, mode, forced)
            tool_names = {sp.name for sp in specs}

        system, turns = to_turns(req.messages)
        extra = [p for p in (tool_prompt, json_mode_prompt(req.response_format)) if p]
        instructions = "\n\n".join(system + extra)
        turns = merge_same_role(turns)
        if not turns:
            if not instructions:
                raise bad_request("messages contain no text", param="messages")
            turns, instructions = [Turn("user", instructions)], ""
        if turns[-1].role != "user":
            raise bad_request("The last message must come from the user or a tool", param="messages")

        # Image: most recent one in the conversation.
        image_url = next((t.images[-1] for t in reversed(turns) if t.images), None)
        image_data = None
        if image_url:
            image_data = await self.fetcher.to_data_url(image_url)
            if not model.vision:
                if s.vision_routing == "auto":
                    notes["vision_routed_from"] = model.id
                    model = self.registry.vision_target()
                elif s.vision_routing == "strict":
                    raise bad_request(f"Model '{model.id}' does not accept images; use '{s.vision_model}'", param="model")

        messages = self._shape(turns, instructions, notes)
        body: dict[str, Any] = {"model": model.upstream, "messages": messages}
        if image_data:
            body["image"] = image_data

        stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        return Prepared(
            model=model,
            requested_model=req.model or model.id,
            body=body,
            tool_names=tool_names,
            legacy_functions=bool(req.functions) and not req.tools,
            stop=[x for x in stop if isinstance(x, str) and x][:16],
            json_mode=bool(req.response_format and req.response_format.get("type") in ("json_object", "json_schema")),
            prompt_tokens=estimate_tokens("\n".join(m["content"] for m in messages)),
            notes=notes,
        )

    # -- shaping -------------------------------------------------------------
    def _shape(self, turns: list[Turn], instructions: str, notes: dict[str, Any]) -> list[dict[str, str]]:
        s = self.s
        if s.history_mode == "flatten":
            return [{"role": "user", "content": self._fit_last(self._flatten(turns, instructions), notes, [])}]

        wrapped = f"{INSTR_OPEN}\n{instructions}\n{INSTR_CLOSE}" if instructions else ""
        prefix: list[Turn] = []
        last = turns[-1]
        history = turns[:-1]
        mode = s.system_mode
        if wrapped and mode == "auto":
            fits = len(wrapped) + 2 + len(last.text) <= s.max_input_chars
            mode = "merge" if fits else "primer"
        if wrapped and mode == "merge":
            last = Turn("user", f"{wrapped}\n\n{last.text}")
        elif wrapped and mode == "primer":
            prefix = [Turn("user", f"{wrapped}\n\nReply with OK to confirm."), Turn("assistant", "OK.")]
        if wrapped:
            notes["system_mode"] = mode

        extra_prefix: list[Turn] = []
        last_text = self._fit_last(last.text, notes, extra_prefix)

        out = [{"role": "system", "content": instructions}] if (instructions and mode == "role") else []
        protected = prefix + extra_prefix
        history = self._trim(history, budget=s.max_context_chars - len(last_text)
                             - sum(len(t.text) for t in protected) - (len(instructions) if mode == "role" else 0),
                             notes=notes)
        ordered = merge_same_role(prefix + history + extra_prefix + [Turn("user", last_text)])
        out.extend({"role": t.role, "content": t.text} for t in ordered)
        return out

    @staticmethod
    def _flatten(turns: list[Turn], instructions: str) -> str:
        blocks = []
        if len(turns) > 1:
            blocks.append("The following is a conversation transcript. Reply to the last [User] message as the Assistant.")
        if instructions:
            blocks.append(f"[System]\n{instructions}")
        for t in turns:
            blocks.append(f"[{'User' if t.role == 'user' else 'Assistant'}]\n{t.text}")
        return "\n\n".join(blocks)

    def _fit_last(self, text: str, notes: dict[str, Any], prefix_out: list[Turn]) -> str:
        limit = self.s.max_input_chars
        if not limit or len(text) <= limit:
            return text
        policy = self.s.overflow_policy
        if policy == "truncate":
            template = "\n\n[... {} characters truncated by glm-bridge ...]\n\n"
            room = max(limit - len(template.format(len(text))), 0)
            head, tail = room // 2, room - room // 2
            removed = len(text) - head - tail
            notes["truncated_chars"] = removed
            return text[:head] + template.format(removed) + (text[-tail:] if tail else "")
        if policy == "split":
            header_room = 120
            size = max(limit - header_room, 1)
            chunks = [text[i:i + size] for i in range(0, len(text), size)]
            n = len(chunks)
            for i, chunk in enumerate(chunks[:-1], start=1):
                prefix_out.append(Turn("user", f"[Part {i}/{n} of a long message. Reply only with OK until the final part arrives.]\n{chunk}"))
                prefix_out.append(Turn("assistant", "OK."))
            notes["split_parts"] = n
            return f"[Part {n}/{n} - final part. Now respond to the whole message.]\n{chunks[-1]}"
        raise BridgeError(
            f"The last message is {len(text)} characters after formatting; the upstream accepts at most {limit}. "
            "Shorten it, or set OVERFLOW_POLICY=truncate or split on the bridge.",
            status=400, type="invalid_request_error", code="context_length_exceeded", param="messages",
        )

    @staticmethod
    def _trim(history: list[Turn], budget: int, notes: dict[str, Any]) -> list[Turn]:
        dropped = 0
        history = list(history)
        total = sum(len(t.text) for t in history)
        while history and total > max(budget, 0):
            total -= len(history.pop(0).text)
            dropped += 1
        while dropped and history and history[0].role != "user":
            history.pop(0)
            dropped += 1
        if dropped:
            notes["history_dropped_turns"] = dropped
        return history
