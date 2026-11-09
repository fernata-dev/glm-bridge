"""Prompt-based tool-calling emulation.

The upstream only produces text, so tools are described in the prompt and the
model is asked to answer with ``<tool_call>{json}</tool_call>`` blocks, which are
parsed back into OpenAI ``tool_calls``. GLM's native ``<arg_key>/<arg_value>``
layout inside ``<tool_call>`` is also accepted, as are bare JSON answers.
This is best-effort: a model can still ignore the format. Unparseable output is
returned as plain text rather than as a broken tool call.
"""

from __future__ import annotations

import json
import re
import secrets
from dataclasses import dataclass
from typing import Any

from .errors import bad_request

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"
_BLOCK = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.S)
_FENCE = re.compile(r"^\s*```[\w-]*\s*\n?(.*?)\n?\s*```\s*$", re.S)
_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


def normalize_tools(tools: list[dict] | None, functions: list[dict] | None) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for item in tools or []:
        if not isinstance(item, dict):
            raise bad_request("Each tool must be an object", param="tools")
        if item.get("type", "function") != "function":
            continue  # non-function tools (e.g. built-in web search) are not emulated
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        specs.append(_spec(fn))
    for fn in functions or []:  # legacy `functions` parameter
        specs.append(_spec(fn))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise bad_request("Tool names must be unique", param="tools")
    return specs


def _spec(fn: Any) -> ToolSpec:
    if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
        raise bad_request("Each tool needs a function name", param="tools")
    params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {"type": "object", "properties": {}}
    return ToolSpec(name=fn["name"], description=str(fn.get("description") or ""), parameters=params)


def tool_choice_mode(tool_choice: Any, function_call: Any = None) -> tuple[str, str | None]:
    """Return (mode, forced_name) with mode in auto|none|required|forced."""
    choice = tool_choice if tool_choice is not None else function_call
    if choice is None or choice == "auto":
        return "auto", None
    if choice == "none":
        return "none", None
    if choice in ("required", "any"):
        return "required", None
    if isinstance(choice, dict):
        fn = choice.get("function") if isinstance(choice.get("function"), dict) else choice
        if isinstance(fn.get("name"), str):
            return "forced", fn["name"]
    raise bad_request("Unsupported tool_choice value", param="tool_choice")


def _compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_tool_prompt(specs: list[ToolSpec], mode: str, forced: str | None) -> str:
    lines = [
        "# Tools",
        "You can call these tools (JSON Schema definitions):",
        "<tools>",
        *(_compact({"name": s.name, "description": s.description, "parameters": s.parameters}) for s in specs),
        "</tools>",
        "",
        "To call a tool, reply with one block per call in exactly this format and nothing after the blocks:",
        OPEN_TAG,
        '{"name": "<tool name>", "arguments": {<arguments as a JSON object>}}',
        CLOSE_TAG,
        "Rules: use only the tools listed, with arguments matching their schema. Never invent a tool result; "
        "results come back to you inside <tool_response> blocks. If no tool is needed, answer normally "
        "without any <tool_call> block.",
    ]
    if mode == "required":
        lines.append("In this reply you MUST call at least one tool.")
    elif mode == "forced" and forced:
        lines.append(f"In this reply you MUST call the tool `{forced}`.")
    return "\n".join(lines)


def render_assistant_tool_calls(tool_calls: list[dict]) -> str:
    blocks = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                pass
        blocks.append(f"{OPEN_TAG}\n{_compact({'name': fn.get('name'), 'arguments': args})}\n{CLOSE_TAG}")
    return "\n".join(blocks)


def render_tool_result(content: str, name: str | None, call_id: str | None) -> str:
    attrs = "".join(f' {k}="{v}"' for k, v in (("name", name), ("id", call_id)) if v)
    return f"<tool_response{attrs}>\n{content}\n</tool_response>"


def new_call_id() -> str:
    return "call_" + secrets.token_hex(12)


def _strip_fence(text: str) -> str:
    m = _FENCE.match(text)
    return m.group(1) if m else text


def _first_json_value(text: str) -> Any:
    text = _strip_fence(text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
                return value
            except json.JSONDecodeError:
                continue
    return None


def _call_from_obj(obj: Any) -> tuple[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    return name, args


def _parse_block(block: str) -> tuple[str, Any] | None:
    block = block.strip()
    if "<arg_key>" in block or (block and not block.lstrip().startswith(("{", "`", "["))):
        # GLM native: first line is the function name, then key/value pairs.
        name = block.split("\n", 1)[0].split("<arg_key>", 1)[0].strip()
        if name and re.fullmatch(r"[\w.\-]+", name):
            args = {}
            for key, raw in _GLM_ARG.findall(block):
                raw = raw.strip()
                try:
                    args[key.strip()] = json.loads(raw)
                except json.JSONDecodeError:
                    args[key.strip()] = raw
            return name, args
    return _call_from_obj(_first_json_value(block))


def _normalize_args(args: Any) -> str | None:
    if args is None:
        return "{}"
    if isinstance(args, str):
        try:
            parsed = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
        return _compact(parsed) if isinstance(parsed, dict) else None
    if isinstance(args, dict):
        return _compact(args)
    return None


def parse_tool_calls(text: str, allowed: set[str]) -> tuple[str, list[dict]] | None:
    """Parse model output. Returns (leading_text, tool_calls) or None if no valid call."""
    candidates: list[tuple[str, Any]] = []
    leading = text
    if OPEN_TAG in text:
        leading = text.split(OPEN_TAG, 1)[0]
        for block in _BLOCK.findall(text):
            parsed = _parse_block(block)
            if parsed:
                candidates.append(parsed)
    else:
        value = _first_json_value(text)
        stripped = _strip_fence(text.strip()).strip()
        if value is not None and stripped[:1] in "{[":
            leading = ""
            if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
                value = value["tool_calls"]
            items = value if isinstance(value, list) else [value]
            for item in items:
                parsed = _call_from_obj(item)
                if parsed:
                    candidates.append(parsed)
    calls = []
    for name, args in candidates:
        if allowed and name not in allowed:
            continue
        arguments = _normalize_args(args)
        if arguments is None:
            continue
        calls.append({"id": new_call_id(), "type": "function", "function": {"name": name, "arguments": arguments}})
    if not calls:
        return None
    return leading.strip(), calls
