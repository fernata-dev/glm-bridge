"""A tolerant Server-Sent Events parser.

Follows the WHATWG framing rules (``data:`` lines joined with ``\\n``, blank line
dispatches, ``:`` comments ignored, CRLF tolerated) and additionally accepts bare
JSON lines, which some PHP backends emit when an error short-circuits the stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass


@dataclass(slots=True)
class SSEEvent:
    event: str
    data: str


class SSEParser:
    def __init__(self) -> None:
        self._data: list[str] = []
        self._event = ""

    def feed_line(self, line: str) -> SSEEvent | None:
        line = line.rstrip("\r\n")
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None
        if line.startswith("{") and not self._data:
            # Bare JSON line outside SSE framing.
            return SSEEvent(event="message", data=line)
        field, sep, value = line.partition(":")
        if not sep:
            return None
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            self._data.append(value)
        elif field == "event":
            self._event = value
        return None

    def flush(self) -> SSEEvent | None:
        return self._dispatch()

    def _dispatch(self) -> SSEEvent | None:
        if not self._data:
            self._event = ""
            return None
        ev = SSEEvent(event=self._event or "message", data="\n".join(self._data))
        self._data = []
        self._event = ""
        return ev


async def iter_sse(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    parser = SSEParser()
    async for line in lines:
        ev = parser.feed_line(line)
        if ev is not None:
            yield ev
    ev = parser.flush()
    if ev is not None:
        yield ev


def parse_sse_text(lines: Iterable[str]) -> list[SSEEvent]:
    """Synchronous helper (used by tests and the probe)."""
    parser = SSEParser()
    out = [ev for line in lines if (ev := parser.feed_line(line)) is not None]
    if (ev := parser.flush()) is not None:
        out.append(ev)
    return out


def decode_json_event(ev: SSEEvent) -> dict | None:
    data = ev.data.strip()
    if not data or data == "[DONE]":
        return None
    try:
        obj = json.loads(data)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None
