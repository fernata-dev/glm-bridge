"""Incremental output processing shared by streaming and non-streaming paths.

Pipeline per text delta:  stop-sequence filter  ->  tool-call detector  ->  emitted text.
Both stages hold back only the minimum suffix that could still turn into a match,
so streaming stays live while results are identical to processing the full text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tools import OPEN_TAG, parse_tool_calls

_LEAD_JSON_CHARS = ("{", "[", "`")


def _partial_suffix(text: str, token: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``token``."""
    for n in range(min(len(token) - 1, len(text)), 0, -1):
        if token.startswith(text[-n:]):
            return n
    return 0


@dataclass
class FinalOutput:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


class OutputProcessor:
    def __init__(self, stop: list[str] | None = None, tool_names: set[str] | None = None) -> None:
        self.stop = [s for s in (stop or []) if s]
        self.hold = max((len(s) for s in self.stop), default=1) - 1
        self.tools = tool_names is not None and len(tool_names) > 0
        self.tool_names = tool_names or set()
        self._stop_buf = ""
        self.stopped = False
        # tool detection state
        self._lead = ""
        self._lead_decided = not self.tools
        self._tag_buf = ""
        self._capturing = False
        self._capture = ""

    # -- stage 1: stop sequences ------------------------------------------
    def feed(self, text: str) -> str:
        """Feed a delta; return text that is safe to emit now."""
        if self.stopped or not text:
            return ""
        if not self.stop:
            return self._tool_stage(text)
        buf = self._stop_buf + text
        hits = [i for i in (buf.find(s) for s in self.stop) if i >= 0]
        if hits:
            self.stopped = True
            self._stop_buf = ""
            return self._tool_stage(buf[: min(hits)])
        cut = len(buf) - self.hold if self.hold else len(buf)
        cut = max(cut, 0)
        self._stop_buf = buf[cut:]
        return self._tool_stage(buf[:cut])

    # -- stage 2: tool-call detection -------------------------------------
    def _tool_stage(self, text: str) -> str:
        if not self.tools or not text:
            return text
        if self._capturing:
            self._capture += text
            return ""
        if not self._lead_decided:
            self._lead += text
            head = self._lead.lstrip()
            if not head:
                return ""
            if len(head) < len(OPEN_TAG) and OPEN_TAG.startswith(head):
                return ""  # could still become "<tool_call>"
            self._lead_decided = True
            text, self._lead = self._lead, ""
            if head[0] in _LEAD_JSON_CHARS:
                # A bare JSON / fenced answer may be a tool call: buffer it until the end.
                self._capturing = True
                self._capture = text
                return ""
            # Everything else (including a leading "<tool_call>") goes through tag detection.
        combined = self._tag_buf + text
        idx = combined.find(OPEN_TAG)
        if idx >= 0:
            self._capturing = True
            self._capture = combined[idx:]
            self._tag_buf = ""
            return combined[:idx]
        keep = _partial_suffix(combined, OPEN_TAG)
        self._tag_buf = combined[len(combined) - keep:] if keep else ""
        return combined[: len(combined) - keep]

    # -- end of stream ------------------------------------------------------
    def finish(self) -> FinalOutput:
        tail = ""
        if self._stop_buf:
            pending, self._stop_buf = self._stop_buf, ""
            tail += self._tool_stage(pending)
        if not self.tools:
            return FinalOutput(text=tail)
        tail += self._tag_buf + self._lead
        self._tag_buf = self._lead = ""
        if self._capturing:
            parsed = parse_tool_calls(self._capture, self.tool_names)
            if parsed is not None:
                leading, calls = parsed
                return FinalOutput(text=tail + leading, tool_calls=calls, finish_reason="tool_calls")
            tail += self._capture
        return FinalOutput(text=tail)


def process_full_text(text: str, stop: list[str] | None, tool_names: set[str] | None) -> tuple[str, FinalOutput]:
    proc = OutputProcessor(stop, tool_names)
    emitted = proc.feed(text)
    final = proc.finish()
    return emitted, final
