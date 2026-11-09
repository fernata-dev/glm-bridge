from __future__ import annotations

import json

import pytest

from conftest import make_settings
from glm_bridge.convert import INSTR_OPEN, PromptBuilder
from glm_bridge.errors import BridgeError
from glm_bridge.images import ImageFetcher
from glm_bridge.models import ModelRegistry
from glm_bridge.output import OutputProcessor
from glm_bridge.schemas import ChatRequest
from glm_bridge.sse import parse_sse_text
from glm_bridge.tools import parse_tool_calls

# -- SSE --------------------------------------------------------------------------


def test_sse_framing_variants():
    lines = ["data: {\"type\":\"delta\",\"text\":\"a\"}\r", "", ": comment", "event: meta", "data: {\"x\":1}",
             "data: {\"y\":2}", "", "{\"type\":\"done\"}", "data: [DONE]"]
    events = parse_sse_text(lines)
    assert events[0].data == '{"type":"delta","text":"a"}'
    assert events[1].event == "meta" and events[1].data == '{"x":1}\n{"y":2}'
    assert events[2].data == '{"type":"done"}'
    assert events[3].data == "[DONE]"  # flushed without trailing blank line


# -- tool parsing -------------------------------------------------------------------

ALLOWED = {"get_weather", "search"}


@pytest.mark.parametrize("text", [
    '<tool_call>{"name":"get_weather","arguments":{"city":"Cairo"}}</tool_call>',
    '<tool_call>\n```json\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Cairo\\"}"}\n```\n</tool_call>',
    '<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Cairo</arg_value>\n</tool_call>',
    '```json\n{"name": "get_weather", "arguments": {"city": "Cairo"}}\n```',
    '{"tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Cairo"}}}]}',
    '<tool_call>{"name":"get_weather","arguments":{"city":"Cairo"}}',  # unterminated
])
def test_parse_tool_call_formats(text):
    leading, calls = parse_tool_calls(text, ALLOWED)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Cairo"}
    assert calls[0]["id"].startswith("call_")


def test_parse_parallel_calls_and_leading_text():
    text = ('Sure.\n<tool_call>{"name":"search","arguments":{"q":"a"}}</tool_call>\n'
            '<tool_call>{"name":"get_weather","arguments":{"city":"b"}}</tool_call>')
    leading, calls = parse_tool_calls(text, ALLOWED)
    assert leading == "Sure." and [c["function"]["name"] for c in calls] == ["search", "get_weather"]


@pytest.mark.parametrize("text", [
    "Just a normal answer.",
    '{"name": "Mohamed", "age": 30}',  # JSON answer, not a known tool
    '<tool_call>{"name":"rm_rf","arguments":{}}</tool_call>',
    '<tool_call>{"name":"search","arguments":"not json"}</tool_call>',
])
def test_parse_rejects_non_calls(text):
    assert parse_tool_calls(text, ALLOWED) is None


# -- output processor ------------------------------------------------------------------

def _stream_all(text: str, stop=None, tools=None, step=1):
    proc = OutputProcessor(stop, tools)
    out = "".join(proc.feed(text[i:i + step]) for i in range(0, len(text), step))
    final = proc.finish()
    return out + final.text, final


@pytest.mark.parametrize("step", [1, 2, 3, 7, 1000])
def test_streaming_equals_full_text(step):
    text = "Answer first.\nObservation: hidden"
    out, final = _stream_all(text, stop=["\nObservation:"], step=step)
    assert out == "Answer first." and final.finish_reason == "stop"
    text2 = 'Checking.<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>'
    out2, final2 = _stream_all(text2, tools=ALLOWED, step=step)
    assert out2 == "Checking." and final2.tool_calls[0]["function"]["name"] == "search"


def test_partial_tag_prefix_is_released_when_not_a_tag():
    out, final = _stream_all("a < b and <tool is fine", tools=ALLOWED, step=1)
    assert out == "a < b and <tool is fine" and not final.tool_calls


def test_json_answer_with_tools_is_buffered_then_released():
    out, final = _stream_all('{"result": 42}', tools=ALLOWED, step=3)
    assert out == '{"result": 42}' and not final.tool_calls


def test_html_answer_with_tools_streams():
    proc = OutputProcessor(None, ALLOWED)
    assert proc.feed("<div>hello</div> and more text") == "<div>hello</div> and more text"


# -- prompt building -------------------------------------------------------------------

async def _build(messages, **overrides):
    settings = make_settings(**overrides)
    builder = PromptBuilder(settings, ModelRegistry(settings), ImageFetcher(settings))
    return await builder.build(ChatRequest(model="glm-5.3", messages=messages))


SYS = {"role": "system", "content": "Be brief."}


async def test_system_auto_merges_when_it_fits():
    prep = await _build([SYS, {"role": "user", "content": "Hi"}])
    msgs = prep.body["messages"]
    assert len(msgs) == 1 and msgs[0]["content"].startswith(INSTR_OPEN) and msgs[0]["content"].endswith("Hi")
    assert prep.notes["system_mode"] == "merge"


async def test_system_auto_uses_primer_when_too_long():
    prep = await _build([SYS, {"role": "user", "content": "x" * 7995}])
    msgs = prep.body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert "Be brief." in msgs[0]["content"] and msgs[2]["content"] == "x" * 7995


async def test_system_role_mode():
    prep = await _build([SYS, {"role": "user", "content": "Hi"}], system_mode="role")
    assert prep.body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hi"}]


async def test_flatten_mode():
    prep = await _build([SYS, {"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                         {"role": "user", "content": "C"}], history_mode="flatten")
    (only,) = prep.body["messages"]
    assert only["role"] == "user" and "[System]\nBe brief." in only["content"]
    assert only["content"].index("[User]\nA") < only["content"].index("[Assistant]\nB") < only["content"].index("[User]\nC")


async def test_consecutive_roles_merged_and_developer_role():
    prep = await _build([{"role": "developer", "content": "Dev rule."}, {"role": "user", "content": "one"},
                         {"role": "user", "content": "two"}], system_mode="role")
    assert prep.body["messages"] == [{"role": "system", "content": "Dev rule."},
                                     {"role": "user", "content": "one\n\ntwo"}]


async def test_overflow_truncate_keeps_head_and_tail():
    text = "HEAD" + "m" * 500 + "TAIL"
    prep = await _build([{"role": "user", "content": text}], max_input_chars=200, overflow_policy="truncate")
    out = prep.body["messages"][-1]["content"]
    assert len(out) <= 200 and out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out
    assert prep.notes["truncated_chars"] > 300


async def test_overflow_split_into_parts():
    text = "".join(str(i % 10) for i in range(500))
    prep = await _build([{"role": "user", "content": text}], max_input_chars=200, overflow_policy="split")
    msgs = prep.body["messages"]
    assert all(len(m["content"]) <= 200 for m in msgs)
    assert [m["role"] for m in msgs][-1] == "user" and msgs[1] == {"role": "assistant", "content": "OK."}
    rebuilt = "".join(m["content"].split("\n", 1)[1] for m in msgs if m["role"] == "user")
    assert rebuilt == text


async def test_history_trimmed_to_budget():
    history = []
    for i in range(10):
        history += [{"role": "user", "content": f"q{i} " + "x" * 100}, {"role": "assistant", "content": f"a{i} " + "y" * 100}]
    prep = await _build(history + [{"role": "user", "content": "final"}], max_context_chars=500)
    msgs = prep.body["messages"]
    assert sum(len(m["content"]) for m in msgs) <= 500
    assert msgs[0]["role"] == "user" and msgs[-1]["content"] == "final"
    assert prep.notes["history_dropped_turns"] > 0


async def test_last_message_must_be_user():
    with pytest.raises(BridgeError) as exc:
        await _build([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "prefill"}])
    assert exc.value.status == 400


async def test_unsupported_content_part():
    with pytest.raises(BridgeError):
        await _build([{"role": "user", "content": [{"type": "input_audio", "input_audio": {}}]}])


async def test_system_only_request_becomes_user_message():
    prep = await _build([{"role": "system", "content": "Summarize: abc"}])
    assert prep.body["messages"] == [{"role": "user", "content": "Summarize: abc"}]


def test_model_registry_resolution():
    reg = ModelRegistry(make_settings(model_aliases='{"gpt-4o-mini": "glm-4.5-flash"}',
                                      model_map='{"glm-4.7": "glm-4.7"}'))
    assert reg.resolve("GLM-5-3-FLASH").id == "glm-5.3-flash"
    assert reg.resolve("gpt-4o-mini").upstream == "glm-4-5-flash"
    assert reg.resolve("glm-4.7").upstream == "glm-4.7"
    assert reg.resolve(None).id == "glm-5.3-flash"
    assert reg.resolve("glm-5.3-flash").vision and not reg.resolve("glm-5.3").vision
    unknown = ModelRegistry(make_settings(allow_unknown_models=True)).resolve("glm-9")
    assert unknown.upstream == "glm-9" and not unknown.known


def test_config_requires_keys():
    from glm_bridge.auth import Authenticator
    with pytest.raises(ValueError):
        Authenticator(make_settings(api_keys=""))
    with pytest.raises(ValueError):
        Authenticator(make_settings(api_keys="short"))
    Authenticator(make_settings(api_keys="", auth_disabled=True))
