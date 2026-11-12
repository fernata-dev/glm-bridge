from __future__ import annotations

import base64
import json

import httpx

from conftest import KEY, PNG, chat, event, parse_sse, reply, sse_body, status, streamed


# -- auth & meta ---------------------------------------------------------------

async def test_health_is_public(make_client):
    client = await make_client()
    r = await client.get("/health", headers={"Authorization": ""})
    assert r.status_code == 200 and r.json()["status"] == "ok"


async def test_auth_required_and_key_formats(make_client):
    client = await make_client()
    r = await client.get("/v1/models", headers={"Authorization": ""})
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_api_key"
    r = await client.get("/v1/models", headers={"Authorization": "Bearer sk-wrong-key-000000000"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    r = await client.get("/v1/models", headers={"Authorization": "", "x-api-key": KEY})
    assert r.status_code == 200


async def test_hashed_key(make_client):
    import hashlib
    client = await make_client(api_keys="sha256:" + hashlib.sha256(KEY.encode()).hexdigest())
    assert (await client.get("/v1/models")).status_code == 200


async def test_rate_limit(make_client):
    client = await make_client(default_rpm=2)
    assert (await client.get("/v1/models")).status_code == 200
    assert (await client.get("/v1/models")).status_code == 200
    r = await client.get("/v1/models")
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1
    assert r.json()["error"]["code"] == "rate_limit_exceeded"


async def test_models_and_aliases(make_client):
    client = await make_client(model_aliases='{"gpt-4o": "glm-5.3"}')
    ids = {m["id"] for m in (await client.get("/v1/models")).json()["data"]}
    assert {"glm-5.3-flash", "glm-4.5-flash", "gpt-4o", "glm-image"} <= ids
    assert (await client.get("/v1/models/glm-5-3-flash")).json()["id"] == "glm-5.3-flash"
    assert (await client.get("/v1/models/nope")).status_code == 404


async def test_routes_work_without_v1_prefix(make_client, fake):
    client = await make_client()
    r = await client.post("/chat/completions", json=chat())
    assert r.status_code == 200


# -- chat: non-streaming ----------------------------------------------------------

async def test_basic_completion(make_client, fake):
    client = await make_client()
    r = await client.post("/v1/chat/completions", json=chat("Hi"))
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Hello world"}
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["total_tokens"] > 0
    assert body["glm_bridge"]["paid_left"] == 3
    assert r.headers["x-bridge-paid-left"] == "3"
    sent = fake.message_bodies[0]
    assert sent == {"model": "glm-5-3-flash", "messages": [{"role": "user", "content": "Hi"}]}
    headers = fake.message_headers[0]
    assert headers["x-glm-nonce"] == "nonce-1"
    assert headers["x-requested-with"] == "com.duckduckgo.mobile.android"
    assert headers["origin"] == "https://glm-ai.chat"


async def test_nonce_cached_between_requests(make_client, fake):
    client = await make_client()
    for _ in range(3):
        assert (await client.post("/v1/chat/completions", json=chat())).status_code == 200
    assert fake.session_calls == 1


async def test_nonce_refreshed_on_403(make_client, fake):
    client = await make_client()
    await client.post("/v1/chat/completions", json=chat())
    fake.expire_nonce()
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200
    assert fake.session_calls == 2
    assert r.json()["glm_bridge"]["nonce_refreshed"] is True


async def test_persistent_403_is_502(make_client, fake):
    client = await make_client()
    fake.script = [status(403), status(403)]
    # nonce valid but server keeps answering 403 (IP block)
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_auth_failed"


async def test_limit_event_maps_to_429(make_client, fake):
    client = await make_client()
    fake.script = [event("limit", message="Daily limit reached", switch="glm-4-7-flash")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 429
    err = r.json()["error"]
    assert err["code"] == "upstream_limit_reached" and err["upstream_switch"] == "glm-4-7-flash"


async def test_follow_switch(make_client, fake):
    client = await make_client(follow_switch=True)
    fake.script = [event("limit", message="limit", switch="glm-4-7-flash"), reply("switched ok")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "switched ok"
    assert fake.message_bodies[1]["model"] == "glm-4-7-flash"
    assert r.json()["glm_bridge"]["switched_to"] == "glm-4-7-flash"


async def test_offline_maps_to_503(make_client, fake):
    client = await make_client()
    fake.script = [event("offline", message="maintenance")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and r.json()["error"]["code"] == "upstream_offline"


async def test_transient_errors_are_retried(make_client, fake):
    client = await make_client(upstream_max_retries=2)
    fake.script = [status(502, "bad gateway"), lambda b: (_ for _ in ()).throw(httpx.ConnectError("boom")),
                   reply("recovered")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "recovered"
    assert r.json()["glm_bridge"]["retries"] == 2


async def test_retries_exhausted(make_client, fake):
    client = await make_client(upstream_max_retries=1)
    fake.script = [status(500), status(500)]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 502 and r.json()["error"]["code"] == "upstream_http_error"


async def test_upstream_400_is_client_error(make_client, fake):
    client = await make_client()
    fake.script = [status(400, json.dumps({"code": "bad", "message": "Message too long"}))]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 400 and "Message too long" in r.json()["error"]["message"]


async def test_empty_upstream_response_is_retried(make_client, fake):
    client = await make_client()
    fake.script = [lambda b: httpx.Response(200, content=b""), reply("second try")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.json()["choices"][0]["message"]["content"] == "second try"


async def test_mid_stream_network_error_non_stream(make_client, fake):
    client = await make_client()
    fake.script = [streamed([sse_body([{"type": "delta", "text": "part"}], done=False), httpx.ReadError("reset")])]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 502


async def test_upstream_model_fallback_meta(make_client, fake):
    client = await make_client()
    fake.script = [reply("x", meta={"fallback": True, "label": "GLM-4.7 Flash"})]
    r = await client.post("/v1/chat/completions", json=chat())
    meta = r.json()["glm_bridge"]
    assert meta["upstream_fallback"] is True and meta["upstream_label"] == "GLM-4.7 Flash"
    assert r.headers["x-bridge-upstream-fallback"] == "1"


async def test_reasoning_events(make_client, fake):
    client = await make_client()
    fake.script = [lambda b: httpx.Response(200, content=sse_body([
        {"type": "reasoning", "text": "thinking..."}, {"type": "delta", "text": "answer"}, {"type": "done"}]))]
    msg = (await client.post("/v1/chat/completions", json=chat())).json()["choices"][0]["message"]
    assert msg["content"] == "answer" and msg["reasoning_content"] == "thinking..."


async def test_stop_sequences(make_client, fake):
    client = await make_client()
    fake.script = [reply("Thought: x\nAction: ", "search\nObserv", "ation: secret")]
    r = await client.post("/v1/chat/completions", json=chat(stop=["\nObservation:"]))
    assert r.json()["choices"][0]["message"]["content"] == "Thought: x\nAction: search"


async def test_json_mode_strips_fences(make_client, fake):
    client = await make_client()
    fake.script = [reply('```json\n{"a": 1}\n```')]
    r = await client.post("/v1/chat/completions", json=chat(response_format={"type": "json_object"}))
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"a": 1}
    assert "valid JSON" in fake.message_bodies[0]["messages"][-1]["content"]


async def test_validation_errors_are_openai_shaped(make_client):
    client = await make_client()
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3", "messages": []})
    assert r.status_code == 400 and r.json()["error"]["type"] == "invalid_request_error"
    r = await client.post("/v1/chat/completions", content=b"not json")
    assert r.status_code == 400
    r = await client.post("/v1/chat/completions", json=chat(n=2))
    assert r.status_code == 400 and r.json()["error"]["param"] == "n"
    r = await client.post("/v1/chat/completions", json={**chat(), "model": "gpt-9"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"


async def test_input_too_long(make_client, fake):
    client = await make_client(max_input_chars=50)
    r = await client.post("/v1/chat/completions", json=chat("x" * 60))
    assert r.status_code == 400 and r.json()["error"]["code"] == "context_length_exceeded"
    assert fake.message_bodies == []


async def test_body_limit(make_client):
    client = await make_client(max_body_bytes=1000)
    r = await client.post("/v1/chat/completions", json=chat("x" * 2000))
    assert r.status_code == 413


# -- chat: streaming --------------------------------------------------------------

async def test_streaming(make_client, fake):
    client = await make_client()
    fake.script = [reply("Hel", "lo")]
    r = await client.post("/v1/chat/completions", json=chat(stream=True, stream_options={"include_usage": True}))
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    chunks, done = parse_sse(r.text)
    assert done
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])
    assert text == "Hello"
    finals = [c for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]]
    assert finals[-1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["completion_tokens"] > 0


async def test_streaming_preflight_error_returns_status(make_client, fake):
    client = await make_client()
    fake.script = [event("limit", message="quota")]
    r = await client.post("/v1/chat/completions", json=chat(stream=True))
    assert r.status_code == 429 and r.json()["error"]["code"] == "upstream_limit_reached"


async def test_streaming_mid_stream_error_chunk(make_client, fake):
    client = await make_client()
    fake.script = [lambda b: httpx.Response(200, content=sse_body([
        {"type": "delta", "text": "partial"}, {"type": "error", "message": "overloaded"}]))]
    r = await client.post("/v1/chat/completions", json=chat(stream=True))
    chunks, done = parse_sse(r.text)
    assert done and chunks[-1]["error"]["code"] == "upstream_error"


async def test_streaming_commits_after_preflight_timeout_and_pings(make_client, fake):
    client = await make_client(preflight_timeout=0.1, keepalive_interval=0.1)
    fake.script = [streamed([0.35, sse_body([{"type": "delta", "text": "late"}, {"type": "done"}])])]
    r = await client.post("/v1/chat/completions", json=chat(stream=True))
    assert r.status_code == 200
    assert ": keep-alive" in r.text
    chunks, done = parse_sse(r.text)
    assert done and "late" in "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"])


# -- tools ------------------------------------------------------------------------

WEATHER_TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


async def test_tool_call_non_stream(make_client, fake):
    client = await make_client()
    fake.script = [reply('<tool_call>\n{"name": "get_weather", ', '"arguments": {"city": "Cairo"}}\n</tool_call>')]
    r = await client.post("/v1/chat/completions", json=chat("Weather in Cairo?", tools=[WEATHER_TOOL]))
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Cairo"}
    assert choice["message"]["content"] is None
    prompt = fake.message_bodies[0]["messages"][-1]["content"]
    assert "get_weather" in prompt and "<tool_call>" in prompt


async def test_tool_call_streaming(make_client, fake):
    client = await make_client()
    fake.script = [reply("Let me check.\n<tool", '_call>{"name":"get_weather","arguments":{"city":"Giza"}}</tool_call>')]
    r = await client.post("/v1/chat/completions", json=chat(stream=True, tools=[WEATHER_TOOL]))
    chunks, done = parse_sse(r.text)
    content = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert content == "Let me check.\n"
    calls = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if c["choices"] and "tool_calls" in c["choices"][0]["delta"]]
    assert calls[0][0]["index"] == 0 and calls[0][0]["function"]["name"] == "get_weather"
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


async def test_tool_round_trip_history(make_client, fake):
    client = await make_client()
    fake.script = [reply("It is sunny in Cairo.")]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Weather in Cairo?"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                               "function": {"name": "get_weather", "arguments": "{\"city\":\"Cairo\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 31}"},
    ]
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3", "messages": messages, "tools": [WEATHER_TOOL]})
    assert r.json()["choices"][0]["message"]["content"] == "It is sunny in Cairo."
    sent = fake.message_bodies[0]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert '<tool_call>\n{"name":"get_weather","arguments":{"city":"Cairo"}}' in sent[1]["content"]
    assert '<tool_response id="call_1">' in sent[2]["content"] and "You are helpful." in sent[2]["content"]


async def test_invalid_tool_call_falls_back_to_text(make_client, fake):
    client = await make_client()
    fake.script = [reply('<tool_call>{"name": "launch_rockets", "arguments": {}}</tool_call>')]
    r = await client.post("/v1/chat/completions", json=chat(tools=[WEATHER_TOOL]))
    msg = r.json()["choices"][0]["message"]
    assert "tool_calls" not in msg and "launch_rockets" in msg["content"]


async def test_legacy_functions(make_client, fake):
    client = await make_client()
    fake.script = [reply('{"name": "get_weather", "arguments": {"city": "Aswan"}}')]
    r = await client.post("/v1/chat/completions", json=chat(functions=[WEATHER_TOOL["function"]]))
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "function_call"
    assert choice["message"]["function_call"]["name"] == "get_weather"


# -- vision ---------------------------------------------------------------------------

def _image_msg(url: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                        {"type": "image_url", "image_url": {"url": url}}]}


async def test_vision_data_url_and_routing(make_client, fake):
    client = await make_client()
    data_url = "data:image/png;base64," + base64.b64encode(PNG).decode()
    r = await client.post("/v1/chat/completions", json={"model": "glm-4.7", "messages": [_image_msg(data_url)]})
    assert r.status_code == 200
    sent = fake.message_bodies[0]
    assert sent["image"] == data_url and sent["model"] == "glm-5-3-flash"
    assert sent["messages"] == [{"role": "user", "content": "What is this?"}]
    assert r.json()["glm_bridge"]["vision_routed_from"] == "glm-4.7"


async def test_vision_remote_url(make_client, fake):
    client = await make_client()
    fake.remote_images["https://img.example.com/cat.png"] = httpx.Response(200, content=PNG)
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3-flash",
                                                        "messages": [_image_msg("https://img.example.com/cat.png")]})
    assert r.status_code == 200
    assert fake.message_bodies[0]["image"].startswith("data:image/png;base64,")


async def test_vision_ssrf_blocked(make_client, fake):
    client = await make_client()
    for url in ("http://127.0.0.1/x.png", "http://internal.example.com/x.png", "http://rebind.example.com/x.png",
                "file:///etc/passwd"):
        r = await client.post("/v1/chat/completions", json={"model": "glm-5.3-flash", "messages": [_image_msg(url)]})
        assert r.status_code == 400, url
    assert fake.message_bodies == []


async def test_vision_redirect_to_private_blocked(make_client, fake):
    client = await make_client()
    fake.remote_images["https://img.example.com/r.png"] = httpx.Response(302, headers={"location": "http://10.1.2.3/x.png"})
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3-flash",
                                                        "messages": [_image_msg("https://img.example.com/r.png")]})
    assert r.status_code == 400


async def test_vision_rejects_non_images_and_big_images(make_client, fake):
    client = await make_client(max_image_bytes=100)
    bad = "data:image/png;base64," + base64.b64encode(b"<html>not an image</html>").decode()
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3-flash", "messages": [_image_msg(bad)]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_image"
    big = "data:image/png;base64," + base64.b64encode(PNG + b"\x00" * 200).decode()
    r = await client.post("/v1/chat/completions", json={"model": "glm-5.3-flash", "messages": [_image_msg(big)]})
    assert r.status_code == 400 and r.json()["error"]["code"] == "image_too_large"


# -- fallback provider ------------------------------------------------------------

FALLBACK = {"fallback_base_url": "https://api.fallback.test/v4", "fallback_api_key": "fb-key",
            "fallback_model": "glm-4.5-flash"}


async def test_fallback_on_offline(make_client, fake):
    client = await make_client(**FALLBACK)
    fake.script = [event("offline", message="down")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from fallback"
    assert r.headers["x-bridge-provider"] == "fallback"
    assert fake.fallback_bodies[0]["body"]["model"] == "glm-4.5-flash"
    assert fake.fallback_bodies[0]["auth"] == "Bearer fb-key"


async def test_fallback_after_partial_non_stream_answer(make_client, fake):
    client = await make_client(**FALLBACK)
    fake.script = [streamed([sse_body([{"type": "delta", "text": "half an ans"}], done=False), httpx.ReadError("reset")])]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from fallback"


async def test_fallback_streaming(make_client, fake):
    client = await make_client(**FALLBACK)
    fake.script = [event("limit", message="quota")]
    r = await client.post("/v1/chat/completions", json=chat(stream=True))
    assert r.status_code == 200 and "from fallback" in r.text and r.text.strip().endswith("[DONE]")


async def test_fallback_not_used_for_client_errors(make_client, fake):
    client = await make_client(**FALLBACK)
    fake.script = [status(400, '{"message": "bad input"}')]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 400 and fake.fallback_bodies == []


async def test_fallback_failure_reports_both(make_client, fake):
    client = await make_client(**FALLBACK)
    fake.script = [event("offline", message="down")]
    fake.fallback_response = lambda body: httpx.Response(401, json={"error": {"message": "bad key"}})
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and "fallback provider also failed" in r.json()["error"]["message"]


# -- images ---------------------------------------------------------------------------

async def test_image_generation_url_passthrough(make_client, fake):
    client = await make_client()
    r = await client.post("/v1/images/generations", json={"prompt": "a red fox"})
    assert r.status_code == 200 and r.json()["data"] == [{"url": "https://cdn.example.com/out.png"}]
    assert fake.image_bodies[0] == {"prompt": "a red fox"}


async def test_image_generation_b64_upstream_served_as_url(make_client, fake):
    client = await make_client()
    b64 = base64.b64encode(PNG).decode()
    fake.image_script = [lambda b: httpx.Response(200, json={"success": True, "data": {"image": f"data:image/png;base64,{b64}"}})]
    r = await client.post("/v1/images/generations", json={"prompt": "fox"})
    url = r.json()["data"][0]["url"]
    assert "/v1/images/files/" in url
    img = await client.get(url.replace("http://bridge", ""), headers={"Authorization": ""})
    assert img.status_code == 200 and img.content == PNG and img.headers["content-type"] == "image/png"


async def test_image_generation_b64_json_fetches_url(make_client, fake):
    client = await make_client()
    fake.remote_images["https://cdn.example.com/out.png"] = httpx.Response(200, content=PNG)
    r = await client.post("/v1/images/generations", json={"prompt": "fox", "response_format": "b64_json"})
    assert base64.b64decode(r.json()["data"][0]["b64_json"]) == PNG


async def test_image_generation_errors(make_client, fake):
    client = await make_client()
    fake.image_script = [lambda b: httpx.Response(200, json={"success": False, "message": "Image quota exceeded"})]
    r = await client.post("/v1/images/generations", json={"prompt": "fox"})
    assert r.status_code == 502 and "Image quota exceeded" in r.json()["error"]["message"]


async def test_image_edit_multipart(make_client, fake):
    client = await make_client()
    r = await client.post("/v1/images/edits", data={"prompt": "make it blue"},
                          files={"image": ("in.png", PNG, "image/png")})
    assert r.status_code == 200
    assert fake.image_bodies[0]["image"].startswith("data:image/png;base64,")


async def test_status_endpoint(make_client, fake):
    client = await make_client()
    await client.post("/v1/chat/completions", json=chat())
    st = (await client.get("/v1/bridge/status")).json()
    assert st["upstream"]["session_ready"] is True and st["upstream"]["paid_left"] == 3
    assert "nonce" not in json.dumps(st["upstream"]["session_info"])


async def test_short_cooldown_is_waited_out(make_client, fake):
    client = await make_client()
    fake.script = [event("limit", message="One moment - send again in 0 s."), reply("after wait")]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "after wait"


FALLBACK_META = {"model": "glm-4-5-flash", "label": "GLM-4.5-Flash (daily limit reached)", "fallback": True}


async def test_reports_actual_model(make_client, fake):
    client = await make_client()
    fake.script = [reply("hi", meta=FALLBACK_META), reply("hi", meta=FALLBACK_META)]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.json()["model"] == "glm-4.5-flash" and r.json()["glm_bridge"]["actual_model"] == "glm-4-5-flash"
    assert r.headers["x-bridge-actual-model"] == "glm-4-5-flash"
    r = await client.post("/v1/chat/completions", json=chat(stream=True))
    chunks, _ = parse_sse(r.text)
    assert {c["model"] for c in chunks} == {"glm-4.5-flash"}


async def test_report_actual_model_can_be_disabled(make_client, fake):
    client = await make_client(report_actual_model=False)
    fake.script = [reply("hi", meta=FALLBACK_META)]
    assert (await client.post("/v1/chat/completions", json=chat())).json()["model"] == "glm-5.3-flash"


async def test_strict_model_rejects_silent_fallback(make_client, fake):
    client = await make_client(strict_model=True)
    fake.script = [reply("hi", meta=FALLBACK_META)]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"


async def test_strict_model_uses_fallback_provider(make_client, fake):
    client = await make_client(strict_model=True, **FALLBACK)
    fake.script = [reply("hi", meta=FALLBACK_META)]
    r = await client.post("/v1/chat/completions", json=chat())
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"] == "from fallback"


async def test_auto_model_uses_site_suggestion(make_client, fake):
    client = await make_client()
    r = await client.post("/v1/chat/completions", json={**chat(), "model": "auto"})
    assert r.status_code == 200 and fake.message_bodies[0]["model"] == "glm-4-5-flash"


async def test_requests_are_paced(make_client, fake):
    import time
    client = await make_client(upstream_min_interval=0.3)
    t0 = time.monotonic()
    for _ in range(3):
        assert (await client.post("/v1/chat/completions", json=chat())).status_code == 200
    assert time.monotonic() - t0 >= 0.6
