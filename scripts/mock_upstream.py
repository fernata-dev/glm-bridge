"""Offline stand-in for glm-ai.chat (same protocol), for wiring up automation tools
without spending the real site's quota.

    uvicorn scripts.mock_upstream:app --port 9001
    UPSTREAM_BASE_URL=http://127.0.0.1:9001 python -m glm_bridge serve

Magic words in the last message: LIMIT -> limit event, SLOW -> 1.6 s before the
first token; a prompt that mentions get_weather -> emulated tool call.
"""
import asyncio
import json

from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

NONCE = {"v": "nonce-A"}
CALLS = []

async def session(request):
    resp = JSONResponse({"nonce": NONCE["v"], "paidLeft": 7})
    resp.set_cookie("glm_visitor", "visitor-1")
    return resp

async def message(request):
    if request.headers.get("x-glm-nonce") != NONCE["v"]:
        return JSONResponse({"code": "rest_cookie_invalid_nonce", "message": "Cookie check failed"}, status_code=403)
    body = await request.json()
    CALLS.append({"cookie": request.headers.get("cookie"), "body": body})
    last = body["messages"][-1]["content"]
    async def gen():
        yield "data: " + json.dumps({"type": "meta", "label": body["model"]}) + "\n\n"
        if "LIMIT" in last:
            yield "data: " + json.dumps({"type": "limit", "message": "Daily limit reached", "switch": "glm-4-7-flash"}) + "\n\n"
            return
        if "SLOW" in last:
            await asyncio.sleep(1.6)
        if "<tool_response id=" in last:
            words = ["It ", "is ", "31°C ", "in ", "Cairo."]
        elif "get_weather" in last:
            words = ['<tool_call>\n{"name": "get_weather", ', '"arguments": {"city": "Cairo"}}\n</tool_call>']
        else:
            words = ["Hello ", "from ", "the ", "fake ", "site. ", "مرحبا"]
        for w in words:
            await asyncio.sleep(0.03)
            yield "data: " + json.dumps({"type": "delta", "text": w}) + "\r\n\r\n"
        yield "data: " + json.dumps({"type": "done", "paidLeft": 6}) + "\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")

async def rotate(request):
    NONCE["v"] = NONCE["v"] + "x"
    return JSONResponse({"nonce": NONCE["v"]})

async def calls(request):
    return JSONResponse(CALLS)

app = Starlette(routes=[
    Route("/wp-json/glm-chat/v1/session", session),
    Route("/wp-json/glm-chat/v1/message", message, methods=["POST"]),
    Route("/_rotate", rotate, methods=["POST"]),
    Route("/_calls", calls),
])
