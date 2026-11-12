"""Command line: serve | keygen | probe."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

import httpx

from . import __version__
from .auth import generate_key, hash_key
from .config import Settings


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name,
               "msg": record.getMessage()}
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def setup_logging(settings: Settings) -> None:
    handler = logging.StreamHandler()
    if settings.log_json:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(settings.log_level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    settings = Settings()
    setup_logging(settings)
    try:
        app = create_app(settings)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    uvicorn.run(app, host=args.host or settings.host, port=args.port or settings.port,
                log_level=settings.log_level.lower(), access_log=False, proxy_headers=True)
    return 0


def cmd_keygen(args: argparse.Namespace) -> int:
    for _ in range(args.count):
        key = generate_key()
        print(f"key:  {key}")
        print(f"hash: {hash_key(key)}")
    print("\nPut the key (or its hash) in API_KEYS; give the key itself to your automation tool.")
    return 0


# -- probe ------------------------------------------------------------------

async def _ask(client: Any, body: dict[str, Any]) -> dict[str, Any]:
    from .upstream import NonceRejected, UpstreamHTTPError

    result: dict[str, Any] = {"text": "", "events": [], "error": None}
    await asyncio.sleep(4)  # free models allow one message every ~3 s
    nonce = await client.nonce()
    for attempt in range(2):
        try:
            async for ev in client.stream_message(body, nonce):
                result["events"].append(ev.type)
                if ev.type == "delta":
                    result["text"] += ev.text
                elif ev.type == "meta":
                    result["meta"] = ev.data
                elif ev.type == "done":
                    result["paid_left"] = ev.data.get("paidLeft")
                elif ev.type in ("error", "limit", "notice", "offline"):
                    result["error"] = f"{ev.type}: {ev.message}"
                    break
            return result
        except NonceRejected:
            if attempt:
                result["error"] = "HTTP 403 (nonce rejected twice)"
                return result
            nonce = await client.nonce(refresh_from=nonce)
        except UpstreamHTTPError as exc:
            result["error"] = f"HTTP {exc.status}: {exc.body[:150]}"
            return result
        except httpx.HTTPError as exc:
            result["error"] = f"network: {exc.__class__.__name__}"
            return result
    return result


async def run_probe(settings: Settings, check_models: bool) -> int:
    from .errors import BridgeError
    from .models import ModelRegistry
    from .upstream import GLMChatClient

    registry = ModelRegistry(settings)
    model = registry.resolve(settings.default_model).upstream
    client = GLMChatClient(settings)

    def show(title: str, res: dict[str, Any]) -> None:
        status = "ERROR " + res["error"] if res["error"] else "ok"
        print(f"  {title}: {status}")
        if res["text"]:
            print(f"    reply: {res['text'][:160]!r}")
        if res.get("meta"):
            print(f"    meta:  {res['meta']}")
        if res.get("paid_left") is not None:
            print(f"    paidLeft: {res['paid_left']}")

    try:
        print(f"glm-bridge {__version__} probe -> {settings.upstream_base_url}\n")
        try:
            await client.nonce()
        except BridgeError as exc:
            print(f"[1] session: FAILED - {exc.message}")
            return 1
        print(f"[1] session: ok (extra fields: {sorted(client.state.session_info)})")

        print(f"[2] basic chat with upstream model '{model}'")
        basic = await _ask(client, {"model": model, "messages": [
            {"role": "user", "content": "Reply with exactly the word PONG and nothing else."}]})
        show("basic", basic)
        if basic["error"] and not basic["text"]:
            return 1

        print("[3] does the upstream accept a 'system' role?")
        sys_res = await _ask(client, {"model": model, "messages": [
            {"role": "system", "content": "Whatever the user says, reply only with the word BANANA."},
            {"role": "user", "content": "What is 2+2?"}]})
        show("system role", sys_res)
        system_ok = not sys_res["error"] and "banana" in sys_res["text"].lower()

        print("[4] does the upstream use earlier messages (history)?")
        hist = await _ask(client, {"model": model, "messages": [
            {"role": "user", "content": "Remember this code word: ZEBRA-4217. Reply only with OK."},
            {"role": "assistant", "content": "OK."},
            {"role": "user", "content": "What code word did I give you? Reply with the code word only."}]})
        show("history", hist)
        history_ok = not hist["error"] and "4217" in hist["text"]

        if check_models:
            print("[5] model ids (uses quota on paid models)")
            for info in registry.models.values():
                candidates = [info.upstream] + ([info.id] if info.id != info.upstream else [])
                verdict = "FAILED"
                for cand in candidates:
                    res = await _ask(client, {"model": cand, "messages": [{"role": "user", "content": "Reply OK."}]})
                    if res["text"] and not res["error"]:
                        meta = res.get("meta") or {}
                        label = f" label={meta.get('label')!r}" if meta.get("label") else ""
                        fb = " FELL BACK" if meta.get("fallback") else ""
                        verdict = f"ok as '{cand}'{label}{fb}"
                        break
                    verdict = f"FAILED ({res['error'] or 'empty reply'})"
                print(f"  {info.id:<16} {verdict}")

        print("\nRecommended settings:")
        print(f"  SYSTEM_MODE={'role' if system_ok else 'auto'}")
        print(f"  HISTORY_MODE={'native' if history_ok else 'flatten'}")
        print("  If every model shows the same label, the site may ignore the model id; fix ids with MODEL_MAP.")
        if not history_ok:
            print("  (history is not honoured: flatten packs the conversation into one message, "
                  "so long chats will hit MAX_INPUT_CHARS sooner)")
        return 0
    finally:
        await client.aclose()


def cmd_probe(args: argparse.Namespace) -> int:
    settings = Settings()
    logging.basicConfig(level=logging.WARNING)
    return asyncio.run(run_probe(settings, args.models))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="glm-bridge", description="OpenAI-compatible bridge for glm-ai.chat")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")

    p_serve = sub.add_parser("serve", help="run the API server (default)")
    p_serve.add_argument("--host")
    p_serve.add_argument("--port", type=int)
    p_serve.set_defaults(func=cmd_serve)

    p_key = sub.add_parser("keygen", help="generate API keys")
    p_key.add_argument("-n", "--count", type=int, default=1)
    p_key.set_defaults(func=cmd_keygen)

    p_probe = sub.add_parser("probe", help="test what the live upstream supports")
    p_probe.add_argument("--models", action="store_true", help="also test every model id (consumes quota)")
    p_probe.set_defaults(func=cmd_probe)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        args = parser.parse_args(["serve", *(argv or sys.argv[1:])])
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
