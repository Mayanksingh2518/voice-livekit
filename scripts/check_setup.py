"""Verify every credential in .env by actually calling each service.

    python scripts/check_setup.py

Nothing here is a guess — each check makes a real (tiny, free) API request and reports what came
back. Run it whenever a key changes, or when something fails and you want to know which service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

OK, FAIL, SKIP = "OK  ", "FAIL", "SKIP"


def check_livekit() -> tuple[str, str]:
    missing = [v for v in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET")
               if not os.getenv(v)]
    if missing:
        return SKIP, f"not set: {', '.join(missing)}"
    import asyncio

    from livekit import api

    async def ping() -> str:
        async with api.LiveKitAPI() as lk:
            res = await lk.room.list_rooms(api.ListRoomsRequest())
            return f"connected, {len(res.rooms)} room(s) active"

    try:
        return OK, asyncio.run(ping())
    except Exception as exc:
        return FAIL, str(exc)[:140]


def check_llm() -> tuple[str, str]:
    from services.model_config import llm_config

    config = llm_config()
    if not config.configured:
        key = "GEMINI_API_KEY" if config.provider == "gemini" else "OPENAI_API_KEY"
        return FAIL, f"{key} not set (LLM_PROVIDER={config.provider})"

    from openai import OpenAI

    client = OpenAI(api_key=config.api_key, base_url=config.base_url)
    try:
        # A tool call, because that is what the agent actually depends on. The prompt is
        # deliberately unambiguous — a model asking for clarification on a vague request is
        # correct behaviour, not a failure, and would make this check noise.
        r = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": "Today is 2026-09-12. When the user agrees to a "
                                              "time, you MUST call book_appointment."},
                {"role": "assistant", "content": "I have Tuesday morning free."},
                {"role": "user", "content": "Yes, Tuesday morning at 10:30 works."},
            ],
            tools=[{
                "type": "function",
                "function": {
                    "name": "book_appointment",
                    "description": "Book the consultation once the user has agreed to a time",
                    "parameters": {
                        "type": "object",
                        "properties": {"preferred_date": {"type": "string"}},
                        "required": ["preferred_date"],
                    },
                },
            }],
        )
        calls = r.choices[0].message.tool_calls
        detail = f"{config.provider}/{config.model}"
        if calls:
            return OK, f"{detail} responding, tool calling works"
        return FAIL, f"{detail} responds but would not call a tool — the agent needs tool "\
                     "calling; try a different LLM_MODEL"
    except Exception as exc:
        msg = str(exc)
        if "insufficient_quota" in msg or "credit_balance" in msg:
            return FAIL, "key valid but the account has NO CREDITS — add credits, "\
                         "or set LLM_PROVIDER=gemini"
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            return FAIL, f"rate limited on {config.model} — free tier quota. Wait, or try "\
                         "another model in LLM_MODEL"
        if "404" in msg and "no longer available" in msg:
            return FAIL, f"model {config.model!r} retired — pick a current one for LLM_MODEL"
        if "invalid_api_key" in msg or "API_KEY_INVALID" in msg or "Incorrect API key" in msg:
            return FAIL, "key rejected"
        return FAIL, msg[:140]


def check_deepgram() -> tuple[str, str]:
    key = os.getenv("DEEPGRAM_API_KEY")
    if not key:
        return SKIP, "DEEPGRAM_API_KEY not set"
    import httpx

    try:
        r = httpx.get(
            "https://api.deepgram.com/v1/projects",
            headers={"Authorization": f"Token {key}"},
            timeout=20,
        )
        if r.status_code == 200:
            return OK, "key valid"
        return FAIL, f"HTTP {r.status_code}: {r.text[:110]}"
    except Exception as exc:
        return FAIL, str(exc)[:140]


def check_tts() -> tuple[str, str]:
    provider = os.getenv("TTS_PROVIDER", "deepgram")
    if provider == "deepgram":
        status, detail = check_deepgram()
        model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-andromeda-en")
        return status, f"Deepgram {model} — {detail}"
    if provider == "openai":
        return SKIP, "using OpenAI TTS — covered by the LLM check if provider is openai"
    key = os.getenv("CARTESIA_API_KEY")
    if not key:
        return FAIL, "CARTESIA_API_KEY not set (or set TTS_PROVIDER=deepgram)"
    import httpx

    try:
        r = httpx.get(
            "https://api.cartesia.ai/voices",
            headers={"X-API-Key": key, "Cartesia-Version": "2024-06-10"},
            timeout=20,
        )
        if r.status_code == 200:
            return OK, "key valid"
        return FAIL, f"HTTP {r.status_code}: {r.text[:110]}"
    except Exception as exc:
        return FAIL, str(exc)[:140]


def check_opik() -> tuple[str, str]:
    if not os.getenv("OPIK_API_KEY") and not os.getenv("OPIK_URL_OVERRIDE"):
        return SKIP, "OPIK_API_KEY not set — calls still run, nothing is logged"
    project = os.getenv("OPIK_PROJECT_NAME", "livekit-voice-agent")
    try:
        import opik

        client = opik.Opik(project_name=project)
        client.rest_client.projects.find_projects(page=1, size=1)
        return OK, f"authenticated, project {project!r}"
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "403" in msg or "Unauthorized" in msg:
            return FAIL, "key or workspace rejected — check OPIK_API_KEY and OPIK_WORKSPACE"
        return FAIL, msg[:140]


def check_sip() -> tuple[str, str]:
    if not os.getenv("SIP_OUTBOUND_TRUNK_ID"):
        return SKIP, "no trunk yet — console mode works without it"
    return OK, f"trunk {os.getenv('SIP_OUTBOUND_TRUNK_ID')}"


CHECKS = [
    ("LiveKit", check_livekit, True),
    ("LLM", check_llm, True),
    ("Deepgram STT", check_deepgram, True),
    ("TTS", check_tts, True),
    ("Opik", check_opik, False),
    ("SIP trunk", check_sip, False),
]


def main() -> int:
    if not Path(".env").exists():
        print("no .env found — run: cp .env.example .env")
        return 1

    print()
    blocking_failures = 0
    for name, fn, required in CHECKS:
        try:
            status, detail = fn()
        except Exception as exc:
            status, detail = FAIL, f"check crashed: {exc}"[:140]
        print(f"  [{status}]  {name:<11} {detail}")
        if status == FAIL and required:
            blocking_failures += 1
    print()

    if blocking_failures:
        print(f"{blocking_failures} required service(s) failing — fix those before running a call.")
        return 1

    print("Ready. Next: python scripts/replay_call.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
