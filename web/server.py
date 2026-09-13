"""Web console for the voice agent.

    uv run python web/server.py        then open http://localhost:8080

Serves one page that starts a call in the browser, streams the live transcript, and shows the
post-call analysis once the agent has finished logging it. The browser is a real LiveKit
participant, so the agent, the tools, the recording and the Opik traces are all the production
path — only the last mile is WebRTC instead of a phone line.

Analysis is read back out of Opik rather than cached here: the traces are the system of record.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aiohttp import web  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

from observability import opik_reader as reader  # noqa: E402
from services import patients  # noqa: E402

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web-console")

AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-caller")
PORT = int(os.getenv("PORT") or os.getenv("WEB_PORT") or 8080)
WEB_DIR = Path(__file__).resolve().parent
INDEX = WEB_DIR / "index.html"
OVERVIEW = WEB_DIR / "overview.html"


async def index(_: web.Request) -> web.StreamResponse:
    return web.FileResponse(INDEX)


async def overview(_: web.Request) -> web.StreamResponse:
    """Static explainer: the call pipeline and the Opik trace topology."""
    return web.FileResponse(OVERVIEW)


async def list_patients(_: web.Request) -> web.Response:
    return web.json_response([
        {
            "id": p.id,
            "name": p.name,
            "care_program": p.care_program,
            "clinician": p.ordering_clinician,
            "phone": patients.mask_phone(p.phone_number),
            "blocked": p.do_not_call or not p.consent_to_call,
            "blocked_reason": (
                "On the do-not-call list" if p.do_not_call
                else "Has not consented to outbound calls" if not p.consent_to_call
                else None
            ),
            "biomarkers": [
                {
                    "name": b.name,
                    "value": b.value,
                    "unit": b.unit,
                    "status": b.status,
                    "reference": b.reference_range,
                    "collected": b.collected_on,
                }
                for b in p.biomarkers
            ],
        }
        for p in patients.list_patients()
    ])


async def start_call(request: web.Request) -> web.Response:
    body = await request.json()
    patient_id = body.get("patient_id", "P001")

    try:
        patient = patients.get_patient(patient_id)
    except patients.PatientNotFound as exc:
        return web.json_response({"error": str(exc)}, status=404)

    # The consent gate runs before a worker is ever involved, exactly as in the CLI dispatcher.
    try:
        patients.assert_contactable(patient)
    except patients.PatientNotContactable as exc:
        return web.json_response({"error": str(exc)}, status=403)

    room_name = f"web-{patient.id}-{uuid.uuid4().hex[:6]}"

    try:
        async with api.LiveKitAPI() as lk:
            await lk.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=AGENT_NAME,
                    room=room_name,
                    metadata=json.dumps({"patient_id": patient.id}),
                )
            )
    except Exception as exc:
        logger.exception("dispatch failed")
        return web.json_response(
            {"error": f"Could not dispatch the agent: {exc}. Is `python main.py dev` running?"},
            status=502,
        )

    token = (
        api.AccessToken(os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET"))
        .with_identity(f"patient-{patient.id}")
        .with_name(patient.name)
        .with_grants(api.VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    logger.info("dispatched %s to %s", AGENT_NAME, room_name)
    return web.json_response({
        "room": room_name,
        "token": token,
        "url": os.getenv("LIVEKIT_URL"),
        "patient": patient.name,
    })


async def end_call(request: web.Request) -> web.Response:
    room_name = (await request.json()).get("room")
    if not room_name:
        return web.json_response({"error": "room is required"}, status=400)
    try:
        async with api.LiveKitAPI() as lk:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))
    except Exception as exc:
        logger.warning("could not delete room %s: %s", room_name, exc)
    return web.json_response({"ok": True})


async def analysis(request: web.Request) -> web.Response:
    """Poll target. Returns ready=False until the agent has flushed the call to Opik."""
    room_name = request.query.get("room")
    if not room_name:
        return web.json_response({"error": "room is required"}, status=400)
    if not reader.configured():
        return web.json_response({"ready": False, "error": "Opik is not configured"})

    try:
        summary = await asyncio.to_thread(reader.find_call_by_room, room_name)
        if summary is None:
            return web.json_response({"ready": False})
        detail = await asyncio.to_thread(reader.get_call, summary.trace_id)
        project_id, _ = await asyncio.to_thread(reader.list_calls, 1)
    except Exception as exc:
        logger.exception("could not read analysis")
        return web.json_response({"ready": False, "error": str(exc)})

    a = detail.analysis
    return web.json_response({
        "ready": True,
        "outcome": summary.outcome,
        "appointment_booked": a.get("appointment_booked"),
        "identity_verified": a.get("identity_verified"),
        "sentiment": a.get("patient_sentiment"),
        "summary": a.get("call_summary"),
        "next_action": a.get("next_action"),
        "safety_violations": a.get("safety_violations") or [],
        "corrections": a.get("_corrections") or [],
        "source": a.get("_source"),
        "biomarkers_communicated": a.get("biomarkers_communicated") or [],
        "appointment": a.get("appointment"),
        "scores": summary.scores,
        "score_reasons": detail.score_reasons,
        "tools": [
            {"name": t["name"], "arguments": t["input"], "result": t["output"]}
            for t in detail.tool_spans
        ],
        "trace_url": reader.trace_url(project_id, summary.trace_id),
        "recording": detail.metadata.get("recording_attached"),
    })


def build_app() -> web.Application:
    app = web.Application()
    app.add_routes([
        web.get("/", index),
        web.get("/overview", overview),
        web.get("/api/patients", list_patients),
        web.post("/api/call", start_call),
        web.post("/api/end", end_call),
        web.get("/api/analysis", analysis),
    ])
    return app


if __name__ == "__main__":
    missing = [
        v for v in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET") if not os.getenv(v)
    ]
    if missing:
        print(f"error: {', '.join(missing)} not set in .env", file=sys.stderr)
        raise SystemExit(1)

    print(f"\n  Care Outreach Console → http://localhost:{PORT}")
    print("  (the agent worker must be running: uv run python main.py dev)\n")
    web.run_app(build_app(), host="0.0.0.0", port=PORT, print=None)
