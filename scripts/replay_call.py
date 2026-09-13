"""Replay a recorded conversation through the full post-call pipeline.

    python scripts/replay_call.py                      # replay data/sample_call.json
    python scripts/replay_call.py --audio call.wav     # attach an audio file too
    python scripts/replay_call.py --no-opik            # analysis only, nothing logged

This exists for two reasons:

  1. It demonstrates the whole post-call flow — analysis, reconciliation, Opik traces, thread,
     tool spans, feedback scores — without placing a phone call or spending telephony credit.
  2. It is the proof that observability/opik_tracer.py is genuinely standalone. Here it is driven
     by a fake session object, with no LiveKit runtime involved at all, and the module does not
     know the difference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from analysis.post_call import analyse_call  # noqa: E402
from observability.opik_tracer import OpikCallTracer  # noqa: E402
from services import patients, scheduler  # noqa: E402
from services.call_state import CallState  # noqa: E402

load_dotenv()

FIXTURE = Path(__file__).resolve().parent.parent / "data" / "sample_call.json"


class FakeSession:
    """Stands in for AgentSession. Records handlers so we can fire synthetic events at them."""

    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event: str, callback: Any) -> Any:
        self.handlers.setdefault(event, []).append(callback)
        return callback

    def emit(self, event: str, payload: Any) -> None:
        for handler in self.handlers.get(event, []):
            handler(payload)


def _item_event(role: str, text: str, at: datetime) -> SimpleNamespace:
    return SimpleNamespace(
        item=SimpleNamespace(role=role, text_content=text, interrupted=False),
        created_at=at,
    )


def _tools_event(tools: list[dict[str, Any]], at: datetime) -> SimpleNamespace:
    calls, outputs = [], []
    for index, tool in enumerate(tools):
        call_id = f"call_{index}_{int(at.timestamp())}"
        calls.append(
            SimpleNamespace(name=tool["name"], call_id=call_id, arguments=tool["arguments"])
        )
        outputs.append(
            SimpleNamespace(
                call_id=call_id, output=tool.get("output"), is_error=tool.get("is_error", False)
            )
        )
    return SimpleNamespace(function_calls=calls, function_call_outputs=outputs, created_at=at)


async def replay(fixture: Path, audio: str | None, use_opik: bool) -> int:
    data = json.loads(fixture.read_text())
    patient = patients.get_patient(data["patient_id"])
    room_name = f"replay-{patient.id}-{datetime.now():%H%M%S}"

    state = CallState(patient=patient, room_name=room_name)
    state.answered_at = datetime.now(timezone.utc)
    variables = patients.build_call_variables(patient)

    session = FakeSession()
    if not use_opik:
        os.environ["OPIK_ENABLED"] = "false"
    tracer = OpikCallTracer.from_env(
        call_id=room_name,
        variables=variables,
        metadata={
            "patient_id": patient.id,
            "llm_model": "gpt-4o-mini",
            "llm_provider": "openai",
            "mode": "replay",
            "source_fixture": fixture.name,
        },
        tags=["voice", "outbound", "healthcare", "replay", f"patient:{patient.id}"],
    )
    cached: dict[str, Any] = {}

    async def build_summary(transcript: list[dict[str, Any]]) -> dict[str, Any]:
        """Analysed once; the tracer and the console output share the same result."""
        if cached:
            return cached
        state.ended_at = datetime.now(timezone.utc)
        state.recording_path = audio
        result = await analyse_call(state, transcript)
        cached.update({
            "outcome": result.analysis.outcome,
            "analysis": result.to_dict(),
            "call_record": state.to_dict(),
            "recording_path": audio,
            "analysis_model": result.model,
            "analysis_usage": result.usage,
        })
        return cached

    tracer.attach(session, ctx=None, finalise=build_summary)

    at = datetime.now(timezone.utc) - timedelta(minutes=4)
    for turn in data["turns"]:
        if turn.get("user"):
            session.emit("conversation_item_added", _item_event("user", turn["user"], at))
            at += timedelta(seconds=3)
        if turn.get("assistant"):
            session.emit("conversation_item_added", _item_event("assistant", turn["assistant"], at))
            at += timedelta(seconds=6)
        for tool in turn.get("tools", []):
            _apply_tool(state, tool)
        if turn.get("tools"):
            session.emit("function_tools_executed", _tools_event(turn["tools"], at))
            at += timedelta(seconds=1)

    print(f"replayed {len(data['turns'])} turns for {patient.name}")
    print("running post-call analysis...")

    summary = await build_summary(tracer.transcript())
    analysis = summary["analysis"]

    print()
    print(f"  outcome            {analysis['outcome']}")
    print(f"  appointment booked {analysis['appointment_booked']}")
    print(f"  sentiment          {analysis['patient_sentiment']}")
    print(f"  identity verified  {analysis['identity_verified']}")
    print(f"  safety violations  {analysis['safety_violations'] or 'none'}")
    print(f"  analysis source    {analysis['_source']}")
    if analysis["_corrections"]:
        print(f"  corrections        {analysis['_corrections']}")
    print(f"  summary            {analysis['call_summary']}")
    print()

    if use_opik and tracer.enabled:
        print("sending to Opik...")
        await tracer.finalise()
        print(f"done — look for thread {tracer.thread_id!r} in your Opik project")
    elif use_opik:
        print("Opik is not configured (set OPIK_API_KEY) — nothing was logged")
    return 0


def _apply_tool(state: CallState, tool: dict[str, Any]) -> None:
    """Mirror the side effects the real tools would have had on CallState."""
    name = tool["name"]
    try:
        args = json.loads(tool["arguments"])
    except (json.JSONDecodeError, TypeError):
        args = {}

    if name == "verify_identity":
        state.identity_confirmed = bool(args.get("is_correct_person"))
    elif name == "detected_answering_machine":
        state.voicemail_detected = True
    elif name == "transfer_to_human":
        state.transfer_requested = True
        state.transfer_reason = args.get("reason")
    elif name == "end_call":
        state.end_reason = args.get("reason")
    elif name == "book_appointment":
        try:
            booking = scheduler.book(
                patient_id=state.patient.id,
                patient_name=state.patient.name,
                specialty=args.get("specialty"),
                preferred_date=args.get("preferred_date", "tomorrow"),
                preferred_window=args.get("preferred_window"),
            )
            state.bookings.append(booking)
            tool["output"] = json.dumps(booking.to_dict())
        except scheduler.SchedulingError as exc:
            tool["output"] = str(exc)
            tool["is_error"] = True

    state.record_tool(
        name,
        args,
        result=None if tool.get("is_error") else tool.get("output"),
        error=tool.get("output") if tool.get("is_error") else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a call through the post-call pipeline")
    parser.add_argument("--fixture", type=Path, default=FIXTURE)
    parser.add_argument("--audio", help="path to an audio file to attach to the Opik trace")
    parser.add_argument("--no-opik", action="store_true", help="skip Opik, print analysis only")
    args = parser.parse_args()
    return asyncio.run(replay(args.fixture, args.audio, not args.no_opik))


if __name__ == "__main__":
    raise SystemExit(main())
