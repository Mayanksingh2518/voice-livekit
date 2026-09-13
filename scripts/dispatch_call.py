"""Trigger one outbound call.

    python scripts/dispatch_call.py --patient P001
    python scripts/dispatch_call.py --patient P001 --phone +919000000000   # override the number
    python scripts/dispatch_call.py --list

The consent and do-not-call gate runs here, before the agent is ever dispatched. That is
deliberate: a call that should not happen should never reach a worker.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

from services import patients  # noqa: E402

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-caller")


def _print_patients() -> None:
    for p in patients.list_patients():
        flags = []
        if p.do_not_call:
            flags.append("DO-NOT-CALL")
        if not p.consent_to_call:
            flags.append("NO-CONSENT")
        marks = "  ".join(f"{b.name} {b.value}{b.unit}" for b in p.biomarkers)
        print(f"{p.id}  {p.name:<16} {patients.mask_phone(p.phone_number):<16} {marks}"
              f"{'   [' + ', '.join(flags) + ']' if flags else ''}")


async def dispatch(patient_id: str, phone: str | None, force: bool) -> int:
    try:
        patient = patients.get_patient(patient_id)
    except patients.PatientNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        patients.assert_contactable(patient)
    except patients.PatientNotContactable as exc:
        if not force:
            print(f"refusing to call: {exc}", file=sys.stderr)
            print("(pass --force only if you have a documented reason)", file=sys.stderr)
            return 2
        print(f"warning: overriding consent gate — {exc}", file=sys.stderr)

    for var in ("LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if not os.getenv(var):
            print(f"error: {var} is not set", file=sys.stderr)
            return 1

    number = phone or patient.phone_number
    room_name = f"call-{patient.id}-{uuid.uuid4().hex[:6]}"
    metadata = {
        "patient_id": patient.id,
        "phone_number": number,
        "transfer_to": os.getenv("TRANSFER_TO"),
    }

    async with api.LiveKitAPI() as lk:
        await lk.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_NAME,
                room=room_name,
                metadata=json.dumps(metadata),
            )
        )

    print(f"dispatched {AGENT_NAME} -> room {room_name}")
    print(f"calling {patient.name} at {patients.mask_phone(number)}")
    print("watch the worker logs; the Opik trace appears when the call ends")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Place an outbound healthcare call")
    parser.add_argument("--patient", default="P001", help="patient id from data/patients.json")
    parser.add_argument("--phone", help="override the number on file (e.g. your test phone)")
    parser.add_argument("--list", action="store_true", help="list patients and exit")
    parser.add_argument("--force", action="store_true", help="bypass the consent gate")
    args = parser.parse_args()

    if args.list:
        _print_patients()
        return 0
    return asyncio.run(dispatch(args.patient, args.phone, args.force))


if __name__ == "__main__":
    raise SystemExit(main())
