"""Place a real outbound call by originating it from Twilio and bridging it into LiveKit.

    python scripts/call_via_twilio.py --patient P001
    python scripts/call_via_twilio.py --patient P001 --to +919170898012
    python scripts/call_via_twilio.py --dry-run        # print the TwiML, call nothing

Twilio's trial plan does not include Elastic SIP Trunking, so LiveKit cannot dial out through it.
Twilio's REST voice API is available on trial, so the call is originated there instead and bridged
into LiveKit over SIP. The patient still receives a genuine phone call.

Run `scripts/setup_inbound_sip.py` once first, and have `python main.py dev` running.
"""

from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from xml.sax.saxutils import quoteattr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from services import patients  # noqa: E402

load_dotenv()

API_ROOT = "https://api.twilio.com/2010-04-01"

# Twilio error codes worth translating — the raw message is not always actionable.
TWILIO_HINTS = {
    "21215": "Twilio has not enabled calling to this country. Console → Voice → Settings → "
             "Geographic Permissions, tick India, save.",
    "21219": "On a trial account you can only call verified numbers. Console → Phone Numbers → "
             "Verified Caller IDs → add this number.",
    "21210": "The From number is not one of your Twilio numbers. Check TWILIO_FROM_NUMBER.",
    "21211": "The To number is not valid E.164 (needs the + and country code).",
    "573002": "Twilio could not route to this number. On trial it must be a verified caller ID, "
              "and it must be in E.164 form with a leading '+'.",
    "21606": "The From number cannot make outbound calls. Check it has Voice capability.",
}

TRIAL_TWIML_HINT = (
    "Trial accounts reject the inline Twiml parameter. Create a TwiML Bin in the Twilio console "
    "(Builder tools -> TwiML host & config -> TwiML Bins) containing the markup printed above "
    "with sip:{{room}}@<host>, then set TWILIO_TWIML_BIN_URL in .env to its handler URL."
)


def to_e164(number: str) -> str:
    """Twilio rejects anything that is not E.164, and a missing '+' is the usual slip.

    Punctuation is stripped, and a leading '+' is added when the digits already include a
    country code. A bare national number is rejected rather than guessed at.
    """
    raw = (number or "").strip()
    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        raise ValueError("no digits in the number")
    if raw.startswith("+"):
        return f"+{digits}"
    if len(digits) >= 11:
        return f"+{digits}"
    raise ValueError(
        f"{number!r} looks like a national number — use E.164 with the country code, "
        "e.g. +919170898012"
    )


def build_twiml(sip_user: str) -> str:
    host = os.getenv("LIVEKIT_SIP_HOST", "")
    username = os.getenv("LIVEKIT_SIP_USERNAME", "")
    password = os.getenv("LIVEKIT_SIP_PASSWORD", "")
    uri = f"sip:{sip_user}@{host}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Dial answerOnBridge="true" timeout="30">'
        f"<Sip username={quoteattr(username)} password={quoteattr(password)}>{uri}</Sip>"
        "</Dial>"
        "</Response>"
    )


def place_call(patient_id: str, to_override: str | None, dry_run: bool) -> int:
    try:
        patient = patients.get_patient(patient_id)
    except patients.PatientNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        patients.assert_contactable(patient)
    except patients.PatientNotContactable as exc:
        print(f"refusing to call: {exc}", file=sys.stderr)
        return 2

    missing = [
        v for v in ("LIVEKIT_SIP_HOST", "LIVEKIT_SIP_USERNAME", "LIVEKIT_SIP_PASSWORD")
        if not os.getenv(v)
    ]
    if missing:
        print(f"error: {', '.join(missing)} not set — run scripts/setup_inbound_sip.py first",
              file=sys.stderr)
        return 1

    # The user part of the SIP URI becomes the room name, which is how the agent learns
    # which patient it is calling. No per-call dispatch rule needed.
    sip_user = f"{patient.id}-{uuid.uuid4().hex[:6]}"
    twiml = build_twiml(sip_user)
    try:
        to_number = to_e164(to_override or patient.phone_number)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"patient   {patient.name} ({patient.id})")
    print(f"to        {to_number}")
    print(f"room      call-{sip_user}")
    if os.getenv("TWILIO_TWIML_BIN_URL"):
        print(f"twiml bin {os.getenv('TWILIO_TWIML_BIN_URL')}?room={sip_user}")
    else:
        print(f"twiml     {twiml}")

    if dry_run:
        print("\ndry run — nothing was dialled")
        return 0

    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    if not (sid and token and from_number):
        print("error: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER in .env",
              file=sys.stderr)
        return 1

    # Trial accounts reject the inline `Twiml` parameter ("limited parameter access"), but accept
    # `Url`. A TwiML Bin hosts the markup and templates {{room}} from the query string, so one
    # static bin serves every call.
    bin_url = os.getenv("TWILIO_TWIML_BIN_URL")
    if bin_url:
        payload = {
            "To": to_number,
            "From": from_number,
            "Url": f"{bin_url}?room={sip_user}",
        }
    else:
        payload = {"To": to_number, "From": from_number, "Twiml": twiml}

    try:
        response = httpx.post(
            f"{API_ROOT}/Accounts/{sid}/Calls.json",
            auth=(sid, token),
            data=payload,
            timeout=30,
        )
    except httpx.HTTPError as exc:
        print(f"error: could not reach Twilio: {exc}", file=sys.stderr)
        return 1

    if response.status_code >= 400:
        body = response.json() if response.headers.get("content-type", "").startswith(
            "application/json"
        ) else {}
        code = str(body.get("code", ""))
        print(f"\nTwilio rejected the call (HTTP {response.status_code})", file=sys.stderr)
        print(f"  {body.get('message') or response.text[:300]}", file=sys.stderr)
        if "limited parameter access" in str(body.get("message", "")):
            print(f"\n  → {TRIAL_TWIML_HINT}", file=sys.stderr)
        elif code in TWILIO_HINTS:
            print(f"\n  → {TWILIO_HINTS[code]}", file=sys.stderr)
        elif code:
            print(f"\n  Twilio error {code}: https://www.twilio.com/docs/errors/{code}",
                  file=sys.stderr)
        return 1

    call = response.json()
    print(f"\ncall {call.get('sid')} queued — status {call.get('status')}")
    print("the patient's phone should ring shortly; watch the worker logs")
    print("the Opik trace appears once the call ends")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Place a call via Twilio, bridged into LiveKit")
    parser.add_argument("--patient", default="P001")
    parser.add_argument("--to", help="override the number on file (must be verified on trial)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return place_call(args.patient, args.to, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
