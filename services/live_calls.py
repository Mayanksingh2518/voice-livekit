"""Control calls that are happening right now.

Opik is the record of finished calls; this is the live view. Two layers can be ended
independently, and a stuck call usually needs both:

  * the Twilio leg — the phone that is ringing or connected
  * the LiveKit room — the agent's session

Ending the Twilio leg hangs up the phone. Deleting the room stops the agent and triggers the
normal shutdown path, so the call is still analysed and logged.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

API_ROOT = "https://api.twilio.com/2010-04-01"
LIVE_TWILIO_STATES = ("queued", "ringing", "in-progress")


@dataclass
class LiveRoom:
    name: str
    participants: int
    created_at: datetime | None


@dataclass
class LiveTwilioCall:
    sid: str
    to: str
    status: str
    started_at: str | None


def _twilio_auth() -> tuple[str, str] | None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    return (sid, token) if sid and token else None


def list_rooms() -> list[LiveRoom]:
    """Rooms currently open on LiveKit — i.e. calls in progress."""
    from livekit import api

    async def _run() -> list[LiveRoom]:
        async with api.LiveKitAPI() as lk:
            res = await lk.room.list_rooms(api.ListRoomsRequest())
            return [
                LiveRoom(
                    name=r.name,
                    participants=r.num_participants,
                    created_at=(
                        datetime.fromtimestamp(r.creation_time) if r.creation_time else None
                    ),
                )
                for r in res.rooms
            ]

    return asyncio.run(_run())


def end_room(room_name: str) -> None:
    """Delete the room. The agent's shutdown path still runs, so the call is analysed and logged."""
    from livekit import api

    async def _run() -> None:
        async with api.LiveKitAPI() as lk:
            await lk.room.delete_room(api.DeleteRoomRequest(room=room_name))

    asyncio.run(_run())


def list_twilio_calls() -> list[LiveTwilioCall]:
    """Twilio legs that are ringing or connected."""
    auth = _twilio_auth()
    if not auth:
        return []
    sid, _ = auth
    calls: dict[str, LiveTwilioCall] = {}
    for status in LIVE_TWILIO_STATES:
        try:
            r = httpx.get(
                f"{API_ROOT}/Accounts/{sid}/Calls.json",
                auth=auth,
                params={"Status": status, "PageSize": 20},
                timeout=20,
            )
            r.raise_for_status()
        except httpx.HTTPError:
            continue
        for c in r.json().get("calls", []):
            calls[c["sid"]] = LiveTwilioCall(
                sid=c["sid"],
                to=c.get("to", ""),
                status=c.get("status", status),
                started_at=c.get("start_time") or c.get("date_created"),
            )
    return list(calls.values())


def end_twilio_call(call_sid: str) -> dict[str, Any]:
    """Hang up the phone leg by completing the call."""
    auth = _twilio_auth()
    if not auth:
        raise RuntimeError("TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set")
    sid, _ = auth
    r = httpx.post(
        f"{API_ROOT}/Accounts/{sid}/Calls/{call_sid}.json",
        auth=auth,
        data={"Status": "completed"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


@dataclass
class CallAttempt:
    """One Twilio call and what became of it — the telephony half of the story."""

    sid: str
    to: str
    status: str
    duration: str
    created: str
    sip_leg: str | None  # the child leg <Dial><Sip> should have created

    @property
    def reached_livekit(self) -> bool:
        return self.sip_leg is not None

    @property
    def diagnosis(self) -> str:
        if self.reached_livekit:
            return "bridged into LiveKit"
        if self.status in ("queued", "ringing"):
            return "still ringing"
        if self.status in ("busy", "no-answer", "failed", "canceled"):
            return f"phone did not answer ({self.status})"
        return "answered, but the SIP bridge to LiveKit was never created"


def recent_attempts(limit: int = 10) -> list[CallAttempt]:
    """Recent Twilio calls, each annotated with whether the SIP bridge actually happened."""
    auth = _twilio_auth()
    if not auth:
        return []
    sid, _ = auth
    try:
        r = httpx.get(
            f"{API_ROOT}/Accounts/{sid}/Calls.json",
            auth=auth,
            params={"PageSize": limit * 3},
            timeout=20,
        )
        r.raise_for_status()
    except httpx.HTTPError:
        return []

    calls = r.json().get("calls", [])
    parents = [c for c in calls if not c.get("parent_call_sid")]
    children = {c.get("parent_call_sid"): c for c in calls if c.get("parent_call_sid")}

    out: list[CallAttempt] = []
    for c in parents[:limit]:
        child = children.get(c["sid"])
        out.append(
            CallAttempt(
                sid=c["sid"],
                to=c.get("to", ""),
                status=c.get("status", ""),
                duration=c.get("duration") or "0",
                created=c.get("date_created", ""),
                sip_leg=child["sid"] if child else None,
            )
        )
    return out


def end_everything() -> tuple[int, int]:
    """Hang up every live call, both layers. Returns (twilio_ended, rooms_ended)."""
    twilio = 0
    for call in list_twilio_calls():
        try:
            end_twilio_call(call.sid)
            twilio += 1
        except Exception:
            pass

    rooms = 0
    for room in list_rooms():
        try:
            end_room(room.name)
            rooms += 1
        except Exception:
            pass
    return twilio, rooms
