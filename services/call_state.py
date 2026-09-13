"""Deterministic record of what actually happened on a call.

This is the ground truth the post-call analysis is reconciled against. The analysis LLM reads the
transcript and forms an opinion; this object records facts. Where they disagree, this wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from services.patients import Patient
from services.scheduler import Booking


@dataclass
class ToolInvocation:
    name: str
    arguments: dict[str, Any]
    result: Any
    error: str | None
    called_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "error": self.error,
            "called_at": self.called_at.isoformat(),
        }


@dataclass
class CallState:
    patient: Patient
    room_name: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    answered_at: datetime | None = None
    ended_at: datetime | None = None

    identity_confirmed: bool = False
    voicemail_detected: bool = False
    do_not_call_requested: bool = False
    transfer_requested: bool = False
    transfer_reason: str | None = None
    end_reason: str | None = None

    sip_status_code: int | None = None
    disconnect_reason: str | None = None
    dial_error: str | None = None

    bookings: list[Booking] = field(default_factory=list)
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    recording_path: str | None = None
    recording_url: str | None = None
    egress_id: str | None = None

    def record_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = None,
        error: str | None = None,
    ) -> None:
        self.tool_invocations.append(
            ToolInvocation(
                name=name,
                arguments=arguments,
                result=result,
                error=error,
                called_at=datetime.now(timezone.utc),
            )
        )

    @property
    def appointment_booked(self) -> bool:
        """Authoritative. A tool call existing is not success — only a booking returning is."""
        return bool(self.bookings)

    @property
    def confirmed_booking(self) -> Booking | None:
        return self.bookings[-1] if self.bookings else None

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None
        return round((self.ended_at - self.started_at).total_seconds(), 2)

    @property
    def talk_time_seconds(self) -> float | None:
        if self.answered_at is None or self.ended_at is None:
            return None
        return round((self.ended_at - self.answered_at).total_seconds(), 2)

    def deterministic_outcome(self) -> str | None:
        """Outcomes that are known without reading the transcript.

        Returns None when the call genuinely needs the analysis LLM to classify it.
        """
        if self.dial_error:
            return self.dial_error
        if self.voicemail_detected:
            return "voicemail"
        if self.do_not_call_requested:
            return "do_not_call_requested"
        if self.transfer_requested:
            return "escalated_to_human"
        if self.appointment_booked:
            return "appointment_booked"
        if self.answered_at and not self.identity_confirmed:
            return "identity_unverified"
        return None

    def to_dict(self) -> dict[str, Any]:
        booking = self.confirmed_booking
        return {
            "room_name": self.room_name,
            "started_at": self.started_at.isoformat(),
            "answered_at": self.answered_at.isoformat() if self.answered_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": self.duration_seconds,
            "talk_time_seconds": self.talk_time_seconds,
            "identity_confirmed": self.identity_confirmed,
            "voicemail_detected": self.voicemail_detected,
            "do_not_call_requested": self.do_not_call_requested,
            "transfer_requested": self.transfer_requested,
            "transfer_reason": self.transfer_reason,
            "end_reason": self.end_reason,
            "sip_status_code": self.sip_status_code,
            "disconnect_reason": self.disconnect_reason,
            "dial_error": self.dial_error,
            "appointment_booked": self.appointment_booked,
            "booking": booking.to_dict() if booking else None,
            "tool_invocations": [t.to_dict() for t in self.tool_invocations],
            "errors": self.errors,
            "recording_url": self.recording_url,
            "recording_path": self.recording_path,
            "egress_id": self.egress_id,
        }
