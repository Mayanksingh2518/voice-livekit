"""Simulated appointment booking.

Stands in for a real scheduling system. Slot availability is generated deterministically from
the date, so a demo reproduces exactly; bookings are held in memory for the life of the process.
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

CLINIC_TZ = ZoneInfo("Asia/Kolkata")

SPECIALTIES: dict[str, str] = {
    "endocrinology": "Dr. Meera Iyer",
    "general medicine": "Dr. Sanjay Rao",
    "diabetology": "Dr. Meera Iyer",
    "nutrition": "Ms. Kavya Nair",
}
DEFAULT_SPECIALTY = "endocrinology"

WINDOWS: dict[str, tuple[time, time]] = {
    "morning": (time(9, 30), time(12, 30)),
    "afternoon": (time(14, 0), time(17, 0)),
    "evening": (time(17, 0), time(19, 30)),
}

SLOT_MINUTES = 30
BOOKING_HORIZON_DAYS = 21
MIN_LEAD_TIME = timedelta(hours=2)

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


class SchedulingError(Exception):
    """Base for scheduling failures the agent is expected to recover from conversationally."""


class SlotUnavailable(SchedulingError):
    def __init__(self, message: str, alternatives: list["Slot"]) -> None:
        super().__init__(message)
        self.alternatives = alternatives


@dataclass(frozen=True)
class Slot:
    specialty: str
    clinician: str
    start: datetime

    @property
    def key(self) -> str:
        return f"{self.specialty}|{self.start.isoformat()}"

    @property
    def spoken(self) -> str:
        return self.start.strftime("%A %d %B at %-I:%M %p")

    def to_dict(self) -> dict[str, Any]:
        return {
            "specialty": self.specialty,
            "clinician": self.clinician,
            "start": self.start.isoformat(),
            "spoken": self.spoken,
        }


@dataclass(frozen=True)
class Booking:
    confirmation_id: str
    patient_id: str
    patient_name: str
    slot: Slot
    booked_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirmation_id": self.confirmation_id,
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "booked_at": self.booked_at.isoformat(),
            **self.slot.to_dict(),
        }


_lock = threading.Lock()
_bookings: dict[str, Booking] = {}


def now() -> datetime:
    return datetime.now(tz=CLINIC_TZ)


def reset() -> None:
    """Test helper — clears in-process bookings."""
    with _lock:
        _bookings.clear()


def resolve_date(text: str, *, today: date | None = None) -> date:
    """Turn whatever the model passed into a real date.

    The prompt asks for YYYY-MM-DD, but models routinely pass 'tomorrow' or a weekday name,
    so those are handled rather than rejected.
    """
    today = today or now().date()
    raw = (text or "").strip().lower()
    if not raw:
        raise SchedulingError("No date was provided.")

    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass

    if raw in {"today", "aaj"}:
        return today
    if raw in {"tomorrow", "kal"}:
        return today + timedelta(days=1)
    if raw in {"day after tomorrow", "overmorrow", "parso"}:
        return today + timedelta(days=2)

    for name, index in _WEEKDAYS.items():
        if name in raw:
            ahead = (index - today.weekday()) % 7 or 7
            return today + timedelta(days=ahead)

    raise SchedulingError(f"Could not understand the date {text!r}.")


def _is_open(day: date) -> bool:
    return day.weekday() < 6  # closed Sundays


def _slot_is_free(slot: Slot) -> bool:
    """Deterministic pseudo-availability, stable across runs for the same slot."""
    digest = hashlib.sha256(slot.key.encode()).digest()[0]
    return digest % 100 >= 35  # ~35% of the grid is pre-booked


def _grid(specialty: str, day: date, window: str | None) -> list[Slot]:
    clinician = SPECIALTIES[specialty]
    windows = [WINDOWS[window]] if window else list(WINDOWS.values())
    slots: list[Slot] = []
    for start_t, end_t in windows:
        cursor = datetime.combine(day, start_t, tzinfo=CLINIC_TZ)
        end = datetime.combine(day, end_t, tzinfo=CLINIC_TZ)
        while cursor < end:
            slots.append(Slot(specialty=specialty, clinician=clinician, start=cursor))
            cursor += timedelta(minutes=SLOT_MINUTES)
    return slots


def normalise_specialty(specialty: str | None) -> str:
    if not specialty:
        return DEFAULT_SPECIALTY
    key = specialty.strip().lower()
    if key in SPECIALTIES:
        return key
    for known in SPECIALTIES:
        if known in key or key in known:
            return known
    return DEFAULT_SPECIALTY


def normalise_window(window: str | None) -> str | None:
    if not window:
        return None
    key = window.strip().lower()
    for known in WINDOWS:
        if known in key:
            return known
    if "am" in key or "early" in key:
        return "morning"
    if "pm" in key or "late" in key:
        return "evening"
    return None


def available_slots(
    specialty: str | None = None,
    on: date | None = None,
    window: str | None = None,
    *,
    limit: int = 3,
) -> list[Slot]:
    """Free slots, searched forward from `on` until `limit` are found or the horizon is reached."""
    spec = normalise_specialty(specialty)
    win = normalise_window(window)
    current = now()
    day = on or current.date()
    horizon = current.date() + timedelta(days=BOOKING_HORIZON_DAYS)

    found: list[Slot] = []
    with _lock:
        taken = set(_bookings)
        while day <= horizon and len(found) < limit:
            if _is_open(day):
                for slot in _grid(spec, day, win):
                    if slot.start - current < MIN_LEAD_TIME:
                        continue
                    if slot.key in taken or not _slot_is_free(slot):
                        continue
                    found.append(slot)
                    if len(found) >= limit:
                        break
            day += timedelta(days=1)
    return found


def book(
    patient_id: str,
    patient_name: str,
    specialty: str | None,
    preferred_date: str,
    preferred_window: str | None = None,
) -> Booking:
    """Book the first free slot matching the request.

    Raises SlotUnavailable (carrying alternatives) rather than silently shifting the appointment —
    the agent must offer the alternative and get the patient to agree.
    """
    spec = normalise_specialty(specialty)
    win = normalise_window(preferred_window)
    day = resolve_date(preferred_date)
    current = now()

    if day < current.date():
        raise SchedulingError(f"{day.isoformat()} is in the past.")
    if day > current.date() + timedelta(days=BOOKING_HORIZON_DAYS):
        raise SchedulingError(
            f"Appointments can only be booked up to {BOOKING_HORIZON_DAYS} days ahead."
        )

    if not _is_open(day):
        raise SlotUnavailable(
            f"The clinic is closed on {day.strftime('%A %d %B')}.",
            available_slots(spec, on=day, window=win, limit=3),
        )

    exact = available_slots(spec, on=day, window=win, limit=1)
    if not exact or exact[0].start.date() != day:
        alternatives = available_slots(spec, on=day, window=None, limit=3)
        window_text = f" in the {win}" if win else ""
        raise SlotUnavailable(
            f"No {spec} slot is free on {day.strftime('%A %d %B')}{window_text}.",
            alternatives,
        )

    slot = exact[0]
    with _lock:
        if slot.key in _bookings:
            raise SlotUnavailable(
                "That slot was just taken.",
                available_slots(spec, on=day, window=win, limit=3),
            )
        booking = Booking(
            confirmation_id=f"APT-{uuid.uuid4().hex[:8].upper()}",
            patient_id=patient_id,
            patient_name=patient_name,
            slot=slot,
            booked_at=current,
        )
        _bookings[slot.key] = booking
    return booking


def bookings_for(patient_id: str) -> list[Booking]:
    with _lock:
        return [b for b in _bookings.values() if b.patient_id == patient_id]
