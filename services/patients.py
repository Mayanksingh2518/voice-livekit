"""Patient record loading, PHI masking, and call-variable construction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "patients.json"


class PatientNotFound(Exception):
    pass


class PatientNotContactable(Exception):
    """Raised when consent or do-not-call flags forbid an outbound call."""


@dataclass(frozen=True)
class Biomarker:
    name: str
    value: float
    unit: str
    reference_range: str
    status: str
    collected_on: str

    @property
    def spoken(self) -> str:
        """How the agent should say this value out loud."""
        if self.unit == "%":
            return f"{self.name} of {self.value} percent"
        return f"{self.name} of {self.value} {self.unit}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "reference_range": self.reference_range,
            "status": self.status,
            "collected_on": self.collected_on,
        }


@dataclass(frozen=True)
class Patient:
    id: str
    name: str
    phone_number: str
    preferred_language: str
    consent_to_call: bool
    do_not_call: bool
    care_program: str
    ordering_clinician: str
    biomarkers: tuple[Biomarker, ...]

    @property
    def first_name(self) -> str:
        return self.name.split()[0]

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "phone_number": mask_phone(self.phone_number) if redact else self.phone_number,
            "care_program": self.care_program,
            "ordering_clinician": self.ordering_clinician,
            "biomarkers": [b.to_dict() for b in self.biomarkers],
        }


def mask_phone(phone: str) -> str:
    """+919876543210 -> +91******3210. Keeps enough to correlate, not enough to dial."""
    if len(phone) <= 6:
        return "*" * len(phone)
    return f"{phone[:3]}{'*' * (len(phone) - 7)}{phone[-4:]}"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, dict[str, Any]]:
    records = json.loads(DATA_FILE.read_text())
    return {record["id"]: record for record in records}


def _build(record: dict[str, Any]) -> Patient:
    return Patient(
        id=record["id"],
        name=record["name"],
        phone_number=record["phone_number"],
        preferred_language=record.get("preferred_language", "en"),
        consent_to_call=record.get("consent_to_call", False),
        do_not_call=record.get("do_not_call", False),
        care_program=record.get("care_program", "General"),
        ordering_clinician=record.get("ordering_clinician", "your doctor"),
        biomarkers=tuple(Biomarker(**b) for b in record.get("biomarkers", [])),
    )


def list_patients() -> list[Patient]:
    return [_build(r) for r in _load_raw().values()]


def get_patient(patient_id: str) -> Patient:
    raw = _load_raw().get(patient_id)
    if raw is None:
        raise PatientNotFound(f"No patient with id {patient_id!r}")
    return _build(raw)


def assert_contactable(patient: Patient) -> None:
    """Consent gate. Runs before dialling, never inside the agent."""
    if patient.do_not_call:
        raise PatientNotContactable(f"{patient.id} is on the do-not-call list")
    if not patient.consent_to_call:
        raise PatientNotContactable(f"{patient.id} has not consented to outbound calls")


def biomarker_briefing(patient: Patient) -> str:
    """The exact wording the agent is allowed to use. Values are never re-derived by the LLM."""
    if not patient.biomarkers:
        return "No biomarker results are available for this patient."

    lines = []
    for b in patient.biomarkers:
        lines.append(
            f"- {b.name}: {b.value} {b.unit} "
            f"(normal range: {b.reference_range}; this result is {b.status}; "
            f"sample collected {b.collected_on})"
        )
    return "\n".join(lines)


def staleness_note(patient: Patient, *, today: date | None = None) -> str | None:
    """Results older than 30 days get called out, so the agent does not present them as current."""
    if not patient.biomarkers:
        return None
    today = today or date.today()
    try:
        oldest = min(date.fromisoformat(b.collected_on) for b in patient.biomarkers)
    except ValueError:
        return None
    age_days = (today - oldest).days
    if age_days > 30:
        return f"These results are {age_days} days old — mention the collection date when you share them."
    return None


def build_call_variables(patient: Patient, *, redact: bool = False) -> dict[str, Any]:
    """The dynamic variables bound into the prompt for this call.

    Logged to Opik verbatim so a trace shows exactly what the agent was told.
    """
    return {
        "patient_id": patient.id,
        "patient_name": patient.name,
        "patient_first_name": patient.first_name,
        "phone_number": mask_phone(patient.phone_number) if redact else patient.phone_number,
        "care_program": patient.care_program,
        "ordering_clinician": patient.ordering_clinician,
        "biomarker_briefing": biomarker_briefing(patient),
        "biomarker_count": len(patient.biomarkers),
        "staleness_note": staleness_note(patient),
    }
