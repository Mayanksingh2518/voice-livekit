"""The outbound care-coordinator agent and its tools.

Every tool writes to CallState before returning. That record — not the transcript — is what the
post-call analysis treats as fact.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import Agent, RunContext, function_tool, get_job_context
from livekit.agents.llm import ToolError

from agent.prompts import VOICEMAIL_MESSAGE, build_instructions
from services import scheduler
from services.call_state import CallState

logger = logging.getLogger("patient-agent")


class PatientOutreachAgent(Agent):
    def __init__(self, state: CallState, variables: dict[str, Any]) -> None:
        today = scheduler.now().date().isoformat()
        super().__init__(instructions=build_instructions(variables, today=today))
        self._state = state
        self._variables = variables

    # ------------------------------------------------------------------ helpers

    async def _hangup(self) -> None:
        """End the call by deleting the room. No-op outside a real job (console mode)."""
        try:
            ctx = get_job_context()
        except RuntimeError:
            logger.info("no job context — skipping hangup")
            return
        try:
            await ctx.delete_room()
        except Exception:
            logger.exception("failed to delete room during hangup")

    # ------------------------------------------------------------------ tools

    @function_tool()
    async def verify_identity(self, context: RunContext, is_correct_person: bool) -> str:
        """Record whether you have confirmed you are speaking to the intended patient.

        Call this as soon as you know, and before sharing any health information.

        Args:
            is_correct_person: True if the person on the call confirmed they are the patient.
        """
        self._state.identity_confirmed = bool(is_correct_person)
        self._state.record_tool(
            "verify_identity", {"is_correct_person": is_correct_person}, result="recorded"
        )
        if is_correct_person:
            return "Identity confirmed. You may now share the results."
        return (
            "Identity not confirmed. Do not share any health information. "
            "Ask for a good time to call back, then use end_call."
        )

    @function_tool()
    async def check_availability(
        self,
        context: RunContext,
        preferred_date: str,
        preferred_window: str | None = None,
        specialty: str | None = None,
    ) -> dict[str, Any]:
        """Find open consultation slots. Use this before booking when the patient is unsure.

        Args:
            preferred_date: The day the patient wants, as YYYY-MM-DD.
            preferred_window: "morning", "afternoon" or "evening" if the patient stated one.
            specialty: The clinic to book with. Defaults to endocrinology.
        """
        args = {
            "preferred_date": preferred_date,
            "preferred_window": preferred_window,
            "specialty": specialty,
        }
        try:
            day = scheduler.resolve_date(preferred_date)
            slots = scheduler.available_slots(specialty, on=day, window=preferred_window, limit=3)
        except scheduler.SchedulingError as exc:
            self._state.record_tool("check_availability", args, error=str(exc))
            raise ToolError(str(exc)) from exc

        result = {
            "slots": [s.to_dict() for s in slots],
            "count": len(slots),
        }
        if not slots:
            result["note"] = "Nothing free that day. Offer the patient a different day."
        self._state.record_tool("check_availability", args, result=result)
        return result

    @function_tool()
    async def book_appointment(
        self,
        context: RunContext,
        preferred_date: str,
        preferred_window: str | None = None,
        specialty: str | None = None,
    ) -> dict[str, Any]:
        """Book the consultation. Only call this once the patient has agreed to a specific time.

        Args:
            preferred_date: The agreed day, as YYYY-MM-DD.
            preferred_window: "morning", "afternoon" or "evening".
            specialty: The clinic to book with. Defaults to endocrinology.
        """
        patient = self._state.patient
        args = {
            "preferred_date": preferred_date,
            "preferred_window": preferred_window,
            "specialty": specialty,
        }
        try:
            booking = scheduler.book(
                patient_id=patient.id,
                patient_name=patient.name,
                specialty=specialty,
                preferred_date=preferred_date,
                preferred_window=preferred_window,
            )
        except scheduler.SlotUnavailable as exc:
            alternatives = [s.spoken for s in exc.alternatives]
            self._state.record_tool("book_appointment", args, error=str(exc))
            if alternatives:
                raise ToolError(
                    f"{exc} Offer these instead and ask the patient to pick one: "
                    + "; ".join(alternatives)
                ) from exc
            raise ToolError(f"{exc} Nothing else is free nearby — offer a callback.") from exc
        except scheduler.SchedulingError as exc:
            self._state.record_tool("book_appointment", args, error=str(exc))
            raise ToolError(str(exc)) from exc

        self._state.bookings.append(booking)
        result = booking.to_dict()
        self._state.record_tool("book_appointment", args, result=result)
        logger.info("booked %s for %s", booking.confirmation_id, patient.id)
        return {
            **result,
            "note": "Booked. Read the day, date and time back to the patient to confirm.",
        }

    @function_tool()
    async def detected_answering_machine(self, context: RunContext) -> None:
        """Call this the moment you realise you have reached a voicemail or answering machine.

        Use it after hearing a recorded greeting or a beep.
        """
        self._state.voicemail_detected = True
        self._state.end_reason = "voicemail"
        self._state.record_tool("detected_answering_machine", {}, result="voicemail")
        logger.info("voicemail detected for %s", self._state.patient.id)

        message = VOICEMAIL_MESSAGE.format(
            clinic="Sehat Clinic", first_name=self._state.patient.first_name
        )
        handle = context.session.say(message, allow_interruptions=False)
        await handle.wait_for_playout()
        await self._hangup()

    @function_tool()
    async def transfer_to_human(self, context: RunContext, reason: str) -> str:
        """Hand the call to a human care manager.

        Use for medical urgency, distress, or anything you are not permitted to answer.

        Args:
            reason: Short description of why the transfer is needed.
        """
        self._state.transfer_requested = True
        self._state.transfer_reason = reason
        self._state.record_tool("transfer_to_human", {"reason": reason})
        logger.info("transfer requested for %s: %s", self._state.patient.id, reason)

        transfer_to = self._variables.get("transfer_to")
        if not transfer_to:
            return (
                "No transfer line is configured. Tell the patient a care manager will call them "
                "back shortly, and if this is urgent they should contact emergency services now."
            )

        handle = context.session.say(
            "Let me connect you to one of our care managers. Please stay on the line."
        )
        await handle.wait_for_playout()
        try:
            ctx = get_job_context()
            participant = await ctx.wait_for_participant()
            await ctx.transfer_sip_participant(participant, f"tel:{transfer_to}")
        except Exception as exc:
            logger.exception("transfer failed")
            self._state.errors.append(f"transfer_failed: {exc}")
            return (
                "The transfer did not go through. Apologise, tell the patient a care manager "
                "will call back shortly, and end the call."
            )
        return "Transferred."

    @function_tool()
    async def end_call(self, context: RunContext, reason: str) -> None:
        """End the call after you have said goodbye.

        Args:
            reason: One of "completed", "declined", "do_not_call", "identity_unverified",
                "callback_requested", or a short phrase describing why.
        """
        normalised = (reason or "completed").strip().lower().replace(" ", "_")
        self._state.end_reason = normalised
        if normalised in {"do_not_call", "dnc", "do_not_contact"}:
            self._state.do_not_call_requested = True
        self._state.record_tool("end_call", {"reason": reason})
        logger.info("ending call for %s: %s", self._state.patient.id, normalised)

        await context.wait_for_playout()
        await self._hangup()
