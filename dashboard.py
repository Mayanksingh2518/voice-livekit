"""Operator console for the outbound voice agent.

    uv run streamlit run dashboard.py

Place a call, then read the transcript, the post-call analysis, the tool calls and the evaluation
scores in one place. Call data is read back out of Opik rather than stored separately — the traces
are the system of record, so what you see here is exactly what was logged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

load_dotenv()

from observability import opik_reader as reader  # noqa: E402
from services import live_calls, patients  # noqa: E402

st.set_page_config(page_title="Care Outreach Console", page_icon="🩺", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2.2rem; max-width: 1300px;}
  .pill {display:inline-block; padding:2px 10px; border-radius:3px; font-size:.75rem;
         font-weight:600; letter-spacing:.02em; font-family:ui-monospace,monospace;}
  .pill-good {background:#DFEFF0; color:#0B6E78;}
  .pill-warn {background:#F7E8E1; color:#A0472A;}
  .pill-flat {background:#EDF2F4; color:#4A5A64;}
  .biomarker {font-family:ui-monospace,monospace; font-size:.85rem;}
  [data-testid="stMetricValue"] {font-size:1.5rem;}
</style>
""", unsafe_allow_html=True)


GOOD_OUTCOMES = {"appointment_booked"}
SOFT_OUTCOMES = {"callback_requested", "escalated_to_human"}


def pill(text: str, kind: str = "flat") -> str:
    return f'<span class="pill pill-{kind}">{text}</span>'


def outcome_pill(outcome: str) -> str:
    kind = "good" if outcome in GOOD_OUTCOMES else "flat" if outcome in SOFT_OUTCOMES else "warn"
    return pill(outcome.replace("_", " "), kind)


def run_script(args: list[str], label: str) -> tuple[bool, str]:
    with st.spinner(label):
        proc = subprocess.run(
            [sys.executable, *args], cwd=ROOT, capture_output=True, text=True, timeout=600
        )
    return proc.returncode == 0, (proc.stdout or "") + (proc.stderr or "")


@st.cache_data(ttl=20, show_spinner=False)
def load_calls():
    return reader.list_calls(limit=30)


@st.cache_data(ttl=20, show_spinner=False)
def load_call(trace_id: str):
    return reader.get_call(trace_id)


# ------------------------------------------------------------------ sidebar

with st.sidebar:
    st.subheader("Care Outreach Console")
    st.caption("Outbound healthcare voice agent · LiveKit + Opik")

    import os

    st.markdown("**Environment**")
    rows = [
        ("LiveKit", bool(os.getenv("LIVEKIT_API_KEY"))),
        ("LLM", bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))),
        ("Deepgram", bool(os.getenv("DEEPGRAM_API_KEY"))),
        ("Opik", reader.configured()),
        ("Telephony", bool(os.getenv("TWILIO_AUTH_TOKEN") and os.getenv("LIVEKIT_SIP_HOST"))
         or bool(os.getenv("SIP_OUTBOUND_TRUNK_ID"))),
    ]
    for name, ok in rows:
        st.markdown(
            f"{pill('ready' if ok else 'not set', 'good' if ok else 'warn')} &nbsp; {name}",
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption(
        f"LLM `{os.getenv('LLM_MODEL', '—')}`  \n"
        f"Analysis `{os.getenv('ANALYSIS_MODEL', '—')}`  \n"
        f"Judge `{os.getenv('OPIK_JUDGE_MODEL', '—')}`"
    )
    if st.button("Refresh call data", width="stretch"):
        st.cache_data.clear()
        st.rerun()


st.title("Care Outreach Console")

if not reader.configured():
    st.error("Opik is not configured. Set `OPIK_API_KEY` in `.env` — call history reads from Opik.")
    st.stop()

# ------------------------------------------------------------------ live call control

try:
    active_rooms = live_calls.list_rooms()
    active_twilio = live_calls.list_twilio_calls()
except Exception:
    active_rooms, active_twilio = [], []

if active_rooms or active_twilio:
    with st.container(border=True):
        head, action = st.columns([4, 1], gap="medium")
        with head:
            st.markdown("#### Call in progress")
            for call in active_twilio:
                st.markdown(
                    f"{pill(call.status, 'warn')} &nbsp; phone `{call.to}` &nbsp; "
                    f"<span style='opacity:.6'>{call.sid}</span>",
                    unsafe_allow_html=True,
                )
            for room in active_rooms:
                st.markdown(
                    f"{pill('connected', 'good')} &nbsp; room `{room.name}` &nbsp; "
                    f"<span style='opacity:.6'>{room.participants} participant(s)</span>",
                    unsafe_allow_html=True,
                )
        with action:
            st.write("")
            if st.button("End call", type="primary", width="stretch"):
                phones, rooms = live_calls.end_everything()
                st.cache_data.clear()
                st.success(f"Hung up {phones} phone leg(s), closed {rooms} room(s).")
                st.rerun()
            st.caption("Hangs up the phone and closes the room. The call is still analysed.")

tab_place, tab_calls, tab_diag = st.tabs(["Place a call", "Call history", "Telephony"])


# ------------------------------------------------------------------ place a call

with tab_place:
    roster = patients.list_patients()
    labels = {p.id: f"{p.id} · {p.name}" for p in roster}
    choice = st.radio(
        "Patient", list(labels), format_func=lambda k: labels[k], horizontal=True
    )
    patient = patients.get_patient(choice)

    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown(f"#### {patient.name}")
        st.caption(
            f"{patient.care_program} · ordering clinician {patient.ordering_clinician} · "
            f"{patients.mask_phone(patient.phone_number)}"
        )
        for b in patient.biomarkers:
            flag = "warn" if b.status in {"high", "low"} else "flat"
            st.markdown(
                f'<div class="biomarker">{b.name} &nbsp; <b>{b.value} {b.unit}</b> &nbsp; '
                f'{pill(b.status, flag)} &nbsp; <span style="opacity:.6">normal '
                f'{b.reference_range} · collected {b.collected_on}</span></div>',
                unsafe_allow_html=True,
            )

        blocked = patient.do_not_call or not patient.consent_to_call
        if blocked:
            st.warning(
                "This patient is on the do-not-call list. The dispatcher refuses the call before "
                "a worker is ever involved."
                if patient.do_not_call
                else "This patient has not consented to outbound calls."
            )

    with right:
        st.markdown("#### Run")
        st.caption(
            "A simulated call replays a scripted conversation through the real post-call "
            "pipeline — analysis, reconciliation, Opik traces and scores. No telephony, no cost."
        )
        if st.button("Run simulated call", type="primary", width="stretch"):
            ok, out = run_script(["scripts/replay_call.py"], "Running call and analysis…")
            st.cache_data.clear()
            if ok:
                st.success("Call logged to Opik. Open **Call history**.")
            else:
                st.error("Run failed.")
            with st.expander("Output", expanded=not ok):
                st.code(out[-3000:] or "(no output)")

        st.divider()
        direct_sip = bool(os.getenv("SIP_OUTBOUND_TRUNK_ID"))
        bridged = bool(os.getenv("TWILIO_AUTH_TOKEN") and os.getenv("LIVEKIT_SIP_HOST"))
        can_call = direct_sip or bridged

        st.caption(
            "A real call rings the patient's phone. Twilio originates it and bridges it into "
            "LiveKit over SIP. Needs `python main.py dev` running in another terminal."
        )
        phone = st.text_input("Dial number", value=patient.phone_number, disabled=not can_call)
        if st.button("Place real call", disabled=not can_call or blocked, width="stretch"):
            script = (
                ["scripts/dispatch_call.py", "--patient", patient.id, "--phone", phone]
                if direct_sip
                else ["scripts/call_via_twilio.py", "--patient", patient.id, "--to", phone]
            )
            ok, out = run_script(script, "Dialling…")
            if ok:
                st.success("Call placed — the phone should ring.")
            else:
                st.error("Call failed.")
            with st.expander("Output", expanded=not ok):
                st.code(out[-3000:] or "(no output)")
        if not can_call:
            st.caption(
                "Set `TWILIO_AUTH_TOKEN` and run `scripts/setup_inbound_sip.py` to enable "
                "real calls."
            )


# ------------------------------------------------------------------ call history

with tab_calls:
    try:
        project_id, calls = load_calls()
    except Exception as exc:
        st.error(f"Could not read from Opik: {exc}")
        st.stop()

    if not calls:
        st.info("No calls logged yet. Run one from **Place a call**.")
        st.stop()

    st.markdown(f"[Open this project in Opik]({reader.threads_url(project_id)})")

    mode_label = {"replay": "simulated", "console": "console",
                  "outbound_sip": "live call", "twilio_bridged": "live call"}
    options = {
        c.trace_id: (
            f"[{mode_label.get(c.mode, c.mode)}]  {c.started_at:%d %b %H:%M} · "
            f"{c.patient_id or '—'} · {c.outcome.replace('_', ' ')}"
        )
        for c in calls
    }
    selected = st.selectbox("Call", list(options), format_func=lambda k: options[k])
    detail = load_call(selected)
    s = detail.summary
    analysis = detail.analysis

    st.markdown(
        f"### {detail.variables.get('patient_name', s.patient_id or 'Call')} &nbsp; "
        f"{outcome_pill(s.outcome)}",
        unsafe_allow_html=True,
    )
    st.caption(f"{s.started_at:%d %B %Y, %H:%M} · thread `{s.thread_id}`")

    if s.mode == "replay":
        st.info(
            "**Simulated call** — the transcript below is the scripted fixture in "
            "`data/sample_call.json`, not a real conversation. No number was dialled. Everything "
            "downstream of it is real: the post-call analysis, the reconciliation and the "
            "evaluation scores were all computed by the live models and logged to Opik.",
            icon="🧪",
        )
    elif s.mode == "console":
        st.info(
            "**Console call** — a real spoken conversation over your microphone, with the full "
            "STT → LLM → TTS pipeline. No telephony involved.",
            icon="🎙️",
        )
    elif s.mode in {"outbound_sip", "twilio_bridged"}:
        st.success(
            "**Live outbound call** — a real phone rang and a real person answered.", icon="📞"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Appointment booked", "Yes" if analysis.get("appointment_booked") else "No")
    m2.metric("Identity verified", "Yes" if analysis.get("identity_verified") else "No")
    m3.metric("Sentiment", (analysis.get("patient_sentiment") or "—").title())
    m4.metric("Turns", detail.metadata.get("turn_count", "—"))

    if s.scores:
        st.markdown("#### Evaluation")
        cols = st.columns(len(s.scores))
        for col, (name, value) in zip(cols, s.scores.items()):
            col.metric(name.replace("_", " ").title(), f"{value:g}")
        for name, why in detail.score_reasons.items():
            with st.expander(f"Why: {name.replace('_', ' ')}"):
                st.write(why)

    body, side = st.columns([3, 2], gap="large")

    with body:
        st.markdown("#### Transcript")
        if detail.transcript:
            for line in detail.transcript:
                with st.chat_message("user" if line["role"] == "patient" else "assistant"):
                    st.write(line["text"])
        else:
            st.caption("No conversation — the call did not connect.")

    with side:
        st.markdown("#### Post-call analysis")
        st.write(analysis.get("call_summary") or "—")
        st.markdown(f"**Next action** · {analysis.get('next_action') or '—'}")

        violations = analysis.get("safety_violations") or []
        if violations:
            st.error("Safety violations: " + "; ".join(violations))
        else:
            st.success("No safety violations found")

        corrections = analysis.get("_corrections") or []
        if corrections:
            st.warning(
                "Reconciled against the tool record:\n\n"
                + "\n".join(f"- {c}" for c in corrections)
            )
        st.caption(f"Analysis source: `{analysis.get('_source', '—')}`")

        if detail.tool_spans:
            st.markdown("#### Tool calls")
            for tool in detail.tool_spans:
                with st.expander(tool["name"]):
                    st.json({"arguments": tool["input"], "result": tool["output"]})

        booking = analysis.get("appointment")
        if booking:
            st.markdown("#### Booking")
            st.json(booking)

        st.link_button(
            "Open trace in Opik", reader.trace_url(project_id, s.trace_id), width="stretch"
        )


# ------------------------------------------------------------------ telephony diagnostics

with tab_diag:
    st.markdown("#### Twilio call attempts")
    st.caption(
        "A call has two halves: Twilio rings the phone, then `<Dial><Sip>` bridges it into "
        "LiveKit. Only when the second half happens does the agent get a room, speak, and "
        "produce a transcript."
    )

    try:
        attempts = live_calls.recent_attempts(limit=10)
    except Exception as exc:
        attempts = []
        st.error(f"Could not read Twilio call history: {exc}")

    if not attempts:
        st.info("No Twilio calls yet.")
    else:
        for a in attempts:
            ok = a.reached_livekit
            with st.container(border=True):
                left, right = st.columns([3, 2], gap="medium")
                with left:
                    st.markdown(
                        f"{pill('bridged' if ok else 'not bridged', 'good' if ok else 'warn')}"
                        f" &nbsp; `{a.to}` &nbsp; <span style='opacity:.6'>{a.created}</span>",
                        unsafe_allow_html=True,
                    )
                    st.caption(f"{a.diagnosis} · {a.duration}s · {a.sid}")
                with right:
                    st.markdown(
                        f"phone leg &nbsp; {pill(a.status, 'flat')}<br>"
                        f"SIP leg &nbsp;&nbsp; "
                        f"{pill(a.sip_leg or 'never created', 'good' if ok else 'warn')}",
                        unsafe_allow_html=True,
                    )

    st.divider()
    st.markdown("#### What we know about this Twilio account")
    st.markdown(
        """
Twilio's trial plan has blocked every SIP route tested so far:

| Route | Result |
|---|---|
| Elastic SIP Trunking (LiveKit dials out) | blocked — console redirects to upgrade |
| Inline `Twiml` parameter | blocked — *"trial accounts have limited parameter access"* |
| `<Dial><Sip>` via a TwiML Bin | phone answers, no SIP leg |
| `<Dial><Sip>` via a Twimlet (no bin, no templating) | phone answers, `<Say>` plays, still no SIP leg |
| Alerts / Events / geo-permission APIs | blocked — cannot read the underlying error |

The last row is the decisive one. Adding a `<Say>` before the `<Dial>` lengthened the call from
5s to 11s, which proves Twilio fetched and executed the TwiML. The `<Dial><Sip>` that followed
produced no child leg and no error: the trial plan drops outbound SIP silently.

The agent's outbound-SIP code is complete and unchanged by any of this — `main.py` still uses
`create_sip_participant` when `SIP_OUTBOUND_TRUNK_ID` is set. What is missing is a telephony
account that permits SIP at all.
        """
    )
