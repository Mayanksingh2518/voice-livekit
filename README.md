# Biomarker Outreach Agent

An outbound healthcare voice agent built on LiveKit. It phones a patient, tells them their HbA1c
and fasting glucose results, and books a follow-up consultation through a tool call. When the call
ends it runs a post-call analysis and ships the whole thing — metadata, transcript, audio, tool
results, analysis, evaluation scores — into [Opik](https://www.comet.com/docs/opik/).

The Opik integration is a single drop-in module, `observability/opik_tracer.py`. It costs the core
application exactly one line.

**Live console:** <https://voice-livekit-m9hu.onrender.com> · **Writeup:** [SUBMISSION.md](SUBMISSION.md)
(what was built, the constraints hit and why, and the production roadmap)

---

## Quick start — no phone number needed

```bash
uv venv --python 3.11        # pin 3.11; silero/onnxruntime wheels lag newer Pythons
uv sync
cp .env.example .env         # fill in the keys below
uv run python main.py download-files   # VAD + turn-detector weights
uv run python scripts/check_setup.py   # calls every service; tells you exactly what is missing
uv run python main.py console
```

Three accounts, all free, no card required:

| Key | Where | Covers |
|---|---|---|
| `LIVEKIT_*` | cloud.livekit.io → Settings → Keys | Rooms, telephony |
| `GEMINI_API_KEY` | aistudio.google.com/apikey | Agent LLM, post-call analysis, Opik judges |
| `DEEPGRAM_API_KEY` | console.deepgram.com | Both speech-to-text and text-to-speech |
| `OPIK_API_KEY` | comet.com/opik | Observability (optional — calls run without it) |

Set `LLM_PROVIDER=openai` with an `OPENAI_API_KEY` to switch providers; both run through the same
OpenAI client, Gemini via its OpenAI-compatible endpoint.

### Running on Gemini's free tier

Free-tier Gemini allows **20 requests per day per model**, and returns 429/503 often enough that
the analysis retries with backoff (`ANALYSIS_MAX_ATTEMPTS`). Two consequences worth knowing before
you start debugging a "broken" run:

- **Each role uses a different model** so they don't share one 20-request budget — the agent
  (`LLM_MODEL`), the analyser (`ANALYSIS_MODEL`) and the judges (`OPIK_JUDGE_MODEL`) are set to
  different `gemini-3.x` models in `.env`. Any of them can be swapped for another.
- **Opik's `GEval` costs about four requests per score**, not one — it samples the judge several
  times to build a continuous score. `Moderation` costs one. Budget roughly six requests per
  completed call, and expect one or two full demo runs per day.

If that becomes the bottleneck, `LLM_PROVIDER=openai` with $5 of credit removes the limit; no other
change is needed.

`console` mode talks to you over your microphone. The whole loop works there — identity check,
biomarker delivery, booking tool, post-call analysis, Opik traces — with no telephony at all.

**Even faster, without talking:** replay a scripted call through the full post-call pipeline.

```bash
uv run python scripts/replay_call.py
```

This drives the Opik module with a fake session object and no LiveKit runtime, which is also the
proof that the module is genuinely standalone.

### Web console

```bash
uv run python web/server.py          # then open http://localhost:8080
```

One page: pick a patient, start a call in the browser, watch the transcript stream, and read the
post-call analysis when it lands. `/overview` explains the pipeline and the trace topology.

The browser joins as a real LiveKit participant, so the agent, the tools, the recording and the
Opik traces are all the production path — only the last mile is WebRTC instead of a phone line.
Call data is read **back out of Opik** rather than cached here: the traces are the system of
record. Patients on the do-not-call list are visibly blocked from dialling.

**Use headphones.** On speakers the agent hears its own voice through the microphone, it gets
transcribed as the patient, and it starts replying to itself.

---

## Architecture

```
main.py                                observability/opik_tracer.py
  entrypoint(ctx)                         OpikCallTracer
   ├─ read ctx.job.metadata                 .attach(session, ctx, finalise=…)   ← the one line
   ├─ build AgentSession                      ├─ on(conversation_item_added) → turns
   ├─ tracer.attach(…)  ──────────────────────┤  on(function_tools_executed) → tool spans
   ├─ create_sip_participant()                ├─ on(metrics_collected)       → usage + cost
   └─ session.start(record={"audio": True})   ├─ on(error)                    → error_info
                                              └─ ctx.add_shutdown_callback()
  agent/patient_agent.py                          ├─ await finalise(transcript)
    verify_identity · check_availability           ├─ turn traces (shared thread_id)
    book_appointment · transfer_to_human           ├─ call trace (+audio, +scores)
    detected_answering_machine · end_call          └─ flush(timeout)

  analysis/post_call.py     structured-output LLM + deterministic reconciliation
  services/scheduler.py     simulated booking — deterministic slots, conflicts, alternatives
  services/call_state.py    the factual record the analysis is reconciled against
```

| Path | Role |
|---|---|
| `main.py` | Worker entrypoint, outbound dial, SIP lifecycle |
| `agent/patient_agent.py` | Agent subclass and its six function tools |
| `agent/prompts.py` | System prompt with the clinical guardrails |
| `services/patients.py` | Patient records, PHI masking, call variables |
| `services/scheduler.py` | Simulated appointment booking |
| `services/call_state.py` | Deterministic record of what actually happened |
| `analysis/post_call.py` | Post-call analysis and reconciliation |
| `observability/opik_tracer.py` | **The Opik integration** (write side) |
| `observability/opik_reader.py` | Reads calls back out of Opik for the console |
| `web/server.py`, `web/index.html` | Web console — start a call, live transcript, analysis |
| `web/overview.html` | How the pipeline and the trace topology work |
| `scripts/` | Dispatch a call, create the SIP trunk, create Opik rules, replay a call |

---

## The Opik integration

```python
from observability.opik_tracer import OpikCallTracer

tracer = OpikCallTracer.from_env(call_id=ctx.room.name, variables=variables, metadata={...})
tracer.attach(session, ctx, finalise=build_summary)
```

That is the entire integration surface. The module imports nothing from this application — it takes
a session-shaped object with an `.on()` method, an optional context with `.add_shutdown_callback()`,
and one async callback that returns a plain dict. Drop it into any LiveKit project unchanged.

### What lands in Opik

| Opik object | Contents |
|---|---|
| `thread_id = call-{room}` | Groups the conversation. In Opik a thread *is* a set of traces sharing this string |
| Turn traces (N) | `input={"user": …}` / `output={"agent": …}`, each with an `llm` span carrying model, provider and token usage |
| Call trace (1) | Variables and transcript in `input`; outcome and analysis in `output`; SIP status, durations, latency metrics, token totals in `metadata`; the recording as an **attachment**; feedback scores |
| Tool spans | `type="tool"` — arguments in, result out, including the simulated booking |
| Analysis span | `type="llm"` — the post-call analyser traces itself, so its prompt and cost are auditable |

### Why one trace per turn

In Opik a thread is formed from sibling traces, not from child spans of one trace. Logging the call
as a single trace with turns as spans looks tidier and quietly disqualifies the conversation from
every thread-level evaluation rule. So turns are separate traces sharing a `thread_id`, and a
separate call-level trace carries the metadata, audio and analysis — deliberately kept *outside* the
thread so the Threads view stays a clean transcript. The call trace carries `thread_id` in its
metadata for cross-navigation.

### Why the manual SDK and not Opik's OpenTelemetry integration

Opik ships a first-class LiveKit integration, but it is OTel-based. It auto-captures LLM, STT and
TTS spans, and then stops short of what this assignment needs: it cannot set a conversation
`thread_id`, cannot carry an audio attachment, and has nowhere to put post-call analysis. Mapping an
OTel attribute onto an Opik thread is still an open upstream feature request. The manual SDK gives
full control over all three. This is a considered choice, not an unawareness of the integration.

### Fail-open

Every handler and the entire shutdown pipeline is wrapped. If Opik is unconfigured, unreachable, or
changes its schema, the module logs a warning and the call is untouched. Collection also runs when
Opik is disabled, so turning telemetry off changes only where data goes — never whether the
post-call analysis runs or what transcript it receives.

Verify it:

```bash
OPIK_ENABLED=false uv run python scripts/replay_call.py
```

The analysis still completes; nothing is sent.

---

## Post-call analysis

`analysis/post_call.py` sends the transcript, the permitted biomarker values and the tool-call
record to an LLM with a Pydantic structured-output schema: outcome, whether an appointment was
booked, sentiment, which biomarkers were communicated, patient concerns, escalation need, safety
violations, summary and next action.

**The reconciliation rule.** `appointment_booked` is read from the *tool-call record*, never from
the model's reading of the transcript. When the two disagree the deterministic value wins and the
disagreement is recorded in `_corrections` on the trace. A model that hallucinates a successful
booking must not be able to mark its own homework. The same applies to identity verification,
voicemail, and do-not-call — see `CallState.deterministic_outcome()`.

If the transcript is empty (nobody answered) no LLM is called at all. If the LLM fails, the analysis
degrades to the known facts and is tagged `_source: "fallback"` rather than being lost.

---

## Online evaluations

```bash
uv run python scripts/setup_opik_rules.py --dry-run   # validate payloads, change nothing
uv run python scripts/setup_opik_rules.py             # create them
```

Rules are created in code through Opik's typed SDK client, not clicked into the UI, so they are
reviewable and reproducible. Three evaluations run at three levels:

| Level | Name | What it judges |
|---|---|---|
| Trace rule (`llm_as_judge`) | **Medical Safety** | Every clinical number stated appears in the permitted results; identity was confirmed before disclosure; no diagnosis or medication advice. Variables map onto `input.transcript` and `input.variables.biomarker_briefing` |
| Thread rule (`trace_thread_llm_as_judge`) | **Booking Effectiveness** | Goal completion, empathy, and whether the patient's questions were answered — over the whole conversation. Thread rules take no variable map; Opik injects the conversation as `{{context}}` |
| SDK, in-process | **Moderation + G-Eval + booking** | Computed at call end and written as feedback scores, so a demo shows results immediately |

**Expected timing.** Trace-level scores and the in-process scores appear as soon as the call is
logged. Thread-level scores only appear once Opik marks the thread inactive, after a cooldown. That
delay is Opik's design, not a broken rule — the in-process scores exist partly to cover the gap.

---

## Placing a real call

LiveKit does not sell phone numbers; it connects a SIP trunk you bring.

**Development — free.** Point the outbound trunk at a softphone (Zoiper, Linphone) instead of the
phone network. `create_sip_participant` runs the identical code path; only the last mile differs.

**Demo — about $20.** A Twilio Elastic SIP Trunk with a US number (~$1/month, roughly $0.03–0.10/min
to India). Trial accounts play a preamble before every call and can only dial verified numbers, so
upgrade before recording the demo.

Either way:

```bash
# fill SIP_TRUNK_* in .env, then
uv run python scripts/setup_sip_trunk.py       # prints SIP_OUTBOUND_TRUNK_ID
# paste that into .env, then
uv run python main.py dev                      # terminal 1: the worker
uv run python scripts/dispatch_call.py --patient P001 --phone +91XXXXXXXXXX   # terminal 2
```

`dispatch_call.py --list` shows the patient fixtures. The consent and do-not-call gate runs in the
dispatcher, before a worker is ever involved — `P003` is flagged do-not-call specifically so you can
watch the call be refused.

---

## Edge cases handled

**Telephony** — no answer (SIP 408/480), busy or declined (486/603) and trunk failure (5xx) are each
mapped to a distinct outcome and still produce an Opik trace; a call nobody answered is a result,
not an absence of data. Patient hangs up mid-call → partial transcript still analysed. Silence →
`user_away_timeout`. `max_call_duration` caps a stuck call. Provider errors mid-call are captured on
the trace rather than killing the session.

**Clinical and privacy** — identity is confirmed through a tool before any health data is shared;
if the wrong person answers nothing is disclosed. Voicemail gets a generic callback message only,
never biomarkers — reading PHI to an unverified recipient is a disclosure. Requests for medical
advice or medication changes are refused and deferred to the clinician. Urgent symptoms abandon the
script and escalate. "Do not call me again" is acknowledged, flagged, and ends the call.

**Booking** — an unavailable slot returns alternatives rather than failing; closed days say so
explicitly; past dates and dates beyond the 21-day horizon are rejected; models that pass "tuesday"
or "tomorrow" instead of `YYYY-MM-DD` are handled; a booking tool call that *errors* still means
`appointment_booked = False`.

**Observability** — Opik unreachable or unconfigured leaves the call untouched; the recorder
finalises the audio file as the session closes, so the pipeline waits for it with a timeout and
falls back to a URL reference; OGG is transcoded to WAV via ffmpeg when available, for reliable
playback in the Opik UI; `REDACT_PHI=true` masks phone numbers before anything leaves the process;
a double shutdown cannot emit duplicate traces; `flush()` is bounded so a worker cannot hang on exit.

## Deployment

Two deployables, because they are different shapes. The agent is a long-lived worker holding a
socket to LiveKit; the console is an ordinary HTTP service.

| Piece | Where | How |
|---|---|---|
| Agent worker | LiveKit Cloud, `ap-south` | `lk agent create --secrets-file .env.agent`, then `lk agent deploy` |
| Web console | Render, free instance | build `pip install .`, start `python web/server.py` |

The repository carries a `Dockerfile` (model weights fetched at build time so the first call is not
delayed) and a `.python-version` pin, since Render otherwise defaults to a Python with no wheels for
parts of this stack.

`pip install .` installs only what the console needs. The agent's speech stack lives behind the
`agent` extra (`pip install ".[agent]"`, `uv sync --extra agent`), which keeps a console deploy
inside a 512 MB instance.

Free hosting cannot run the agent: Render's background workers are paid-only, Fly.io's free tier is
gone for new accounts, Koyeb's free tier excludes worker services, and platforms that sleep on
absent inbound traffic kill a worker whose connection is outbound. LiveKit Cloud's own agent
hosting is purpose-built for this and has a free allowance.

## Testing

```bash
uv run --group dev pytest
```

34 tests over the two pieces of logic worth testing directly: the scheduler (conflicts,
alternatives, closed days, relative dates, the booking horizon) and the analysis reconciliation
(the model claiming a booking that did not happen, missing one that did, and every deterministic
override). Both are pure functions, so they need no LiveKit runtime.

## Scope boundaries

Deliberately not built, and worth naming rather than hiding:

- **Retry campaigns.** A `no_answer` outcome is recorded with a next action, but scheduling the
  retry is a campaign-layer concern, not the agent's.
- **Persistent storage.** Bookings live in process memory; a real deployment would talk to the
  scheduling system rather than `services/scheduler.py`.
- **AMD.** LiveKit 1.8 ships built-in answering-machine detection. This uses the documented
  LLM-tool approach instead, which is simpler and provider-independent; `AgentSession.amd` is the
  production upgrade path.

## Stack

`livekit-agents` 1.8.1 · `opik` 2.2.59 · Python 3.11 · Gemini LLM (switchable to OpenAI) ·
Deepgram nova-3 STT · Deepgram Aura TTS · Silero VAD · LiveKit English turn detector · aiohttp console
