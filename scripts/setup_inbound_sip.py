"""Set up LiveKit to *receive* the SIP leg of a Twilio-originated call.

    python scripts/setup_inbound_sip.py          # create trunk + dispatch rule
    python scripts/setup_inbound_sip.py --list   # show what exists

Why inbound rather than outbound: Twilio's trial plan does not include Elastic SIP Trunking, so
LiveKit cannot dial out through it. Twilio's REST voice API *is* available on trial, so the call is
originated from Twilio instead and bridged into LiveKit over SIP:

    scripts/call_via_twilio.py  ──REST──>  Twilio  ──dials──>  patient's phone
                                                                    │ answers
                              LiveKit  <──SIP INVITE──────────────┘
                                 └── dispatch rule opens room "call-<patient>-<id>"
                                     and dispatches the agent into it

The dispatch rule is `callee` type: the user part of the SIP URI Twilio dials becomes the room
name, so the patient id can be carried in the URI without creating a rule per call.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

load_dotenv()

TRUNK_NAME = "healthcare-inbound"
RULE_NAME = "healthcare-callee-rule"
ROOM_PREFIX = "call-"
AGENT_NAME = os.getenv("AGENT_NAME", "healthcare-outbound-caller")


def sip_host() -> str | None:
    """LiveKit Cloud serves SIP at <project>.sip.livekit.cloud, matching the ws URL."""
    explicit = os.getenv("LIVEKIT_SIP_HOST")
    if explicit:
        return explicit
    url = os.getenv("LIVEKIT_URL", "")
    host = url.split("://")[-1].strip("/")
    if host.endswith(".livekit.cloud"):
        return host.replace(".livekit.cloud", ".sip.livekit.cloud")
    return None


async def show() -> int:
    async with api.LiveKitAPI() as lk:
        trunks = await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
        rules = await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())

    if not trunks.items:
        print("no inbound trunks")
    for t in trunks.items:
        print(f"trunk  {t.sip_trunk_id}  {t.name!r}  user={t.auth_username or '(none)'}")
    if not rules.items:
        print("no dispatch rules")
    for r in rules.items:
        kind = r.rule.WhichOneof("rule")
        print(f"rule   {r.sip_dispatch_rule_id}  {r.name!r}  {kind}  trunks={list(r.trunk_ids)}")
    return 0


async def create() -> int:
    host = sip_host()
    if not host:
        print("error: could not derive the SIP host from LIVEKIT_URL.", file=sys.stderr)
        print("Set LIVEKIT_SIP_HOST explicitly (LiveKit Cloud → Settings → SIP).", file=sys.stderr)
        return 1

    username = os.getenv("LIVEKIT_SIP_USERNAME") or "twilio"
    password = os.getenv("LIVEKIT_SIP_PASSWORD") or secrets.token_urlsafe(18)

    async with api.LiveKitAPI() as lk:
        existing = await lk.sip.list_inbound_trunk(api.ListSIPInboundTrunkRequest())
        trunk = next((t for t in existing.items if t.name == TRUNK_NAME), None)

        if trunk is None:
            trunk = await lk.sip.create_inbound_trunk(
                api.CreateSIPInboundTrunkRequest(
                    trunk=api.SIPInboundTrunkInfo(
                        name=TRUNK_NAME,
                        auth_username=username,
                        auth_password=password,
                        krisp_enabled=True,
                    )
                )
            )
            print(f"created inbound trunk {trunk.sip_trunk_id}")
        else:
            print(f"reusing inbound trunk {trunk.sip_trunk_id}")
            password = os.getenv("LIVEKIT_SIP_PASSWORD") or "(unchanged — see your .env)"

        rules = await lk.sip.list_dispatch_rule(api.ListSIPDispatchRuleRequest())
        rule = next((r for r in rules.items if r.name == RULE_NAME), None)
        if rule is None:
            rule = await lk.sip.create_dispatch_rule(
                api.CreateSIPDispatchRuleRequest(
                    name=RULE_NAME,
                    trunk_ids=[trunk.sip_trunk_id],
                    rule=api.SIPDispatchRule(
                        dispatch_rule_callee=api.SIPDispatchRuleCallee(
                            room_prefix=ROOM_PREFIX,
                            randomize=False,
                        )
                    ),
                    room_config=api.RoomConfiguration(
                        agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
                    ),
                )
            )
            print(f"created dispatch rule {rule.sip_dispatch_rule_id}")
        else:
            print(f"reusing dispatch rule {rule.sip_dispatch_rule_id}")

    print()
    print("Add these to your .env:")
    print(f"LIVEKIT_SIP_HOST={host}")
    print(f"LIVEKIT_SIP_USERNAME={username}")
    print(f"LIVEKIT_SIP_PASSWORD={password}")
    print()
    print(f"Twilio will dial: sip:<patient>-<id>@{host}")
    print(f"which opens room: {ROOM_PREFIX}<patient>-<id>  and dispatches agent {AGENT_NAME!r}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure LiveKit inbound SIP for Twilio")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    return asyncio.run(show() if args.list else create())


if __name__ == "__main__":
    raise SystemExit(main())
