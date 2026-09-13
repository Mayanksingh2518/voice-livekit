"""Create the LiveKit outbound SIP trunk from environment variables.

    python scripts/setup_sip_trunk.py          # create (or show) the trunk
    python scripts/setup_sip_trunk.py --list   # list existing trunks

Reads:
    SIP_TRUNK_ADDRESS    e.g. my-trunk.pstn.twilio.com   (no sip: prefix)
    SIP_TRUNK_NUMBER     the caller ID, E.164            e.g. +15105550123
    SIP_TRUNK_USERNAME / SIP_TRUNK_PASSWORD              trunk credentials

Print the resulting trunk id into SIP_OUTBOUND_TRUNK_ID in your .env.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402
from livekit import api  # noqa: E402

load_dotenv()


async def list_trunks() -> int:
    async with api.LiveKitAPI() as lk:
        response = await lk.sip.list_outbound_trunk(api.ListSIPOutboundTrunkRequest())
    if not response.items:
        print("no outbound trunks configured")
        return 0
    for trunk in response.items:
        print(f"{trunk.sip_trunk_id}  {trunk.name!r}  address={trunk.address}  "
              f"numbers={list(trunk.numbers)}")
    return 0


async def create() -> int:
    address = os.getenv("SIP_TRUNK_ADDRESS")
    number = os.getenv("SIP_TRUNK_NUMBER")
    if not address or not number:
        print("error: set SIP_TRUNK_ADDRESS and SIP_TRUNK_NUMBER in .env", file=sys.stderr)
        return 1

    trunk = api.SIPOutboundTrunkInfo(
        name=os.getenv("SIP_TRUNK_NAME", "healthcare-outbound"),
        address=address,
        numbers=[number],
        auth_username=os.getenv("SIP_TRUNK_USERNAME", ""),
        auth_password=os.getenv("SIP_TRUNK_PASSWORD", ""),
    )

    async with api.LiveKitAPI() as lk:
        created = await lk.sip.create_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(trunk=trunk)
        )

    print(f"created trunk {created.sip_trunk_id}")
    print()
    print("add this to your .env:")
    print(f"SIP_OUTBOUND_TRUNK_ID={created.sip_trunk_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Manage the LiveKit outbound SIP trunk")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    return asyncio.run(list_trunks() if args.list else create())


if __name__ == "__main__":
    raise SystemExit(main())
