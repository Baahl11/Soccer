from __future__ import annotations

import os
from typing import Any

from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v13 as v13

MODEL_VERSION = "SOCCER EDGE ENGINE v1.3"
AUTOMATION_VERSION = "2.3.0"

# Deep-dive count is a scheduler work-cap, NOT an API quota. Increasing it lets
# cached/Galaxy-first fixtures continue through the lifecycle without raising
# the hard provider-call ceiling enforced by v4/v6 (currently 20 calls/tick in
# NORMAL mode, lower in protected daily-budget modes).
ELASTIC_DEEP_DIVE_CAP = int(os.getenv("SOCCER_EDGE_ELASTIC_DEEP_DIVE_CAP", "12"))
BASE_DEEP_DIVE_CAP = v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK


async def run_tick() -> dict[str, Any]:
    previous_cap = v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK
    v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK = max(previous_cap, ELASTIC_DEEP_DIVE_CAP)
    try:
        payload = await v13.run_tick()
    finally:
        v5.MAX_DEEP_DIVE_FIXTURES_PER_TICK = previous_cap

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["deep_dive_capacity_policy"] = (
        "ELASTIC_WORK_CAP_WITH_UNCHANGED_PROVIDER_API_HARD_CAP"
    )
    payload["base_deep_dive_cap"] = BASE_DEEP_DIVE_CAP
    payload["elastic_deep_dive_cap"] = ELASTIC_DEEP_DIVE_CAP
    payload["provider_api_cap_unchanged"] = True

    due = int(payload.get("urgent_late_shortlist_due") or 0)
    processed = int(payload.get("urgent_late_shortlist_processed") or 0)
    payload["urgent_window_completion_rate_this_tick"] = (
        round(processed / due, 4) if due > 0 else 1.0
    )
    payload["urgent_window_policy"] = (
        "EXISTING_T10_T20_CLOSE_SHORTLIST_FIRST; T40_DISCOVERY_AFTER; "
        "ELASTIC_DEEP_DIVE_CAP_MAY_USE_GALAXY_OR_CACHE_BUT_CANNOT_BYPASS_PROVIDER_CALL_BUDGET"
    )
    return payload
