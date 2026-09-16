from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v29 as v29
from mcp_gateway.galaxy_builder_v2 import build as build_galaxy_builder_v2

MODEL_VERSION = v29.MODEL_VERSION
AUTOMATION_VERSION = "3.7.0"


async def run_tick() -> dict[str, Any]:
    payload = await v29.run_tick()
    builder = build_galaxy_builder_v2(payload)
    payload["galaxy_builder"] = builder
    payload["galaxy_builder_candidate_count_this_tick"] = int(builder.get("candidate_count") or 0)
    payload["galaxy_builder_same_game_count_this_tick"] = int(builder.get("same_game_candidate_count") or 0)
    payload["galaxy_builder_multi_match_count_this_tick"] = int(builder.get("multi_match_candidate_count") or 0)
    payload["galaxy_builder_policy"] = builder.get("policy")
    payload["galaxy_builder_actionable_scope"] = (
        "V0.2 SHADOW/RESEARCH; SGP AND MULTI-MATCH PARLAYS; TARGET +110 OR BETTER; "
        "NO BET PROMOTION WITHOUT FINAL VERIFIED SPORTSBOOK PARLAY QUOTE AND ALL MODEL GATES"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
