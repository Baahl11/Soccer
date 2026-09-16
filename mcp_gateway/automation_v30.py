from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v29 as v29
from mcp_gateway.galaxy_builder import build as build_galaxy_builder

MODEL_VERSION = v29.MODEL_VERSION
AUTOMATION_VERSION = "3.6.0"


async def run_tick() -> dict[str, Any]:
    payload = await v29.run_tick()
    builder = build_galaxy_builder(payload)
    payload["galaxy_builder"] = builder
    payload["galaxy_builder_candidate_count_this_tick"] = int(builder.get("candidate_count") or 0)
    payload["galaxy_builder_policy"] = builder.get("policy")
    payload["galaxy_builder_actionable_scope"] = "RESEARCH_ONLY; ZERO_BET_PROMOTION_IN_V0.1"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
