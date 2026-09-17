from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v48 as v48
from mcp_gateway import team_corners_intelligence

MODEL_VERSION = v48.MODEL_VERSION
AUTOMATION_VERSION = "3.25.0"


async def run_tick() -> dict[str, Any]:
    payload = await v48.run_tick()
    metrics = team_corners_intelligence.attach(payload)
    payload["team_corners_intelligence"] = {
        "schema_version": team_corners_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "TEAM_CORNERS_POISSON_FROM_LIVE_CORNER_LAMBDAS_v0.1",
        "policy": "TEAM-SPECIFIC CORNER LAMBDAS FROM SPORT MODEL; EXACT OBSERVED TEAM CORNER LINES; SETTLEMENT-AWARE; ZERO DECISION WEIGHT",
    }
    payload["v325_provider_requests_added"] = 0
    payload["v325_model_weights_changed"] = False
    payload["v325_canonical_bet_logic_changed"] = False
    payload["v325_team_corners_checkpoint"] = "LIVE_RESEARCH_TEAM_CORNERS_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
