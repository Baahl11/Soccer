from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v47 as v47
from mcp_gateway import corners_intelligence

MODEL_VERSION = v47.MODEL_VERSION
AUTOMATION_VERSION = "3.24.0"


async def run_tick() -> dict[str, Any]:
    payload = await v47.run_tick()
    metrics = corners_intelligence.attach(payload)
    payload["corners_intelligence"] = {
        "schema_version": corners_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "LEAGUE_TEAM_CORNERS_POISSON_LIVE_v0.1",
        "formation_challenger_live_weight": 0.0,
        "policy": "FINALIZED-HISTORY TEAM/LEAGUE CORNERS REGISTRY; EXACT OBSERVED FT CORNERS TOTALS; SETTLEMENT-AWARE; FORMATION CHALLENGER SHADOW ONLY",
    }
    payload["v324_provider_requests_added"] = 0
    payload["v324_state_registry_reads_max"] = 1
    payload["v324_model_weights_changed"] = False
    payload["v324_canonical_bet_logic_changed"] = False
    payload["v324_corners_checkpoint"] = "LIVE_RESEARCH_BASELINE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
