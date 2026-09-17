from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v46 as v46
from mcp_gateway import two_h_goals_intelligence

MODEL_VERSION = v46.MODEL_VERSION
AUTOMATION_VERSION = "3.23.0"


async def run_tick() -> dict[str, Any]:
    payload = await v46.run_tick()
    metrics = two_h_goals_intelligence.attach(payload)
    payload["two_h_goals_intelligence"] = {
        "schema_version": two_h_goals_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "HIERARCHICAL_2H_POISSON_STRENGTH_LIVE_v0.1",
        "model_timing": "PREGAME_ONLY_NOT_HALFTIME_CONDITIONED",
        "policy": "PERIOD-SPECIFIC FINALIZED-HISTORY REGISTRY; NEVER REUSE FT/1H PROBABILITY; EXACT OBSERVED 2H LINES; SETTLEMENT-AWARE; ZERO DECISION WEIGHT",
    }
    payload["v323_provider_requests_added"] = 0
    payload["v323_state_registry_reads_shared"] = True
    payload["v323_model_weights_changed"] = False
    payload["v323_canonical_bet_logic_changed"] = False
    payload["v323_2h_pregame_checkpoint"] = "LIVE_PERIOD_SPECIFIC_PREGAME_RESEARCH_MODEL_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
