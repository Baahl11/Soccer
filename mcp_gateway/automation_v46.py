from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v45 as v45
from mcp_gateway import one_h_goals_intelligence

MODEL_VERSION = v45.MODEL_VERSION
AUTOMATION_VERSION = "3.22.0"


async def run_tick() -> dict[str, Any]:
    payload = await v45.run_tick()
    metrics = one_h_goals_intelligence.attach(payload)
    payload["one_h_goals_intelligence"] = {
        "schema_version": one_h_goals_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "HIERARCHICAL_1H_POISSON_STRENGTH_LIVE_v0.1",
        "policy": "PERIOD-SPECIFIC FINALIZED-HISTORY REGISTRY; NEVER REUSE FT PROBABILITY; EXACT OBSERVED 1H LINES; SETTLEMENT-AWARE; ZERO DECISION WEIGHT",
    }
    payload["v322_provider_requests_added"] = 0
    payload["v322_state_registry_reads_max"] = 1
    payload["v322_model_weights_changed"] = False
    payload["v322_canonical_bet_logic_changed"] = False
    payload["v322_1h_checkpoint"] = "LIVE_PERIOD_SPECIFIC_RESEARCH_MODEL_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
