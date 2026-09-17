from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v38 as v38
from mcp_gateway import team_totals_intelligence

MODEL_VERSION = v38.MODEL_VERSION
AUTOMATION_VERSION = "3.15.0"


async def run_tick() -> dict[str, Any]:
    payload = await v38.run_tick()
    metrics = team_totals_intelligence.attach(payload)
    payload["team_totals_intelligence"] = {
        "schema_version": team_totals_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "supported_lines": list(team_totals_intelligence.SUPPORTED_LINES),
        "model": "CANONICAL_TEAM_LAMBDA_POISSON_DERIVATION_v0.1",
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "policy": (
            "DERIVE TEAM-TOTAL PROBABILITIES FROM EXISTING CANONICAL HOME/AWAY LAMBDAS; "
            "PRICE ONLY EXACT OBSERVED TEAM-TOTAL MARKETS; HALF-GOAL LINES ONLY V1; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION"
        ),
    }
    payload["v315_provider_requests_added"] = 0
    payload["v315_model_weights_changed"] = False
    payload["v315_canonical_bet_logic_changed"] = False
    payload["v315_team_totals_checkpoint"] = "LIVE_RESEARCH_BRIDGE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
