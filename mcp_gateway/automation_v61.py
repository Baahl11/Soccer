from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v60 as v60
from mcp_gateway import player_shots_intelligence

MODEL_VERSION = v60.MODEL_VERSION
AUTOMATION_VERSION = "3.37.0"


async def run_tick() -> dict[str, Any]:
    payload = await v60.run_tick()
    metrics = player_shots_intelligence.attach(payload)
    payload["player_shots_intelligence"] = {
        "schema_version": player_shots_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "SHOTS HALF-LINE FAIR PROBABILITIES ONLY; OPPONENT SHOT SUPPRESSION SHRUNK; FORMATION NUMERIC EFFECT DISABLED; NO MARKET PRICE/EV/PROMOTION",
    }
    payload["v337_provider_requests_added"] = 0
    payload["v337_model_weights_changed"] = False
    payload["v337_canonical_bet_logic_changed"] = False
    payload["v337_shots_props_checkpoint"] = "PLAYER_SHOTS_DISTRIBUTION_AND_HALF_LINE_PROBABILITIES_ADDED; NO MARKET_PRICE_OR_PRODUCTION_WEIGHT"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
