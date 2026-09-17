from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v62 as v62
from mcp_gateway import player_goalscorer_intelligence

MODEL_VERSION = v62.MODEL_VERSION
AUTOMATION_VERSION = "3.39.0"


async def run_tick() -> dict[str, Any]:
    payload = await v62.run_tick()
    metrics = player_goalscorer_intelligence.attach(payload)
    payload["player_goalscorer_intelligence"] = {
        "schema_version": player_goalscorer_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "ANYTIME GOAL FAIR PROBABILITY ONLY; OPPONENT GOALS-ALLOWED SHRUNK; PENALTY/GK/FORMATION NUMERIC EFFECTS DISABLED; NO MARKET PRICE/EV/PROMOTION",
    }
    payload["v339_provider_requests_added"] = 0
    payload["v339_model_weights_changed"] = False
    payload["v339_canonical_bet_logic_changed"] = False
    payload["v339_goalscorer_checkpoint"] = "ANYTIME_GOAL_PROBABILITY_ADDED_RESEARCH_ONLY; VERIFIED_PRICE_AND_OOS_PROMOTION_PENDING"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
