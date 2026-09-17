from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v61 as v61
from mcp_gateway import player_sot_intelligence

MODEL_VERSION = v61.MODEL_VERSION
AUTOMATION_VERSION = "3.38.0"


async def run_tick() -> dict[str, Any]:
    payload = await v61.run_tick()
    metrics = player_sot_intelligence.attach(payload)
    payload["player_sot_intelligence"] = {
        "schema_version": player_sot_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "SOT HALF-LINE FAIR PROBABILITIES ONLY; OPPONENT SOT SUPPRESSION SHRUNK; FORMATION NUMERIC EFFECT DISABLED; NO MARKET PRICE/EV/PROMOTION",
    }
    payload["v338_provider_requests_added"] = 0
    payload["v338_model_weights_changed"] = False
    payload["v338_canonical_bet_logic_changed"] = False
    payload["v338_sot_props_checkpoint"] = "PLAYER_SOT_DISTRIBUTION_AND_HALF_LINE_PROBABILITIES_ADDED; NO MARKET_PRICE_OR_PRODUCTION_WEIGHT"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
