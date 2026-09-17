from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v63 as v63
from mcp_gateway import player_assists_intelligence

MODEL_VERSION = v63.MODEL_VERSION
AUTOMATION_VERSION = "3.40.0"


async def run_tick() -> dict[str, Any]:
    payload = await v63.run_tick()
    metrics = player_assists_intelligence.attach(payload)
    payload["player_assists_intelligence"] = {
        "schema_version": player_assists_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "1+/2+ ASSIST FAIR PROBABILITIES FROM SHRUNK ASSIST RATE + TEAM SCORING ENVIRONMENT; NO XA OR MARKET PRICE/EV/PROMOTION",
    }
    payload["v340_provider_requests_added"] = 0
    payload["v340_model_weights_changed"] = False
    payload["v340_canonical_bet_logic_changed"] = False
    payload["v340_assists_checkpoint"] = "ASSIST_PROBABILITIES_ADDED_RESEARCH_ONLY; XA_AND_VERIFIED_PRICE_REMAIN_GAPS"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
