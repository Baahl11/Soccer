from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v64 as v64
from mcp_gateway import gk_saves_intelligence

MODEL_VERSION = v64.MODEL_VERSION
AUTOMATION_VERSION = "3.41.0"


async def run_tick() -> dict[str, Any]:
    payload = await v64.run_tick()
    metrics = gk_saves_intelligence.attach(payload)
    payload["gk_saves_intelligence"] = {
        "schema_version": gk_saves_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "player_prop_bet_eligibility_changed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "PROJECTED OPPONENT SOT X SHRUNK SAVE RESULT PROXY; NO PSxG CLAIM; NO MARKET PRICE/EV/PROMOTION",
    }
    payload["v341_provider_requests_added"] = 0
    payload["v341_model_weights_changed"] = False
    payload["v341_canonical_bet_logic_changed"] = False
    payload["v341_gk_saves_checkpoint"] = "GK_SAVE_LINE_PROBABILITIES_ADDED_RESEARCH_ONLY; PSXG_AND_VERIFIED_PRICE_REMAIN_GAPS"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
