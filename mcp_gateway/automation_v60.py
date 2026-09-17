from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v59 as v59
from mcp_gateway import player_trend_intelligence

MODEL_VERSION = v59.MODEL_VERSION
AUTOMATION_VERSION = "3.36.0"


async def run_tick() -> dict[str, Any]:
    payload = await v59.run_tick()
    metrics = player_trend_intelligence.attach(payload)
    payload["player_trend_intelligence"] = {
        "schema_version": player_trend_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "market_line_probabilities_created": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "decision_weight": 0.0,
        "policy": "ROLE/MINUTES-ADJUSTED SHRUNK PLAYER RATE RESEARCH ONLY; NO SHOTS/SOT/GOAL/ASSIST LINE PROBABILITIES; PRODUCTION UNCHANGED",
    }
    payload["v336_provider_requests_added"] = 0
    payload["v336_model_weights_changed"] = False
    payload["v336_canonical_bet_logic_changed"] = False
    payload["v336_player_trends_checkpoint"] = "PROBABILISTIC_ROLE_MINUTES_AND_SHRUNK_RATE_CONTEXT_ADDED; PROP_LINE_MODELS_REMAIN_SEPARATE"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
