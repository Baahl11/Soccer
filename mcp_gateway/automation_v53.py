from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v52 as v52
from mcp_gateway import red_cards_intelligence

MODEL_VERSION = v52.MODEL_VERSION
AUTOMATION_VERSION = "3.29.0"


async def run_tick() -> dict[str, Any]:
    payload = await v52.run_tick()
    metrics = red_cards_intelligence.attach(payload)
    payload["red_cards_intelligence"] = {
        "schema_version": red_cards_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "EMPIRICAL_BAYES_ANY_RED_CARD_MATCH_v0.1",
        "scope": "ANY_RED_CARD_IN_MATCH_YES_NO_ONLY",
        "policy": "RED CARDS SEPARATE FROM YELLOWS; STRONG LOW-FREQUENCY SHRINKAGE; EXPLICIT RED-CARD YES/NO MARKETS ONLY; ZERO DECISION WEIGHT",
    }
    payload["v329_provider_requests_added"] = 0
    payload["v329_model_weights_changed"] = False
    payload["v329_canonical_bet_logic_changed"] = False
    payload["v329_red_cards_checkpoint"] = "LIVE_RESEARCH_ANY_RED_CARD_MODEL_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
