from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v50 as v50
from mcp_gateway import team_cards_intelligence

MODEL_VERSION = v50.MODEL_VERSION
AUTOMATION_VERSION = "3.27.0"


async def run_tick() -> dict[str, Any]:
    payload = await v50.run_tick()
    metrics = team_cards_intelligence.attach(payload)
    payload["team_cards_intelligence"] = {
        "schema_version": team_cards_intelligence.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "TEAM_YELLOW_CARD_POISSON_FROM_DISCIPLINE_LAMBDAS_v0.1",
        "scope": "TEAM_YELLOW_CARDS_ONLY; REDS SEPARATE",
        "policy": "EXACT EXPLICIT TEAM-YELLOW MARKETS ONLY; SETTLEMENT-AWARE; ZERO DECISION WEIGHT",
    }
    payload["v327_provider_requests_added"] = 0
    payload["v327_model_weights_changed"] = False
    payload["v327_canonical_bet_logic_changed"] = False
    payload["v327_team_cards_checkpoint"] = "LIVE_RESEARCH_TEAM_YELLOW_CARDS_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
