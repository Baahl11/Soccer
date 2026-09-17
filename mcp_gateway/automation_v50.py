from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v49 as v49
from mcp_gateway import cards_intelligence_live

MODEL_VERSION = v49.MODEL_VERSION
AUTOMATION_VERSION = "3.26.0"


async def run_tick() -> dict[str, Any]:
    payload = await v49.run_tick()
    metrics = cards_intelligence_live.attach(payload)
    payload["cards_intelligence_live"] = {
        "schema_version": cards_intelligence_live.SCHEMA_VERSION,
        **metrics,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "TEAM_DISCIPLINE_PLUS_OPTIONAL_REFEREE_YELLOW_CARDS_LIVE_v0.1",
        "scope": "TOTAL_YELLOW_CARDS_ONLY; RED_CARDS_SEPARATE",
        "market_rule_policy": "GENERIC CARDS/BOOKINGS ARE BLOCKED UNTIL SPORTSBOOK SCORING RULE IS EXPLICITLY MAPPED",
    }
    payload["v326_provider_requests_added"] = 0
    payload["v326_model_weights_changed"] = False
    payload["v326_canonical_bet_logic_changed"] = False
    payload["v326_cards_checkpoint"] = "LIVE_RESEARCH_YELLOW_CARDS_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
