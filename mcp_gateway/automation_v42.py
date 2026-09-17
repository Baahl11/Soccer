from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v41 as v41
from mcp_gateway import one_x_two_intelligence

MODEL_VERSION = v41.MODEL_VERSION
AUTOMATION_VERSION = "3.18.0"


async def run_tick() -> dict[str, Any]:
    payload = await v41.run_tick()
    metrics = one_x_two_intelligence.attach(payload)
    payload["one_x_two_intelligence"] = {
        "schema_version": one_x_two_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_MODEL_SELECTION_PENDING",
        "model_selection_policy": (
            "CANONICAL + RELATIVE_STRENGTH LIVE SHADOW; DIXON_COLES AND CLASS_PRIOR OFFLINE OOS; "
            "NO WINNER UNTIL SAME-FIXTURE HEAD-TO-HEAD AND CALIBRATION/CLV GATES PASS"
        ),
        "market_policy": (
            "OBSERVED THREE-WAY 1X2 ONLY; NO-VIG ONLY WHEN HOME/DRAW/AWAY ARE ALL PRESENT; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION"
        ),
    }
    payload["v318_provider_requests_added"] = 0
    payload["v318_model_weights_changed"] = False
    payload["v318_canonical_bet_logic_changed"] = False
    payload["v318_1x2_checkpoint"] = "LIVE_RESEARCH_MODEL_SELECTION_BRIDGE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
