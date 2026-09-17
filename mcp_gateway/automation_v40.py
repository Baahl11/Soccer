from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v39 as v39
from mcp_gateway import correct_score_intelligence

MODEL_VERSION = v39.MODEL_VERSION
AUTOMATION_VERSION = "3.16.0"


async def run_tick() -> dict[str, Any]:
    payload = await v39.run_tick()
    metrics = correct_score_intelligence.attach(payload)
    payload["correct_score_intelligence"] = {
        "schema_version": correct_score_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "model": "CANONICAL_HOME_AWAY_LAMBDA_INDEPENDENT_POISSON_EXACT_SCORE_v0.1",
        "policy": (
            "DERIVE EXACT-SCORE PROBABILITIES FROM EXISTING CANONICAL HOME/AWAY LAMBDAS; "
            "COMPARE ONLY TO OBSERVED CORRECT_SCORE/EXACT_SCORE PRICES; ZERO DECISION WEIGHT; "
            "NO BET_LEAN_GALAXY PROMOTION BEFORE OOS/CALIBRATION/CLV GATES"
        ),
    }
    payload["v316_provider_requests_added"] = 0
    payload["v316_model_weights_changed"] = False
    payload["v316_canonical_bet_logic_changed"] = False
    payload["v316_correct_score_checkpoint"] = "LIVE_RESEARCH_BRIDGE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
