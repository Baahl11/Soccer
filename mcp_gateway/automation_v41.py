from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v40 as v40
from mcp_gateway import btts_intelligence

MODEL_VERSION = v40.MODEL_VERSION
AUTOMATION_VERSION = "3.17.0"


async def run_tick() -> dict[str, Any]:
    payload = await v40.run_tick()
    metrics = btts_intelligence.attach(payload)
    payload["btts_intelligence"] = {
        "schema_version": btts_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "model": "CANONICAL_SCORE_MATRIX_BTTS_v0.1",
        "policy": (
            "USE EXISTING CANONICAL SCORE-MATRIX BTTS PROBABILITY; COMPARE ONLY TO OBSERVED YES/NO BTTS MARKETS; "
            "NO-VIG ONLY WITH BOTH SIDES; ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION BEFORE OOS/CLV GATES"
        ),
    }
    payload["v317_provider_requests_added"] = 0
    payload["v317_model_weights_changed"] = False
    payload["v317_canonical_bet_logic_changed"] = False
    payload["v317_btts_checkpoint"] = "LIVE_RESEARCH_CALIBRATION_BRIDGE_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
