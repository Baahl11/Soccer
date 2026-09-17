from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v43 as v43
from mcp_gateway import dnb_intelligence

MODEL_VERSION = v43.MODEL_VERSION
AUTOMATION_VERSION = "3.20.0"


async def run_tick() -> dict[str, Any]:
    payload = await v43.run_tick()
    metrics = dnb_intelligence.attach(payload)
    payload["dnb_intelligence"] = {
        "schema_version": dnb_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "RESEARCH_ONLY_PARENT_1X2_NOT_APPROVED",
        "model": "CANONICAL_1X2_CONDITIONAL_DNB_v0.1",
        "settlement_policy": (
            "DRAW IS PUSH; FAIR DNB PRICE USES P(WIN|NO_DRAW); PUSH-AWARE EV=P(WIN)*PRICE+P(DRAW)-1"
        ),
        "promotion_policy": (
            "PARENT 1X2 MUST BE PRODUCTION-APPROVED FIRST; DNB THEN REQUIRES OWN NON-DRAW OOS/CALIBRATION/CLV/SHRINKAGE REVIEW"
        ),
    }
    payload["v320_provider_requests_added"] = 0
    payload["v320_model_weights_changed"] = False
    payload["v320_canonical_bet_logic_changed"] = False
    payload["v320_dnb_checkpoint"] = "LIVE_RESEARCH_PUSH_AWARE_DNB_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
