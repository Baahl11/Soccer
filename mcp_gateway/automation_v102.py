from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v101 as v101
from mcp_gateway import confidence_engine_v4

MODEL_VERSION = v101.MODEL_VERSION
AUTOMATION_VERSION = "4.11.0-v4.015"


def _annotate_v4_015(payload: dict[str, Any]) -> None:
    payload["v4_015_confidence_engine_v1"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_FRAMEWORK_IMPLEMENTED",
        "model_version": confidence_engine_v4.MODEL_VERSION,
        "inputs_required": [
            "oos_rows",
            "calibration_ece",
            "disagreement_max_range",
            "feature_missing_rate",
        ],
        "minimum_oos_rows": confidence_engine_v4.MIN_OOS_ROWS,
        "confidence_thresholds": {
            "high": confidence_engine_v4.HIGH_CONFIDENCE_SCORE,
            "moderate": confidence_engine_v4.MODERATE_CONFIDENCE_SCORE,
        },
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Confidence is research-only and combines OOS sample strength, calibration quality, "
            "cross-model agreement and feature completeness. Odds, price and execution readiness "
            "are intentionally excluded."
        ),
    }
    payload["v4_015_provider_requests_added"] = 0
    payload["v4_015_model_weights_changed"] = False
    payload["v4_015_canonical_bet_logic_changed"] = False
    payload["v4_015_checkpoint"] = (
        "CONFIDENCE ENGINE v1 IMPLEMENTED: research confidence score/bands from OOS sample, "
        "calibration, disagreement and feature completeness only. No market fields, provider "
        "calls, production weight, thresholds, tiers, stakes or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v101.run_tick()
    _annotate_v4_015(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
