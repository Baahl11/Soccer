from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v96 as v96
from mcp_gateway import bivariate_poisson_v4

MODEL_VERSION = v96.MODEL_VERSION
AUTOMATION_VERSION = "4.6.0-v4.010"


def _annotate_v4_010(payload: dict[str, Any]) -> None:
    payload["v4_010_bivariate_poisson_baseline"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_BASELINE_IMPLEMENTED",
        "model_version": bivariate_poisson_v4.MODEL_VERSION,
        "method": "BIVARIATE_POISSON_SHARED_COMPONENT_FRACTION_ON_POINT_IN_TIME_GOAL_RATE_FEATURES",
        "shared_fraction_grid_min": min(bivariate_poisson_v4.SHARED_FRACTION_GRID),
        "shared_fraction_grid_max": max(bivariate_poisson_v4.SHARED_FRACTION_GRID),
        "shared_fraction_grid_step": 0.01,
        "minimum_training_rows": bivariate_poisson_v4.MIN_TRAIN_ROWS,
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "runtime_prediction_weight": 0.0,
        "validation_endpoint": "/internal/bivariate-poisson/validate",
    }
    payload["v4_010_provider_requests_added"] = 0
    payload["v4_010_model_weights_changed"] = False
    payload["v4_010_canonical_bet_logic_changed"] = False
    payload["v4_010_checkpoint"] = (
        "BIVARIATE POISSON BASELINE IMPLEMENTED AS RESEARCH ONLY. The shared "
        "component is fit from prior point-in-time training rows and compared "
        "walk-forward against independent Poisson. No market fields, post-kickoff "
        "features, production weights, thresholds, tiers, stakes, or BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v96.run_tick()
    _annotate_v4_010(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
