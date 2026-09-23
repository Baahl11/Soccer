from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v95 as v95
from mcp_gateway import dixon_coles_v4

MODEL_VERSION = v95.MODEL_VERSION
AUTOMATION_VERSION = "4.5.0-v4.009"


def _annotate_v4_009(payload: dict[str, Any]) -> None:
    dataset = payload.get("v4_008_reproducible_training_dataset")
    if not isinstance(dataset, dict):
        dataset = {}

    payload["v4_009_dixon_coles_baseline"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_BASELINE_IMPLEMENTED",
        "model_version": dixon_coles_v4.MODEL_VERSION,
        "method": "DIXON_COLES_LOW_SCORE_CORRECTION_ON_POINT_IN_TIME_GOAL_RATE_FEATURES",
        "rho_grid_min": min(dixon_coles_v4.RHO_GRID),
        "rho_grid_max": max(dixon_coles_v4.RHO_GRID),
        "rho_grid_step": 0.01,
        "minimum_training_rows": dixon_coles_v4.MIN_TRAIN_ROWS,
        "training_dataset_version": dataset.get("dataset_version") or "4.0.0",
        "feature_schema_version": dataset.get("feature_schema_version") or "4.0.0",
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "runtime_prediction_weight": 0.0,
        "validation_endpoint": "/internal/dixon-coles/validate",
    }
    payload["v4_009_provider_requests_added"] = 0
    payload["v4_009_model_weights_changed"] = False
    payload["v4_009_canonical_bet_logic_changed"] = False
    payload["v4_009_checkpoint"] = (
        "DIXON-COLES BASELINE IMPLEMENTED AS RESEARCH ONLY. It fits rho from prior "
        "point-in-time training rows and compares against independent Poisson in "
        "walk-forward evaluation. No market fields, post-kickoff features, production "
        "weights, canonical thresholds, tiers, stakes, or BET logic are changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v95.run_tick()
    _annotate_v4_009(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
