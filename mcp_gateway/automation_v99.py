from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v98 as v98
from mcp_gateway import ensemble_v4

MODEL_VERSION = v98.MODEL_VERSION
AUTOMATION_VERSION = "4.8.0-v4.012"


def _annotate_v4_012(payload: dict[str, Any]) -> None:
    dataset_meta = payload.get("v4_008_reproducible_training_dataset")
    if not isinstance(dataset_meta, dict):
        dataset_meta = {}
    current_rows = int(dataset_meta.get("current_tick_valid_feature_snapshots") or 0)

    payload["v4_012_ensemble_v1"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_FRAMEWORK_IMPLEMENTED",
        "model_version": ensemble_v4.MODEL_VERSION,
        "required_components": list(ensemble_v4.REQUIRED_COMPONENTS),
        "minimum_component_oos": ensemble_v4.MIN_COMPONENT_OOS,
        "same_fixture_oos_required": True,
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "current_tick_feature_snapshot_count": current_rows,
        "note": (
            "No ensemble probability is emitted until component predictions share an "
            "adequate same-fixture OOS cohort. Research framework only."
        ),
    }
    payload["v4_012_provider_requests_added"] = 0
    payload["v4_012_model_weights_changed"] = False
    payload["v4_012_canonical_bet_logic_changed"] = False
    payload["v4_012_checkpoint"] = (
        "ENSEMBLE v1 FRAMEWORK IMPLEMENTED WITH SAME-COHORT OOS GATES. Dixon-Coles, "
        "bivariate Poisson and LightGBM must each meet minimum OOS evidence before "
        "research combination. No market input, production weight, thresholds, tiers, "
        "stakes, or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v98.run_tick()
    _annotate_v4_012(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
