from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v97 as v97
from mcp_gateway import lightgbm_goals_v4

MODEL_VERSION = v97.MODEL_VERSION
AUTOMATION_VERSION = "4.7.0-v4.011"


def _annotate_v4_011(payload: dict[str, Any]) -> None:
    payload["v4_011_lightgbm_goals_challenger"] = {
        "schema_version": "1.0.0",
        "status": "OFFLINE_RESEARCH_CHALLENGER_IMPLEMENTED",
        "model_version": lightgbm_goals_v4.MODEL_VERSION,
        "training_mode": "GITHUB_ACTIONS_OFFLINE_ONLY",
        "runtime_imports_lightgbm": False,
        "minimum_training_rows": lightgbm_goals_v4.MIN_TRAIN_ROWS,
        "minimum_oos_rows": lightgbm_goals_v4.MIN_OOS_ROWS,
        "targets": [
            "home_goals",
            "away_goals",
            "home_win",
            "draw",
            "away_win",
            "btts",
            "over_1_5",
            "over_2_5",
            "over_3_5",
        ],
        "explicit_missingness_indicators": True,
        "silent_imputation_used": False,
        "market_fields_used": False,
        "post_kickoff_features_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "validation_workflow": ".github/workflows/v4-011-lightgbm-validation.yml",
    }
    payload["v4_011_provider_requests_added"] = 0
    payload["v4_011_model_weights_changed"] = False
    payload["v4_011_canonical_bet_logic_changed"] = False
    payload["v4_011_checkpoint"] = (
        "LIGHTGBM GOALS CHALLENGER IMPLEMENTED OFFLINE ONLY. Training consumes the "
        "reproducible point-in-time V4 dataset in GitHub Actions so the 512 MB Render "
        "tick worker does not load LightGBM. Explicit missingness is preserved and "
        "market/post-kickoff fields are excluded. No production promotion is allowed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v97.run_tick()
    _annotate_v4_011(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
