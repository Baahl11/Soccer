from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v94 as v94
from mcp_gateway import training_dataset_v4

MODEL_VERSION = v94.MODEL_VERSION
AUTOMATION_VERSION = "4.4.0-v4.008"


def _annotate_v4_008(payload: dict[str, Any]) -> None:
    feature = payload.get("v4_007_feature_snapshot_schema")
    if not isinstance(feature, dict):
        feature = {}

    payload["v4_008_reproducible_training_dataset"] = {
        "dataset_version": training_dataset_v4.DATASET_VERSION,
        "feature_schema_version": feature.get("schema_version") or "4.0.0",
        "status": "LIVE_VALIDATION",
        "policy": training_dataset_v4.POLICY,
        "source_tables": [
            "soccer_fixtures",
            "soccer_feature_snapshots",
            "soccer_results",
        ],
        "snapshot_selection": "LATEST_VALID_SNAPSHOT_AT_OR_BEFORE_KICKOFF",
        "targets_source": "FINAL_RESULT_ONLY",
        "row_fingerprint": "SHA256_CANONICAL_JSON",
        "dataset_fingerprint": "SHA256_ORDERED_ROWS_CANONICAL_JSON",
        "market_fields_included": False,
        "post_kickoff_features_allowed": False,
        "explicit_missingness_preserved": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "current_tick_valid_feature_snapshots": int(feature.get("valid_snapshot_count") or 0),
        "current_tick_invalid_feature_snapshots": int(feature.get("invalid_snapshot_count") or 0),
    }
    payload["v4_008_provider_requests_added"] = 0
    payload["v4_008_model_weights_changed"] = False
    payload["v4_008_canonical_bet_logic_changed"] = False
    payload["v4_008_checkpoint"] = (
        "REPRODUCIBLE TRAINING DATASET v4: deterministic rows are derived from the latest "
        "valid feature snapshot at or before kickoff plus finalized match results only. "
        "Market fields and post-kickoff features are excluded; explicit missingness and "
        "feature provenance are preserved; rows/datasets are SHA-256 fingerprinted."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v94.run_tick()
    _annotate_v4_008(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
