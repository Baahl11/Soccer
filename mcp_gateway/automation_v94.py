from __future__ import annotations

from collections import Counter
from typing import Any

from mcp_gateway import automation_v93 as v93
from mcp_gateway import feature_snapshot_v4

MODEL_VERSION = v93.MODEL_VERSION
AUTOMATION_VERSION = "4.3.0-v4.007"


def _annotate_feature_snapshot_schema(payload: dict[str, Any]) -> None:
    valid = 0
    invalid = 0
    missing_total = 0
    feature_total = 0
    stage_counts: Counter[str] = Counter()

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        snapshot = feature_snapshot_v4.build(payload, event)
        errors = feature_snapshot_v4.validate(snapshot)
        if errors:
            invalid += 1
            continue
        valid += 1
        missing_total += int(snapshot.get("missing_feature_count") or 0)
        feature_total += int(snapshot.get("feature_count") or 0)
        stage_counts[str(snapshot.get("stage") or "UNKNOWN")] += 1

    payload["v4_007_feature_snapshot_schema"] = {
        "schema_version": feature_snapshot_v4.SCHEMA_VERSION,
        "status": "LIVE_VALIDATION",
        "valid_snapshot_count": valid,
        "invalid_snapshot_count": invalid,
        "feature_cells_count": feature_total,
        "missing_feature_cells_count": missing_total,
        "missing_feature_pct": (
            round(missing_total / feature_total * 100.0, 2) if feature_total else None
        ),
        "stage_counts": dict(stage_counts),
        "sport_first": True,
        "market_fields_included": False,
        "feature_envelope_fields": [
            "value",
            "source",
            "captured_at",
            "sample_n",
            "freshness",
            "confidence",
            "missing_reason",
        ],
        "database_table": "soccer_feature_snapshots",
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
    }
    payload["v4_007_provider_requests_added"] = 0
    payload["v4_007_model_weights_changed"] = False
    payload["v4_007_canonical_bet_logic_changed"] = False
    payload["v4_007_checkpoint"] = (
        "FEATURE SNAPSHOT SCHEMA v4 LIVE: versioned sport-first feature envelopes "
        "with explicit provenance, freshness, sample size, confidence and missing_reason. "
        "No odds/market fields are admitted into the feature snapshot."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v93.run_tick()
    _annotate_feature_snapshot_schema(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
