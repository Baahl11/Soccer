from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Any, Iterable

from mcp_gateway import calibration_v4
from mcp_gateway import one_x_two_multiclass_oos_v4 as multiclass

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_OOS_STAGE_DIAGNOSTICS_V4_1.0.0"
DIRECTIONAL_MIN = 20
REVIEW_MIN = 50
TARGET_KEYS = ("home_win", "draw", "away_win", "btts", "over_2_5")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _semantic_version(value: str) -> tuple[int, ...] | None:
    match = re.search(r"v([0-9]+(?:[.][0-9]+)*)", str(value), re.I)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def _current_model_version(rows: Iterable[dict[str, Any]]) -> str | None:
    counts = Counter(
        str(row.get("model_version") or "UNKNOWN")
        for row in rows
        if isinstance(row, dict)
    )
    if not counts:
        return None
    semantic = [
        (parsed, version)
        for version in counts
        if (parsed := _semantic_version(version)) is not None
    ]
    if semantic:
        return max(semantic, key=lambda item: item[0])[1]
    return max(counts, key=counts.get)


def _sample_status(n: int) -> str:
    if n >= REVIEW_MIN:
        return "REVIEW_SAMPLE"
    if n >= DIRECTIONAL_MIN:
        return "DIRECTIONAL_SAMPLE"
    return "DATA_BLOCKED"


def _binary_stage_metrics(rows: list[dict[str, Any]], target: str) -> dict[str, Any]:
    observations = []
    for row in rows:
        predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
        outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}
        probability = _num(predictions.get(target))
        outcome = outcomes.get(target)
        if probability is None or not 0.0 <= probability <= 1.0 or outcome not in (0, 1):
            continue
        observations.append({"probability": probability, "outcome": int(outcome)})

    metrics = calibration_v4.reliability_metrics(observations)
    positives = sum(int(row["outcome"]) for row in observations)
    return {
        "rows": len(observations),
        "positive_count": positives,
        "negative_count": len(observations) - positives,
        "sample_status": _sample_status(len(observations)),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "ece": metrics.get("ece"),
        "mce": metrics.get("mce"),
    }


def _multiclass_stage_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for row in rows:
        item = multiclass._extract(row)
        if item is not None:
            samples.append(item)
    metrics = multiclass._metrics(samples)
    return {
        **metrics,
        "sample_status": _sample_status(int(metrics.get("n") or 0)),
    }


def _stage_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        stage = str(row.get("run_type") or "UNKNOWN").upper()
        by_stage[stage].append(row)

    out: dict[str, Any] = {}
    for stage, group in sorted(by_stage.items()):
        out[stage] = {
            "rows": len(group),
            "unique_fixtures": len({
                row.get("fixture_id") for row in group
                if row.get("fixture_id") is not None
            }),
            "sample_status": _sample_status(len(group)),
            "binary_targets": {
                target: _binary_stage_metrics(group, target)
                for target in TARGET_KEYS
            },
            "multiclass_1x2": _multiclass_stage_metrics(group),
        }
    return out


def build_report(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    clean = [row for row in rows if isinstance(row, dict)]
    current_version = _current_model_version(clean)
    current_rows = [
        row for row in clean
        if str(row.get("model_version") or "UNKNOWN") == str(current_version)
    ]

    all_stages = _stage_report(clean)
    current_stages = _stage_report(current_rows)

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "OOS_STAGE_DIAGNOSTICS_ACTIVE",
        "current_source_model_version": current_version,
        "all_model_rows": len(clean),
        "current_model_rows": len(current_rows),
        "stage_counts_all_models": dict(sorted(Counter(
            str(row.get("run_type") or "UNKNOWN").upper() for row in clean
        ).items())),
        "stage_counts_current_model": dict(sorted(Counter(
            str(row.get("run_type") or "UNKNOWN").upper() for row in current_rows
        ).items())),
        "all_models_by_stage": all_stages,
        "current_model_by_stage": current_stages,
        "sample_policy": {
            "directional_minimum": DIRECTIONAL_MIN,
            "review_minimum": REVIEW_MIN,
            "primary_unit": "LATEST_PREKICKOFF_PREDICTION_PER_FIXTURE",
        },
        "anti_leakage": {
            "uses_canonical_oos_rows_only": True,
            "market_fields_used": False,
            "stage_calibrators_fitted": False,
            "runtime_weights_changed": False,
        },
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "notes": [
            "Stage metrics are diagnostics only; no stage-specific calibrator is fitted or applied.",
            "Current-model stage metrics are reported separately to avoid mixing historical runtime model versions.",
            "Multiclass 1X2 metrics use the same normalized probability simplex as the canonical multiclass OOS validator.",
        ],
    }


def _load_rows(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("rows"), list):
        return []
    return [row for row in value["rows"] if isinstance(row, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Research-only OOS diagnostics by scheduler stage.")
    parser.add_argument("--oos-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(_load_rows(args.oos_report))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "model_version": report["model_version"],
        "status": report["status"],
        "current_source_model_version": report["current_source_model_version"],
        "all_model_rows": report["all_model_rows"],
        "current_model_rows": report["current_model_rows"],
        "stage_counts_current_model": report["stage_counts_current_model"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
