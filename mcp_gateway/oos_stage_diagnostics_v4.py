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
MODEL_VERSION = "SOCCER_OOS_STAGE_DIAGNOSTICS_V4_1.3.0"
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
    discrimination = _auc_discrimination(observations)
    return {
        "rows": len(observations),
        "positive_count": positives,
        "negative_count": len(observations) - positives,
        "sample_status": _sample_status(len(observations)),
        "brier": metrics.get("brier"),
        "log_loss": metrics.get("log_loss"),
        "ece": metrics.get("ece"),
        "mce": metrics.get("mce"),
        "discrimination": discrimination,
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





def _auc_discrimination(observations: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in observations if row.get("outcome") == 1]
    negatives = [row for row in observations if row.get("outcome") == 0]
    n_pos = len(positives)
    n_neg = len(negatives)
    if n_pos == 0 or n_neg == 0:
        return {
            "auc": None,
            "auc_standard_error": None,
            "auc_lower_95": None,
            "positive_count": n_pos,
            "negative_count": n_neg,
            "discrimination_ready": False,
        }

    ranked = sorted(
        (
            (float(row["probability"]), int(row["outcome"]))
            for row in observations
        ),
        key=lambda item: item[0],
    )
    rank = 1
    sum_positive_ranks = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        count = end - index
        average_rank = (rank + (rank + count - 1)) / 2.0
        for _, outcome in ranked[index:end]:
            if outcome == 1:
                sum_positive_ranks += average_rank
        rank += count
        index = end

    auc = (
        sum_positive_ranks - (n_pos * (n_pos + 1) / 2.0)
    ) / (n_pos * n_neg)

    # Hanley-McNeil large-sample AUC standard error. We use the lower
    # confidence bound as a conservative research gate so calibration that
    # merely collapses to the base rate cannot create fixture-level edges.
    q1 = auc / (2.0 - auc) if auc < 2.0 else 0.0
    q2 = (2.0 * auc * auc) / (1.0 + auc) if auc > -1.0 else 0.0
    variance = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    standard_error = math.sqrt(max(variance, 0.0))
    lower_95 = max(0.0, auc - 1.96 * standard_error)

    return {
        "auc": round(auc, 8),
        "auc_standard_error": round(standard_error, 8),
        "auc_lower_95": round(lower_95, 8),
        "positive_count": n_pos,
        "negative_count": n_neg,
        "discrimination_ready": lower_95 > 0.5,
    }


def _current_model_deployment_calibrators(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for target in TARGET_KEYS:
        observations = []
        for row in rows:
            predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
            outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}
            probability = _num(predictions.get(target))
            outcome = outcomes.get(target)
            if probability is None or not 0.0 <= probability <= 1.0 or outcome not in (0, 1):
                continue
            observations.append({"probability": probability, "outcome": int(outcome)})
        report = calibration_v4.calibration_report(observations)
        calibrator = report.get("calibrator") if isinstance(report.get("calibrator"), dict) else {}
        brier_delta = report.get("brier_delta")
        log_loss_delta = report.get("log_loss_delta")
        discrimination = _auc_discrimination(observations)
        eligible = (
            calibrator.get("status") == "RESEARCH_CALIBRATOR_FITTED"
            and isinstance(brier_delta, (int, float))
            and isinstance(log_loss_delta, (int, float))
            and brier_delta < 0
            and log_loss_delta < 0
            and discrimination.get("discrimination_ready") is True
        )
        out[target] = {
            "rows": len(observations),
            "eligible_for_phase16_research": eligible,
            "brier_delta": brier_delta,
            "log_loss_delta": log_loss_delta,
            "discrimination": discrimination,
            "calibrator": calibrator,
            "production_promotion_allowed": False,
            "runtime_prediction_weight": 0.0,
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
    current_model_deployment_calibrators = _current_model_deployment_calibrators(current_rows)

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
        "current_model_deployment_calibrators": current_model_deployment_calibrators,
        "sample_policy": {
            "directional_minimum": DIRECTIONAL_MIN,
            "review_minimum": REVIEW_MIN,
            "primary_unit": "LATEST_PREKICKOFF_PREDICTION_PER_FIXTURE",
        },
        "anti_leakage": {
            "uses_canonical_oos_rows_only": True,
            "market_fields_used": False,
            "stage_calibrators_fitted": False,
            "current_model_full_oos_research_calibrators_fitted": True,
            "runtime_weights_changed": False,
        },
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "notes": [
            "Stage metrics are diagnostics only; no stage-specific calibrator is fitted or applied.",
            "Each stage now reports AUC and conservative AUC lower-95 discrimination for every binary target; stage diagnostics never change runtime weights or eligibility by themselves.",
            "Current-model stage metrics are reported separately to avoid mixing historical runtime model versions.",
            "Multiclass 1X2 metrics use the same normalized probability simplex as the canonical multiclass OOS validator.",
            "Current-model full-OOS binary calibrators are persisted only for downstream Phase16 research ranking; stage metrics remain diagnostic and production prediction weights remain unchanged.",
            "Phase16 binary calibration eligibility additionally requires AUC 95% lower confidence bound above 0.50, preventing base-rate-only calibration from creating fixture-level research edges.",
            "Phase16 binary calibration eligibility requires Brier and Log Loss improvement plus a conservative discrimination gate: AUC 95% lower bound must exceed 0.50.",
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
