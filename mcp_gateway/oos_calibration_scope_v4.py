from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from typing import Any, Iterable

from mcp_gateway import calibration_v4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_OOS_CALIBRATION_SCOPE_V4_1.0.0"
TARGET_KEYS = ("home_win", "draw", "away_win", "btts", "over_2_5")


def _num_probability(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) and 0.0 <= out <= 1.0 else None


def _semantic_version(value: str) -> tuple[int, ...] | None:
    match = re.search(r"v([0-9]+(?:[.][0-9]+)*)", str(value), re.I)
    if not match:
        return None
    try:
        return tuple(int(part) for part in match.group(1).split("."))
    except ValueError:
        return None


def current_model_version(rows: Iterable[dict[str, Any]]) -> str | None:
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


def build_targets(rows: Iterable[dict[str, Any]], *, source_model_version: str | None) -> tuple[dict[str, Any], int, int]:
    source = [row for row in rows if isinstance(row, dict)]
    targets: dict[str, Any] = {}
    ready_targets = 0
    improving_targets = 0

    for target in TARGET_KEYS:
        observations: list[dict[str, Any]] = []
        for row in source:
            predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
            outcomes = row.get("outcomes") if isinstance(row.get("outcomes"), dict) else {}
            probability = _num_probability(predictions.get(target))
            outcome = outcomes.get(target)
            if probability is None or outcome not in (0, 1):
                continue
            observations.append({"probability": probability, "outcome": int(outcome)})

        report = calibration_v4.calibration_report(observations)
        raw_metrics = report.get("raw_metrics") if isinstance(report.get("raw_metrics"), dict) else {}
        calibrated_metrics = report.get("calibrated_metrics") if isinstance(report.get("calibrated_metrics"), dict) else {}
        calibrator = report.get("calibrator") if isinstance(report.get("calibrator"), dict) else {}
        fitted = calibrator.get("status") == "RESEARCH_CALIBRATOR_FITTED"
        if fitted:
            ready_targets += 1

        brier_delta = report.get("brier_delta")
        log_loss_delta = report.get("log_loss_delta")
        improves_both = (
            isinstance(brier_delta, (int, float))
            and isinstance(log_loss_delta, (int, float))
            and brier_delta < 0
            and log_loss_delta < 0
        )
        if improves_both:
            improving_targets += 1

        targets[target] = {
            "rows": len(observations),
            "source_model_version": source_model_version,
            "status": report.get("status"),
            "raw_metrics": raw_metrics,
            "calibrator": calibrator,
            "calibrated_metrics": calibrated_metrics or None,
            "brier_delta": brier_delta,
            "log_loss_delta": log_loss_delta,
            "calibration_improves_brier_and_log_loss": improves_both,
            "production_promotion_allowed": False,
        }

    return targets, ready_targets, improving_targets


def apply_current_model_scope(report: dict[str, Any]) -> dict[str, Any]:
    out = dict(report) if isinstance(report, dict) else {}
    rows = [row for row in out.get("rows", []) if isinstance(row, dict)]
    current_version = current_model_version(rows)
    current_rows = [
        row for row in rows
        if str(row.get("model_version") or "UNKNOWN") == str(current_version)
    ]

    historical_targets = out.get("targets") if isinstance(out.get("targets"), dict) else {}
    historical_ready = int(out.get("ready_target_count") or 0)
    historical_improving = int(out.get("targets_improving_brier_and_log_loss") or 0)

    targets, ready_targets, improving_targets = build_targets(
        current_rows,
        source_model_version=current_version,
    )

    out["schema_version"] = SCHEMA_VERSION
    out["calibration_scope_model_version"] = MODEL_VERSION
    out["current_source_model_version"] = current_version
    out["all_model_rows"] = len(rows)
    out["current_model_rows"] = len(current_rows)
    out["canonical_target_scope"] = "CURRENT_SOURCE_MODEL_VERSION_ONLY"
    out["targets"] = targets
    out["ready_target_count"] = ready_targets
    out["targets_improving_brier_and_log_loss"] = improving_targets
    out["historical_all_models_targets"] = historical_targets
    out["historical_all_models_ready_target_count"] = historical_ready
    out["historical_all_models_targets_improving_brier_and_log_loss"] = historical_improving
    out["status"] = (
        "OOS_CALIBRATION_MATERIALIZED"
        if current_rows and ready_targets == len(TARGET_KEYS)
        else "COLLECTING_OOS_PREDICTIONS"
    )
    out["provider_requests_added"] = 0
    out["production_promotion_allowed"] = False
    out["runtime_prediction_weight"] = 0.0
    out["canonical_bet_logic_changed"] = False

    notes = list(out.get("notes")) if isinstance(out.get("notes"), list) else []
    notes.extend([
        "Canonical calibration targets are fitted only on the latest semantic source model version; historical model versions remain available for diagnostics but cannot satisfy current-model readiness gates.",
        "The complete merged OOS row ledger is preserved unchanged for provenance, audit, and historical comparisons.",
        "This scope correction changes no provider calls, runtime prediction weights, betting thresholds, canonical BET logic, or automatic promotion behavior.",
    ])
    out["notes"] = notes
    return out


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scope canonical OOS calibration targets to the current source model version.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = apply_current_model_scope(_load_json(args.input))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "calibration_scope_model_version": report.get("calibration_scope_model_version"),
        "status": report.get("status"),
        "canonical_target_scope": report.get("canonical_target_scope"),
        "current_source_model_version": report.get("current_source_model_version"),
        "all_model_rows": report.get("all_model_rows"),
        "current_model_rows": report.get("current_model_rows"),
        "ready_target_count": report.get("ready_target_count"),
        "targets_improving_brier_and_log_loss": report.get("targets_improving_brier_and_log_loss"),
        "provider_requests_added": report.get("provider_requests_added"),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
