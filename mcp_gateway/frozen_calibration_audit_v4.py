from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Iterable

MODEL_VERSION = "SOCCER_FROZEN_CALIBRATION_AUDIT_V4_1.0.0"
SCHEMA_VERSION = "1.0.0"


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _artifact(provenance: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "phase16_calibrator_artifact",
        "calibrator_artifact",
        "calibration_artifact",
    ):
        value = provenance.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _artifact_fingerprint(artifact: dict[str, Any]) -> str | None:
    for key in ("fingerprint_sha256", "sha256", "artifact_id", "fingerprint"):
        value = artifact.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _source_model_version(provenance: dict[str, Any]) -> str | None:
    artifact = _artifact(provenance)
    for value in (
        artifact.get("source_model_version"),
        provenance.get("source_model_version"),
    ):
        if value is not None and str(value).strip():
            return str(value).strip()

    binary = provenance.get("binary_calibration_diagnostics")
    if isinstance(binary, dict):
        value = binary.get("source_model_version")
        if value is not None and str(value).strip():
            return str(value).strip()

    one_x_two = provenance.get("one_x_two_class_discrimination_diagnostics")
    if isinstance(one_x_two, dict):
        for row in one_x_two.values():
            if not isinstance(row, dict):
                continue
            value = row.get("source_model_version")
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def _model_version_match(provenance: dict[str, Any]) -> bool | None:
    artifact = _artifact(provenance)
    value = artifact.get("source_model_version_matches")
    if isinstance(value, bool):
        return value

    binary = provenance.get("binary_calibration_diagnostics")
    if isinstance(binary, dict) and isinstance(binary.get("source_model_version_matches"), bool):
        return bool(binary["source_model_version_matches"])

    one_x_two = provenance.get("one_x_two_class_discrimination_diagnostics")
    if isinstance(one_x_two, dict):
        states = [
            row.get("source_model_version_matches")
            for row in one_x_two.values()
            if isinstance(row, dict) and isinstance(row.get("source_model_version_matches"), bool)
        ]
        if states:
            return all(states)
    return None


def _is_calibrated(provenance: dict[str, Any]) -> bool:
    fields = provenance.get("calibrated_probability_fields")
    if isinstance(fields, dict) and fields:
        return True
    status = str(provenance.get("calibration_status") or "").strip().upper()
    return status == "RESEARCH_CALIBRATION_APPLIED"


def audit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total_rows = 0
    provenance_rows = 0
    calibrated_rows = 0
    calibrated_with_source_model = 0
    calibrated_with_source = 0
    calibrated_with_policy = 0
    calibrated_with_artifact = 0
    calibrated_with_fingerprint = 0
    artifact_kind_counts: Counter[str] = Counter()
    match_counts: Counter[str] = Counter()
    gap_counts: Counter[str] = Counter()

    for row in rows:
        if not isinstance(row, dict):
            continue
        total_rows += 1
        provenance = row.get("phase16_calibration_provenance")
        if not isinstance(provenance, dict) or not provenance:
            continue
        provenance_rows += 1
        if not _is_calibrated(provenance):
            continue
        calibrated_rows += 1

        source_model = _source_model_version(provenance)
        if source_model:
            calibrated_with_source_model += 1
        else:
            gap_counts["MISSING_SOURCE_MODEL_VERSION"] += 1

        calibration_source = provenance.get("calibration_source") or provenance.get("phase16_calibration_source")
        if calibration_source:
            calibrated_with_source += 1
        else:
            gap_counts["MISSING_CALIBRATION_SOURCE"] += 1

        calibration_policy = provenance.get("calibration_policy") or provenance.get("phase16_calibration_policy")
        if calibration_policy:
            calibrated_with_policy += 1
        else:
            gap_counts["MISSING_CALIBRATION_POLICY"] += 1

        artifact = _artifact(provenance)
        if artifact:
            calibrated_with_artifact += 1
            artifact_kind_counts[str(artifact.get("kind") or "UNKNOWN")] += 1
        else:
            gap_counts["MISSING_CALIBRATOR_ARTIFACT"] += 1

        if artifact and _artifact_fingerprint(artifact):
            calibrated_with_fingerprint += 1
        else:
            gap_counts["MISSING_IMMUTABLE_ARTIFACT_FINGERPRINT"] += 1

        match = _model_version_match(provenance)
        match_counts["MATCH" if match is True else "MISMATCH" if match is False else "UNKNOWN"] += 1

    if calibrated_rows == 0:
        status = "NOT_VERIFIED_NO_CALIBRATED_ROWS"
    elif calibrated_with_fingerprint == calibrated_rows:
        status = "OK_FROZEN_ARTIFACT_IDENTITY_COMPLETE"
    else:
        status = "WATCH_FROZEN_ARTIFACT_IDENTITY_GAPS"

    coverage = (
        round(calibrated_with_fingerprint / calibrated_rows, 8)
        if calibrated_rows
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": status,
        "total_rows": total_rows,
        "rows_with_phase16_calibration_provenance": provenance_rows,
        "calibrated_provenance_rows": calibrated_rows,
        "calibrated_rows_with_source_model_version": calibrated_with_source_model,
        "calibrated_rows_with_calibration_source": calibrated_with_source,
        "calibrated_rows_with_calibration_policy": calibrated_with_policy,
        "calibrated_rows_with_calibrator_artifact": calibrated_with_artifact,
        "calibrated_rows_with_immutable_artifact_fingerprint": calibrated_with_fingerprint,
        "immutable_artifact_coverage": coverage,
        "source_model_version_match_counts": dict(sorted(match_counts.items())),
        "artifact_kind_counts": dict(sorted(artifact_kind_counts.items())),
        "gap_counts": dict(sorted(gap_counts.items())),
        "historical_rows_mutated": False,
        "historical_probabilities_recomputed": False,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "strict_close_semantics_changed": False,
        "policy": (
            "AUDIT_ONLY; POINT_IN_TIME_PROVENANCE; NO_RETROACTIVE_RECALIBRATION; "
            "NO_PROVIDER_CALLS; NO_DECISION_WEIGHT; NO_PROMOTION"
        ),
    }


def _iter_jsonl(path: str) -> Iterable[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def run(input_path: str, output_path: str) -> dict[str, Any]:
    report = audit_rows(_iter_jsonl(input_path))
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="soccer_edge_state/analysis/signal_ledger.jsonl",
    )
    parser.add_argument(
        "--output",
        default="soccer_edge_state/analysis/frozen_calibration_audit_v4.json",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
