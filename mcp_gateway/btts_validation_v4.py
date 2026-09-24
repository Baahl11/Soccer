from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_BTTS_CALIBRATION_VALIDATION_V4_1.1.0"
MIN_CALIBRATION_SAMPLE = 300
MIN_TRUE_CLV_ROWS = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if isinstance(row, dict)
        and _norm(row.get("market")) in {"both teams score", "both teams to score", "btts"}
    ]
    values = [_num(row.get("clv_probability_pp")) for row in selected]
    valid = [value for value in values if value is not None]
    return {
        "rows": len(selected),
        "unique_fixtures": len({row.get("fixture_id") for row in selected if row.get("fixture_id") is not None}),
        "avg_probability_clv_pp": round(sum(valid) / len(valid), 6) if valid else None,
        "positive_rows": sum(1 for value in valid if value > 0),
        "negative_rows": sum(1 for value in valid if value < 0),
        "flat_rows": sum(1 for value in valid if math.isclose(value, 0.0, abs_tol=1e-12)),
    }


def calibration_error(by_bucket: dict[str, Any]) -> dict[str, Any]:
    total_n = 0
    weighted_gap = 0.0
    max_gap = 0.0
    usable = 0
    for bucket in by_bucket.values():
        if not isinstance(bucket, dict):
            continue
        n = int(bucket.get("n") or 0)
        mean_p = _num(bucket.get("mean_probability"))
        observed = _num(bucket.get("observed_rate"))
        if n <= 0 or mean_p is None or observed is None:
            continue
        gap = abs(mean_p - observed)
        total_n += n
        weighted_gap += gap * n
        max_gap = max(max_gap, gap)
        usable += 1
    return {
        "n": total_n,
        "usable_buckets": usable,
        "ece": round(weighted_gap / total_n, 8) if total_n else None,
        "mce": round(max_gap, 8) if total_n else None,
    }


def build_report(validation: dict[str, Any], true_clv_rows: Iterable[dict[str, Any]], oos_calibration: dict[str, Any] | None = None) -> dict[str, Any]:
    overall = validation.get("overall") if isinstance(validation.get("overall"), dict) else {}
    sample = int(overall.get("n") or validation.get("evaluated_fixtures") or 0)
    bucket_metrics = calibration_error(
        validation.get("by_probability_bucket")
        if isinstance(validation.get("by_probability_bucket"), dict)
        else {}
    )
    gate = validation.get("promotion_gate") if isinstance(validation.get("promotion_gate"), dict) else {}
    clv = summarize_true_clv(true_clv_rows)
    oos_calibration = oos_calibration if isinstance(oos_calibration, dict) else {}
    oos_target = (
        oos_calibration.get("targets", {}).get("btts", {})
        if isinstance(oos_calibration.get("targets"), dict)
        else {}
    )
    oos_calibrator = oos_target.get("calibrator") if isinstance(oos_target.get("calibrator"), dict) else {}
    canonical_oos_rows = int(oos_target.get("rows") or 0)
    canonical_oos_ready = (
        canonical_oos_rows >= MIN_CALIBRATION_SAMPLE
        and oos_target.get("status") == "RESEARCH_CALIBRATION_AVAILABLE"
        and oos_calibrator.get("status") == "RESEARCH_CALIBRATOR_FITTED"
        and oos_target.get("calibration_improves_brier_and_log_loss") is True
    )

    blockers: list[str] = []
    warnings: list[str] = []
    if sample < MIN_CALIBRATION_SAMPLE:
        blockers.append(f"CALIBRATION_SAMPLE_{sample}_LT_{MIN_CALIBRATION_SAMPLE}")
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"BTTS_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if not canonical_oos_ready:
        blockers.append("BTTS_CANONICAL_OOS_CALIBRATION_NOT_READY")
    ece = _num(bucket_metrics.get("ece"))
    if ece is not None and ece >= 0.10:
        warnings.append("CALIBRATION_ECE_GE_0_10_REQUIRES_REVIEW")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-018",
        "status": "CALIBRATION_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "calibration_sample": {
            "n": sample,
            "minimum_required": MIN_CALIBRATION_SAMPLE,
            "brier": _num(overall.get("brier")),
            "log_loss": _num(overall.get("log_loss")),
            "mean_probability": _num(overall.get("mean_probability")),
            "observed_rate": _num(overall.get("observed_rate")),
            **bucket_metrics,
        },
        "source_promotion_gate": gate,
        "canonical_oos_calibration": {
            "available": canonical_oos_ready,
            "rows": canonical_oos_rows,
            "target_status": oos_target.get("status"),
            "calibrator_status": oos_calibrator.get("status"),
            "brier_delta": oos_target.get("brier_delta"),
            "log_loss_delta": oos_target.get("log_loss_delta"),
            "improves_brier_and_log_loss": oos_target.get("calibration_improves_brier_and_log_loss") is True,
            "source_model_version": oos_calibration.get("model_version"),
        },
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "BTTS calibration is evaluated independently from FT totals and 1X2.",
            "True CLV counts only canonical BTTS rows; other market families cannot satisfy this gate.",
            "Existing legacy calibration diagnostics remain descriptive; canonical OOS calibration can satisfy the calibration layer only when its BTTS target is fitted and improves both Brier and log loss.",
            "Canonical OOS calibration does not substitute for family-specific true CLV, settlements, or manual promotion review.",
        ],
    }


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="V4-018 BTTS calibration validation gate.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--v4-oos-calibration-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.validation),
        _load_jsonl(args.true_clv_tracking),
        _load_json(args.v4_oos_calibration_report) if args.v4_oos_calibration_report else {},
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
