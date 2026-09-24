from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1X2_CALIBRATION_VALIDATION_V4_1.1.0"
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
        if isinstance(row, dict) and _norm(row.get("market")) in {"match winner", "winner"}
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


def build_report(
    calibration: dict[str, Any],
    prior_calibration: dict[str, Any],
    market_summary: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
    multiclass_oos: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = int(calibration.get("sample_fixtures") or 0)
    top1 = _num(calibration.get("top1_accuracy"))
    brier = _num(calibration.get("multiclass_brier"))
    log_loss = _num(calibration.get("multiclass_log_loss"))

    baseline = prior_calibration.get("baseline") if isinstance(prior_calibration.get("baseline"), dict) else {}
    challenger = prior_calibration.get("challenger") if isinstance(prior_calibration.get("challenger"), dict) else {}
    improvement = prior_calibration.get("improvement") if isinstance(prior_calibration.get("improvement"), dict) else {}

    brier_delta = _num(improvement.get("brier_delta"))
    log_loss_delta = _num(improvement.get("log_loss_delta"))
    accuracy_delta_pp = _num(improvement.get("accuracy_delta_pp"))
    legacy_challenger_better = (
        brier_delta is not None
        and log_loss_delta is not None
        and brier_delta < 0
        and log_loss_delta < 0
    )

    multiclass_oos = multiclass_oos if isinstance(multiclass_oos, dict) else {}
    canonical_source_rows = int(multiclass_oos.get("source_rows_current_model") or 0)
    canonical_eval_rows = int(multiclass_oos.get("evaluated_rows") or 0)
    canonical_brier_delta = _num(multiclass_oos.get("brier_delta"))
    canonical_log_loss_delta = _num(multiclass_oos.get("log_loss_delta"))
    canonical_ready = (
        multiclass_oos.get("status") == "RESEARCH_MULTICLASS_CALIBRATION_AVAILABLE"
        and canonical_source_rows >= MIN_CALIBRATION_SAMPLE
        and canonical_eval_rows >= 100
        and multiclass_oos.get("improves_brier_and_log_loss") is True
        and canonical_brier_delta is not None
        and canonical_log_loss_delta is not None
        and canonical_brier_delta < 0
        and canonical_log_loss_delta < 0
    )
    challenger_better = canonical_ready if multiclass_oos else legacy_challenger_better
    effective_sample = canonical_source_rows if multiclass_oos else sample

    clv = summarize_true_clv(true_clv_rows)
    family = (
        (market_summary.get("by_market_family") or {}).get("FT_1X2")
        if isinstance(market_summary.get("by_market_family"), dict)
        else {}
    )
    family = family if isinstance(family, dict) else {}

    blockers: list[str] = []
    warnings: list[str] = []
    if effective_sample < MIN_CALIBRATION_SAMPLE:
        blockers.append(f"CALIBRATION_SAMPLE_{effective_sample}_LT_{MIN_CALIBRATION_SAMPLE}")
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"1X2_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if not challenger_better:
        blockers.append(
            "MULTICLASS_OOS_CHALLENGER_NOT_READY"
            if multiclass_oos
            else "CALIBRATION_CHALLENGER_DOES_NOT_IMPROVE_BRIER_AND_LOG_LOSS"
        )
    if int(family.get("settled") or 0) < 20:
        warnings.append("ACTIONABLE_SETTLEMENT_SAMPLE_LT_20")
    draw_bins = (
        (calibration.get("calibration_by_outcome") or {}).get("D")
        if isinstance(calibration.get("calibration_by_outcome"), dict)
        else {}
    )
    if isinstance(draw_bins, dict):
        sparse_draw_bins = sum(1 for value in draw_bins.values() if isinstance(value, dict) and int(value.get("n") or 0) < 20)
        if sparse_draw_bins:
            warnings.append("DRAW_CALIBRATION_HAS_SPARSE_BUCKETS")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-017",
        "status": "CALIBRATION_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "calibration_sample": {
            "n": effective_sample,
            "legacy_diagnostic_n": sample,
            "minimum_required": MIN_CALIBRATION_SAMPLE,
            "top1_accuracy": top1,
            "multiclass_brier": brier,
            "multiclass_log_loss": log_loss,
        },
        "canonical_multiclass_oos": {
            "available": canonical_ready,
            "model_version": multiclass_oos.get("model_version"),
            "source_model_version": multiclass_oos.get("source_model_version"),
            "source_rows_current_model": canonical_source_rows,
            "evaluated_rows": canonical_eval_rows,
            "walk_forward_folds": int(multiclass_oos.get("walk_forward_folds") or 0),
            "baseline": multiclass_oos.get("baseline"),
            "temperature_scaled": multiclass_oos.get("temperature_scaled"),
            "brier_delta": canonical_brier_delta,
            "log_loss_delta": canonical_log_loss_delta,
            "improves_brier_and_log_loss": multiclass_oos.get("improves_brier_and_log_loss") is True,
        },
        "prior_calibration_challenger": {
            "baseline": baseline,
            "challenger": challenger,
            "improvement": improvement,
            "improves_brier_and_log_loss": challenger_better,
        },
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "settlement_context": {
            "settled": int(family.get("settled") or 0),
            "hit_rate_ex_push": _num(family.get("hit_rate_ex_push")),
            "roi_units": _num(family.get("roi_units")),
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Canonical 1X2 calibration evidence uses current-model chronological walk-forward temperature scaling and must improve both multiclass Brier and log loss.",
            "The historical class-prior challenger is retained only as descriptive legacy context when canonical multiclass OOS is supplied.",
            "True CLV is counted only from canonical 1X2/Match Winner observations.",
            "This gate validates calibration evidence; it does not alter runtime probabilities or bet classification.",
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
    parser = argparse.ArgumentParser(description="V4-017 1X2 calibration validation gate.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--prior-calibration", required=True)
    parser.add_argument("--market-summary", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--multiclass-oos-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.calibration),
        _load_json(args.prior_calibration),
        _load_json(args.market_summary),
        _load_jsonl(args.true_clv_tracking),
        _load_json(args.multiclass_oos_report) if args.multiclass_oos_report else {},
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
