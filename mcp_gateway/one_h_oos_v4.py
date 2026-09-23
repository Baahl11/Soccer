from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_1H_OOS_VALIDATION_V4_1.0.0"
MIN_CALIBRATED_OOS = 100
MIN_TRUE_CLV_ROWS = 50
REQUIRED_LINES = (0.5, 1.0, 1.25, 1.5, 1.75, 2.0)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_one_h_total_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    selection = _norm(row.get("selection"))
    period = any(token in market for token in ("first half", "1st half", "1h"))
    total = any(token in market for token in ("over/under", "over under", "total")) or "over" in selection or "under" in selection
    return period and total


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_one_h_total_market(row)]
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


def build_report(calibration: dict[str, Any], true_clv_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    baseline = calibration.get("baseline_same_sample") if isinstance(calibration.get("baseline_same_sample"), dict) else {}
    challenger = calibration.get("challenger") if isinstance(calibration.get("challenger"), dict) else {}
    gate = calibration.get("promotion_gate") if isinstance(calibration.get("promotion_gate"), dict) else {}

    n = int(challenger.get("n") or baseline.get("n") or 0)
    baseline_brier = _num(baseline.get("brier"))
    challenger_brier = _num(challenger.get("brier"))
    baseline_ll = _num(baseline.get("log_loss"))
    challenger_ll = _num(challenger.get("log_loss"))

    brier_improved = (
        baseline_brier is not None and challenger_brier is not None and challenger_brier <= baseline_brier
    )
    log_loss_improved = (
        baseline_ll is not None and challenger_ll is not None and challenger_ll <= baseline_ll
    )

    clv = summarize_true_clv(true_clv_rows)
    observed_lines = [1.5] if n > 0 else []
    missing_lines = [line for line in REQUIRED_LINES if line not in observed_lines]

    blockers: list[str] = []
    warnings: list[str] = []
    if n < MIN_CALIBRATED_OOS:
        blockers.append(f"CALIBRATED_OOS_{n}_LT_{MIN_CALIBRATED_OOS}")
    if not brier_improved:
        blockers.append("CHALLENGER_BRIER_NOT_BETTER_THAN_BASELINE")
    if not log_loss_improved:
        blockers.append("CHALLENGER_LOG_LOSS_NOT_BETTER_THAN_BASELINE")
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"1H_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if gate.get("enabled") is not True:
        blockers.append("SOURCE_1H_PROMOTION_GATE_DISABLED")
    if missing_lines:
        warnings.append("1H_LINE_COVERAGE_INCOMPLETE")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-020",
        "status": "OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "calibration": {
            "n": n,
            "minimum_calibrated_oos": MIN_CALIBRATED_OOS,
            "baseline": baseline,
            "challenger": challenger,
            "challenger_brier_improved": brier_improved,
            "challenger_log_loss_improved": log_loss_improved,
        },
        "required_lines": list(REQUIRED_LINES),
        "observed_lines": observed_lines,
        "missing_required_lines": missing_lines,
        "asian_settlement_required": True,
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "source_promotion_gate": gate,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Current persisted 1H calibration evidence is for Over 1.5 only; other roadmap lines remain unvalidated.",
            "The current challenger is not promoted unless it is no worse on both Brier and log loss.",
            "Dedicated 1H exact-market history and true CLV are mandatory and cannot be substituted by FT market closes.",
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
    parser = argparse.ArgumentParser(description="V4-020 1H OOS validation gate.")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_json(args.calibration), _load_jsonl(args.true_clv_tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
