from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_2H_OOS_VALIDATION_V4_1.0.0"
MIN_OOS = 200
MIN_TRUE_CLV_ROWS = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_two_h_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    selection = _norm(row.get("selection"))
    period = any(token in market for token in ("second half", "2nd half", "2h"))
    derivative = any(token in market for token in ("over/under", "over under", "total", "both teams", "btts")) or any(
        token in selection for token in ("over", "under", "yes", "no")
    )
    return period and derivative


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_two_h_market(row)]
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


def build_report(model_report: dict[str, Any], true_clv_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    n = int(model_report.get("walk_forward_evaluated") or 0)
    baseline = model_report.get("baseline") if isinstance(model_report.get("baseline"), dict) else {}
    challenger = model_report.get("challenger") if isinstance(model_report.get("challenger"), dict) else {}
    improvement = model_report.get("improvement") if isinstance(model_report.get("improvement"), dict) else {}
    gate = model_report.get("promotion_gate") if isinstance(model_report.get("promotion_gate"), dict) else {}

    brier_delta = _num(improvement.get("brier_delta_baseline_minus_conditioned"))
    log_loss_delta = _num(improvement.get("log_loss_delta_baseline_minus_conditioned"))
    mae_delta = _num(improvement.get("mae_delta_baseline_minus_conditioned"))
    challenger_better_all = (
        brier_delta is not None and brier_delta > 0
        and log_loss_delta is not None and log_loss_delta > 0
        and mae_delta is not None and mae_delta > 0
    )

    clv = summarize_true_clv(true_clv_rows)
    blockers: list[str] = []
    warnings: list[str] = []

    if n < MIN_OOS:
        blockers.append(f"OOS_{n}_LT_{MIN_OOS}")
    if not challenger_better_all:
        blockers.append("HALFTIME_CONDITIONED_CHALLENGER_DOES_NOT_BEAT_BASELINE")
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"2H_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if gate.get("enabled") is not True:
        blockers.append("SOURCE_2H_PROMOTION_GATE_DISABLED")
    live_status = str(model_report.get("live_status") or "")
    if "NO_DEDICATED_HT_RESEARCH_STAGE" in live_status:
        blockers.append("DEDICATED_HT_RESEARCH_STAGE_MISSING")
    missing_context = model_report.get("not_yet_conditioned_on")
    if isinstance(missing_context, list) and missing_context:
        warnings.append("LIVE_CONTEXT_FEATURES_INCOMPLETE")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-021",
        "status": "OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "oos_sample": {
            "walk_forward_evaluated": n,
            "minimum_required": MIN_OOS,
        },
        "baseline": baseline,
        "challenger": challenger,
        "improvement": improvement,
        "challenger_beats_baseline_on_brier_logloss_mae": challenger_better_all,
        "conditioning_features": model_report.get("conditioning_features") or [],
        "not_yet_conditioned_on": model_report.get("not_yet_conditioned_on") or [],
        "live_status": model_report.get("live_status"),
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "source_promotion_gate": gate,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "V4-021 evaluates the halftime-conditioned 2H challenger against its pregame 2H baseline.",
            "Negative baseline-minus-conditioned deltas mean the challenger is worse and therefore cannot be promoted.",
            "Dedicated 2H market history and true CLV are mandatory; FT market closes cannot satisfy this gate.",
            "A dedicated halftime scheduler/research stage is required before any live production review.",
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
    parser = argparse.ArgumentParser(description="V4-021 2H OOS validation gate.")
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_json(args.model_report), _load_jsonl(args.true_clv_tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
