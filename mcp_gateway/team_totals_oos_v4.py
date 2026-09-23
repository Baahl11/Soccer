from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_TEAM_TOTALS_OOS_V4_1.0.0"
MIN_RESEARCH_FIXTURES = 100
MIN_ACTIONABLE_REVIEW_FIXTURES = 200
MIN_TRUE_CLV_ROWS = 50
REQUIRED_LINES = (0.5, 1.5, 2.5)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_team_total_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    return "team total" in market or "team goals" in market


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_team_total_market(row)]
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


def role_line_calibration(summary: dict[str, Any]) -> dict[str, Any]:
    total_n = 0
    weighted_gap = 0.0
    max_gap = 0.0
    worst_key = None
    for key, row in summary.items():
        if not isinstance(row, dict):
            continue
        n = int(row.get("n") or 0)
        mean_p = _num(row.get("mean_probability"))
        observed = _num(row.get("observed_rate"))
        if n <= 0 or mean_p is None or observed is None:
            continue
        gap = abs(mean_p - observed)
        total_n += n
        weighted_gap += gap * n
        if gap > max_gap:
            max_gap = gap
            worst_key = key
    return {
        "n": total_n,
        "weighted_absolute_calibration_gap": round(weighted_gap / total_n, 8) if total_n else None,
        "max_absolute_calibration_gap": round(max_gap, 8) if total_n else None,
        "worst_segment": worst_key,
    }


def build_report(validation: dict[str, Any], true_clv_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    fixtures = int(validation.get("evaluated_fixtures") or 0)
    probability_rows = int(validation.get("evaluated_probability_rows") or 0)
    by_line = validation.get("by_line") if isinstance(validation.get("by_line"), dict) else {}
    by_role_line_selection = (
        validation.get("by_role_line_selection")
        if isinstance(validation.get("by_role_line_selection"), dict)
        else {}
    )
    gate = validation.get("promotion_gate") if isinstance(validation.get("promotion_gate"), dict) else {}
    clv = summarize_true_clv(true_clv_rows)
    calibration = role_line_calibration(by_role_line_selection)

    observed_lines = {
        float(line)
        for line, row in by_line.items()
        if _num(line) is not None and isinstance(row, dict) and int(row.get("n") or 0) > 0
    }
    missing_lines = [line for line in REQUIRED_LINES if line not in observed_lines]

    blockers: list[str] = []
    warnings: list[str] = []
    if fixtures < MIN_RESEARCH_FIXTURES:
        blockers.append(f"OOS_FIXTURES_{fixtures}_LT_RESEARCH_{MIN_RESEARCH_FIXTURES}")
    if fixtures < MIN_ACTIONABLE_REVIEW_FIXTURES:
        blockers.append(f"OOS_FIXTURES_{fixtures}_LT_ACTIONABLE_{MIN_ACTIONABLE_REVIEW_FIXTURES}")
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"TEAM_TOTALS_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if gate.get("enabled") is not True:
        blockers.append("SOURCE_TEAM_TOTALS_PROMOTION_GATE_DISABLED")
    if missing_lines:
        blockers.append("REQUIRED_HALF_GOAL_LINE_COVERAGE_INCOMPLETE")

    max_gap = _num(calibration.get("max_absolute_calibration_gap"))
    if max_gap is not None and max_gap >= 0.10:
        warnings.append("ROLE_LINE_CALIBRATION_MAX_GAP_GE_0_10")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-019",
        "status": "OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "oos_sample": {
            "evaluated_fixtures": fixtures,
            "evaluated_probability_rows": probability_rows,
            "minimum_research_fixtures": MIN_RESEARCH_FIXTURES,
            "minimum_actionable_review_fixtures": MIN_ACTIONABLE_REVIEW_FIXTURES,
        },
        "required_lines": list(REQUIRED_LINES),
        "observed_lines": sorted(observed_lines),
        "missing_required_lines": missing_lines,
        "overall": validation.get("overall") if isinstance(validation.get("overall"), dict) else {},
        "by_team_role": validation.get("by_team_role") if isinstance(validation.get("by_team_role"), dict) else {},
        "by_line": by_line,
        "role_line_calibration": calibration,
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "source_promotion_gate": gate,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "The current OOS probability sample is large enough for research/actionable review counts, but market evidence remains a separate gate.",
            "Team-total true CLV must come from exact team-total line/price history; FT totals/1X2/BTTS closes cannot satisfy this requirement.",
            "No production promotion occurs from probability calibration alone.",
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
    parser = argparse.ArgumentParser(description="V4-019 Team Totals OOS validation gate.")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_json(args.validation), _load_jsonl(args.true_clv_tracking))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
