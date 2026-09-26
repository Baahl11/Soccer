from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_FT_TOTALS_VALIDATION_V4_1.1.0"
SUPPORTED_LINES = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)
MIN_DIRECTIONAL_SETTLED = 20
MIN_ACTIONABLE_REVIEW_SETTLED = 50
MIN_TRUE_CLV_ROWS = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _single_leg_grade(total_goals: int, side: str, line: float) -> str:
    normalized = str(side or "").upper()
    if normalized not in {"OVER", "UNDER"}:
        raise ValueError("SIDE_MUST_BE_OVER_OR_UNDER")
    if math.isclose(float(total_goals), float(line), abs_tol=1e-9):
        return "PUSH"
    if normalized == "OVER":
        return "WIN" if float(total_goals) > float(line) else "LOSS"
    return "WIN" if float(total_goals) < float(line) else "LOSS"


def split_asian_line(line: float) -> tuple[float, ...]:
    value = round(float(line), 2)
    quarter = round((value * 4) % 2, 8)
    if math.isclose(quarter, 1.0, abs_tol=1e-8):
        lower = math.floor(value * 2) / 2
        return (round(lower, 2), round(lower + 0.5, 2))
    return (value,)


def settle_asian_total(total_goals: int, side: str, line: float) -> dict[str, Any]:
    legs = split_asian_line(line)
    grades = [_single_leg_grade(total_goals, side, leg) for leg in legs]
    if len(grades) == 1:
        settlement = grades[0]
    else:
        wins = grades.count("WIN")
        losses = grades.count("LOSS")
        pushes = grades.count("PUSH")
        if wins == 2:
            settlement = "WIN"
        elif losses == 2:
            settlement = "LOSS"
        elif wins == 1 and pushes == 1:
            settlement = "HALF_WIN"
        elif losses == 1 and pushes == 1:
            settlement = "HALF_LOSS"
        elif wins == 1 and losses == 1:
            settlement = "HALF_WIN_HALF_LOSS"
        else:
            settlement = "PUSH"
    return {
        "total_goals": int(total_goals),
        "side": str(side).upper(),
        "line": float(line),
        "legs": list(legs),
        "leg_grades": grades,
        "settlement": settlement,
    }


def settlement_return_units(settlement: str, decimal_price: Any) -> float | None:
    price = _num(decimal_price)
    if price is None or price <= 1.0:
        return None
    profit = price - 1.0
    mapping = {
        "WIN": profit,
        "LOSS": -1.0,
        "PUSH": 0.0,
        "HALF_WIN": profit / 2.0,
        "HALF_LOSS": -0.5,
        "HALF_WIN_HALF_LOSS": (profit - 1.0) / 2.0,
    }
    value = mapping.get(str(settlement or "").upper())
    return round(value, 6) if value is not None else None


def is_ft_totals_clv_row(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    return market in {"goals over/under", "over/under", "goals over under"}


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_ft_totals_clv_row(row)]
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
    ft_validation: dict[str, Any],
    market_summary: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
    *,
    v4_oos_calibration_available: bool = False,
) -> dict[str, Any]:
    actionable = ft_validation.get("actionable") if isinstance(ft_validation.get("actionable"), dict) else {}
    research = ft_validation.get("watch_research") if isinstance(ft_validation.get("watch_research"), dict) else {}
    market_ft = (
        (market_summary.get("by_market_family") or {}).get("FT_TOTALS")
        if isinstance(market_summary.get("by_market_family"), dict)
        else {}
    )
    market_ft = market_ft if isinstance(market_ft, dict) else {}

    commercial_settled = int(market_ft.get("settled") or actionable.get("n") or 0)
    actionable_n = int(actionable.get("n") or 0)
    research_n = int(research.get("n") or 0)
    model_settled = int(
        ft_validation.get("evaluated_decisions")
        or (actionable_n + research_n)
        or 0
    )
    roi_units = _num(market_ft.get("roi_units"))
    hit_rate = _num(market_ft.get("hit_rate_ex_push"))
    brier = _num(actionable.get("mean_brier"))
    log_loss = _num(actionable.get("mean_log_loss"))

    by_line = ft_validation.get("by_line") if isinstance(ft_validation.get("by_line"), dict) else {}
    observed_lines = sorted({
        float(line)
        for line, summary in by_line.items()
        if _num(line) is not None and isinstance(summary, dict) and int(summary.get("n") or 0) > 0
    })
    required_line_coverage = {
        str(line): {
            "observed": line in observed_lines,
            "n": int((by_line.get(str(line)) or {}).get("n") or 0)
            if isinstance(by_line.get(str(line)), dict)
            else 0,
        }
        for line in SUPPORTED_LINES
    }
    missing_required_lines = [
        line for line in SUPPORTED_LINES if line not in observed_lines
    ]

    clv = summarize_true_clv(true_clv_rows)
    blockers: list[str] = []
    warnings: list[str] = []

    if model_settled < MIN_DIRECTIONAL_SETTLED:
        blockers.append(
            f"MODEL_SETTLED_{model_settled}_LT_DIRECTIONAL_{MIN_DIRECTIONAL_SETTLED}"
        )
    if model_settled < MIN_ACTIONABLE_REVIEW_SETTLED:
        blockers.append(
            f"MODEL_SETTLED_{model_settled}_LT_REVIEW_{MIN_ACTIONABLE_REVIEW_SETTLED}"
        )
    if clv["rows"] < MIN_TRUE_CLV_ROWS:
        blockers.append(f"FT_TOTALS_TRUE_CLV_{clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    if not v4_oos_calibration_available:
        blockers.append("V4_OOS_CALIBRATION_NOT_MATERIALIZED")
    if missing_required_lines:
        warnings.append("ASIAN_TOTAL_LINE_COVERAGE_INCOMPLETE")
    if commercial_settled < MIN_ACTIONABLE_REVIEW_SETTLED:
        warnings.append(
            f"COMMERCIAL_SETTLED_{commercial_settled}_LT_{MIN_ACTIONABLE_REVIEW_SETTLED}"
        )
    if roi_units is not None and roi_units <= 0:
        warnings.append("NON_POSITIVE_SETTLED_ROI")

    status = "PRODUCTION_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD"
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-016",
        "status": status,
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "provider_requests_added": 0,
        "supported_lines": list(SUPPORTED_LINES),
        "asian_quarter_settlement_supported": True,
        "sample": {
            "model_settled": model_settled,
            "actionable_settled": actionable_n,
            "research_settled": research_n,
            "commercial_settled": commercial_settled,
            "minimum_directional_model_settled": MIN_DIRECTIONAL_SETTLED,
            "minimum_model_review_settled": MIN_ACTIONABLE_REVIEW_SETTLED,
            "hit_rate_ex_push_commercial": hit_rate,
            "roi_units_commercial": roi_units,
            "mean_brier_legacy_actionable_signal_probability": brier,
            "mean_log_loss_legacy_actionable_signal_probability": log_loss,
            "sample_policy": (
                "MODEL_VALIDATION_USES_ALL_RESOLVED_FT_TOTALS_RESEARCH_AND_ACTIONABLE_DECISIONS; "
                "COMMERCIAL_PERFORMANCE_REMAINS_BET_LEAN_ONLY"
            ),
        },
        "line_coverage": required_line_coverage,
        "missing_required_lines": missing_required_lines,
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": True,
        },
        "v4_oos_calibration_available": bool(v4_oos_calibration_available),
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Resolved WATCH/RESEARCH FT Totals decisions are valid model-evidence outcomes and may satisfy sample-size gates; BET/LEAN settlements remain a separate commercial-performance metric.",
            "Legacy p_shrunk Brier/log-loss are descriptive and are not treated as V4 ensemble OOS calibration evidence.",
            "True CLV is counted only for canonical FT totals rows; 1X2/BTTS close observations cannot satisfy this gate.",
            "Quarter-goal lines use half-stake Asian settlement across adjacent integer/half lines.",
            "No automatic promotion is permitted. Eligible status only means manual production review may begin.",
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
    parser = argparse.ArgumentParser(description="V4-016 FT totals production-validation gate.")
    parser.add_argument("--ft-validation", required=True)
    parser.add_argument("--market-summary", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--v4-oos-calibration-available", action="store_true")
    args = parser.parse_args()

    report = build_report(
        _load_json(args.ft_validation),
        _load_json(args.market_summary),
        _load_jsonl(args.true_clv_tracking),
        v4_oos_calibration_available=args.v4_oos_calibration_available,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
