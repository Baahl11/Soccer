from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_CORNERS_OOS_VALIDATION_V4_1.4.0"
MIN_FT_OOS = 150
MIN_FORMATION_ADJUSTED = 100
MIN_TEAM_ROWS = 400
MIN_TRUE_CLV_ROWS = 50
FT_LINES = (8.5, 9.5, 10.5)
TEAM_LINES = (3.5, 4.5, 5.5)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_corners_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    return "corner" in market


def _corner_family(row: dict[str, Any]) -> str | None:
    family = str(row.get("market_family") or "").strip().upper()
    if family in {"FT_CORNERS", "TEAM_CORNERS"}:
        return family
    market = _norm(row.get("market"))
    if "corner" not in market:
        return None
    if "team" in market or "home corners" in market or "away corners" in market:
        return "TEAM_CORNERS"
    return "FT_CORNERS"


def _summarize_selected_true_clv(selected: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_num(row.get("clv_probability_pp")) for row in selected]
    valid = [value for value in values if value is not None]
    fixtures = [row.get("fixture_id") for row in selected if row.get("fixture_id") is not None]
    return {
        "rows": len(selected),
        "unique_fixtures": len(set(fixtures)),
        "avg_probability_clv_pp": round(sum(valid) / len(valid), 6) if valid else None,
        "positive_rows": sum(1 for value in valid if value > 0),
        "negative_rows": sum(1 for value in valid if value < 0),
        "flat_rows": sum(1 for value in valid if math.isclose(value, 0.0, abs_tol=1e-12)),
    }


def summarize_true_clv(rows: Iterable[dict[str, Any]], family: str | None = None) -> dict[str, Any]:
    selected = [
        row for row in rows
        if isinstance(row, dict)
        and is_corners_market(row)
        and (family is None or _corner_family(row) == family)
    ]
    return _summarize_selected_true_clv(selected)


def _line_improvement(baseline: dict[str, Any], challenger: dict[str, Any], line: float) -> dict[str, Any]:
    key = str(line)
    b = (baseline.get("lines") or {}).get(key) if isinstance(baseline.get("lines"), dict) else None
    c = (challenger.get("lines") or {}).get(key) if isinstance(challenger.get("lines"), dict) else None
    b = b if isinstance(b, dict) else {}
    c = c if isinstance(c, dict) else {}
    brier_b = _num(b.get("brier"))
    brier_c = _num(c.get("brier"))
    ll_b = _num(b.get("log_loss"))
    ll_c = _num(c.get("log_loss"))
    return {
        "line": line,
        "baseline_brier": brier_b,
        "challenger_brier": brier_c,
        "brier_delta_challenger_minus_baseline": round(brier_c - brier_b, 6) if brier_b is not None and brier_c is not None else None,
        "baseline_log_loss": ll_b,
        "challenger_log_loss": ll_c,
        "log_loss_delta_challenger_minus_baseline": round(ll_c - ll_b, 6) if ll_b is not None and ll_c is not None else None,
        "improves_brier": brier_b is not None and brier_c is not None and brier_c <= brier_b,
        "improves_log_loss": ll_b is not None and ll_c is not None and ll_c <= ll_b,
    }


def build_report(
    baseline_report: dict[str, Any],
    team_report: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    baseline = baseline_report.get("baseline") if isinstance(baseline_report.get("baseline"), dict) else {}
    challenger = baseline_report.get("formation_challenger") if isinstance(baseline_report.get("formation_challenger"), dict) else {}
    ft_gate = baseline_report.get("promotion_gate") if isinstance(baseline_report.get("promotion_gate"), dict) else {}
    team_gate = team_report.get("promotion_gate") if isinstance(team_report.get("promotion_gate"), dict) else {}

    ft_n = int(baseline_report.get("walk_forward_evaluations") or baseline.get("n") or 0)
    formation_n = int(baseline_report.get("formation_adjusted_evaluations") or 0)
    team_rows = int(team_report.get("evaluated_rows") or 0)
    team_fixtures = int(team_report.get("evaluated_fixtures") or 0)

    line_improvements = [_line_improvement(baseline, challenger, line) for line in FT_LINES]
    all_ft_lines_improve = all(
        row["improves_brier"] and row["improves_log_loss"]
        for row in line_improvements
    )

    mae_b = _num(baseline.get("mae_total_corners"))
    mae_c = _num(challenger.get("mae_total_corners"))
    mae_improves = mae_b is not None and mae_c is not None and mae_c <= mae_b

    role_lines = team_report.get("by_role_line") if isinstance(team_report.get("by_role_line"), dict) else {}
    observed_team_lines = sorted({
        float(key.split("|")[-1])
        for key, row in role_lines.items()
        if isinstance(row, dict) and "|" in str(key) and _num(str(key).split("|")[-1]) is not None and int(row.get("n") or 0) > 0
    })
    missing_team_lines = [line for line in TEAM_LINES if line not in observed_team_lines]

    true_clv_rows = [row for row in true_clv_rows if isinstance(row, dict)]
    clv = summarize_true_clv(true_clv_rows)
    ft_clv = summarize_true_clv(true_clv_rows, "FT_CORNERS")
    team_clv = summarize_true_clv(true_clv_rows, "TEAM_CORNERS")
    blockers: list[str] = []
    warnings: list[str] = []

    ft_blockers: list[str] = []
    team_blockers: list[str] = []

    if ft_n < MIN_FT_OOS:
        ft_blockers.append(f"FT_CORNERS_OOS_{ft_n}_LT_{MIN_FT_OOS}")
    if formation_n < MIN_FORMATION_ADJUSTED:
        ft_blockers.append(f"FORMATION_ADJUSTED_{formation_n}_LT_{MIN_FORMATION_ADJUSTED}")
    if not all_ft_lines_improve:
        ft_blockers.append("FORMATION_CHALLENGER_NOT_BETTER_ON_ALL_FT_LINES")
    if not mae_improves:
        ft_blockers.append("FORMATION_CHALLENGER_MAE_NOT_BETTER")
    if ft_clv["rows"] < MIN_TRUE_CLV_ROWS:
        ft_blockers.append(f"FT_CORNERS_TRUE_CLV_{ft_clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    league_lift = baseline_report.get("formation_lift_by_league")
    if not isinstance(league_lift, dict) or not league_lift:
        ft_blockers.append("FT_CORNERS_LEAGUE_LIFT_NOT_MATERIALIZED")
    elif league_lift.get("review_ready") is not True:
        ft_blockers.append("FT_CORNERS_LEAGUE_LIFT_NOT_STABLE")

    if team_rows < MIN_TEAM_ROWS:
        team_blockers.append(f"TEAM_CORNERS_ROWS_{team_rows}_LT_{MIN_TEAM_ROWS}")
    if team_clv["rows"] < MIN_TRUE_CLV_ROWS:
        team_blockers.append(f"TEAM_CORNERS_TRUE_CLV_{team_clv['rows']}_LT_{MIN_TRUE_CLV_ROWS}")
    parent_ft_review_ready = (
        ft_n >= MIN_FT_OOS
        and formation_n >= MIN_FORMATION_ADJUSTED
        and all_ft_lines_improve
        and mae_improves
        and ft_clv["rows"] >= MIN_TRUE_CLV_ROWS
        and "FT_CORNERS_LEAGUE_LIFT_NOT_MATERIALIZED" not in ft_blockers
        and "FT_CORNERS_LEAGUE_LIFT_NOT_STABLE" not in ft_blockers
    )
    if not parent_ft_review_ready:
        team_blockers.append("PARENT_FT_CORNERS_NOT_REVIEW_READY")

    league_rows = team_report.get("by_league") if isinstance(team_report.get("by_league"), dict) else {}
    venue_stability = team_report.get("league_venue_stability")
    if not league_rows or not isinstance(venue_stability, dict) or not venue_stability:
        team_blockers.append("TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_MATERIALIZED")
    elif venue_stability.get("review_ready") is not True:
        team_blockers.append("TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_READY")
    if missing_team_lines:
        warnings.append("TEAM_CORNERS_LINE_COVERAGE_INCOMPLETE")

    blockers = sorted(set(ft_blockers + team_blockers))

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "ticket": "V4-022",
        "status": "OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "ft_corners": {
            "walk_forward_evaluations": ft_n,
            "minimum_oos": MIN_FT_OOS,
            "formation_adjusted_evaluations": formation_n,
            "minimum_formation_adjusted": MIN_FORMATION_ADJUSTED,
            "formation_lift_by_league": league_lift if isinstance(league_lift, dict) else {},
            "baseline_mae_total_corners": mae_b,
            "challenger_mae_total_corners": mae_c,
            "mae_improves": mae_improves,
            "line_improvements": line_improvements,
            "all_required_lines_improve_brier_and_log_loss": all_ft_lines_improve,
            "required_lines": list(FT_LINES),
        },
        "team_corners": {
            "evaluated_fixtures": team_fixtures,
            "evaluated_rows": team_rows,
            "minimum_team_rows": MIN_TEAM_ROWS,
            "required_lines": list(TEAM_LINES),
            "observed_lines": observed_team_lines,
            "missing_required_lines": missing_team_lines,
            "overall": team_report.get("overall") if isinstance(team_report.get("overall"), dict) else {},
            "league_venue_stability": venue_stability if isinstance(venue_stability, dict) else {},
        },
        "true_clv": {
            **clv,
            "minimum_rows": MIN_TRUE_CLV_ROWS,
            "family_specific": False,
            "scope": "ALL_CORNERS_DIAGNOSTIC_ONLY",
        },
        "family_views": {
            "FT_CORNERS": {
                "status": "OOS_REVIEW_ELIGIBLE" if not ft_blockers else "RESEARCH_HOLD",
                "blockers": ft_blockers,
                "warnings": [],
                "true_clv": {
                    **ft_clv,
                    "minimum_rows": MIN_TRUE_CLV_ROWS,
                    "family_specific": True,
                },
            },
            "TEAM_CORNERS": {
                "status": "OOS_REVIEW_ELIGIBLE" if not team_blockers else "RESEARCH_HOLD",
                "blockers": team_blockers,
                "warnings": warnings,
                "true_clv": {
                    **team_clv,
                    "minimum_rows": MIN_TRUE_CLV_ROWS,
                    "family_specific": True,
                },
            },
        },
        "source_ft_promotion_gate": ft_gate,
        "source_team_promotion_gate": team_gate,
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Corners OOS is evaluated separately from goals markets because market microstructure and variance differ.",
            "Formation adjustment currently improves FT corners Brier/log-loss on 8.5/9.5/10.5 and total-corners MAE, but its adjusted sample remains too small.",
            "Verified exact corners prices and family-specific true CLV are mandatory before any production promotion.",
            "Team corners require stable home/away line calibration and cannot inherit evidence from FT corners or goals.",
            "FT_CORNERS and TEAM_CORNERS true-CLV evidence is reported in separate family_views; aggregate corners CLV remains diagnostic only.",
            "Historical source promotion_gate.enabled flags are descriptive metadata only; V4-022 blockers now name the missing evidence explicitly.",
            "FT Corners requires formation lift to be materialized and stable across at least two review-sized leagues before review; Team Corners requires an adequate parent FT model plus materialized league/venue stability.",
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
    parser = argparse.ArgumentParser(description="V4-022 Corners OOS validation gate.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--team-validation", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.baseline),
        _load_json(args.team_validation),
        _load_jsonl(args.true_clv_tracking),
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
