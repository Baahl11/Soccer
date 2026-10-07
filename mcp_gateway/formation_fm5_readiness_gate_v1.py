from __future__ import annotations

import argparse
import json
import os
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_FM5_READINESS_GATE_V1.0.0"
MIN_STYLE_HISTORY_ROWS = 100
MIN_TARGET_OOS_ROWS = 100
MIN_PERSONNEL_ROWS = 100
MIN_CORNERS_FORMATION_ROWS = 100
MIN_STABLE_CORNERS_LEAGUES = 2


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _target_style_ready(target: dict[str, Any]) -> bool:
    improvement = _dict(target.get("improvement"))
    return (
        _int(target.get("eligible_fixtures")) >= MIN_TARGET_OOS_ROWS
        and improvement.get("home_mae_improves") is True
        and improvement.get("away_mae_improves") is True
        and improvement.get("total_mae_improves") is True
    )


def evaluate_fm4(report: dict[str, Any]) -> dict[str, Any]:
    style = _dict(report.get("style_profile"))
    density = _dict(style.get("prior_density"))
    health = _dict(report.get("health"))
    personnel = _dict(report.get("personnel_overlay"))
    coverage = _dict(personnel.get("coverage"))
    targets = _dict(report.get("targets"))

    prior_style_rows = _int(density.get("rows_with_both_min_field_n_ge_3"))
    leakage_clean = (
        health.get("odds_consumed") is False
        and health.get("market_prices_consumed") is False
        and health.get("current_match_postgame_style_consumed") is False
        and health.get("inferred_player_roles_used") is False
    )

    target_rows: dict[str, Any] = {}
    style_ready_targets: list[str] = []
    for name in ("GOALS", "SHOTS", "SOT"):
        target = _dict(targets.get(name))
        ready = (
            prior_style_rows >= MIN_STYLE_HISTORY_ROWS
            and leakage_clean
            and _target_style_ready(target)
        )
        improvement = _dict(target.get("improvement"))
        target_rows[name] = {
            "eligible_fixtures": _int(target.get("eligible_fixtures")),
            "minimum_required": MIN_TARGET_OOS_ROWS,
            "home_mae_improves": improvement.get("home_mae_improves") is True,
            "away_mae_improves": improvement.get("away_mae_improves") is True,
            "total_mae_improves": improvement.get("total_mae_improves") is True,
            "style_component_ready_for_fm5_review": ready,
            "source_blockers": _list(target.get("blockers")),
        }
        if ready:
            style_ready_targets.append(name)

    both_prior_xi = _int(coverage.get("rows_with_both_prior_confirmed_xi"))
    both_coach = _int(coverage.get("rows_with_both_previous_coach_comparable"))
    both_last3 = _int(coverage.get("rows_with_both_last3_core_return_rate"))
    personnel_sample_ready = (
        both_prior_xi >= MIN_PERSONNEL_ROWS
        and both_coach >= MIN_PERSONNEL_ROWS
        and both_last3 >= MIN_PERSONNEL_ROWS
    )

    # FM-4 currently materializes continuity diagnostics but does not yet
    # materialize a validated personnel outcome ablation. This explicit flag
    # is required before personnel can ever enter FM-5.
    personnel_outcome_ablation_ready = (
        personnel.get("outcome_ablation_ready") is True
        and health.get("personnel_outcome_ablation_used") is True
    )
    personnel_ready = (
        leakage_clean
        and personnel_sample_ready
        and personnel_outcome_ablation_ready
    )

    blockers: list[str] = []
    if prior_style_rows < MIN_STYLE_HISTORY_ROWS:
        blockers.append(
            f"FM4_STYLE_HISTORY_{prior_style_rows}_LT_{MIN_STYLE_HISTORY_ROWS}"
        )
    if not leakage_clean:
        blockers.append("FM4_LEAKAGE_CONTRACT_NOT_CLEAN")
    for name, row in target_rows.items():
        if not row["style_component_ready_for_fm5_review"]:
            blockers.append(f"FM4_{name}_STYLE_OOS_NOT_READY")
    if not personnel_sample_ready:
        blockers.append("FM4_PERSONNEL_SAMPLE_NOT_READY")
    if not personnel_outcome_ablation_ready:
        blockers.append("FM4_PERSONNEL_OUTCOME_ABLATION_NOT_READY")

    return {
        "source_model_version": report.get("model_version"),
        "prior_style_rows_both_teams_n_ge_3": prior_style_rows,
        "minimum_prior_style_rows": MIN_STYLE_HISTORY_ROWS,
        "leakage_contract_clean": leakage_clean,
        "targets": target_rows,
        "style_ready_targets": style_ready_targets,
        "personnel": {
            "rows_with_both_prior_confirmed_xi": both_prior_xi,
            "rows_with_both_previous_coach_comparable": both_coach,
            "rows_with_both_last3_core_return_rate": both_last3,
            "minimum_required": MIN_PERSONNEL_ROWS,
            "sample_ready": personnel_sample_ready,
            "outcome_ablation_ready": personnel_outcome_ablation_ready,
            "component_ready_for_fm5_review": personnel_ready,
            "source_blockers": _list(personnel.get("blockers")),
        },
        "blockers": blockers,
    }


def evaluate_corners(report: dict[str, Any]) -> dict[str, Any]:
    ft = _dict(report.get("ft_corners"))
    lift = _dict(ft.get("formation_lift_by_league"))
    side = _dict(ft.get("side_specific_formation_challenger"))
    team = _dict(report.get("team_corners"))
    team_stability = _dict(team.get("league_venue_stability"))
    health = _dict(report.get("formation_matchup_health"))

    stable_leagues = _list(lift.get("stable_lift_leagues"))
    ft_rows = _int(ft.get("formation_adjusted_evaluations"))
    ft_sport_ready = (
        ft_rows >= MIN_CORNERS_FORMATION_ROWS
        and ft.get("mae_improves") is True
        and ft.get("all_required_lines_improve_brier_and_log_loss") is True
        and len(stable_leagues) >= MIN_STABLE_CORNERS_LEAGUES
        and health.get("odds_consumed") is False
    )

    side_rows = _int(side.get("formation_adjusted_evaluations"))
    team_sport_ready = (
        side_rows >= MIN_CORNERS_FORMATION_ROWS
        and side.get("home_mae_improves") is True
        and side.get("away_mae_improves") is True
        and side.get("both_side_mae_improve") is True
        and side.get("total_mae_improves") is True
        and team_stability.get("review_ready") is True
        and ft_sport_ready
        and health.get("odds_consumed") is False
    )

    blockers: list[str] = []
    if not ft_sport_ready:
        blockers.append("FM2_FT_CORNERS_SPORT_OOS_NOT_READY")
    if not team_sport_ready:
        blockers.append("FM2_TEAM_CORNERS_SPORT_OOS_NOT_READY")

    return {
        "source_model_version": report.get("model_version"),
        "ft_corners": {
            "formation_adjusted_evaluations": ft_rows,
            "minimum_required": MIN_CORNERS_FORMATION_ROWS,
            "mae_improves": ft.get("mae_improves") is True,
            "all_required_lines_improve_brier_and_log_loss": (
                ft.get("all_required_lines_improve_brier_and_log_loss") is True
            ),
            "stable_lift_league_count": len(stable_leagues),
            "minimum_stable_lift_leagues": MIN_STABLE_CORNERS_LEAGUES,
            "component_ready_for_fm5_review": ft_sport_ready,
        },
        "team_corners": {
            "formation_adjusted_evaluations": side_rows,
            "minimum_required": MIN_CORNERS_FORMATION_ROWS,
            "home_mae_improves": side.get("home_mae_improves") is True,
            "away_mae_improves": side.get("away_mae_improves") is True,
            "both_side_mae_improve": side.get("both_side_mae_improve") is True,
            "total_mae_improves": side.get("total_mae_improves") is True,
            "league_venue_stability_review_ready": (
                team_stability.get("review_ready") is True
            ),
            "component_ready_for_fm5_review": team_sport_ready,
        },
        "blockers": blockers,
    }


def build_report(
    fm4_report: dict[str, Any],
    corners_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fm4 = evaluate_fm4(_dict(fm4_report))
    corners = evaluate_corners(_dict(corners_report)) if corners_report else None

    ready_components: list[str] = []
    ready_components.extend(
        f"FM4_STYLE_{name}" for name in fm4.get("style_ready_targets", [])
    )
    if fm4["personnel"]["component_ready_for_fm5_review"]:
        ready_components.append("FM4_PERSONNEL")
    if corners:
        if corners["ft_corners"]["component_ready_for_fm5_review"]:
            ready_components.append("FM2_FT_CORNERS")
        if corners["team_corners"]["component_ready_for_fm5_review"]:
            ready_components.append("FM2_TEAM_CORNERS")

    blockers = list(fm4.get("blockers") or [])
    if corners:
        blockers.extend(corners.get("blockers") or [])

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "FM5_COMPONENT_REVIEW_READY"
            if ready_components
            else "FM5_BLOCKED_EVIDENCE_GATES"
        ),
        "policy": (
            "SPORT_FIRST; NO_MARKET_INPUT_TO_RAW_PROJECTION; "
            "READINESS_ONLY; NO_AUTOMATIC_WEIGHT_OR_PRODUCTION_CHANGE; "
            "EACH_COMPONENT_REQUIRES VERSIONED_MANUAL_INTEGRATION_AFTER_OOS_GATE"
        ),
        "fm4": fm4,
        "corners": corners,
        "ready_components": sorted(ready_components),
        "blockers": sorted(set(blockers)),
        "integration_review_allowed": bool(ready_components),
        "automatic_integration_allowed": False,
        "production_enabled": False,
        "decision_weight": 0.0,
        "market_prices_consumed": False,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "historical_predictions_rewritten": False,
        "production_promotion_allowed": False,
    }


def _load(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate evidence-only readiness for FM-5 sporting integration."
    )
    parser.add_argument("--fm4-report", required=True)
    parser.add_argument("--corners-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(
        _load(args.fm4_report),
        _load(args.corners_report) if args.corners_report else None,
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
