from __future__ import annotations

from copy import deepcopy

from mcp_gateway import formation_fm5_readiness_gate_v1 as gate


def _fm4_base():
    return {
        "model_version": "FORMATION_MATCHUP_FM4_STYLE_ABLATION_V1.0.0",
        "style_profile": {
            "prior_density": {
                "rows_with_both_min_field_n_ge_3": 0,
            }
        },
        "targets": {
            name: {
                "eligible_fixtures": 0,
                "improvement": {
                    "home_mae_improves": False,
                    "away_mae_improves": False,
                    "total_mae_improves": False,
                },
                "blockers": ["STYLE_ABLATION_0_LT_100"],
            }
            for name in ("GOALS", "SHOTS", "SOT")
        },
        "personnel_overlay": {
            "coverage": {
                "rows_with_both_prior_confirmed_xi": 0,
                "rows_with_both_previous_coach_comparable": 0,
                "rows_with_both_last3_core_return_rate": 0,
            },
            "blockers": ["PERSONNEL_OUTCOME_ABLATION_NOT_YET_VALIDATED"],
        },
        "health": {
            "odds_consumed": False,
            "market_prices_consumed": False,
            "current_match_postgame_style_consumed": False,
            "inferred_player_roles_used": False,
            "personnel_outcome_ablation_used": False,
        },
    }


def _corners_base():
    return {
        "model_version": "SOCCER_CORNERS_OOS_VALIDATION_V4_1.5.0",
        "formation_matchup_health": {
            "odds_consumed": False,
        },
        "ft_corners": {
            "formation_adjusted_evaluations": 43,
            "mae_improves": True,
            "all_required_lines_improve_brier_and_log_loss": True,
            "formation_lift_by_league": {
                "stable_lift_leagues": [],
            },
            "side_specific_formation_challenger": {
                "formation_adjusted_evaluations": 43,
                "home_mae_improves": True,
                "away_mae_improves": True,
                "both_side_mae_improve": True,
                "total_mae_improves": True,
            },
        },
        "team_corners": {
            "league_venue_stability": {
                "review_ready": False,
            }
        },
    }


def test_current_like_evidence_keeps_fm5_blocked_and_zero_weight():
    report = gate.build_report(_fm4_base(), _corners_base())

    assert report["status"] == "FM5_BLOCKED_EVIDENCE_GATES"
    assert report["integration_review_allowed"] is False
    assert report["automatic_integration_allowed"] is False
    assert report["production_enabled"] is False
    assert report["decision_weight"] == 0.0
    assert report["market_prices_consumed"] is False
    assert report["provider_requests_added"] == 0
    assert report["model_weights_changed"] is False
    assert report["canonical_bet_logic_changed"] is False
    assert report["historical_predictions_rewritten"] is False
    assert report["production_promotion_allowed"] is False


def test_style_target_becomes_review_ready_only_after_history_sample_and_all_side_lift():
    fm4 = _fm4_base()
    fm4["style_profile"]["prior_density"]["rows_with_both_min_field_n_ge_3"] = 120
    target = fm4["targets"]["GOALS"]
    target["eligible_fixtures"] = 130
    target["improvement"] = {
        "home_mae_improves": True,
        "away_mae_improves": True,
        "total_mae_improves": True,
    }

    report = gate.build_report(fm4)

    assert report["status"] == "FM5_COMPONENT_REVIEW_READY"
    assert "FM4_STYLE_GOALS" in report["ready_components"]
    assert report["integration_review_allowed"] is True
    assert report["automatic_integration_allowed"] is False
    assert report["production_enabled"] is False


def test_style_target_does_not_pass_when_only_total_improves():
    fm4 = _fm4_base()
    fm4["style_profile"]["prior_density"]["rows_with_both_min_field_n_ge_3"] = 120
    target = fm4["targets"]["SHOTS"]
    target["eligible_fixtures"] = 130
    target["improvement"] = {
        "home_mae_improves": False,
        "away_mae_improves": True,
        "total_mae_improves": True,
    }

    report = gate.build_report(fm4)

    assert "FM4_STYLE_SHOTS" not in report["ready_components"]
    assert "FM4_SHOTS_STYLE_OOS_NOT_READY" in report["blockers"]


def test_personnel_never_passes_from_sample_depth_alone():
    fm4 = _fm4_base()
    fm4["personnel_overlay"]["coverage"] = {
        "rows_with_both_prior_confirmed_xi": 150,
        "rows_with_both_previous_coach_comparable": 150,
        "rows_with_both_last3_core_return_rate": 150,
    }

    report = gate.build_report(fm4)

    assert report["fm4"]["personnel"]["sample_ready"] is True
    assert report["fm4"]["personnel"]["outcome_ablation_ready"] is False
    assert report["fm4"]["personnel"]["component_ready_for_fm5_review"] is False
    assert "FM4_PERSONNEL" not in report["ready_components"]


def test_personnel_requires_explicit_validated_outcome_ablation_flag():
    fm4 = _fm4_base()
    fm4["personnel_overlay"]["coverage"] = {
        "rows_with_both_prior_confirmed_xi": 150,
        "rows_with_both_previous_coach_comparable": 150,
        "rows_with_both_last3_core_return_rate": 150,
    }
    fm4["personnel_overlay"]["outcome_ablation_ready"] = True
    fm4["health"]["personnel_outcome_ablation_used"] = True

    report = gate.build_report(fm4)

    assert report["fm4"]["personnel"]["component_ready_for_fm5_review"] is True
    assert "FM4_PERSONNEL" in report["ready_components"]


def test_market_leakage_blocks_style_even_with_good_oos_metrics():
    fm4 = _fm4_base()
    fm4["style_profile"]["prior_density"]["rows_with_both_min_field_n_ge_3"] = 150
    for target in fm4["targets"].values():
        target["eligible_fixtures"] = 150
        target["improvement"] = {
            "home_mae_improves": True,
            "away_mae_improves": True,
            "total_mae_improves": True,
        }
    fm4["health"]["market_prices_consumed"] = True

    report = gate.build_report(fm4)

    assert report["ready_components"] == []
    assert "FM4_LEAKAGE_CONTRACT_NOT_CLEAN" in report["blockers"]


def test_ft_corners_requires_sample_metrics_and_two_stable_leagues():
    corners = _corners_base()
    corners["ft_corners"]["formation_adjusted_evaluations"] = 120
    corners["ft_corners"]["formation_lift_by_league"]["stable_lift_leagues"] = [
        39,
        140,
    ]

    report = gate.build_report(_fm4_base(), corners)

    assert report["corners"]["ft_corners"]["component_ready_for_fm5_review"] is True
    assert "FM2_FT_CORNERS" in report["ready_components"]


def test_team_corners_requires_ft_parent_and_league_venue_stability():
    corners = _corners_base()
    corners["ft_corners"]["formation_adjusted_evaluations"] = 120
    corners["ft_corners"]["formation_lift_by_league"]["stable_lift_leagues"] = [
        39,
        140,
    ]
    side = corners["ft_corners"]["side_specific_formation_challenger"]
    side["formation_adjusted_evaluations"] = 120
    corners["team_corners"]["league_venue_stability"]["review_ready"] = True

    report = gate.build_report(_fm4_base(), corners)

    assert report["corners"]["team_corners"]["component_ready_for_fm5_review"] is True
    assert "FM2_TEAM_CORNERS" in report["ready_components"]


def test_gate_does_not_mutate_source_reports():
    fm4 = _fm4_base()
    corners = _corners_base()
    before_fm4 = deepcopy(fm4)
    before_corners = deepcopy(corners)

    gate.build_report(fm4, corners)

    assert fm4 == before_fm4
    assert corners == before_corners
