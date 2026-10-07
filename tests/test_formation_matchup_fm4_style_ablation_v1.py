from copy import deepcopy
from datetime import datetime, timedelta, timezone

from mcp_gateway import formation_matchup_fm4_style_ablation_v1 as fm4


def _row(i: int) -> dict:
    home_shots = 12 + (i % 5)
    away_shots = 8 + (i % 4)
    home_sot = 4 + (i % 3)
    away_sot = 2 + (i % 2)
    return {
        "fixture_id": i,
        "kickoff_local": f"2026-08-{(i % 28) + 1:02d}T{(i // 28) % 24:02d}:00:00+00:00",
        "league_id": 39,
        "league": "Test League",
        "home_team_id": 10,
        "home_team": "Home",
        "away_team_id": 20,
        "away_team": "Away",
        "home_formation": "4-3-3",
        "away_formation": "4-2-3-1",
        "matchup_key": "4-3-3 vs 4-2-3-1",
        "home_goals": 1 + (i % 3),
        "away_goals": i % 2,
        "home_shots": home_shots,
        "away_shots": away_shots,
        "home_sot": home_sot,
        "away_sot": away_sot,
        "home_blocked_shots": 3 + (i % 2),
        "away_blocked_shots": 2 + (i % 2),
        "home_shots_inside_box": 7 + (i % 3),
        "away_shots_inside_box": 4 + (i % 3),
        "home_possession": 52 + (i % 6),
        "away_possession": 48 - (i % 6),
        "home_fouls": 10 + (i % 4),
        "away_fouls": 11 + (i % 3),
        "home_yellow_cards": 1 + (i % 2),
        "away_yellow_cards": 1 + ((i + 1) % 2),
    }


def _source(n=80):
    rows = [_row(i) for i in range(1, n + 1)]
    rows.sort(key=lambda row: row["fixture_id"])
    start = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    for idx, row in enumerate(rows):
        row["kickoff_local"] = (start + timedelta(hours=12 * idx)).isoformat()
    return {
        "model_version": "FORMATION_MATCHUP_ENGINE_V1.0.0",
        "status": "RESEARCH_ONLY_FORMATION_MATCHUP_ENGINE",
        "fixtures_with_verified_formation_pair_and_final": n,
        "rows": rows,
    }


def _eval(report, target, fixture_id):
    return [
        row
        for row in report["evaluation_rows"]
        if row["target"] == target and row["fixture_id"] == fixture_id
    ][0]


def test_geometry_is_literal_only_and_does_not_infer_roles():
    assert fm4._formation_geometry("4-2-3-1") == {
        "back_line": 4.0,
        "front_line": 1.0,
        "outfield_lines": 4.0,
        "central_layers": 5.0,
    }
    assert fm4._formation_geometry("wingbacks") is None
    assert fm4._formation_geometry("4-3") is None


def test_current_match_postgame_style_is_not_used_for_its_own_prediction():
    source = _source(85)
    base = fm4.build_report(source)
    fixture_id = 70
    before = _eval(base, "SHOTS", fixture_id)

    changed = deepcopy(source)
    row = [r for r in changed["rows"] if r["fixture_id"] == fixture_id][0]
    row["home_possession"] = 99
    row["away_possession"] = 1
    row["home_shots_inside_box"] = row["home_shots"]
    row["away_shots_inside_box"] = 0
    row["home_blocked_shots"] = 0
    row["away_blocked_shots"] = row["away_shots"]

    after = _eval(fm4.build_report(changed), "SHOTS", fixture_id)
    for key in (
        "baseline_home",
        "baseline_away",
        "style_residual_home",
        "style_residual_away",
        "style_home",
        "style_away",
    ):
        assert after[key] == before[key]


def test_future_match_does_not_rewrite_prior_prediction():
    source = _source(85)
    base = fm4.build_report(source)
    before = _eval(base, "SOT", 70)

    changed = deepcopy(source)
    future = [r for r in changed["rows"] if r["fixture_id"] == 85][0]
    future["home_possession"] = 100
    future["away_possession"] = 0
    future["home_sot"] = 20
    future["away_sot"] = 15
    future["home_goals"] = 10
    future["away_goals"] = 9

    after = _eval(fm4.build_report(changed), "SOT", 70)
    assert after["style_home"] == before["style_home"]
    assert after["style_away"] == before["style_away"]


def test_personnel_is_explicitly_not_materialized_and_market_is_not_used():
    report = fm4.build_report(_source(80))

    assert report["status"] == "RESEARCH_HOLD_FM4_STYLE_PERSONNEL_ABLATION"
    assert report["personnel_overlay"]["status"] == "NOT_MATERIALIZED"
    assert report["health"]["odds_consumed"] is False
    assert report["health"]["market_prices_consumed"] is False
    assert report["health"]["current_match_postgame_style_consumed"] is False
    assert report["health"]["inferred_player_roles_used"] is False
    assert report["health"]["coach_continuity_used"] is False
    assert report["health"]["production_enabled"] is False
    assert report["health"]["decision_weight"] == 0.0
    assert "PERSONNEL_OVERLAY_NOT_MATERIALIZED" in report["health"]["blockers"]


def test_style_ablation_has_forward_only_evaluations_after_training_gate():
    report = fm4.build_report(_source(90))

    for target in ("SHOTS", "SOT", "GOALS"):
        row = report["targets"][target]
        assert row["eligible_fixtures"] > 0
        assert row["production_enabled"] is False
        assert row["decision_weight"] == 0.0
    shot_rows = [
        row for row in report["evaluation_rows"] if row["target"] == "SHOTS"
    ]
    assert min(row["training_rows_home"] for row in shot_rows) >= fm4.MIN_STYLE_TRAIN_N
    assert min(row["training_rows_away"] for row in shot_rows) >= fm4.MIN_STYLE_TRAIN_N


def _lineup_team(team_id: int, start: int, coach_id: int = 900) -> dict:
    return {
        "team_id": team_id,
        "team": f"Team {team_id}",
        "formation": "4-3-3",
        "coach_id": coach_id,
        "coach": f"Coach {coach_id}",
        "starters": [
            {
                "id": start + offset,
                "name": f"P{start + offset}",
                "pos": "G" if offset == 0 else "D" if offset < 5 else "M" if offset < 9 else "F",
                "grid": f"{1 + offset // 4}:{1 + offset % 4}",
            }
            for offset in range(11)
        ],
    }


def test_verified_personnel_features_use_player_ids_and_provider_positions_only():
    previous = _lineup_team(10, 100, coach_id=900)
    current = _lineup_team(10, 100, coach_id=900)
    current["starters"][-1] = {
        "id": 999,
        "name": "New",
        "pos": "F",
        "grid": "4:3",
    }

    features = fm4._personnel_features(current, [previous])

    assert features is not None
    assert features["prior_confirmed_xi_count"] == 1
    assert features["previous_xi_overlap_count"] == 10
    assert features["previous_xi_overlap_rate"] == round(10 / 11, 6)
    assert features["new_starter_count_vs_previous"] == 1
    assert features["coach_same_as_previous"] is True
    assert features["coach_consecutive_prior_matches"] == 1
    assert features["position_comparable_starters"] == 10
    assert features["position_continuity_rate"] == 1.0


def test_build_report_materializes_personnel_continuity_without_decision_weight(monkeypatch):
    source = _source(5)
    personnel_events = []
    for row in source["rows"]:
        fixture_id = int(row["fixture_id"])
        home_start = 100
        if fixture_id >= 2:
            home = _lineup_team(10, 100, coach_id=900)
            home["starters"][-1] = {
                "id": 9000 + fixture_id,
                "name": "Rotated",
                "pos": "F",
                "grid": "4:3",
            }
        else:
            home = _lineup_team(10, home_start, coach_id=900)
        away = _lineup_team(20, 200, coach_id=901)
        personnel_events.append(
            {
                "fixture_id": fixture_id,
                "kickoff": fm4._dt(row["kickoff_local"]),
                "captured_at": fm4._dt(row["kickoff_local"]) - timedelta(minutes=20),
                "stage": "T-20",
                "teams": [home, away],
            }
        )

    monkeypatch.setattr(
        fm4,
        "_history_events",
        lambda _: ([], personnel_events),
    )
    report = fm4.build_report(source, history_dir="unused")

    personnel = report["personnel_overlay"]
    coverage = personnel["coverage"]
    assert personnel["status"] == "RESEARCH_ONLY_PERSONNEL_CONTINUITY"
    assert coverage["current_both_xi_confirmed_rows"] == 5
    assert coverage["rows_with_both_prior_confirmed_xi"] == 4
    assert coverage["rows_with_both_previous_coach_comparable"] == 4
    assert coverage["rows_with_both_last3_core_return_rate"] == 2
    assert personnel["production_enabled"] is False
    assert personnel["decision_weight"] == 0.0
    assert "PERSONNEL_OUTCOME_ABLATION_NOT_YET_VALIDATED" in personnel["blockers"]
    assert "PERSONNEL_OVERLAY_NOT_MATERIALIZED" not in personnel["blockers"]
    assert report["health"]["personnel_continuity_materialized"] is True
    assert report["health"]["personnel_outcome_ablation_used"] is False
    assert report["health"]["coach_continuity_used"] is False
    assert report["health"]["inferred_player_roles_used"] is False


def test_fm4_exposes_history_depth_blocker_without_relaxing_style_gate():
    source = _source(8)
    report = fm4.build_report(source, history_dir=None)

    style = report["style_profile"]
    assert style["history_depth_status"] == "INSUFFICIENT_PRIOR_DEPTH"
    assert "Do not relax MIN_TEAM_STYLE_N" in style["history_depth_policy"]
    assert any(
        blocker.startswith("STYLE_HISTORY_DEPTH_BOTH_TEAMS_N_GE_3_")
        for blocker in report["health"]["blockers"]
    )
    assert report["health"]["production_enabled"] is False
    assert report["health"]["decision_weight"] == 0.0
