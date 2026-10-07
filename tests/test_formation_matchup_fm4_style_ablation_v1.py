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
