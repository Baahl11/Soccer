from copy import deepcopy
from datetime import datetime, timedelta, timezone

from mcp_gateway import formation_personnel_outcome_ablation_v1 as ablation


def _source(n: int = 90) -> list[dict]:
    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    rows = []
    for i in range(1, n + 1):
        signal = (i % 11) / 10.0
        home_goals = 1 + int(signal > 0.5) + (1 if i % 17 == 0 else 0)
        away_goals = int(signal < 0.3)
        rows.append(
            {
                "fixture_id": i,
                "kickoff_local": (start + timedelta(hours=12 * i)).isoformat(),
                "league_id": 39,
                "home_team_id": 10,
                "away_team_id": 20,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_shots": 10 + home_goals * 3 + (i % 3),
                "away_shots": 8 + away_goals * 3 + (i % 2),
                "home_sot": 3 + home_goals + (i % 2),
                "away_sot": 2 + away_goals,
            }
        )
    return rows


def _personnel(n: int = 90) -> list[dict]:
    start = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    rows = []
    for i in range(1, n + 1):
        signal = (i % 11) / 10.0
        rows.append(
            {
                "fixture_id": i,
                "kickoff_local": (start + timedelta(hours=12 * i)).isoformat(),
                "home": {
                    "previous_xi_overlap_rate": 0.65 + signal * 0.3,
                    "last3_core_return_rate": 0.60 + signal * 0.35,
                    "coach_same_as_previous": i % 9 != 0,
                    "new_starter_count_vs_previous": 4 - int(signal * 3),
                },
                "away": {
                    "previous_xi_overlap_rate": 0.9 - signal * 0.25,
                    "last3_core_return_rate": 0.88 - signal * 0.2,
                    "coach_same_as_previous": i % 13 != 0,
                    "new_starter_count_vs_previous": 1 + int(signal * 2),
                },
            }
        )
    return rows


def _eval(report: dict, target: str, fixture_id: int) -> dict:
    return [
        row
        for row in report["evaluation_rows"]
        if row["target"] == target and row["fixture_id"] == fixture_id
    ][0]


def test_personnel_feature_contract_requires_complete_prior_only_inputs():
    row = _personnel(1)[0]
    values = ablation._personnel_features(row)
    assert values is not None
    assert len(values) == 8

    broken = deepcopy(row)
    broken["home"]["last3_core_return_rate"] = None
    assert ablation._personnel_features(broken) is None


def test_personnel_ablation_is_forward_only_for_current_fixture_outcome():
    source = _source(90)
    personnel = _personnel(90)
    base = ablation.build_report(source, personnel)
    fixture_id = base["evaluation_rows"][10]["fixture_id"]
    before = _eval(base, "GOALS", fixture_id)

    changed = deepcopy(source)
    row = [value for value in changed if value["fixture_id"] == fixture_id][0]
    row["home_goals"] = 9
    row["away_goals"] = 8

    after = _eval(ablation.build_report(changed, personnel), "GOALS", fixture_id)
    for key in (
        "baseline_home",
        "baseline_away",
        "personnel_residual_home",
        "personnel_residual_away",
        "personnel_home",
        "personnel_away",
    ):
        assert after[key] == before[key]


def test_future_outcome_does_not_rewrite_prior_personnel_prediction():
    source = _source(90)
    personnel = _personnel(90)
    base = ablation.build_report(source, personnel)
    fixture_id = base["evaluation_rows"][5]["fixture_id"]
    before = _eval(base, "SHOTS", fixture_id)

    changed = deepcopy(source)
    changed[-1]["home_shots"] = 45
    changed[-1]["away_shots"] = 37
    changed[-1]["home_goals"] = 8
    changed[-1]["away_goals"] = 7

    after = _eval(ablation.build_report(changed, personnel), "SHOTS", fixture_id)
    assert after["personnel_home"] == before["personnel_home"]
    assert after["personnel_away"] == before["personnel_away"]


def test_personnel_ablation_stays_research_only_below_review_sample():
    report = ablation.build_report(_source(80), _personnel(80))

    assert report["status"] == "RESEARCH_HOLD"
    assert report["goals_ready_for_fm5"] is False
    assert report["production_enabled"] is False
    assert report["decision_weight"] == 0.0
    assert report["odds_consumed"] is False
    assert report["market_prices_consumed"] is False
    assert report["current_match_outcome_used_as_input"] is False
    assert report["inferred_player_roles_used"] is False
    assert report["provider_requests_added"] == 0
    assert report["production_promotion_allowed"] is False
    assert any(
        blocker.startswith("PERSONNEL_GOALS_ABLATION_")
        for blocker in report["blockers"]
    )


def test_evaluations_only_start_after_training_gate():
    report = ablation.build_report(_source(90), _personnel(90))
    assert report["evaluation_rows"]
    assert min(
        row["training_rows_home"] for row in report["evaluation_rows"]
    ) >= ablation.MIN_TRAIN_N
    assert min(
        row["training_rows_away"] for row in report["evaluation_rows"]
    ) >= ablation.MIN_TRAIN_N


def test_inputs_are_not_mutated():
    source = _source(50)
    personnel = _personnel(50)
    before_source = deepcopy(source)
    before_personnel = deepcopy(personnel)

    ablation.build_report(source, personnel)

    assert source == before_source
    assert personnel == before_personnel
