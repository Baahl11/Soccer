from copy import deepcopy

import pytest

from mcp_gateway import feature_snapshot_v4, training_dataset_v4


def _tick():
    return {
        "generated_at_utc": "2026-09-23T04:55:00+00:00",
        "model_version": "SOCCER EDGE ENGINE v1.0",
    }


def _event():
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-40",
        "model_version": "SOCCER EDGE ENGINE v1.0",
        "fixture": {
            "fixture_id": 12345,
            "league_id": 128,
            "season": 2026,
            "home_team_id": 10,
            "away_team_id": 20,
            "venue": "Test Stadium",
            "city": "Test City",
        },
        "coverage": {"data_tier": "A"},
        "availability_confidence": 0.9,
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
            "teams": [
                {"team_id": 10, "formation": "4-3-3"},
                {"team_id": 20, "formation": "4-2-3-1"},
            ],
        },
        "injuries": [],
        "raw_projection": {
            "status": "MODELED_LIMITED",
            "sample": {
                "home_home_played": 8,
                "away_away_played": 7,
                "minimum_split_sample": 7,
            },
            "raw_home_goal_rate": 1.72,
            "raw_away_goal_rate": 1.03,
            "raw_total_goals": 2.75,
        },
    }


def _row():
    snapshot = feature_snapshot_v4.build(_tick(), _event())
    return training_dataset_v4.build_row(
        snapshot=snapshot,
        kickoff="2026-09-23T05:30:00+00:00",
        final_status="FT",
        home_goals=2,
        away_goals=1,
    )


def test_v4_008_row_is_valid_and_sport_first():
    row = _row()
    assert row["dataset_version"] == "4.0.0"
    assert row["feature_schema_version"] == "4.0.0"
    assert row["fixture_id"] == 12345
    assert row["market_fields_included"] is False
    assert training_dataset_v4.validate_row(row) == []


def test_v4_008_targets_are_derived_only_from_final_score():
    row = _row()
    assert row["targets"] == {
        "final_status": "FT",
        "home_goals": 2,
        "away_goals": 1,
        "total_goals": 3,
        "home_win": 1,
        "draw": 0,
        "away_win": 0,
        "btts": 1,
        "over_1_5": 1,
        "over_2_5": 1,
        "over_3_5": 0,
    }


def test_v4_008_rejects_post_kickoff_snapshot():
    snapshot = feature_snapshot_v4.build(_tick(), _event())
    with pytest.raises(ValueError, match="FEATURE_SNAPSHOT_AFTER_KICKOFF"):
        training_dataset_v4.build_row(
            snapshot=snapshot,
            kickoff="2026-09-23T04:30:00+00:00",
            final_status="FT",
            home_goals=2,
            away_goals=1,
        )


def test_v4_008_row_fingerprint_is_deterministic():
    first = _row()
    second = _row()
    assert first["row_fingerprint"] == second["row_fingerprint"]
    assert training_dataset_v4.dataset_fingerprint([first]) == training_dataset_v4.dataset_fingerprint([second])


def test_v4_008_fingerprint_detects_mutation():
    row = _row()
    mutated = deepcopy(row)
    mutated["features"]["team_performance.home_goal_rate_blend"] = 9.99
    assert "ROW_FINGERPRINT_MISMATCH" in training_dataset_v4.validate_row(mutated)


def test_v4_008_market_fields_do_not_enter_training_row():
    event = _event()
    event["market"] = {"markets": [{"market": "Match Winner", "price": 1.8}]}
    event["market_decision"] = {"best_decision": {"decimal_price": 1.8}}
    snapshot = feature_snapshot_v4.build(_tick(), event)
    row = training_dataset_v4.build_row(
        snapshot=snapshot,
        kickoff="2026-09-23T05:30:00+00:00",
        final_status="FT",
        home_goals=1,
        away_goals=1,
    )
    serialized = str(row).lower()
    assert "decimal_price" not in serialized
    assert "match winner" not in serialized
    assert row["market_fields_included"] is False
