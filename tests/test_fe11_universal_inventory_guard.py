from __future__ import annotations

from scripts.fe11_universal_inventory_guard import verify


def example():
    return {
        "fe11_history_inventory": {
            "status": "VERIFIED_READ",
            "research_rows_persisted": 3,
            "production_bets_authorized": False,
            "provider_requests_added": 0,
        },
        "fe11_universal_inventory": {
            "status": "VERIFIED_READ",
            "eligible_pregame_baselines": 23,
            "matched_official_final_fixtures": 0,
            "current_future_baselines": 23,
            "official_period_complete_fixtures": 0,
            "official_ft_corners_verified_fixtures": 0,
            "official_period_and_corners_complete_fixtures": 0,
            "joint_goals_corners_handicap_training_rows": 0,
            "provider_requests_added": 0,
            "production_bets_authorized": False,
        },
    }


def test_real_verified_initial_state_is_accepted_without_fake_postgame():
    x = verify(example())
    assert x["status"] == "VERIFIED_AGGREGATE_CONSISTENCY"
    assert x["closure_state"] == "NO_MATCHED_OFFICIAL_FINAL_AS_OF_AUDIT"
    assert x["eligible_pregame_baselines"] == 23
    assert x["matched_official_final_fixtures"] == 0
    assert x["blockers"] == []
    assert x["provider_requests_added"] == 0
    assert x["database_writes_added"] == 0
    assert x["bet_eligible"] is False


def test_later_legal_postgame_with_no_corners_is_not_declared_missing_zero():
    source = example()
    inv = source["fe11_universal_inventory"]
    inv["current_future_baselines"] = 19
    inv["matched_official_final_fixtures"] = 4
    inv["official_period_complete_fixtures"] = 3
    x = verify(source)
    assert x["status"] == "VERIFIED_AGGREGATE_CONSISTENCY"
    assert x["closure_state"] == "OFFICIAL_POSTGAME_LINKAGE_OBSERVED"
    assert x["official_ft_corners_verified_fixtures"] == 0


def test_valid_completed_verified_90m_data_and_joint_gate_separate():
    source = example()
    inv = source["fe11_universal_inventory"]
    inv.update(current_future_baselines=0,
               matched_official_final_fixtures=23,
               official_period_complete_fixtures=18,
               official_ft_corners_verified_fixtures=15,
               official_period_and_corners_complete_fixtures=12)
    x = verify(source)
    assert x["status"] == "VERIFIED_AGGREGATE_CONSISTENCY"
    assert x["official_period_and_corners_complete_fixtures"] == 12
    assert x["joint_goals_corners_handicap_training_rows"] == 0


def test_rejects_preexisting_history_or_universal_floor_regression():
    source = example()
    source["fe11_history_inventory"]["research_rows_persisted"] = 2
    assert "HISTORICAL_PROFILE_REGRESSION" in verify(source)["blockers"]
    source = example()
    source["fe11_universal_inventory"]["eligible_pregame_baselines"] = 22
    assert "VERIFIED_PREMATCH_BASELINE_REGRESSION" in verify(source)["blockers"]


def test_rejects_inconsistent_counts_and_fake_corners():
    for name, value in (
        ("current_future_baselines", 24),
        ("matched_official_final_fixtures", 24),
        ("official_period_complete_fixtures", 1),
        ("official_ft_corners_verified_fixtures", 1),
        ("official_period_and_corners_complete_fixtures", 1),
    ):
        source = example()
        source["fe11_universal_inventory"][name] = value
        assert verify(source)["status"] == "NOT_VERIFIED", name


def test_rejects_false_verifications_and_boolean_coercion():
    for value in (True, "23", -1, None):
        source = example()
        source["fe11_universal_inventory"]["eligible_pregame_baselines"] = value
        assert verify(source)["status"] == "NOT_VERIFIED"
    source = example()
    source["fe11_universal_inventory"]["status"] = "NOT_VERIFIED"
    assert "UNIVERSAL_DB_READ_FAILED" in verify(source)["blockers"]


def test_rejects_bet_promotion_and_any_provider_calls_during_read_only_audit():
    for field, value in (
        ("production_bets_authorized", True),
        ("provider_requests_added", 1),
    ):
        source = example()
        source["fe11_universal_inventory"][field] = value
        assert verify(source)["status"] == "NOT_VERIFIED"
        source = example()
        source["fe11_history_inventory"][field] = value
        assert verify(source)["status"] == "NOT_VERIFIED"


def test_missing_input_fails_closed():
    assert verify(None)["status"] == "NOT_VERIFIED"
    assert verify({})["status"] == "NOT_VERIFIED"
