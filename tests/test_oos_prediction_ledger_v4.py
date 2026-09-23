from mcp_gateway import oos_prediction_ledger_v4 as v


def test_outcomes_binary_targets():
    result = v.outcomes(2, 1)
    assert result == {
        "home_win": 1,
        "draw": 0,
        "away_win": 0,
        "btts": 1,
        "over_2_5": 1,
    }


def test_normalize_row_requires_probabilities_in_bounds():
    row = {
        "fixture_id": 1,
        "run_timestamp": "2026-09-23T10:00:00+00:00",
        "kickoff": "2026-09-23T11:00:00+00:00",
        "run_type": "T-40",
        "model_version": "TEST",
        "raw_projection": {
            "raw_home_win_prob": 0.50,
            "raw_draw_prob": 0.25,
            "raw_away_win_prob": 0.25,
            "raw_btts_yes_prob": 0.55,
            "raw_over_2_5_prob": 1.5,
        },
        "home_goals": 2,
        "away_goals": 1,
    }
    normalized = v.normalize_row(row)
    assert normalized is not None
    assert normalized["predictions"]["home_win"] == 0.5
    assert normalized["predictions"]["over_2_5"] is None
    assert normalized["outcomes"]["over_2_5"] == 1
    assert normalized["anti_leakage"] is True


def test_target_contract_is_stable():
    assert set(v.TARGET_KEYS) == {"home_win", "draw", "away_win", "btts", "over_2_5"}
    assert set(v.RAW_PROBABILITY_KEYS) == set(v.TARGET_KEYS)
