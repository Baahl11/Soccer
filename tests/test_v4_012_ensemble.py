from mcp_gateway import ensemble_v4


P1 = {
    "home_win": 0.50,
    "draw": 0.28,
    "away_win": 0.22,
    "btts": 0.54,
    "over_1_5": 0.75,
    "over_2_5": 0.52,
    "over_3_5": 0.29,
}
P2 = {
    "home_win": 0.47,
    "draw": 0.30,
    "away_win": 0.23,
    "btts": 0.57,
    "over_1_5": 0.77,
    "over_2_5": 0.55,
    "over_3_5": 0.31,
}
P3 = {
    "home_win": 0.53,
    "draw": 0.25,
    "away_win": 0.22,
    "btts": 0.51,
    "over_1_5": 0.73,
    "over_2_5": 0.49,
    "over_3_5": 0.27,
}


def test_v4_012_blocks_when_component_oos_is_insufficient():
    result = ensemble_v4.combine(
        {
            "DIXON_COLES": P1,
            "BIVARIATE_POISSON": P2,
            "LIGHTGBM_GOALS": P3,
        },
        {
            "DIXON_COLES": 11,
            "BIVARIATE_POISSON": 11,
            "LIGHTGBM_GOALS": 0,
        },
    )
    assert result["status"] == "INSUFFICIENT_COMPONENT_OOS"
    assert result["probabilities"] is None
    assert result["production_promotion_allowed"] is False


def test_v4_012_equal_weight_research_ensemble_is_normalized():
    result = ensemble_v4.combine(
        {
            "DIXON_COLES": P1,
            "BIVARIATE_POISSON": P2,
            "LIGHTGBM_GOALS": P3,
        },
        {name: 120 for name in ensemble_v4.REQUIRED_COMPONENTS},
    )
    assert result["status"] == "RESEARCH_ENSEMBLE_AVAILABLE"
    probs = result["probabilities"]
    assert abs(probs["home_win"] + probs["draw"] + probs["away_win"] - 1.0) < 1e-7
    assert result["market_fields_used"] is False
    assert result["production_promotion_allowed"] is False


def test_v4_012_weights_are_explicit_and_normalized():
    result = ensemble_v4.combine(
        {
            "DIXON_COLES": P1,
            "BIVARIATE_POISSON": P2,
            "LIGHTGBM_GOALS": P3,
        },
        {name: 150 for name in ensemble_v4.REQUIRED_COMPONENTS},
        weights={
            "DIXON_COLES": 2.0,
            "BIVARIATE_POISSON": 1.0,
            "LIGHTGBM_GOALS": 1.0,
        },
    )
    assert result["weights"] == {
        "DIXON_COLES": 0.5,
        "BIVARIATE_POISSON": 0.25,
        "LIGHTGBM_GOALS": 0.25,
    }
    assert result["weight_policy"] == "CALLER_SUPPLIED_RESEARCH_WEIGHTS"


def test_v4_012_current_dataset_gate_is_conservatively_blocked():
    gate = ensemble_v4.current_gate(11)
    assert gate["status"] == "DATA_BLOCKED"
    assert gate["component_oos_upper_bound"] == 0
    assert gate["production_promotion_allowed"] is False
