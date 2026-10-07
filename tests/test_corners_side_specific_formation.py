from mcp_gateway import analyze_corners_baseline as corners


def _row():
    return {
        "baseline_home_lambda": 5.0,
        "baseline_away_lambda": 4.0,
        "baseline_total_lambda": 9.0,
        "challenger_total_lambda": 9.4,
        "side_challenger_home_lambda": 5.4,
        "side_challenger_away_lambda": 3.8,
        "side_challenger_total_lambda": 9.2,
        "actual_home_corners": 6.0,
        "actual_away_corners": 3.0,
        "actual_total_corners": 9.0,
        "prior_matchup_n": 8,
        "base_over_8_5": 0.55,
        "base_over_9_5": 0.45,
        "base_over_10_5": 0.35,
        "challenger_over_8_5": 0.60,
        "challenger_over_9_5": 0.50,
        "challenger_over_10_5": 0.40,
        "side_challenger_over_8_5": 0.58,
        "side_challenger_over_9_5": 0.48,
        "side_challenger_over_10_5": 0.38,
    }


def test_side_specific_metrics_are_computed_independently():
    row = _row()
    base = corners.side_specific_metrics([row], "base")
    challenger = corners.side_specific_metrics([row], "side_challenger")

    assert base["mae_home_corners"] == 1.0
    assert base["mae_away_corners"] == 1.0
    assert challenger["mae_home_corners"] == 0.6
    assert challenger["mae_away_corners"] == 0.8
    assert challenger["mae_total_corners"] == 0.2


def test_total_probability_metrics_support_side_challenger_prefix():
    metrics = corners.metrics_for_evals([_row()], "side_challenger")
    assert metrics["n"] == 1
    assert metrics["mae_total_corners"] == 0.2
    assert set(metrics["lines"]) == {"8.5", "9.5", "10.5"}
