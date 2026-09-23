from mcp_gateway import team_totals_oos_v4 as v


def test_v4_019_blocks_without_family_specific_clv():
    report = v.build_report(
        {
            "evaluated_fixtures": 500,
            "evaluated_probability_rows": 6000,
            "by_line": {
                "0.5": {"n": 2000},
                "1.5": {"n": 2000},
                "2.5": {"n": 2000},
            },
            "by_role_line_selection": {
                "HOME:0.5:OVER": {"n": 500, "mean_probability": 0.72, "observed_rate": 0.81},
                "AWAY:0.5:OVER": {"n": 500, "mean_probability": 0.67, "observed_rate": 0.75},
            },
            "promotion_gate": {"enabled": False},
        },
        [{"market": "Goals Over/Under", "fixture_id": 1, "clv_probability_pp": 0.02}],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert report["oos_sample"]["evaluated_fixtures"] == 500
    assert report["true_clv"]["rows"] == 0
    assert "TEAM_TOTALS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_019_team_total_clv_matcher_is_strict():
    summary = v.summarize_true_clv([
        {"market": "Home Team Total Goals", "fixture_id": 1, "clv_probability_pp": 0.03},
        {"market": "Away Team Goals Over/Under", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "fixture_id": 3, "clv_probability_pp": 0.2},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.01


def test_v4_019_role_line_calibration_summary():
    metrics = v.role_line_calibration({
        "HOME:0.5:OVER": {"n": 100, "mean_probability": 0.70, "observed_rate": 0.80},
        "AWAY:0.5:OVER": {"n": 100, "mean_probability": 0.65, "observed_rate": 0.70},
    })
    assert metrics["n"] == 200
    assert metrics["weighted_absolute_calibration_gap"] == 0.075
    assert metrics["max_absolute_calibration_gap"] == 0.1
