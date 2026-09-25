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


def test_v4_019_rejects_cards_corners_and_period_team_totals_from_true_clv():
    summary = v.summarize_true_clv([
        {
            "market_family": "CARDS",
            "market": "Home Team Total Cards",
            "fixture_id": 10,
            "clv_probability_pp": 0.04,
        },
        {
            "market_family": "TEAM_CORNERS",
            "market": "Away Team Total Corners",
            "fixture_id": 11,
            "clv_probability_pp": 0.03,
        },
        {
            "market_family": "1H",
            "market": "Home Team Total Goals - First Half",
            "fixture_id": 12,
            "clv_probability_pp": 0.02,
        },
        {
            "market_family": "HOME_TT",
            "market": "Home Team Total Goals",
            "fixture_id": 13,
            "clv_probability_pp": 0.01,
        },
    ])

    assert summary["rows"] == 1
    assert summary["unique_fixtures"] == 1
    assert summary["avg_probability_clv_pp"] == 0.01


def test_v4_019_role_line_calibration_summary():
    metrics = v.role_line_calibration({
        "HOME:0.5:OVER": {"n": 100, "mean_probability": 0.70, "observed_rate": 0.80},
        "AWAY:0.5:OVER": {"n": 100, "mean_probability": 0.65, "observed_rate": 0.70},
    })
    assert metrics["n"] == 200
    assert metrics["weighted_absolute_calibration_gap"] == 0.075
    assert metrics["max_absolute_calibration_gap"] == 0.1


def test_v4_019_becomes_review_eligible_only_with_real_clv_and_calibration():
    validation = {
        "evaluated_fixtures": 500,
        "evaluated_probability_rows": 6000,
        "by_line": {
            "0.5": {"n": 2000},
            "1.5": {"n": 2000},
            "2.5": {"n": 2000},
        },
        "by_role_line_selection": {
            "HOME:0.5:OVER": {"n": 500, "mean_probability": 0.72, "observed_rate": 0.80},
            "AWAY:0.5:OVER": {"n": 500, "mean_probability": 0.67, "observed_rate": 0.74},
            "HOME:1.5:OVER": {"n": 500, "mean_probability": 0.48, "observed_rate": 0.53},
            "AWAY:1.5:OVER": {"n": 500, "mean_probability": 0.42, "observed_rate": 0.47},
        },
        "promotion_gate": {"enabled": False},
    }
    clv_rows = [
        {
            "market": "Home Team Total Goals" if i % 2 == 0 else "Away Team Total Goals",
            "fixture_id": i,
            "clv_probability_pp": 0.01,
        }
        for i in range(50)
    ]

    report = v.build_report(validation, clv_rows)

    assert report["status"] == "OOS_REVIEW_ELIGIBLE"
    assert report["blockers"] == []
    assert report["review_gate"]["review_eligible"] is True
    assert report["review_gate"]["true_clv_sample_ready"] is True
    assert report["review_gate"]["role_line_calibration_ready"] is True
    assert report["source_promotion_gate"]["enabled"] is False
    assert report["production_promotion_allowed"] is False
    assert report["manual_review_required"] is True


def test_v4_019_blocks_calibration_gap_at_or_above_ten_points():
    validation = {
        "evaluated_fixtures": 500,
        "evaluated_probability_rows": 6000,
        "by_line": {
            "0.5": {"n": 2000},
            "1.5": {"n": 2000},
            "2.5": {"n": 2000},
        },
        "by_role_line_selection": {
            "HOME:0.5:OVER": {"n": 500, "mean_probability": 0.70, "observed_rate": 0.81},
        },
        "promotion_gate": {"enabled": False},
    }
    clv_rows = [
        {"market": "Home Team Total Goals", "fixture_id": i, "clv_probability_pp": 0.01}
        for i in range(50)
    ]

    report = v.build_report(validation, clv_rows)

    assert report["status"] == "RESEARCH_HOLD"
    assert any(blocker.startswith("ROLE_LINE_CALIBRATION_MAX_GAP_") for blocker in report["blockers"])
    assert report["review_gate"]["role_line_calibration_ready"] is False
    assert report["production_promotion_allowed"] is False
