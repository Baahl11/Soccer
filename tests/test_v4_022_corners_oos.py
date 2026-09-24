from mcp_gateway import corners_oos_v4 as v


def test_v4_022_blocks_current_small_formation_sample_and_no_clv():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.612748,
                "lines": {
                    "8.5": {"brier": 0.253428, "log_loss": 0.701139},
                    "9.5": {"brier": 0.252232, "log_loss": 0.697662},
                    "10.5": {"brier": 0.221616, "log_loss": 0.63546},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.581188,
                "lines": {
                    "8.5": {"brier": 0.252486, "log_loss": 0.699201},
                    "9.5": {"brier": 0.248863, "log_loss": 0.690901},
                    "10.5": {"brier": 0.221125, "log_loss": 0.634773},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "promotion_gate": {"enabled": False},
        },
        [],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert report["ft_corners"]["all_required_lines_improve_brier_and_log_loss"] is True
    assert report["ft_corners"]["mae_improves"] is True
    assert "FORMATION_ADJUSTED_39_LT_100" in report["blockers"]
    assert "FT_CORNERS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert "TEAM_CORNERS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_022_true_clv_matcher_is_corners_only():
    rows = [
        {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "fixture_id": 3, "clv_probability_pp": 0.20},
    ]
    summary = v.summarize_true_clv(rows)
    ft = v.summarize_true_clv(rows, "FT_CORNERS")
    team = v.summarize_true_clv(rows, "TEAM_CORNERS")
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
    assert ft["rows"] == 1
    assert ft["unique_fixtures"] == 1
    assert ft["avg_probability_clv_pp"] == 0.02
    assert team["rows"] == 1
    assert team["unique_fixtures"] == 1
    assert team["avg_probability_clv_pp"] == -0.01


def test_v4_022_family_views_do_not_share_ft_and_team_clv():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.6,
                "lines": {
                    "8.5": {"brier": 0.25, "log_loss": 0.70},
                    "9.5": {"brier": 0.25, "log_loss": 0.70},
                    "10.5": {"brier": 0.22, "log_loss": 0.63},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.5,
                "lines": {
                    "8.5": {"brier": 0.24, "log_loss": 0.69},
                    "9.5": {"brier": 0.24, "log_loss": 0.69},
                    "10.5": {"brier": 0.21, "log_loss": 0.62},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "promotion_gate": {"enabled": False},
        },
        [
            *[
                {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
            *[
                {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(10)
            ],
        ],
    )
    assert report["family_views"]["FT_CORNERS"]["true_clv"]["rows"] == 60
    assert report["family_views"]["TEAM_CORNERS"]["true_clv"]["rows"] == 10
    assert "FT_CORNERS_TRUE_CLV_60_LT_50" not in report["family_views"]["FT_CORNERS"]["blockers"]
    assert "TEAM_CORNERS_TRUE_CLV_10_LT_50" in report["family_views"]["TEAM_CORNERS"]["blockers"]


def test_v4_022_replaces_historical_disabled_flags_with_explicit_evidence_blockers():
    report = v.build_report(
        {
            "walk_forward_evaluations": 244,
            "formation_adjusted_evaluations": 39,
            "baseline": {
                "mae_total_corners": 2.6,
                "lines": {
                    "8.5": {"brier": 0.25, "log_loss": 0.70},
                    "9.5": {"brier": 0.25, "log_loss": 0.70},
                    "10.5": {"brier": 0.22, "log_loss": 0.63},
                },
            },
            "formation_challenger": {
                "mae_total_corners": 2.5,
                "lines": {
                    "8.5": {"brier": 0.24, "log_loss": 0.69},
                    "9.5": {"brier": 0.24, "log_loss": 0.69},
                    "10.5": {"brier": 0.21, "log_loss": 0.62},
                },
            },
            "promotion_gate": {"enabled": False},
        },
        {
            "evaluated_fixtures": 244,
            "evaluated_rows": 1464,
            "by_role_line": {
                "HOME|3.5": {"n": 244},
                "HOME|4.5": {"n": 244},
                "HOME|5.5": {"n": 244},
                "AWAY|3.5": {"n": 244},
                "AWAY|4.5": {"n": 244},
                "AWAY|5.5": {"n": 244},
            },
            "by_league": {"46": {"n": 90}},
            "promotion_gate": {"enabled": False},
        },
        [
            *[
                {"market_family": "FT_CORNERS", "market": "Corners Over/Under", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
            *[
                {"market_family": "TEAM_CORNERS", "market": "Home Team Corners", "fixture_id": i, "clv_probability_pp": 0.1}
                for i in range(60)
            ],
        ],
    )

    ft_blockers = report["family_views"]["FT_CORNERS"]["blockers"]
    team_blockers = report["family_views"]["TEAM_CORNERS"]["blockers"]
    assert "SOURCE_FT_CORNERS_PROMOTION_GATE_DISABLED" not in ft_blockers
    assert "SOURCE_TEAM_CORNERS_PROMOTION_GATE_DISABLED" not in team_blockers
    assert "FORMATION_ADJUSTED_39_LT_100" in ft_blockers
    assert "FT_CORNERS_LEAGUE_LIFT_NOT_MATERIALIZED" in ft_blockers
    assert "PARENT_FT_CORNERS_NOT_REVIEW_READY" in team_blockers
    assert "TEAM_CORNERS_LEAGUE_VENUE_STABILITY_NOT_MATERIALIZED" in team_blockers
