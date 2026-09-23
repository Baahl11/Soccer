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
    assert "CORNERS_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_022_true_clv_matcher_is_corners_only():
    summary = v.summarize_true_clv([
        {"market": "Corners Over/Under", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Home Team Corners", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
