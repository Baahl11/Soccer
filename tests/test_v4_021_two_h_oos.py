from mcp_gateway import two_h_oos_v4 as v


def test_v4_021_blocks_current_worse_challenger_and_missing_live_path():
    report = v.build_report(
        {
            "walk_forward_evaluated": 604,
            "baseline": {"brier_o1_5": 0.253033, "log_loss_o1_5": 0.699253, "mae_2h_lambda": 0.931935},
            "challenger": {"brier_o1_5": 0.254011, "log_loss_o1_5": 0.701232, "mae_2h_lambda": 0.933602},
            "improvement": {
                "brier_delta_baseline_minus_conditioned": -0.000978,
                "log_loss_delta_baseline_minus_conditioned": -0.001979,
                "mae_delta_baseline_minus_conditioned": -0.001667,
            },
            "live_status": "OFFLINE_MODEL_BUILT; SCHEDULER_HAS_NO_DEDICATED_HT_RESEARCH_STAGE_YET",
            "promotion_gate": {"enabled": False},
            "not_yet_conditioned_on": ["halftime red cards"],
        },
        [],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "HALFTIME_CONDITIONED_CHALLENGER_DOES_NOT_BEAT_BASELINE" in report["blockers"]
    assert "DEDICATED_HT_RESEARCH_STAGE_MISSING" in report["blockers"]
    assert "2H_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_021_true_clv_matcher_is_2h_only():
    summary = v.summarize_true_clv([
        {"market": "Second Half Goals Over/Under", "selection": "Over 1.5", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "2H Both Teams To Score", "selection": "Yes", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "selection": "Over 2.5", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
