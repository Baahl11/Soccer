from mcp_gateway import one_h_oos_v4 as v


def test_v4_020_blocks_current_challenger_and_missing_market_evidence():
    report = v.build_report(
        {
            "baseline_same_sample": {"n": 220, "brier": 0.2459, "log_loss": 0.6853},
            "challenger": {"n": 220, "brier": 0.2484, "log_loss": 0.6907},
            "promotion_gate": {"enabled": False},
        },
        [{"market": "Goals Over/Under", "fixture_id": 1, "clv_probability_pp": 0.1}],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "CHALLENGER_BRIER_NOT_BETTER_THAN_BASELINE" in report["blockers"]
    assert "CHALLENGER_LOG_LOSS_NOT_BETTER_THAN_BASELINE" in report["blockers"]
    assert "1H_TRUE_CLV_0_LT_50" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_020_true_clv_matcher_is_period_specific():
    summary = v.summarize_true_clv([
        {"market": "First Half Goals Over/Under", "selection": "Over 1.5", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "1H Total", "selection": "Under 1.5", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Goals Over/Under", "selection": "Over 2.5", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005
