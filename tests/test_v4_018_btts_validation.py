from mcp_gateway import btts_validation_v4 as v


def test_v4_018_blocks_when_true_clv_is_too_small():
    report = v.build_report(
        {
            "overall": {"n": 500, "brier": 0.276102, "log_loss": 0.771148, "mean_probability": 0.495624, "observed_rate": 0.588},
            "by_probability_bucket": {
                "40-49%": {"n": 100, "mean_probability": 0.45, "observed_rate": 0.56},
                "60-69%": {"n": 100, "mean_probability": 0.64, "observed_rate": 0.64},
            },
            "promotion_gate": {"enabled": False},
        },
        [{"market": "Both Teams Score", "fixture_id": i, "clv_probability_pp": 0.0} for i in range(25)],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "BTTS_TRUE_CLV_25_LT_50" in report["blockers"]
    assert "SOURCE_BTTS_PROMOTION_GATE_DISABLED" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_018_true_clv_is_btts_only():
    summary = v.summarize_true_clv([
        {"market": "Both Teams Score", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Both Teams To Score", "fixture_id": 2, "clv_probability_pp": -0.01},
        {"market": "Match Winner", "fixture_id": 3, "clv_probability_pp": 0.20},
    ])
    assert summary["rows"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005


def test_v4_018_calibration_error_is_weighted():
    metrics = v.calibration_error({
        "a": {"n": 100, "mean_probability": 0.4, "observed_rate": 0.5},
        "b": {"n": 50, "mean_probability": 0.8, "observed_rate": 0.7},
    })
    assert metrics["n"] == 150
    assert metrics["ece"] == 0.1
    assert metrics["mce"] == 0.1
