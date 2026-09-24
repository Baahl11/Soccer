from mcp_gateway import one_x_two_validation_v4 as v


def test_v4_017_blocks_when_challenger_worsens_scores():
    report = v.build_report(
        {
            "sample_fixtures": 500,
            "top1_accuracy": 0.466,
            "multiclass_brier": 0.6446,
            "multiclass_log_loss": 1.072,
            "calibration_by_outcome": {"D": {"20-30%": {"n": 331}}},
        },
        {
            "baseline": {"brier": 0.6487, "log_loss": 1.0779},
            "challenger": {"brier": 0.6499, "log_loss": 1.0801},
            "improvement": {"brier_delta": 0.0012, "log_loss_delta": 0.0022, "accuracy_delta_pp": 0.43},
        },
        {"by_market_family": {"FT_1X2": {"settled": 10, "roi_units": 11.2, "hit_rate_ex_push": 0.5}}},
        [{"market": "Match Winner", "fixture_id": i, "clv_probability_pp": 0.0} for i in range(60)],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert "CALIBRATION_CHALLENGER_DOES_NOT_IMPROVE_BRIER_AND_LOG_LOSS" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_v4_017_true_clv_is_1x2_only():
    summary = v.summarize_true_clv([
        {"market": "Match Winner", "fixture_id": 1, "clv_probability_pp": 0.02},
        {"market": "Both Teams Score", "fixture_id": 2, "clv_probability_pp": 0.20},
        {"market": "Winner", "fixture_id": 3, "clv_probability_pp": -0.01},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.005


def test_v4_017_accepts_canonical_multiclass_oos_challenger_but_keeps_clv_gate():
    report = v.build_report(
        {
            "sample_fixtures": 500,
            "top1_accuracy": 0.466,
            "multiclass_brier": 0.6446,
            "multiclass_log_loss": 1.072,
            "calibration_by_outcome": {"D": {"20-30%": {"n": 331}}},
        },
        {
            "baseline": {"brier": 0.6487, "log_loss": 1.0779},
            "challenger": {"brier": 0.6499, "log_loss": 1.0801},
            "improvement": {"brier_delta": 0.0012, "log_loss_delta": 0.0022, "accuracy_delta_pp": 0.43},
        },
        {"by_market_family": {"FT_1X2": {"settled": 10, "roi_units": 11.2, "hit_rate_ex_push": 0.5}}},
        [{"market": "Match Winner", "fixture_id": i, "clv_probability_pp": 0.01} for i in range(27)],
        {
            "model_version": "SOCCER_1X2_MULTICLASS_OOS_V4_1.0.3",
            "status": "RESEARCH_MULTICLASS_CALIBRATION_AVAILABLE",
            "source_model_version": "SOCCER EDGE ENGINE v1.7",
            "source_rows_current_model": 407,
            "evaluated_rows": 207,
            "walk_forward_folds": 5,
            "baseline": {"multiclass_brier": 0.67090004, "multiclass_log_loss": 1.10764206, "n": 207},
            "temperature_scaled": {"multiclass_brier": 0.65940541, "multiclass_log_loss": 1.08928822, "n": 207},
            "brier_delta": -0.01149463,
            "log_loss_delta": -0.01835384,
            "improves_brier_and_log_loss": True,
        },
    )
    assert report["canonical_multiclass_oos"]["available"] is True
    assert "CALIBRATION_CHALLENGER_DOES_NOT_IMPROVE_BRIER_AND_LOG_LOSS" not in report["blockers"]
    assert "MULTICLASS_OOS_CHALLENGER_NOT_READY" not in report["blockers"]
    assert "1X2_TRUE_CLV_27_LT_50" in report["blockers"]
    assert report["calibration_sample"]["n"] == 407
    assert report["status"] == "RESEARCH_HOLD"
    assert report["production_promotion_allowed"] is False
