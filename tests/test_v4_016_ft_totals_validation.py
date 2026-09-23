from mcp_gateway import ft_totals_validation_v4 as v


def test_quarter_line_splits_and_settles_half_win():
    assert v.split_asian_line(2.25) == (2.0, 2.5)
    settled = v.settle_asian_total(3, "OVER", 2.25)
    assert settled["settlement"] == "WIN"
    settled = v.settle_asian_total(2, "OVER", 2.25)
    assert settled["settlement"] == "HALF_LOSS"


def test_quarter_line_half_win_under():
    settled = v.settle_asian_total(3, "UNDER", 3.25)
    assert settled["settlement"] == "HALF_WIN"
    assert v.settlement_return_units("HALF_WIN", 2.0) == 0.5
    assert v.settlement_return_units("HALF_LOSS", 2.0) == -0.5


def test_report_blocks_current_small_sample_and_no_ft_true_clv():
    report = v.build_report(
        {
            "actionable": {"n": 15, "mean_brier": 0.15, "mean_log_loss": 0.49},
            "by_line": {
                "1.5": {"n": 5},
                "2.5": {"n": 7},
                "3.5": {"n": 3},
            },
        },
        {
            "by_market_family": {
                "FT_TOTALS": {
                    "settled": 15,
                    "hit_rate_ex_push": 0.7333,
                    "roi_units": 3.34,
                }
            }
        },
        [
            {
                "market": "Match Winner",
                "clv_probability_pp": 0.01,
                "fixture_id": 1,
            }
        ],
    )
    assert report["status"] == "RESEARCH_HOLD"
    assert report["true_clv"]["rows"] == 0
    assert "V4_OOS_CALIBRATION_NOT_MATERIALIZED" in report["blockers"]
    assert report["production_promotion_allowed"] is False


def test_ft_true_clv_is_family_specific():
    summary = v.summarize_true_clv([
        {"market": "Goals Over/Under", "fixture_id": 1, "clv_probability_pp": 0.03},
        {"market": "Match Winner", "fixture_id": 1, "clv_probability_pp": 0.40},
        {"market": "Over/Under", "fixture_id": 2, "clv_probability_pp": -0.01},
    ])
    assert summary["rows"] == 2
    assert summary["unique_fixtures"] == 2
    assert summary["avg_probability_clv_pp"] == 0.01
