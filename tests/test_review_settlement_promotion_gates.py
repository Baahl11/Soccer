from __future__ import annotations

from mcp_gateway.review_settlement_promotion_gates import build_report, clv_review


def _segments() -> dict:
    return {
        "status": "SEGMENTATION_RESEARCH_ONLY",
        "dimensions": {
            "market_family__stage_bucket": {
                "FT_TOTALS|T_MINUS_0_30": {"settled": 12, "roi_units": 3.0, "roi_per_settled_units": 0.25, "hit_rate_ex_push": 0.65},
                "FT_TOTALS|T_MINUS_31_60": {"settled": 11, "roi_units": 2.0, "roi_per_settled_units": 0.18, "hit_rate_ex_push": 0.60},
            },
            "market_family__league": {
                "FT_TOTALS|39|Premier League": {"settled": 15, "roi_units": 4.0, "roi_per_settled_units": 0.27, "hit_rate_ex_push": 0.66},
                "FT_TOTALS|140|LaLiga": {"settled": 12, "roi_units": 1.0, "roi_per_settled_units": 0.08, "hit_rate_ex_push": 0.58},
            },
        },
    }


def test_clv_review_blocks_negative_true_clv() -> None:
    review = clv_review({"true_clv_rows": 80, "avg_true_clv_probability_pp": -0.2}, {})
    assert review["status"] == "CLV_NEGATIVE_WARNING"


def test_small_sample_is_hold_review() -> None:
    report = build_report(
        _segments(),
        {"by_market_family": {"FT_TOTALS": {"n": 12, "settled": 12, "hit_rate_ex_push": 0.7, "roi_units": 5.0}}},
        {"true_clv_rows": 80, "avg_true_clv_probability_pp": 0.1, "status": "ACTIVE_TRUE_CLV_SAMPLE"},
        {},
    )
    item = report["market_family_reviews"][0]
    assert item["recommendation"] == "HOLD_OR_DEMOTE_REVIEW"
    assert "sample_below_tier_b_minimum" in item["blockers"]


def test_negative_roi_is_hold_review() -> None:
    report = build_report(
        _segments(),
        {"by_market_family": {"FT_TOTALS": {"n": 25, "settled": 25, "hit_rate_ex_push": 0.48, "roi_units": -1.5}}},
        {"true_clv_rows": 80, "avg_true_clv_probability_pp": 0.1, "status": "ACTIVE_TRUE_CLV_SAMPLE"},
        {},
    )
    item = report["market_family_reviews"][0]
    assert item["recommendation"] == "HOLD_OR_DEMOTE_REVIEW"
    assert "non_positive_roi" in item["blockers"]


def test_positive_sample_without_true_clv_is_tier_b_only() -> None:
    report = build_report(
        _segments(),
        {"by_market_family": {"FT_TOTALS": {"n": 35, "settled": 35, "hit_rate_ex_push": 0.62, "roi_units": 8.0}}},
        {},
        {"tracked_signal_rows": 300, "positive_clv_rate": 0.03},
    )
    item = report["market_family_reviews"][0]
    assert item["recommendation"] == "TIER_B_CANDIDATE_MANUAL_REVIEW"
    assert "clv_not_strong_enough_for_tier_a_or_s" in item["warnings"]


def test_tier_a_candidate_requires_non_negative_true_clv() -> None:
    report = build_report(
        _segments(),
        {"by_market_family": {"FT_TOTALS": {"n": 60, "settled": 60, "hit_rate_ex_push": 0.61, "roi_units": 12.0}}},
        {"true_clv_rows": 80, "avg_true_clv_probability_pp": 0.05, "status": "ACTIVE_TRUE_CLV_SAMPLE"},
        {},
    )
    item = report["market_family_reviews"][0]
    assert item["recommendation"] == "TIER_A_CANDIDATE_MANUAL_REVIEW"


def test_tier_s_candidate_requires_large_true_clv_and_stability() -> None:
    report = build_report(
        _segments(),
        {"by_market_family": {"FT_TOTALS": {"n": 130, "settled": 130, "hit_rate_ex_push": 0.59, "roi_units": 20.0}}},
        {"true_clv_rows": 220, "avg_true_clv_probability_pp": 0.08, "status": "ACTIVE_TRUE_CLV_SAMPLE"},
        {},
    )
    item = report["market_family_reviews"][0]
    assert item["recommendation"] == "TIER_S_CANDIDATE_MANUAL_REVIEW"
