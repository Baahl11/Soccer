from __future__ import annotations

from mcp_gateway.analyze_settlement_segments import build_report, odds_band, stage_bucket


def test_stage_bucket_groups_urgent_windows() -> None:
    assert stage_bucket("T-10") == "T_MINUS_0_30"
    assert stage_bucket("T-45") == "T_MINUS_31_60"
    assert stage_bucket("T-90") == "T_MINUS_61_90"
    assert stage_bucket("EARLY_RESEARCH") == "EARLY_RESEARCH"


def test_odds_band() -> None:
    assert odds_band(None) == "NO_PRICE"
    assert odds_band(1.42) == "LT_1_50"
    assert odds_band(1.66) == "1_50_1_79"
    assert odds_band(1.95) == "1_80_2_19"
    assert odds_band(2.5) == "2_20_2_99"
    assert odds_band(3.1) == "GE_3_00"


def test_build_report_segments_by_market_class_stage_odds() -> None:
    rows = [
        {
            "market_family": "FT_TOTALS",
            "classification": "LEAN",
            "stage": "T-20",
            "decimal_price": 1.91,
            "data_tier": "A",
            "league_id": 39,
            "league": "Premier League",
            "settlement_status": "WIN",
            "roi_units": 0.91,
        },
        {
            "market_family": "FT_TOTALS",
            "classification": "LEAN",
            "stage": "T-20",
            "decimal_price": 1.91,
            "data_tier": "A",
            "league_id": 39,
            "league": "Premier League",
            "settlement_status": "LOSS",
            "roi_units": -1.0,
        },
        {
            "market_family": "FT_BTTS",
            "classification": "BET",
            "stage": "T-60",
            "decimal_price": 2.25,
            "data_tier": "B",
            "league_id": 140,
            "league": "La Liga",
            "settlement_status": "PUSH",
            "roi_units": 0.0,
        },
    ]
    report = build_report(rows)
    assert report["input_rows"] == 3
    assert report["settled_rows"] == 3
    totals_lean = report["dimensions"]["market_family__classification"]["FT_TOTALS|LEAN"]
    assert totals_lean["n"] == 2
    assert totals_lean["settled"] == 2
    assert totals_lean["hit_rate_ex_push"] == 0.5
    assert report["dimensions"]["market_family__odds_band"]["FT_TOTALS|1_80_2_19"]["n"] == 2
