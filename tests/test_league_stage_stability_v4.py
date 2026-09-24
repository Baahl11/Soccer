from mcp_gateway import league_stage_stability_v4 as v


def _row(fixture_id, league, stage, family, clv, price_clv=None):
    return {
        "fixture_id": fixture_id,
        "league": league,
        "stage": stage,
        "market_family": family,
        "probability_clv": clv,
        "price_clv": price_clv,
    }


def test_fixture_weighting_prevents_multi_line_sample_inflation():
    rows = []
    for fixture_id in range(1, 11):
        for _ in range(20):
            rows.append(_row(fixture_id, "League A", "T-10", "1H", 1.0))
    report = v.build_report(rows)
    family = report["families"]["1H"]

    assert family["overall"]["rows"] == 200
    assert family["overall"]["unique_fixtures"] == 10
    assert family["overall"]["sample_status"] == "DATA_BLOCKED"
    assert family["status"] == "DATA_BLOCKED"


def test_directional_negative_stage_is_flagged_after_20_unique_fixtures():
    rows = []
    for fixture_id in range(1, 26):
        rows.append(_row(fixture_id, "League A", "T-20", "FT_CORNERS", -1.0))
    report = v.build_report(rows)
    family = report["families"]["FT_CORNERS"]

    assert family["status"] == "DIRECTIONAL_ONLY"
    assert family["negative_directional_stages"] == ["T-20"]


def test_review_ready_requires_50_unique_fixtures_and_no_negative_directional_segments():
    rows = []
    for fixture_id in range(1, 61):
        league = "League A" if fixture_id <= 30 else "League B"
        stage = "T-20" if fixture_id % 2 else "T-10"
        rows.append(_row(fixture_id, league, stage, "1X2", 0.8, 1.2))
    report = v.build_report(rows)
    family = report["families"]["1X2"]

    assert family["overall"]["unique_fixtures"] == 60
    assert family["status"] == "STABILITY_REVIEW_READY"
    assert family["max_league_fixture_share"] == 0.5
    assert report["provider_requests_added"] == 0
