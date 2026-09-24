from mcp_gateway import analyze_corners_clv_concentration as v


def test_corners_clv_concentration_separates_ft_and_team_fixtures():
    report = v.build_report([
        {"fixture_id": 1, "market_family": "FT_CORNERS", "market": "Corners Over/Under", "line": 9.5, "selection": "OVER", "bookmaker": "A"},
        {"fixture_id": 1, "market_family": "FT_CORNERS", "market": "Corners Over/Under", "line": 10.5, "selection": "UNDER", "bookmaker": "A"},
        {"fixture_id": 2, "market_family": "TEAM_CORNERS", "market": "Home Team Corners", "line": 4.5, "selection": "OVER", "bookmaker": "B"},
        {"fixture_id": 3, "market_family": "FT_TOTALS", "market": "Goals Over/Under", "line": 2.5},
    ])
    assert report["families"]["FT_CORNERS"]["rows"] == 2
    assert report["families"]["FT_CORNERS"]["unique_fixtures"] == 1
    assert report["families"]["FT_CORNERS"]["rows_per_fixture"]["max"] == 2
    assert report["families"]["TEAM_CORNERS"]["rows"] == 1
    assert report["families"]["TEAM_CORNERS"]["unique_fixtures"] == 1
    assert report["ignored_rows_by_family"]["FT_TOTALS"] == 1
    assert report["production_promotion_allowed"] is False
