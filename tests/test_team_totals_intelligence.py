from mcp_gateway import team_totals_intelligence as tti


def _base_event(values):
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-20",
        "fixture": {
            "fixture_id": 123,
            "home_team": "Home FC",
            "away_team": "Away FC",
            "home_team_id": 1,
            "away_team_id": 2,
        },
        "raw_projection": {
            "raw_home_goal_rate": 1.8,
            "raw_away_goal_rate": 1.1,
        },
        "market_provenance": {
            "fresh": True,
            "source": "TEST",
        },
        "market": {
            "markets": [
                {
                    "market": "Home Team Total Goals",
                    "bookmaker": "Book",
                    "bookmaker_id": 99,
                    "market_id": 10,
                    "values": values,
                }
            ]
        },
    }


def test_team_totals_accepts_normalized_selection_line_decimal_price():
    event = _base_event([
        {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
        {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
    ])
    report = tti.build(event)
    rows = report["observed_exact_market_rows"]

    assert report["observed_exact_market_count"] == 2
    assert {(row["selection"], row["line"]) for row in rows} == {
        ("OVER", 1.5),
        ("UNDER", 1.5),
    }
    assert {row["decimal_price"] for row in rows} == {1.95, 1.85}
    assert all(row["team_role"] == "HOME" for row in rows)


def test_team_totals_keeps_legacy_embedded_line_price_support():
    event = _base_event([
        {"selection": "Over 1.5", "price": 1.95},
        {"selection": "Under 1.5", "price": 1.85},
    ])
    report = tti.build(event)
    rows = report["observed_exact_market_rows"]

    assert report["observed_exact_market_count"] == 2
    assert {(row["selection"], row["line"]) for row in rows} == {
        ("OVER", 1.5),
        ("UNDER", 1.5),
    }


def test_team_totals_rejects_period_and_non_goal_team_total_markets():
    event = _base_event([
        {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
        {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
    ])
    event["market"]["markets"] = [
        {
            "market": "Home Team Total Cards",
            "bookmaker": "Book",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
            ],
        },
        {
            "market": "Away Team Total Corners",
            "bookmaker": "Book",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
            ],
        },
        {
            "market": "Home Team Total Goals - First Half",
            "bookmaker": "Book",
            "values": [
                {"selection": "Over", "line": 1.5, "decimal_price": 1.95},
                {"selection": "Under", "line": 1.5, "decimal_price": 1.85},
            ],
        },
    ]

    report = tti.build(event)

    assert report["observed_exact_market_count"] == 0
    assert report["observed_exact_market_rows"] == []
