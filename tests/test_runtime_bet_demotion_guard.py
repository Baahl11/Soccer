from mcp_gateway import runtime_bet_demotion_guard as guard


def test_ft_totals_bet_demoted_to_watch():
    payload = {
        "events": [
            {
                "classification": "BET",
                "best_market": {"market": "Goals Over/Under", "selection": "under", "line": 2.5},
                "stake_units": 1.0,
                "bet_eligible": True,
                "tier": "B",
            }
        ]
    }
    metrics = guard.apply(payload)
    row = payload["events"][0]
    assert metrics["events_demoted"] == 1
    assert row["classification"] == "WATCH"
    assert row["original_classification"] == "BET"
    assert row["bet_eligible"] is False
    assert row["stake_units"] == 0.0
    assert row["tier"] is None
    assert row["runtime_demotion_guard"]["market_family"] == "FT_TOTALS"


def test_period_bet_markets_are_demoted():
    payload = {
        "events": [
            {"classification": "BET", "best_market": {"market": "Goals Over/Under - Second Half", "selection": "over", "line": 2.5}},
            {"classification": "BET", "best_market": {"market": "Both Teams To Score - Second Half", "selection": "yes"}},
            {"classification": "BET", "best_market": {"market": "Both Teams Score - First Half", "selection": "yes"}},
        ]
    }
    metrics = guard.apply(payload)
    assert metrics["events_demoted"] == 3
    assert [row["classification"] for row in payload["events"]] == ["WATCH", "WATCH", "WATCH"]
    assert [row["runtime_demotion_guard"]["market_family"] for row in payload["events"]] == [
        "2H_TOTALS",
        "2H_BTTS",
        "1H_OTHER",
    ]


def test_lean_rows_are_preserved_even_for_ft_totals():
    payload = {
        "events": [
            {
                "classification": "LEAN",
                "best_market": {"market": "Goals Over/Under", "selection": "over", "line": 2.5},
                "stake_units": 0.0,
                "bet_eligible": False,
            }
        ]
    }
    metrics = guard.apply(payload)
    row = payload["events"][0]
    assert metrics["events_demoted"] == 0
    assert row["classification"] == "LEAN"
    assert "runtime_demotion_guard" not in row


def test_match_table_rows_are_demoted_too():
    payload = {
        "match_table_rows": [
            {
                "classification": "BET",
                "market": "Goals Over/Under",
                "selection": "under",
                "line": 2.5,
                "stake_units": 1.0,
                "tier": "B",
            }
        ]
    }
    metrics = guard.apply(payload)
    row = payload["match_table_rows"][0]
    assert metrics["match_table_rows_demoted"] == 1
    assert row["classification"] == "WATCH"
    assert row["stake_units"] == 0.0
    assert row["tier"] is None


def test_non_target_bet_family_is_preserved():
    payload = {
        "events": [
            {"classification": "BET", "best_market": {"market": "Match Winner", "selection": "home"}}
        ]
    }
    metrics = guard.apply(payload)
    assert metrics["events_demoted"] == 0
    assert payload["events"][0]["classification"] == "BET"
