import asyncio

from mcp_gateway import automation_v125
from mcp_gateway import btts_paid_odds_intelligence as btts
from mcp_gateway import price_resolver_v4


def _market(yes_price=2.0, no_price=1.8):
    return {
        "fixture_id": 101,
        "bookmaker_id": 8,
        "bookmaker": "Book",
        "market_id": 8,
        "market": "Both Teams Score",
        "provider_update": "2026-09-29T20:00:00+00:00",
        "source": "API_FOOTBALL_ODDS_V3",
        "values": [
            {"selection": "Yes", "decimal_price": yes_price},
            {"selection": "No", "decimal_price": no_price},
        ],
    }


def _event():
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "T-20",
        "classification": "WATCH",
        "fixture": {
            "fixture_id": 101,
            "kickoff": "2026-09-29T21:00:00+00:00",
            "league": "League",
            "country": "Country",
            "home_team": "Home",
            "away_team": "Away",
            "status": "NS",
        },
        "coverage": {"data_tier": "B"},
        "raw_projection": {"raw_btts_yes_prob": 0.57},
        "sporting_shortlist": {
            "tracks": ["SIDE", "TWO_WAY"],
            "side_edge_score": 71.0,
            "goal_environment_score": 74.0,
            "two_way_scoring_score": 82.0,
            "rank": 82.0,
        },
    }


def test_strict_btts_market_and_devig_offer():
    assert btts.is_strict_btts_market(_market()) is True
    assert btts.is_strict_btts_market({"market": "Both Teams Score - First Half"}) is False
    offer = btts._offer_from_market(_market(2.0, 1.8))
    assert offer is not None
    assert offer["selection"] == "yes"
    assert offer["decimal_price"] == 2.0
    assert round(offer["p_market_fair"], 6) == round((1 / 2.0) / ((1 / 2.0) + (1 / 1.8)), 6)


def test_attach_adds_one_research_only_priced_btts_row():
    payload = {"events": [_event()], "match_table_rows": []}
    result = btts.attach(payload, {101: [_market()]})
    assert result["entry_rows_added"] == 1
    assert result["provider_requests_added"] == 0
    assert result["production_promotion_allowed"] is False
    row = payload["match_table_rows"][0]
    assert row["fixture_id"] == 101
    assert row["market_family"] == "BTTS"
    assert row["market"] == "Both Teams Score"
    assert row["selection"] == "yes"
    assert row["price"] == 2.0
    assert row["classification"] == "WATCH"
    assert row["bet_eligible"] is False
    assert row["market_use"] == "BTTS_TRUE_CLV_ENTRY_RESEARCH_ONLY"
    intel = payload["events"][0]["btts_paid_odds_intelligence"]
    assert intel["observed_market_rows"][0]["p_raw"] == 0.57


def test_attach_does_not_duplicate_existing_priced_btts_row():
    payload = {
        "events": [_event()],
        "match_table_rows": [
            {
                "fixture_id": 101,
                "stage": "T-20",
                "classification": "WATCH",
                "market_family": "BTTS",
                "market": "Both Teams Score",
                "selection": "yes",
                "price": 1.95,
                "bookmaker": "Book",
            }
        ],
    }
    result = btts.attach(payload, {101: [_market()]})
    assert result["entry_rows_added"] == 0
    assert result["already_priced_btts_rows"] == 1
    assert len(payload["match_table_rows"]) == 1


def test_attach_requires_existing_two_way_signal():
    event = _event()
    event["sporting_shortlist"]["tracks"] = ["SIDE"]
    payload = {"events": [event], "match_table_rows": []}
    result = btts.attach(payload, {101: [_market()]})
    assert result["entry_rows_added"] == 0
    assert result["missing_signal_fixtures"] == 1


def test_v207_wrapper_observes_existing_fetch_without_adding_calls(monkeypatch):
    calls = {"count": 0}

    async def fake_fetch(_client, fixture_id, *, api_key, remaining_calls):
        calls["count"] += 1
        assert fixture_id == 101
        return [_market()], 1, "PRICE_API_RESOLVED", 4000

    async def fake_upstream_tick():
        await price_resolver_v4._fetch_fixture_odds(
            None,
            101,
            api_key="test",
            remaining_calls=1,
        )
        return {"events": [_event()], "match_table_rows": []}

    monkeypatch.setattr(price_resolver_v4, "_fetch_fixture_odds", fake_fetch)
    monkeypatch.setattr(automation_v125.v124, "run_tick", fake_upstream_tick)

    payload = asyncio.run(automation_v125.run_tick())
    assert calls["count"] == 1
    assert payload["version"] == "4.34.0-btts-paid-entry"
    assert payload["v207_btts_paid_odds_entry_capture"]["entry_rows_added"] == 1
    assert payload["v207_btts_paid_odds_entry_capture"]["provider_requests_added"] == 0
    assert price_resolver_v4._fetch_fixture_odds is fake_fetch
