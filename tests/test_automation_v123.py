import asyncio
from datetime import datetime, timezone

from mcp_gateway import automation as base
from mcp_gateway import automation_v123 as v
from mcp_gateway import automation_v7 as v7


def test_v123_exposes_spillover_checkpoint_and_ft_team_totals(monkeypatch):
    payload = {
        "events": [{
            "fixture": {
                "fixture_id": 1001,
                "kickoff": "2026-09-24T23:00:00+00:00",
                "home_team": "Home",
                "away_team": "Away",
                "home_team_id": 10,
                "away_team_id": 20,
            },
            "raw_projection": {
                "raw_home_goal_rate": 1.7,
                "raw_away_goal_rate": 1.1,
            },
            "stage": "T-40",
            "event_type": "SOCCER_REFRESH",
            "classification": "WATCH",
        }],
        "match_table_rows": [],
    }

    async def fake_run_tick():
        return payload

    async def fake_resolve(target, *, max_api_calls=None):
        assert max_api_calls == 25
        target["events"][0]["market"] = {
            "source": "API_FOOTBALL_ODDS_V3",
            "resolution_status": "PRICE_CACHE_HIT_RESEARCH_ONLY",
            "markets": [{
                "market": "Total - Home",
                "market_id": 16,
                "bookmaker": "1xBet",
                "values": [
                    {"selection": "Over", "line": 1.5, "decimal_price": 1.83},
                    {"selection": "Under", "line": 1.5, "decimal_price": 1.80},
                ],
            }],
        }
        target["price_resolution_v4"] = {
            "status": "ACTIVE_PRICE_RESOLVER",
            "candidate_rows": 0,
            "unique_candidate_fixtures": 0,
            "api_calls_added": 0,
            "resolution_counts": {},
            "research_spillover_candidate_fixtures": 1,
            "research_spillover_cache_hits": 1,
            "research_spillover_api_calls_added": 0,
            "research_spillover_fixtures_fetched": 0,
            "research_spillover_market_rows_fetched": 0,
        }
        return target["price_resolution_v4"]

    monkeypatch.setattr(v.v121, "run_tick", fake_run_tick)
    monkeypatch.setattr(v.price_resolver_v4, "resolve_payload", fake_resolve)

    out = asyncio.run(v.run_tick())

    checkpoint = out["price_resolution_checkpoint"]
    assert checkpoint["research_spillover_candidate_fixtures"] == 1
    assert checkpoint["research_spillover_cache_hits"] == 1
    assert checkpoint["research_spillover_api_calls_added"] == 0
    assert checkpoint["research_spillover_fixtures_fetched"] == 0
    assert out["team_totals_post_resolution"]["observed_exact_market_rows"] == 2
    rows = out["events"][0]["team_totals_intelligence"]["observed_exact_market_rows"]
    assert {row["team_role"] for row in rows} == {"HOME"}
    assert {row["market"] for row in rows} == {"Total - Home"}
    assert out["price_resolver_leftover_budget"] == 25
    assert out["version"] == "4.31.8-team-totals-diversity-catchup"


def test_v123_price_budget_is_global_leftover():
    payload = {
        "api_calls_this_tick": 63,
        "effective_max_api_calls_per_tick": 70,
        "max_api_calls_per_tick": 90,
    }
    assert v._leftover_price_budget(payload) == 7

    payload["api_calls_this_tick"] = 75
    assert v._leftover_price_budget(payload) == 0

    payload = {"api_calls_this_tick": 3, "max_api_calls_per_tick": 70}
    assert v._leftover_price_budget(payload) == 25



def test_v123_diversity_catchup_adds_only_guarded_daily_surplus():
    plan = v._price_budget_plan({
        "api_calls_this_tick": 70,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 7002,
        "daily_budget_mode": "NORMAL",
    })
    assert plan["standard_leftover_budget"] == 0
    assert plan["diversity_catchup_overflow_budget"] == 20
    assert plan["total_price_resolver_budget"] == 20
    assert plan["primary_tick_cap_unchanged"] is True

    partial = v._price_budget_plan({
        "api_calls_this_tick": 63,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 7002,
        "daily_budget_mode": "NORMAL",
    })
    assert partial["standard_leftover_budget"] == 7
    assert partial["diversity_catchup_overflow_budget"] == 18
    assert partial["total_price_resolver_budget"] == 25


def test_v123_diversity_catchup_is_disabled_outside_normal_budget_mode_or_floor():
    low_daily = v._price_budget_plan({
        "api_calls_this_tick": 70,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 5000,
        "daily_budget_mode": "NORMAL",
    })
    assert low_daily["diversity_catchup_overflow_budget"] == 0
    assert low_daily["total_price_resolver_budget"] == 0

    reduced = v._price_budget_plan({
        "api_calls_this_tick": 70,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 7002,
        "daily_budget_mode": "REDUCED",
    })
    assert reduced["diversity_catchup_overflow_budget"] == 0
    assert reduced["total_price_resolver_budget"] == 0


def test_v7_exposes_only_future_upcoming_market_capture_fixtures():
    now = datetime(2026, 9, 25, 4, 0, tzinfo=timezone.utc)
    fixtures = [
        {
            "fixture_id": 1,
            "kickoff": "2026-09-25T05:00:00+00:00",
            "status": "NS",
        },
        {
            "fixture_id": 2,
            "kickoff": "2026-09-25T03:00:00+00:00",
            "status": "FT",
        },
        {
            "fixture_id": 3,
            "kickoff": "2026-09-25T06:00:00+00:00",
            "status": "PST",
        },
        {
            "fixture_id": 4,
            "kickoff": "2026-09-25T04:30:00+00:00",
            "status": "NS",
        },
    ]

    rows = v7._upcoming_market_capture_fixtures(fixtures, now)

    assert [row["fixture_id"] for row in rows] == [4, 1]



def test_primary_odds_compactor_preserves_api_football_ft_team_totals_without_displacing_primary():
    primary_bets = [
        {
            "id": 1000 + i,
            "name": f"Corners Market {i}",
            "values": [{"value": "Over 8.5", "odd": "1.90"}],
        }
        for i in range(70)
    ]
    payload = {
        "response": [{
            "update": "2026-09-25T04:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": primary_bets + [
                    {
                        "id": 16,
                        "name": "Total - Home",
                        "values": [
                            {"value": "Over 1.5", "odd": "1.85"},
                            {"value": "Under 1.5", "odd": "1.95"},
                        ],
                    },
                    {
                        "id": 17,
                        "name": "Total - Away",
                        "values": [
                            {"value": "Over 0.5", "odd": "1.70"},
                            {"value": "Under 0.5", "odd": "2.10"},
                        ],
                    },
                ],
            }],
        }],
    }

    compact = base._compact_odds(payload)

    assert compact["primary_market_rows"] == 60
    assert compact["ft_team_total_rows"] == 2
    assert compact["ft_team_totals_reused_from_same_provider_response"] is True
    assert len(compact["markets"]) == 62
    assert {row["market_id"] for row in compact["markets"][-2:]} == {16, 17}
