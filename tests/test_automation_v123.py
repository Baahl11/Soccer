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
            "primary_clv_maturation_source": "TEST_PRIMARY",
            "primary_clv_maturation_candidates": 4,
            "primary_clv_maturation_candidate_family_counts": {"1X2": 2, "BTTS": 1, "FT_TOTALS": 1},
            "primary_clv_maturation_max_calls_per_tick": 8,
            "primary_clv_maturation_api_calls_added": 2,
            "primary_clv_maturation_fixtures_refreshed": 2,
            "primary_clv_maturation_family_refresh_counts": {"1X2": 1, "BTTS": 1},
            "primary_clv_maturation_cache_replays_ignored": 1,
            "primary_clv_maturation_unchanged_provider_updates": 1,
            "primary_clv_maturation_budget_exhausted": 0,
            "primary_clv_maturation_primary_payload_reuse_fixtures": 1,
            "primary_clv_maturation_synthetic_events_added": 1,
            "research_spillover_maturation_source": "TEST_MATURATION",
            "research_spillover_maturation_candidates": 3,
            "research_spillover_maturation_max_calls_per_tick": 12,
            "research_spillover_maturation_api_calls_added": 2,
            "research_spillover_maturation_later_real_quote_refreshes": 1,
            "research_spillover_maturation_cache_replays_ignored": 2,
            "research_spillover_maturation_unchanged_provider_updates": 1,
            "research_spillover_maturation_budget_exhausted": 0,
            "research_spillover_clv_maturation_continues_after_diversity_target": True,
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
    assert checkpoint["primary_clv_maturation_candidates"] == 4
    assert checkpoint["primary_clv_maturation_api_calls_added"] == 2
    assert checkpoint["primary_clv_maturation_fixtures_refreshed"] == 2
    assert checkpoint["primary_clv_maturation_family_refresh_counts"] == {"1X2": 1, "BTTS": 1}
    assert checkpoint["research_spillover_maturation_source"] == "TEST_MATURATION"
    assert checkpoint["research_spillover_maturation_candidates"] == 3
    assert checkpoint["research_spillover_maturation_api_calls_added"] == 2
    assert checkpoint["research_spillover_maturation_later_real_quote_refreshes"] == 1
    assert checkpoint["research_spillover_maturation_cache_replays_ignored"] == 2
    assert checkpoint["research_spillover_clv_maturation_continues_after_diversity_target"] is True
    assert out["team_totals_post_resolution"]["observed_exact_market_rows"] == 2
    rows = out["events"][0]["team_totals_intelligence"]["observed_exact_market_rows"]
    assert {row["team_role"] for row in rows} == {"HOME"}
    assert {row["market"] for row in rows} == {"Total - Home"}
    assert out["price_resolver_leftover_budget"] == 25
    expected_global_cap = v.v6._BASE_MAX_API_CALLS_PER_TICK
    expected_reserve = min(20, max(0, expected_global_cap - 1))
    assert out["pre_price_pipeline_api_cap"] == expected_global_cap - expected_reserve
    assert out["global_api_cap_after_daily_policy"] == expected_global_cap
    assert out["primary_price_reserve_calls"] == expected_reserve
    assert out["team_totals_diversity_catchup_overflow_budget"] == 0
    assert out["version"] == "4.32.7-dedicated-ht-research"


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



def test_v123_price_budget_plan_never_exceeds_global_tick_leftover():
    exhausted = v._price_budget_plan({
        "api_calls_this_tick": 70,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 7002,
        "daily_budget_mode": "NORMAL",
    })
    assert exhausted["standard_leftover_budget"] == 0
    assert exhausted["diversity_catchup_overflow_budget"] == 0
    assert exhausted["total_price_resolver_budget"] == 0
    assert exhausted["overflow_above_global_tick_cap_allowed"] is False

    partial = v._price_budget_plan({
        "api_calls_this_tick": 63,
        "effective_max_api_calls_per_tick": 70,
        "last_daily_remaining": 7002,
        "daily_budget_mode": "NORMAL",
    })
    assert partial["standard_leftover_budget"] == 7
    assert partial["diversity_catchup_overflow_budget"] == 0
    assert partial["total_price_resolver_budget"] == 7
    assert partial["overflow_above_global_tick_cap_allowed"] is False


def test_v123_reserves_capacity_inside_live_elastic_cap(monkeypatch):
    seen = {}
    original_elastic = v.v90._elastic_request_cap
    global_cap, _reason = original_elastic(7000)
    expected_pre_cap, expected_reserve = v._reserve_from_elastic_cap(global_cap)
    assert global_cap == 70
    assert expected_pre_cap == 50
    assert expected_reserve == 20

    async def fake_run_tick():
        upstream_cap, upstream_reason = v.v90._elastic_request_cap(7000)
        seen["pre_price_cap_during_upstream"] = upstream_cap
        seen["pre_price_reason_during_upstream"] = upstream_reason
        return {
            "events": [],
            "match_table_rows": [],
            "api_calls_this_tick": upstream_cap,
            "max_api_calls_per_tick": upstream_cap,
            "effective_max_api_calls_per_tick": upstream_cap,
            "elastic_request_cap": upstream_cap,
            "last_daily_remaining": 7000,
            "daily_budget_mode": "NORMAL",
        }

    async def fake_resolve(target, *, max_api_calls=None):
        seen["price_budget"] = max_api_calls
        target["price_resolution_v4"] = {
            "status": "ACTIVE_PRICE_RESOLVER",
            "candidate_rows": 0,
            "unique_candidate_fixtures": 0,
            "api_calls_added": 0,
            "resolution_counts": {},
        }
        return target["price_resolution_v4"]

    monkeypatch.setattr(v.v121, "run_tick", fake_run_tick)
    monkeypatch.setattr(v.price_resolver_v4, "resolve_payload", fake_resolve)

    out = asyncio.run(v.run_tick())

    assert seen["pre_price_cap_during_upstream"] == 50
    assert "PRICE_RESERVE_20" in seen["pre_price_reason_during_upstream"]
    assert seen["price_budget"] == 20
    assert v.v90._elastic_request_cap is original_elastic
    assert out["api_calls_this_tick"] == 50
    assert out["pre_price_pipeline_api_cap"] == 50
    assert out["elastic_request_cap_upstream_observed"] == 50
    assert out["global_api_cap_after_daily_policy"] == 70
    assert out["elastic_request_cap"] == 70
    assert out["primary_price_reserve_calls"] == 20
    assert out["effective_max_api_calls_per_tick"] == 70
    assert out["price_resolver_leftover_budget"] == 20
    assert out["team_totals_diversity_catchup_overflow_budget"] == 0


def test_v123_reserve_preserves_minimum_upstream_capacity():
    upstream_cap, reserve = v._reserve_from_elastic_cap(25)
    assert upstream_cap == 8
    assert reserve == 17

    upstream_cap, reserve = v._reserve_from_elastic_cap(70)
    assert upstream_cap == 50
    assert reserve == 20


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



def test_v7_halftime_handoff_requires_verified_ht_score():
    fixtures = [
        {
            "fixture_id": 1,
            "status": "HT",
            "score": {"halftime": {"home": 1, "away": 0}},
        },
        {
            "fixture_id": 2,
            "status": "NS",
            "score": {"halftime": {"home": 0, "away": 0}},
        },
        {
            "fixture_id": 3,
            "status": "HT",
            "score": {"halftime": {"home": None, "away": None}},
        },
    ]
    rows = v7._current_halftime_research_fixtures(fixtures)
    assert [row["fixture_id"] for row in rows] == [1]


def test_v123_adds_research_only_ht_event_without_provider_calls(monkeypatch):
    payload = {
        "events": [],
        "current_halftime_research_fixture_count": 1,
        "current_halftime_research_fixtures": [{
            "fixture_id": 9501,
            "status": "HT",
            "score": {"halftime": {"home": 1, "away": 1}},
            "league_id": 39,
            "season": 2026,
            "home_team_id": 1,
            "away_team_id": 2,
        }],
    }

    monkeypatch.setattr(v.halftime_2h_intelligence, "attach", lambda target: {
        "halftime_events": 1,
        "modeled_halftime_events": 1,
        "halftime_state_registry_loaded": True,
        "period_rate_registry_loaded": True,
        "provider_requests_added": 0,
    })

    result = v._attach_dedicated_ht_research(payload)

    assert result["status"] == "DEDICATED_HT_RESEARCH_STAGE_ACTIVE"
    assert result["synthetic_ht_events_added"] == 1
    assert result["provider_requests_added"] == 0
    assert result["decision_weight"] == 0.0
    assert result["production_promotion_allowed"] is False
    assert "current_halftime_research_fixtures" not in payload
    event = payload["events"][0]
    assert event["event_type"] == "SOCCER_REFRESH"
    assert event["stage"] == "HT"
    assert event["classification"] == "RESEARCH_ONLY"
    assert event["bet_eligible"] is False
    assert event["decision_weight"] == 0.0


def test_v123_ht_handoff_dedupes_existing_ht_event(monkeypatch):
    payload = {
        "events": [{
            "event_type": "SOCCER_REFRESH",
            "stage": "HT",
            "fixture": {
                "fixture_id": 9502,
                "status": "HT",
                "score": {"halftime": {"home": 0, "away": 0}},
            },
        }],
        "current_halftime_research_fixtures": [{
            "fixture_id": 9502,
            "status": "HT",
            "score": {"halftime": {"home": 0, "away": 0}},
        }],
    }
    monkeypatch.setattr(v.halftime_2h_intelligence, "attach", lambda target: {
        "halftime_events": 1,
        "modeled_halftime_events": 0,
        "provider_requests_added": 0,
    })
    result = v._attach_dedicated_ht_research(payload)
    assert result["synthetic_ht_events_added"] == 0
    assert len(payload["events"]) == 1
