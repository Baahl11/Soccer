import asyncio
from datetime import datetime, timezone

from mcp_gateway import automation as base
from mcp_gateway import automation_v123 as v
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v5 as v5
from mcp_gateway import automation_v7 as v7
from mcp_gateway import persistence as persistence_base


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
    assert out["version"] == "4.32.9-hard-budget-reserve"


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


def test_v123_configures_hard_v90_reserve_and_restores_it(monkeypatch):
    seen = {}
    original_reserve = v.v90._REQUEST_CAP_RESERVE_CALLS
    original_min_upstream = v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS

    async def fake_run_tick():
        seen["reserve_during_upstream"] = v.v90._REQUEST_CAP_RESERVE_CALLS
        seen["min_upstream_during_upstream"] = v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS
        global_cap, _ = v.v90._elastic_request_cap(7000)
        upstream_cap, reserve = v.v90._request_cap_with_reserve(global_cap)
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

    assert seen["reserve_during_upstream"] == 20
    assert seen["min_upstream_during_upstream"] == v.MIN_UPSTREAM_API_CALLS
    assert seen["price_budget"] == 20
    assert v.v90._REQUEST_CAP_RESERVE_CALLS == original_reserve
    assert v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS == original_min_upstream
    assert out["api_calls_this_tick"] == 50
    assert out["pre_price_pipeline_api_cap"] == 50
    assert out["elastic_request_cap_upstream_observed"] == 50
    assert out["global_api_cap_after_daily_policy"] == 70
    assert out["elastic_request_cap"] == 70
    assert out["primary_price_reserve_calls"] == 20
    assert out["effective_max_api_calls_per_tick"] == 70
    assert out["price_resolver_leftover_budget"] == 20
    assert out["team_totals_diversity_catchup_overflow_budget"] == 0


def test_v90_hard_gate_stops_upstream_at_reserved_cap(monkeypatch):
    original_adaptive = v.v90.v6._adaptive_paced_api_get
    original_run_tick = v.v90.v89.run_tick
    original_reserve = v.v90._REQUEST_CAP_RESERVE_CALLS
    original_min = v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS
    original_calls = v.v90.v2._API_CALLS_THIS_TICK
    original_remaining = v.v90.v2._LAST_DAILY_REMAINING
    original_cap = v.v90.v2.MAX_API_CALLS_PER_TICK

    async def fake_original_adaptive(endpoint, params):
        if v.v90.v2._API_CALLS_THIS_TICK >= v.v90.v2.MAX_API_CALLS_PER_TICK:
            raise v.v90.v2.TickBudgetExceeded("hard cap")
        v.v90.v2._API_CALLS_THIS_TICK += 1
        remaining = 4403 - v.v90.v2._API_CALLS_THIS_TICK
        v.v90.v2._LAST_DAILY_REMAINING = remaining
        return {"quota": {"daily_remaining": remaining}, "response": []}

    async def fake_v89_run_tick():
        v.v90.v2._API_CALLS_THIS_TICK = 0
        v.v90.v2._LAST_DAILY_REMAINING = None
        v.v90.v2.MAX_API_CALLS_PER_TICK = v.v90.v6._BASE_MAX_API_CALLS_PER_TICK
        while True:
            try:
                await v.v90.v6._adaptive_paced_api_get("fixtures", {"date": "2026-09-25"})
            except v.v90.v2.TickBudgetExceeded:
                break
        return {
            "events": [],
            "due_fixture_count": 0,
            "api_calls_this_tick": v.v90.v2._API_CALLS_THIS_TICK,
            "max_api_calls_per_tick": v.v90.v2.MAX_API_CALLS_PER_TICK,
            "last_daily_remaining": v.v90.v2._LAST_DAILY_REMAINING,
        }

    monkeypatch.setattr(v.v90.v6, "_adaptive_paced_api_get", fake_original_adaptive)
    monkeypatch.setattr(v.v90.v89, "run_tick", fake_v89_run_tick)
    v.v90._REQUEST_CAP_RESERVE_CALLS = 20
    v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS = 8
    try:
        out = asyncio.run(v.v90.run_tick())
    finally:
        v.v90._REQUEST_CAP_RESERVE_CALLS = original_reserve
        v.v90._REQUEST_CAP_MIN_UPSTREAM_CALLS = original_min
        v.v90.v6._adaptive_paced_api_get = original_adaptive
        v.v90.v89.run_tick = original_run_tick
        v.v90.v2._API_CALLS_THIS_TICK = original_calls
        v.v90.v2._LAST_DAILY_REMAINING = original_remaining
        v.v90.v2.MAX_API_CALLS_PER_TICK = original_cap

    assert out["api_calls_this_tick"] == 25
    assert out["elastic_global_request_cap"] == 45
    assert out["elastic_request_cap"] == 25
    assert out["elastic_request_cap_reserve"] == 20
    assert "PRICE_RESERVE_20" in out["elastic_request_cap_reason"]
    assert out["v4_005_runtime_request_cap_observed"] == 25


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



def test_odds_compactor_retains_cards_and_player_props_without_displacing_existing_rows():
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
            "update": "2026-09-25T06:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": primary_bets + [
                    {
                        "id": 16,
                        "name": "Total - Home",
                        "values": [{"value": "Over 1.5", "odd": "1.85"}],
                    },
                    {
                        "id": 200,
                        "name": "Total Yellow Cards",
                        "values": [{"value": "Over 4.5", "odd": "1.90"}],
                    },
                    {
                        "id": 201,
                        "name": "Player Shots",
                        "values": [{"value": "Player A Over 2.5", "odd": "1.95"}],
                    },
                    {
                        "id": 202,
                        "name": "Goalkeeper Saves",
                        "values": [{"value": "Keeper A Over 3.5", "odd": "1.88"}],
                    },
                    {
                        "id": 203,
                        "name": "Player Cards",
                        "values": [{"value": "Player B To Be Booked", "odd": "2.10"}],
                    },
                ],
            }],
        }],
    }

    compact = base._compact_odds(payload)

    assert compact["primary_market_rows"] == 60
    assert compact["ft_team_total_rows"] == 1
    assert compact["card_research_market_rows"] == 1
    assert compact["player_prop_research_market_rows"] == 3
    assert compact["research_derivative_sidecar_rows"] == 4
    assert compact["research_derivative_sidecar_provider_requests_added"] == 0
    canonical_names = [row["market"] for row in compact["markets"]]
    research_names = [row["market"] for row in compact["research_cards_props_markets"]]
    assert "Total - Home" in canonical_names
    assert "Total Yellow Cards" not in canonical_names
    assert "Player Shots" not in canonical_names
    assert "Goalkeeper Saves" not in canonical_names
    assert "Player Cards" not in canonical_names
    assert len(compact["markets"]) == 61
    assert set(research_names) == {
        "Total Yellow Cards",
        "Player Shots",
        "Goalkeeper Saves",
        "Player Cards",
    }
    assert all(row["research_only"] is True for row in compact["research_cards_props_markets"])
    assert all(row["decision_weight"] == 0.0 for row in compact["research_cards_props_markets"])
    player_shots = next(row for row in compact["research_cards_props_markets"] if row["market"] == "Player Shots")
    assert player_shots["values"][0]["parsed_line"] == 2.5
    player_cards = next(row for row in compact["research_cards_props_markets"] if row["market"] == "Player Cards")
    assert player_cards["values"][0]["parsed_line"] is None


def test_research_derivative_sidecar_summary_is_zero_call_and_research_only():
    payload = {
        "events": [
            {"market": {"card_research_market_rows": 2, "player_prop_research_market_rows": 3}},
            {"market": {"card_research_market_rows": 1, "player_prop_research_market_rows": 0}},
            {"market": "NOT VERIFIED"},
        ]
    }
    result = v._summarize_research_derivative_sidecars(payload)
    assert result["events_with_sidecar"] == 2
    assert result["card_market_rows"] == 3
    assert result["player_prop_market_rows"] == 3
    assert result["total_market_rows"] == 6
    assert result["provider_requests_added"] == 0
    assert result["production_promotion_allowed"] is False
    assert result["decision_weight"] == 0.0



def test_research_derivative_classifier_does_not_steal_team_totals_or_corners():
    assert base._is_ft_team_total_bet({"id": 16, "name": "Total - Home"}) is True
    assert base._is_player_prop_research_bet({"name": "Player Shots On Target"}) is True
    assert base._is_player_prop_research_bet({"name": "Anytime Goalscorer"}) is True
    assert base._is_player_prop_research_bet({"name": "Home Anytime Goal Scorer"}) is True
    assert base._is_player_prop_research_bet({"name": "Away First Goal Scorer"}) is True
    assert base._is_player_prop_research_bet({"name": "Home Last Goal Scorer"}) is True
    assert base._player_prop_research_subfamily({"name": "Home Anytime Goal Scorer"}) == "GOALSCORER_ANYTIME"
    assert base._player_prop_research_subfamily({"name": "Away First Goal Scorer"}) == "GOALSCORER_FIRST"
    assert base._player_prop_research_subfamily({"name": "Home Last Goal Scorer"}) == "GOALSCORER_LAST"
    assert base._is_player_prop_research_bet({"name": "Goalkeeper Saves"}) is True
    assert base._is_card_research_bet({"name": "Total Yellow Cards"}) is True
    assert base._is_card_research_bet({"name": "Booking Points"}) is True
    assert base._is_card_research_bet({"name": "Cards Asian Handicap"}) is True
    assert base._is_card_research_bet({"name": "Cards European Handicap"}) is True
    assert base._is_card_research_bet({"name": "First Card Received (3 way)"}) is True
    assert base._is_card_research_bet({"name": "RCARD"}) is True
    assert base._is_card_research_bet({"name": "Player Cards"}) is False
    assert base._is_player_prop_research_bet({"name": "Player Cards"}) is True
    assert base._is_card_research_bet({"name": "Total Corners"}) is False
    assert base._is_player_prop_research_bet({"name": "ShotOnTarget Handicap"}) is False
    assert base._is_player_prop_research_bet({"name": "Total Shots"}) is False
    assert base._is_player_prop_research_bet({"name": "Total - Home"}) is False


def test_booking_points_are_captured_but_flagged_for_scoring_rule_mapping():
    payload = {
        "response": [{
            "update": "2026-09-25T06:10:00+00:00",
            "bookmakers": [{
                "id": 7,
                "name": "Book",
                "bets": [{
                    "id": 300,
                    "name": "Booking Points",
                    "values": [{"value": "Over 35.5", "odd": "1.91"}],
                }],
            }],
        }],
    }
    compact = base._compact_odds(payload)
    assert compact["markets"] == []
    assert compact["card_research_market_rows"] == 1
    row = compact["research_cards_props_markets"][0]
    assert row["research_family"] == "CARDS"
    assert row["bookmaker_scoring_rule_required"] is True
    assert row["values"][0]["parsed_line"] == 35.5



def test_research_derivative_summary_excludes_cache_replay_from_new_capture():
    payload = {
        "events": [
            {
                "market": {
                    "source": "API_FOOTBALL_ODDS_V3",
                    "resolution_status": "PRICE_API_RESOLVED",
                    "card_research_market_rows": 2,
                    "player_prop_research_market_rows": 3,
                }
            },
            {
                "market": {
                    "source": "LOCAL_ODDS_CACHE",
                    "resolution_status": "PRICE_CACHE_HIT_LOCAL",
                    "card_research_market_rows": 4,
                    "player_prop_research_market_rows": 5,
                }
            },
        ]
    }
    result = v._summarize_research_derivative_sidecars(payload)
    assert result["events_with_sidecar"] == 2
    assert result["fresh_provider_events_with_sidecar"] == 1
    assert result["cache_replay_events_with_sidecar"] == 1
    assert result["observed_card_market_rows"] == 6
    assert result["observed_player_prop_market_rows"] == 8
    assert result["card_market_rows"] == 2
    assert result["player_prop_market_rows"] == 3
    assert result["total_market_rows"] == 5
    assert result["cache_replay_rows_excluded_from_new_evidence"] == 9


def test_v5_odds_cache_marks_replay_and_fresh_provider(monkeypatch):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    compact = {
        "markets": [],
        "card_research_market_rows": 1,
        "player_prop_research_market_rows": 0,
    }

    monkeypatch.setattr(v5.base, "_cache_get", lambda *args, **kwargs: dict(compact))
    cached = asyncio.run(v5._odds_7m(999, now))
    assert cached["source"] == "LOCAL_ODDS_CACHE"
    assert cached["resolution_status"] == "PRICE_CACHE_HIT_LOCAL"

    writes = []
    monkeypatch.setattr(v5.base, "_cache_get", lambda *args, **kwargs: None)
    monkeypatch.setattr(v5.base, "_cache_set", lambda *args, **kwargs: writes.append(args))

    async def fake_api_get(endpoint, params):
        assert endpoint == "odds"
        return {"response": []}

    monkeypatch.setattr(v5.base, "_api_get", fake_api_get)
    monkeypatch.setattr(v5.base, "_compact_odds", lambda payload, lineup=None: dict(compact))
    fresh = asyncio.run(v5._odds_7m(999, now))
    assert fresh["source"] == "API_FOOTBALL_ODDS_V3"
    assert fresh["resolution_status"] == "PRICE_API_RESOLVED"
    assert writes



def test_observed_goal_scorer_and_card_derivatives_are_isolated_in_sidecar():
    payload = {
        "response": [{
            "update": "2026-09-25T14:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [
                    {"id": 401, "name": "Home Anytime Goal Scorer", "values": [{"value": "Player A", "odd": "2.20"}]},
                    {"id": 402, "name": "Away First Goal Scorer", "values": [{"value": "Player B", "odd": "6.50"}]},
                    {"id": 403, "name": "Home Last Goal Scorer", "values": [{"value": "Player C", "odd": "7.00"}]},
                    {"id": 404, "name": "Cards Asian Handicap", "values": [{"value": "Home +0.5", "odd": "1.90"}]},
                    {"id": 405, "name": "First Card Received (3 way)", "values": [{"value": "Home", "odd": "1.80"}]},
                    {"id": 406, "name": "RCARD", "values": [{"value": "Yes", "odd": "3.50"}]},
                    {"id": 407, "name": "ShotOnTarget Handicap", "values": [{"value": "Home +0.5", "odd": "1.90"}]},
                ],
            }],
        }],
    }

    compact = base._compact_odds(payload)
    canonical_names = {row["market"] for row in compact["markets"]}
    research = {row["market"]: row for row in compact["research_cards_props_markets"]}

    assert "Home Anytime Goal Scorer" not in canonical_names
    assert "Away First Goal Scorer" not in canonical_names
    assert "Home Last Goal Scorer" not in canonical_names
    assert "Cards Asian Handicap" not in canonical_names
    assert "First Card Received (3 way)" not in canonical_names
    assert "RCARD" not in canonical_names
    assert "ShotOnTarget Handicap" in canonical_names

    assert research["Home Anytime Goal Scorer"]["research_subfamily"] == "GOALSCORER_ANYTIME"
    assert research["Away First Goal Scorer"]["research_subfamily"] == "GOALSCORER_FIRST"
    assert research["Home Last Goal Scorer"]["research_subfamily"] == "GOALSCORER_LAST"
    assert research["Cards Asian Handicap"]["research_family"] == "CARDS"
    assert research["First Card Received (3 way)"]["research_family"] == "CARDS"
    assert research["RCARD"]["research_family"] == "CARDS"


def test_player_prop_capture_aligns_values_to_confirmed_xi_without_extra_calls():
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [
                    {"id": 501, "name": "Ángel Di María", "pos": "F"},
                    {"id": 502, "name": "Keeper One", "pos": "G"},
                ],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [
                    {"id": 601, "name": "Player B", "pos": "M"},
                    {"id": 602, "name": "Keeper Two", "pos": "G"},
                ],
            },
        ],
    }
    payload = {
        "response": [{
            "update": "2026-09-25T14:00:00+00:00",
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [
                    {
                        "id": 801,
                        "name": "Player Shots",
                        "values": [
                            {"value": "Angel Di Maria Over 2.5", "odd": "1.90"},
                            {"value": "Bench Player Over 1.5", "odd": "1.80"},
                        ],
                    },
                    {
                        "id": 802,
                        "name": "Goalkeeper Saves",
                        "values": [
                            {"value": "Keeper One Over 3.5", "odd": "1.88"},
                        ],
                    },
                ],
            }],
        }],
    }

    compact = base._compact_odds(payload, lineup=lineup)

    assert compact["player_prop_research_market_rows"] == 2
    rows = {
        row["research_subfamily"]: row
        for row in compact["research_cards_props_markets"]
        if row.get("research_family") == "PLAYER_PROPS"
    }

    shots = rows["SHOTS"]
    assert shots["confirmed_xi_at_quote"] is True
    assert shots["xi_aligned_value_rows"] == 1
    assert shots["decision_weight"] == 0.0
    assert shots["production_promotion_allowed"] is False
    assert shots["values"][0]["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
    assert shots["values"][0]["player_id"] == 501
    assert shots["values"][0]["team_id"] == 10
    assert shots["values"][0]["parsed_line"] == 2.5
    assert shots["values"][1]["xi_alignment_status"] == "PLAYER_NOT_MATCHED_TO_CONFIRMED_XI"

    gk = rows["GK_SAVES"]
    assert gk["confirmed_xi_at_quote"] is True
    assert gk["xi_aligned_value_rows"] == 1
    assert gk["values"][0]["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
    assert gk["values"][0]["position"] == "G"
    assert gk["values"][0]["player_id"] == 502


def test_player_prop_capture_marks_unaligned_when_confirmed_xi_is_unavailable():
    payload = {
        "response": [{
            "bookmakers": [{
                "id": 1,
                "name": "Book",
                "bets": [{
                    "id": 801,
                    "name": "Player Shots",
                    "values": [{"value": "Player A Over 2.5", "odd": "1.90"}],
                }],
            }],
        }],
    }

    compact = base._compact_odds(payload)
    row = next(
        row for row in compact["research_cards_props_markets"]
        if row.get("research_subfamily") == "SHOTS"
    )

    assert row["confirmed_xi_at_quote"] is False
    assert row["xi_aligned_value_rows"] == 0
    assert row["values"][0]["xi_alignment_status"] == "NO_CONFIRMED_XI_AT_QUOTE"
    assert row["decision_weight"] == 0.0
    assert row["production_promotion_allowed"] is False


def test_production_event_passes_confirmed_xi_into_odds_compaction(monkeypatch):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    fixture = {
        "fixture_id": 99001,
        "league_id": 39,
        "season": 2026,
        "kickoff": "2026-09-25T12:10:00+00:00",
        "home_team_id": 10,
        "away_team_id": 20,
        "home_team": "Home",
        "away_team": "Away",
    }
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }

    async def fake_coverage(*args, **kwargs):
        return {
            "data_tier": "A",
            "injuries": False,
            "lineups": True,
            "odds": True,
            "statistics_fixtures": False,
        }

    async def fake_sport_bundle(*args, **kwargs):
        return {"sport_data": "UNAVAILABLE"}

    async def fake_api_get(endpoint, params):
        if endpoint == "fixtures/lineups":
            return {"response": []}
        if endpoint == "odds":
            return {"response": []}
        raise AssertionError(endpoint)

    seen = {}

    def fake_compact_lineups(payload):
        return lineup

    def fake_compact_odds(payload, lineup=None):
        seen["lineup"] = lineup
        return {
            "markets": [],
            "research_cards_props_markets": [],
            "player_prop_research_market_rows": 0,
        }

    monkeypatch.setattr(v2.base, "_coverage", fake_coverage)
    monkeypatch.setattr(v2.base, "_sport_bundle", fake_sport_bundle)
    monkeypatch.setattr(v2.base, "_api_get", fake_api_get)
    monkeypatch.setattr(v2.base, "_compact_lineups", fake_compact_lineups)
    monkeypatch.setattr(v2.base, "_compact_odds", fake_compact_odds)

    event = asyncio.run(v2._event_for_fixture(fixture, "T-10", now))

    assert seen["lineup"] is lineup
    assert event["lineups"]["both_xi_confirmed"] is True
    assert event["market"]["xi_alignment_input_available"] is True


def test_v5_cached_player_props_are_realigned_when_confirmed_xi_arrives(monkeypatch):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    cached = {
        "markets": [],
        "research_cards_props_markets": [{
            "research_only": True,
            "research_family": "PLAYER_PROPS",
            "research_subfamily": "SHOTS",
            "market": "Player Shots",
            "values": [{
                "selection": "Player A Over 2.5",
                "price": "1.90",
                "parsed_line": 2.5,
                "xi_alignment_status": "NO_CONFIRMED_XI_AT_QUOTE",
            }],
        }],
        "player_prop_research_market_rows": 1,
    }
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }

    monkeypatch.setattr(v5.base, "_cache_get", lambda *args, **kwargs: dict(cached))
    result = asyncio.run(v5._odds_7m(999, now, lineup=lineup))

    value = result["research_cards_props_markets"][0]["values"][0]
    assert result["source"] == "LOCAL_ODDS_CACHE"
    assert value["xi_alignment_status"] == "MATCHED_CONFIRMED_XI"
    assert value["player_id"] == 501
    assert result["research_cards_props_markets"][0]["xi_aligned_value_rows"] == 1


def test_v5_research_only_xi_capture_runs_for_t20_shortlist_miss(monkeypatch):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    fixture = {
        "fixture_id": 88001,
        "league_id": 39,
        "season": 2026,
        "kickoff": "2026-09-25T12:20:00+00:00",
        "home_team_id": 10,
        "away_team_id": 20,
        "home_team": "Home",
        "away_team": "Away",
    }
    coverage = {
        "data_tier": "A",
        "lineups": True,
        "odds": True,
        "injuries": False,
        "statistics_fixtures": True,
        "statistics_players": True,
    }
    lineup = {
        "both_xi_confirmed": True,
        "both_goalkeepers_confirmed": True,
        "lineup_state": "CONFIRMED_API",
        "teams": [
            {
                "team_id": 10,
                "team": "Home",
                "starters": [{"id": 501, "name": "Player A", "pos": "F"}],
            },
            {
                "team_id": 20,
                "team": "Away",
                "starters": [{"id": 601, "name": "Keeper B", "pos": "G"}],
            },
        ],
    }
    market = {
        "source": "API_FOOTBALL_ODDS_V3",
        "resolution_status": "PRICE_API_RESOLVED",
        "markets": [],
        "research_cards_props_markets": [{
            "research_family": "PLAYER_PROPS",
            "research_subfamily": "SHOTS",
            "market": "Player Shots",
            "values": [{
                "selection": "Player A Over 2.5",
                "price": "1.90",
                "parsed_line": 2.5,
                "xi_alignment_status": "MATCHED_CONFIRMED_XI",
                "player_id": 501,
            }],
        }],
    }

    async def fake_cheap(*args, **kwargs):
        return {"sport_data": "AVAILABLE"}

    def fake_projection(*args, **kwargs):
        return {
            "status": "MODELED_LIMITED",
            "screen_scores": {
                "side_edge_score": 10.0,
                "goal_environment_score": 60.0,
                "two_way_scoring_score": 10.0,
            },
        }

    async def fake_lineup(*args, **kwargs):
        return lineup

    async def fake_odds(fixture_id, when, lineup=None):
        assert fixture_id == 88001
        assert lineup is not None and lineup["both_xi_confirmed"] is True
        return market

    monkeypatch.setattr(v5, "_cheap_sport_bundle", fake_cheap)
    monkeypatch.setattr(v5, "build_raw_projection", fake_projection)
    monkeypatch.setattr(v5, "_lineup_cached", fake_lineup)
    monkeypatch.setattr(v5, "_odds_7m", fake_odds)
    monkeypatch.setattr(v5, "_shortlist_set", lambda *args, **kwargs: None)

    v5._PLAYER_PROPS_XI_RESEARCH_ATTEMPTS = 0
    v5._PLAYER_PROPS_XI_RESEARCH_CAPTURED = 0

    event = asyncio.run(v5._priority_event(fixture, "T-20", coverage, now))

    assert event["classification"] == "RESEARCH_ONLY"
    assert event["bet_eligible"] is False
    assert event["decision_weight"] == 0.0
    assert event["market_use"] == "PLAYER_PROPS_XI_RESEARCH_ONLY"
    assert event["research_player_props_xi_capture"]["status"] == "XI_CONFIRMED_RESEARCH_CAPTURED"
    assert event["research_player_props_xi_capture"]["player_prop_market_rows"] == 1
    assert event["research_player_props_xi_capture"]["xi_aligned_value_rows"] == 1
    assert event["market_skipped_by_sport_screen"] is False
    assert v5._PLAYER_PROPS_XI_RESEARCH_ATTEMPTS == 1
    assert v5._PLAYER_PROPS_XI_RESEARCH_CAPTURED == 1


def test_v5_research_only_xi_capture_does_not_request_odds_without_confirmed_xi(monkeypatch):
    now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
    fixture = {"fixture_id": 88002}
    coverage = {"lineups": True, "odds": True}
    event = {"notes": []}

    async def fake_lineup(*args, **kwargs):
        return {
            "both_xi_confirmed": False,
            "both_goalkeepers_confirmed": False,
            "lineup_state": "PENDING",
            "teams": [],
        }

    async def forbidden_odds(*args, **kwargs):
        raise AssertionError("odds must not be requested without confirmed XI")

    monkeypatch.setattr(v5, "_lineup_cached", fake_lineup)
    monkeypatch.setattr(v5, "_odds_7m", forbidden_odds)
    v5._PLAYER_PROPS_XI_RESEARCH_ATTEMPTS = 0
    v5._PLAYER_PROPS_XI_RESEARCH_CAPTURED = 0

    captured = asyncio.run(
        v5._try_player_props_xi_research(event, fixture, "T-20", coverage, now)
    )

    assert captured is False
    assert event["research_player_props_xi_capture"]["status"] == "XI_NOT_CONFIRMED"
    assert v5._PLAYER_PROPS_XI_RESEARCH_ATTEMPTS == 1
    assert v5._PLAYER_PROPS_XI_RESEARCH_CAPTURED == 0


def test_persistence_disambiguates_same_fixture_stage_events_by_microsecond(monkeypatch):
    executed = []

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            executed.append((sql, params))

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def cursor(self):
            return FakeCursor()

    monkeypatch.setattr(persistence_base, "persistence_configured", lambda: True)
    monkeypatch.setattr(persistence_base, "ensure_schema", lambda: None)
    monkeypatch.setattr(persistence_base, "_connect", lambda: FakeConnection())

    generated = "2026-09-25T18:14:19.376270+00:00"
    fixture = {
        "fixture_id": 1569941,
        "kickoff": "2026-09-25T18:40:00+00:00",
        "home_team_id": 10,
        "away_team_id": 20,
    }
    tick = {
        "generated_at_utc": generated,
        "generated_at_local": "2026-09-25T12:14:19.376270-06:00",
        "timezone": "America/Mexico_City",
        "fixture_scan_count": 1,
        "event_count": 2,
        "actionable_refresh_count": 0,
        "events": [
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-20",
                "fixture": dict(fixture),
                "classification": "WATCH",
                "bet_eligible": False,
            },
            {
                "event_type": "SOCCER_REFRESH",
                "stage": "T-20",
                "fixture": dict(fixture),
                "classification": "RESEARCH_ONLY",
                "bet_eligible": False,
            },
        ],
    }

    assert persistence_base.persist_tick(tick) is True

    refresh_params = [
        params
        for sql, params in executed
        if "INSERT INTO soccer_refresh_events" in sql
    ]
    assert len(refresh_params) == 2
    assert refresh_params[0][7] == generated
    assert refresh_params[1][7] != generated
    assert str(refresh_params[1][7]).endswith("376271+00:00")
    assert tick["events"][1]["persistence"]["same_fixture_stage_ordinal"] == 1
    assert tick["events"][1]["persistence"]["generated_at_microsecond_offset"] == 1
