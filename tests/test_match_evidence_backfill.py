"""Data retention, low-coverage sporting research and immutable BET gates."""
# Regression gate: venue W/D/L completeness and postkickoff provenance.
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from mcp_gateway import automation_v2, feature_snapshot_v4, research_backfill_v1, subscriber_contract_v2


def test_feature_snapshot_captures_verified_home_away_splits_without_imputation():
    tick = {"generated_at_utc": "2026-10-08T22:00:00Z", "model_version": "test"}
    event = {
        "fixture": {"fixture_id": 1612077, "home_team_id": 18681,
                    "away_team_id": 18683, "league_id": 906, "season": 2026},
        "stage": "RESEARCH_BACKFILL", "coverage": {"data_tier": "D"},
        "sporting": {
            "home_stats": {
                "fixtures": {"played": {"home": 16, "total": 31},
                             "wins": {"home": 11}, "draws": {"home": 1}, "loses": {"home": 4}},
                "goals": {"for": {"total": {"home": 28}},
                          "against": {"total": {"home": 6}}},
            },
            "away_stats": {
                "fixtures": {"played": {"away": 14, "total": 29},
                             "wins": {"away": 2}, "draws": {"away": 3}, "loses": {"away": 9}},
                "goals": {"for": {"total": {"away": 10}},
                          "against": {"total": {"away": 20}}},
            },
        },
    }
    snap = feature_snapshot_v4.build(tick, event)
    features = snap["features"]
    assert snap["data_tier"] == "D"
    assert features["team_performance.home_wins_split"]["value"] == 11
    assert features["team_performance.home_wins_split"]["sample_n"] == 16
    assert features["team_performance.away_wins_split"]["value"] == 2
    assert features["team_performance.away_wins_split"]["sample_n"] == 14
    assert features["team_performance.home_goals_for_split"]["value"] == 28
    assert features["team_performance.away_goals_against_split"]["value"] == 20
    assert features["team_performance.home_goal_rate_blend"]["value"] is None
    assert features["xg.xgf"]["value"] is None
    assert feature_snapshot_v4.validate(snap) == []


def test_sparse_new_snapshot_never_erases_prior_verified_statistics():
    old = {"captured_at": "2026-10-08T18:00:00Z", "data_tier": "D", "payload": {
        "features": {
            "team_performance.home_wins_split": {
                "value": 11, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 16,
                "captured_at": "2026-10-08T18:00:00Z",
            },
            "team_performance.away_wins_split": {
                "value": 2, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 14,
                "captured_at": "2026-10-08T18:00:00Z",
            },
        },
    }}
    new = {"captured_at": "2026-10-08T21:00:00Z", "data_tier": "RESEARCH_ONLY", "payload": {
        "features": {
            "team_performance.home_wins_split": {"value": None},
            "context.league_id": {
                "value": 906, "source": "API_FOOTBALL_FIXTURE",
                "captured_at": "2026-10-08T21:00:00Z",
            },
        },
    }}
    groups = subscriber_contract_v2._match_evidence_sections({"feature_snapshots": [new, old]})
    assert [g["category"] for g in groups] == ["TEAMS", "CONTEXT"]
    team = {i["key"]: i for i in groups[0]["items"]}
    assert team["team_performance.home_wins_split"]["value"] == 11
    assert team["team_performance.away_wins_split"]["value"] == 2
    assert team["team_performance.home_wins_split"]["captured_at"] == "2026-10-08T18:00:00Z"
    assert groups[1]["items"][0]["captured_at"] == "2026-10-08T21:00:00Z"


def test_tier_d_collects_display_only_stats_and_never_promotes_bet(monkeypatch):
    async def fake_coverage(*_args):
        return {"data_tier": "D", "odds": True, "lineups": True}

    async def fake_stats(team_id, *_args):
        return {"form": "WW", "fixtures": {"played": {"total": 2}},
                "team_id": team_id}

    from mcp_gateway import automation as base
    monkeypatch.setattr(base, "_coverage", fake_coverage)
    monkeypatch.setattr(base, "_team_stats", fake_stats)
    monkeypatch.setattr(automation_v2, "_API_CALLS_THIS_TICK", 0)
    monkeypatch.setattr(automation_v2, "_LAST_DAILY_REMAINING", 200)
    monkeypatch.setattr(automation_v2, "_RESEARCH_FIXTURES_THIS_TICK", 0)
    event = asyncio.run(automation_v2._event_for_fixture(
        {"fixture_id": 12, "league_id": 906, "season": 2026,
         "home_team_id": 1, "away_team_id": 2},
        "T-90", datetime.now(timezone.utc),
    ))
    assert event["classification"] == "PASS"
    assert event["bet_eligible"] is False
    assert event["sporting"]["home_stats"]["team_id"] == 1
    assert event["sporting"]["away_stats"]["team_id"] == 2
    assert "raw_projection" not in event
    assert "market" not in event
    assert "market_decision" not in event
    assert event["availability_confidence"] is None


def test_backfill_emits_only_independent_nonbet_sport_evidence(monkeypatch):
    fixture = {
        "fixture_id": 1612077, "league_id": 906, "season": 2026,
        "home_team_id": 18681, "away_team_id": 18683,
        "home_team": "Boca Juniors Res.", "away_team": "Colón Res.",
        "kickoff": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(), "status": "NS",
        "league": "Reserve League", "country": "Argentina",
        "round": "Clausura - 12", "venue": None, "city": None,
        "status_long": "Not Started", "data_tier": "D",
    }
    async def fake_stats(team_id, *_args):
        return {"fixtures": {"played": {"total": 10}}, "team_id": team_id}

    from mcp_gateway import automation as base
    monkeypatch.setattr(base, "_team_stats", fake_stats)
    monkeypatch.setattr(research_backfill_v1, "pending_fixtures", lambda limit: [dict(fixture)])
    monkeypatch.setattr(automation_v2, "_API_CALLS_THIS_TICK", 0)
    monkeypatch.setattr(automation_v2, "_LAST_DAILY_REMAINING", 200)
    previous = {"classification": "BET", "fixture": {"fixture_id": 999}}
    tick = {"events": [previous]}
    events = asyncio.run(research_backfill_v1.collect(tick))
    assert len(events) == 1
    row = events[0]
    assert row["event_type"] == "SPORT_FEATURE_RESEARCH_BACKFILL"
    assert row["stage"] == "RESEARCH_BACKFILL"
    assert row["classification"] is None
    assert row["bet_eligible"] is False
    assert row["market_decision"] is None
    assert row["raw_projection"] is None
    assert row["fixture"]["status_long"] == "Not Started"
    assert row["sporting"]["home_stats"]["team_id"] == 18681
    assert row["sporting"]["away_stats"]["team_id"] == 18683
    assert tick["events"] == [previous]


def test_backfill_quota_guard_blocks_provider_calls(monkeypatch):
    monkeypatch.setattr(automation_v2, "_API_CALLS_THIS_TICK", automation_v2.MAX_API_CALLS_PER_TICK - 2)
    monkeypatch.setattr(automation_v2, "_LAST_DAILY_REMAINING", 200)
    monkeypatch.setattr(research_backfill_v1, "pending_fixtures", lambda limit: (_ for _ in ()).throw(AssertionError("should not query DB")))
    assert asyncio.run(research_backfill_v1.collect({})) == []
def _no_analysis_payload():
    return {"generated_at_utc": "2026-10-08T21:00:00Z", "events": []}




def test_final_score_sync_uses_official_batch_without_regrading_predictions():
    from mcp_gateway import persistence

    class Cursor:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append((sql, params))

    cursor = Cursor()
    tick = {
        "generated_at_utc": "2026-10-09T04:30:00Z",
        "observed_final_fixtures": [
            {"fixture_id": 1612077, "status": "FT",
             "status_long": "Match Finished", "goals": {"home": 1, "away": 0},
             "score": {"fulltime": {"home": 1, "away": 0}}},
            {"fixture_id": 1612078, "status": "NS",
             "goals": {"home": None, "away": None}},
        ],
    }
    persistence._persist_final_observations(cursor, tick)
    assert len(cursor.calls) == 2
    update_sql, (update_json,) = cursor.calls[0]
    insert_sql, params = cursor.calls[1]
    assert "UPDATE soccer_fixtures" in update_sql
    assert "INSERT INTO soccer_results" in insert_sql
    assert "ON CONFLICT (fixture_id) DO NOTHING" in insert_sql
    assert "UPDATE soccer_model_runs" not in update_sql + insert_sql
    assert "UPDATE soccer_market_snapshots" not in update_sql + insert_sql
    import json
    observed = json.loads(update_json)
    assert len(observed) == 1
    assert observed[0]["fixture_id"] == 1612077
    assert params[1] == "2026-10-09T04:30:00Z"


def test_final_registry_score_is_separate_from_pregame_probabilities():
    registry = {
        "fixture_id": 1612077, "league": "Reserve League",
        "home_team_id": 18681, "home_team": "Boca Juniors Res.",
        "away_team_id": 18683, "away_team": "Colón Res.",
        "kickoff": "2026-10-08T22:00:00Z",
        "status": "FT", "final_home_goals": 1, "final_away_goals": 0,
        "final_observed_at": "2026-10-09T04:30:00Z",
    }
    packet = subscriber_contract_v2.build_match_contract(
        _no_analysis_payload(), 1612077, registry_fixture=registry,
        relational_evidence={"model_runs": [{
            "run_timestamp": "2026-10-08T20:00:00Z",
            "raw_projection": {
                "raw_home_win_prob": 0.595,
                "raw_draw_prob": 0.258,
                "raw_away_win_prob": 0.147,
                "raw_home_goal_rate": 1.51,
                "raw_away_goal_rate": 0.60,
            },
        }], "counts": {"model_runs": 1}},
    )
    assert packet["fixture"]["fixture_status"] == "FT"
    assert packet["fixture"]["final_home_goals"] == 1
    assert packet["fixture"]["final_away_goals"] == 0
    assert packet["sport_context"]["outcome_probabilities"]["home"] == 0.595
    assert packet["projection_ladder"]["fair_market_probability"] is None
    assert packet["decision_summary"]["classification"] is None
    assert packet["model_weights_changed"] is False
    assert packet["canonical_bet_logic_changed"] is False



def test_registry_final_display_keeps_score_separate(monkeypatch):
    """Only terminal, stored scores qualify for display; stale rows do not."""
    from datetime import datetime, timezone

    class DummyCursor:
        def __init__(self):
            self.sql = ""
            self.params = None
        def execute(self, sql, params):
            self.sql, self.params = sql, params
        def fetchone(self):
            return (
                1612077, datetime(2026, 10, 8, 22, tzinfo=timezone.utc),
                "FT", "Reserve League", "Argentina", 18681, "Boca Juniors Res.",
                18683, "Colón Res.", 1, 0,
                datetime(2026, 10, 9, 4, 30, tzinfo=timezone.utc),
            )
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    class DummyConnection:
        def __init__(self):
            self.cursor_instance = DummyCursor()
        def cursor(self):
            return self.cursor_instance
        def __enter__(self):
            return self
        def __exit__(self, *_):
            return False

    conn = DummyConnection()
    monkeypatch.setattr(subscriber_contract_v2.persistence_base, "_connect", lambda: conn)
    result = subscriber_contract_v2._load_registry_fixture(1612077)
    assert result["status"] == "FT"
    assert result["final_home_goals"] == 1
    assert result["final_away_goals"] == 0
    assert conn.cursor_instance.params == (1612077,)
    assert "WHEN f.status IN ('FT','AET','PEN')" in conn.cursor_instance.sql
    assert "CASE WHEN r.final_status IN ('FT','AET','PEN') THEN r.home_goals" in conn.cursor_instance.sql


def test_venue_splits_are_preserved_as_separate_persisted_fields():
    """No substitution of season-wide wins for venue-specific wins."""
    snap = {
        "captured_at": "2026-10-08T18:00:00Z", "data_tier": "D",
        "payload": {"features": {
            "team_performance.home_played_split": {
                "value": 16, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 31,
            },
            "team_performance.home_wins_total": {
                "value": 17, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 31,
            },
            "team_performance.home_wins_split": {
                "value": 11, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 16,
            },
            "team_performance.away_wins_total": {
                "value": 7, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 29,
            },
            "team_performance.away_wins_split": {
                "value": 2, "source": "API_FOOTBALL_TEAM_STATS", "sample_n": 14,
            },
        }},
    }
    groups = subscriber_contract_v2._match_evidence_sections({"feature_snapshots":[snap]})
    by_key = {item["key"]:item for item in groups[0]["items"]}
    assert by_key["team_performance.home_wins_split"]["value"] == 11
    assert by_key["team_performance.home_wins_total"]["value"] == 17
    assert by_key["team_performance.away_wins_split"]["value"] == 2
    assert by_key["team_performance.away_wins_total"]["value"] == 7
    assert by_key["team_performance.home_wins_split"]["sample_n"] == 16


def test_research_backfill_missing_venue_wdl_triggers_new_snapshot():
    query = research_backfill_v1._PENDING_SQL
    assert "team_performance.home_wins_split,value" in query
    assert "team_performance.away_wins_split,value" in query
    assert "team_performance.home_draws_split,value" in query
    assert "team_performance.away_losses_split,value" in query
    # psycopg treats a bare '%' as a query placeholder even inside LIKE literals.
    # Enumerate the terminal capture stages instead of a wildcard.
    assert "s.stage IN (" in query
    assert "'RESEARCH_BACKFILL_POST_KICKOFF'" in query
    assert "'RESEARCH_BACKFILL_POSTGAME'" in query
    assert "RESEARCH_BACKFILL%" not in query
    assert "interval '24 hours'" in query


def test_finished_fixture_backfill_has_postgame_scope(monkeypatch):
    fixture = {
        "fixture_id": 1612077, "league_id": 906, "season": 2026,
        "home_team_id": 18681, "away_team_id": 18683,
        "home_team": "Boca Juniors Res.", "away_team": "Colón Res.",
        "kickoff": "2026-10-08T22:00:00Z", "status": "FT",
        "league": "Reserve League", "country": "Argentina",
        "round": "Clausura - 12", "venue": None, "city": None,
        "status_long": "Match Finished", "data_tier": "RESEARCH_ONLY",
    }
    async def fake_stats(team_id, *_args):
        return {"team_id":team_id, "fixtures": {
            "played":{"home":17,"away":15,"total":32},
            "wins":{"home":12,"away":2}, "draws":{"home":1,"away":3},
            "loses":{"home":4,"away":10},
        }}
    from mcp_gateway import automation as base
    monkeypatch.setattr(base,"_team_stats",fake_stats)
    monkeypatch.setattr(research_backfill_v1,"pending_fixtures",lambda limit:[dict(fixture)])
    monkeypatch.setattr(automation_v2,"_API_CALLS_THIS_TICK",0)
    monkeypatch.setattr(automation_v2,"_LAST_DAILY_REMAINING",200)
    events=asyncio.run(research_backfill_v1.collect({"events":[]}))
    assert len(events)==1
    event=events[0]
    assert event["stage"]=="RESEARCH_BACKFILL_POSTGAME"
    assert event["sporting"]["collection_scope"]=="POSTGAME_OBSERVATION"
    assert event["classification"] is None and event["bet_eligible"] is False
    assert event["raw_projection"] is None and event["market_decision"] is None
    snapshot=feature_snapshot_v4.build({"generated_at_utc":"2026-10-09T05:00:00Z"},event)
    assert snapshot["observation_scope"]=="POSTGAME_OBSERVATION"
    assert snapshot["features"]["team_performance.home_wins_split"]["value"]==12
    assert snapshot["features"]["team_performance.away_losses_split"]["value"]==10
    assert feature_snapshot_v4.validate(snapshot)==[]
    sections=subscriber_contract_v2._match_evidence_sections({"feature_snapshots":[{
        "captured_at":snapshot["captured_at"],"payload":snapshot,
    }]})
    team={item["key"]:item for item in sections[0]["items"]}
    assert team["team_performance.home_wins_split"]["observation_scope"]=="POSTGAME_OBSERVATION"


def test_backfill_rotates_three_queues_without_budget_overrun():
    """Queue fairness must not change sport projections or exceed the API cap."""
    sql = research_backfill_v1._PENDING_SQL
    assert "NOW() - interval '14 days'" in sql
    assert "NOW() - interval '48 hours'" in sql
    assert "PARTITION BY queue_lane" in sql
    assert "ROW_NUMBER() OVER" in sql
    assert "MOD(queue_lane - MOD(" in sql
    assert "ORDER BY lane_rank ASC" in sql
    assert "LIMIT %s" in sql
    assert "RESEARCH_BACKFILL%" not in sql  # psycopg placeholder trap
    assert all(token in sql for token in (
        "team_performance.home_wins_split,value",
        "team_performance.home_draws_split,value",
        "team_performance.home_losses_split,value",
        "team_performance.away_wins_split,value",
        "team_performance.away_draws_split,value",
        "team_performance.away_losses_split,value",
    ))
    assert research_backfill_v1.MAX_BACKFILL_FIXTURES_PER_TICK <= 2


def test_backfill_emits_selected_fixture_ids_and_protects_quota(monkeypatch):
    fixture = {
        "fixture_id": 1612077, "league_id": 906, "season": 2026,
        "home_team_id": 18681, "away_team_id": 18683,
        "home_team": "Boca Juniors Res.", "away_team": "Colón Res.",
        "kickoff": "2026-10-08T22:00:00Z", "status": "FT",
        "league": "Reserve League", "country": "Argentina",
        "round": "Clausura - 12", "venue": None, "city": None,
        "status_long": "Match Finished", "data_tier": "RESEARCH_ONLY",
    }
    calls = []
    async def fake_stats(team_id, *_args):
        calls.append(team_id)
        return {"fixtures":{"played":{"total":12}}, "team_id":team_id}

    from mcp_gateway import automation as base
    monkeypatch.setattr(base, "_team_stats", fake_stats)
    monkeypatch.setattr(research_backfill_v1,"pending_fixtures",lambda limit:[dict(fixture)])
    monkeypatch.setattr(automation_v2,"_API_CALLS_THIS_TICK",0)
    monkeypatch.setattr(automation_v2,"_LAST_DAILY_REMAINING",200)
    tick = {"events":[]}
    got = asyncio.run(research_backfill_v1.collect(tick))
    assert len(got) == 1
    assert calls == [18681, 18683]
    assert tick["research_backfill_status"] == "ATTEMPTED"
    assert tick["research_backfill_selected_fixture_ids"] == [1612077]
    assert tick["research_backfill_emitted_fixture_ids"] == [1612077]
    assert got[0]["sporting"]["collection_scope"] == "POSTGAME_OBSERVATION"
    assert got[0]["bet_eligible"] is False
    assert got[0]["raw_projection"] is None

    monkeypatch.setattr(
        automation_v2, "_API_CALLS_THIS_TICK",
        automation_v2.MAX_API_CALLS_PER_TICK - 2,
    )
    guarded = {}
    assert asyncio.run(research_backfill_v1.collect(guarded)) == []
    assert guarded["research_backfill_status"] == "DEFERRED_TICK_BUDGET"
    assert calls == [18681, 18683]


def test_candidate_query_keeps_original_16_columns_for_persistence(monkeypatch):
    """The new ranking columns must never leak into the persisted fixture dict."""
    from mcp_gateway import persistence
    row = (
        1612077, 906, "Reserve League", "Argentina", 2026, "Clausura - 12",
        datetime(2026,10,8,22,tzinfo=timezone.utc), "FT", "Finished",
        18681, "Boca Juniors Res.", 18683, "Colón Res.", None, None,
        "RESEARCH_ONLY",
    )
    class Cursor:
        def __init__(self): self.query = ""
        def execute(self,query,args):
            self.query = query
            assert args == (2,)
        def fetchall(self): return [row]
        def __enter__(self): return self
        def __exit__(self,*_): return False
    class Conn:
        def __init__(self): self.cursor_obj = Cursor()
        def cursor(self): return self.cursor_obj
        def __enter__(self): return self
        def __exit__(self,*_): return False
    conn = Conn()
    monkeypatch.setattr(persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(persistence, "_connect", lambda: conn)
    chosen = research_backfill_v1.pending_fixtures(2)
    assert len(chosen) == 1
    assert chosen[0]["fixture_id"] == 1612077
    assert "queue_lane" not in chosen[0]
    assert "lane_rank" not in chosen[0]
    assert "PARTITION BY queue_lane" in conn.cursor_obj.query
