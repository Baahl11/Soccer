"""Data retention, low-coverage sporting research and immutable BET gates."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

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
        "kickoff": "2026-10-08T22:00:00Z", "status": "NS",
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
