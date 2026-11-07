"""Offline verified postgame FT corners queue for all FE11 approved fixtures."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from mcp_gateway import fe11_postgame_corners_v1 as collector
from mcp_gateway import fe11_corners_persistence_v1 as writer


def provider_rows(home=0, away="5", *, team_a=101, team_b=202):
    return [
        {"team": {"id": team_a}, "statistics": [
            {"type": "Corner Kicks", "value": home}
        ]},
        {"team": {"id": team_b}, "statistics": [
            {"type": "Corner Kicks", "value": away}
        ]},
    ]


def test_strict_provider_team_ids_exact_and_real_zero_is_not_missing():
    assert collector.parse_verified_ft_corners(
        provider_rows(), 101, 202
    ) == {"home": 0, "away": 5}
    assert collector.parse_verified_ft_corners(
        provider_rows(), 202, 101
    ) == {"home": 5, "away": 0}
    assert collector.parse_verified_ft_corners(provider_rows(team_b=999), 101, 202) is None
    assert collector.parse_verified_ft_corners(provider_rows(team_b=101), 101, 202) is None
    assert collector.parse_verified_ft_corners(provider_rows(away=None), 101, 202) is None
    assert collector.parse_verified_ft_corners(provider_rows(away="-1"), 101, 202) is None
    assert collector.parse_verified_ft_corners(provider_rows(away="not available"), 101, 202) is None


def test_ambiguous_corner_rows_are_rejected_even_if_first_value_valid():
    rows = provider_rows()
    rows[0]["statistics"].append({"type": "Corner Kicks", "value": 9})
    assert collector.parse_verified_ft_corners(rows, 101, 202) is None
    assert collector.parse_verified_ft_corners(rows[:1], 101, 202) is None


class CursorSpy:
    def __init__(self):
        self.statements = []
    def execute(self, sql, values):
        self.statements.append((sql, values))


def tick():
    return {"generated_at_utc": "2026-11-07T05:30:00+00:00"}


def selected():
    return [{
        "fixture_id": 333,
        "home_team_id": 101, "away_team_id": 202,
        "kickoff": "2026-11-07T01:00:00+00:00",
    }]


def test_successful_queued_ft_corners_have_exact_identity_and_no_market(monkeypatch):
    import mcp_gateway.automation_v2 as v2
    monkeypatch.setattr(collector, "pending", lambda limit: selected())
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 3)
    monkeypatch.setattr(v2, "_LAST_DAILY_REMAINING", 5000)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 28)

    async def api(endpoint, params):
        assert endpoint == "fixtures/statistics"
        assert params == {"fixture": 333}
        v2._API_CALLS_THIS_TICK += 1
        return {"response": provider_rows()}

    monkeypatch.setattr(v2, "_budgeted_api_get", api)
    t = tick()
    asyncio.run(collector.collect(t))
    s = t["fe11_postgame_corner_summary"]
    assert s["status"] == "VERIFIED_CAPTURES"
    assert s["selected"] == s["captured"] == 1
    assert s["provider_requests_added"] == 1
    assert t["fe11_postgame_corner_observations"][0]["home_corners"] == 0
    assert t["fe11_postgame_corner_observations"][0]["away_corners"] == 5
    assert t["fe11_postgame_corner_observations"][0]["bet_eligible"] is False
    spy = CursorSpy()
    writer.persist(spy, t)
    assert len(spy.statements) == 2
    assert "JOIN soccer_fe11_pregame_baselines" in spy.statements[1][0]
    assert "o.final_status = 'FT'" in spy.statements[1][0]
    assert "ON CONFLICT (fixture_id, observed_at) DO NOTHING" in spy.statements[1][0]


def test_missing_stats_cause_attempt_only_never_corner_snapshot(monkeypatch):
    import mcp_gateway.automation_v2 as v2
    monkeypatch.setattr(collector, "pending", lambda limit: selected())
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 1)
    monkeypatch.setattr(v2, "_LAST_DAILY_REMAINING", 5000)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 28)
    async def api(endpoint, params):
        v2._API_CALLS_THIS_TICK += 1
        return {"response": provider_rows(away=None)}
    monkeypatch.setattr(v2, "_budgeted_api_get", api)
    t = tick()
    asyncio.run(collector.collect(t))
    assert t["fe11_postgame_corner_observations"] == []
    assert t["fe11_postgame_corner_attempts"][0]["status"] == "NOT_VERIFIED_PROVIDER_STATS"
    spy = CursorSpy()
    writer.persist(spy, t)
    assert len(spy.statements) == 1
    assert "soccer_fe11_corner_capture_attempts" in spy.statements[0][0]


def test_provider_quota_reduces_work_without_inferred_corners(monkeypatch):
    import mcp_gateway.automation_v2 as v2
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 25)
    monkeypatch.setattr(v2, "_LAST_DAILY_REMAINING", 5000)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 28)
    t = tick()
    asyncio.run(collector.collect(t))
    assert t["fe11_postgame_corner_summary"]["status"] == "DEFERRED_TICK_BUDGET"
    assert t["fe11_postgame_corner_attempts"] == []


def test_daily_quota_can_block_all_postgame_stats(monkeypatch):
    import mcp_gateway.automation_v2 as v2
    monkeypatch.setattr(v2, "_API_CALLS_THIS_TICK", 2)
    monkeypatch.setattr(v2, "_LAST_DAILY_REMAINING", 75)
    monkeypatch.setattr(v2, "MAX_API_CALLS_PER_TICK", 28)
    t = tick()
    asyncio.run(collector.collect(t))
    assert t["fe11_postgame_corner_summary"]["status"] == "DEFERRED_DAILY_QUOTA"


def test_persistence_rejects_tampered_bookmaker_offer_and_missing_source():
    time = datetime.now(timezone.utc).isoformat()
    sample = {
        "fixture_id": 333,
        "kickoff": (datetime.now(timezone.utc)-timedelta(hours=3)).isoformat(),
        "home_team_id": 101, "away_team_id": 202,
        "observed_at_utc": time,
        "home_corners": 0, "away_corners": 5,
        "period": "FT_90", "source": "UNTRUSTED",
        "source_response_verified": True, "bet_eligible": False,
    }
    t = {
        "fe11_postgame_corner_summary": {"version": collector.VERSION,
                                         "bet_eligible": False,
                                         "market_inputs_used": False,
                                         "captured": 1},
        "fe11_postgame_corner_observations": [sample],
        "fe11_postgame_corner_attempts": [{
            "fixture_id": 333, "kickoff": sample["kickoff"],
            "observed_at_utc": time, "status": "VERIFIED_FT_CORNERS",
            "provider_requests_added": 1
        }],
    }
    spy = CursorSpy()
    writer.persist(spy, t)
    assert spy.statements == []


def test_queue_is_all_matched_scores_not_top_three_deepdives():
    sql = collector.PENDING_SQL
    assert "soccer_fe11_pregame_baselines" in sql
    assert "soccer_fe11_official_outcome_snapshots" in sql
    assert "final_status = 'FT'" in sql
    assert "soccer_fe11_corner_capture_attempts" in sql
    assert "12 hours" in sql and "30 days" in sql
    assert "LIMIT %s" in sql
    worker = (Path(__file__).resolve().parents[1]/"mcp_gateway"/"tick_worker.py").read_text()
    assert "await fe11_postgame_corners_v1.collect(payload)" in worker
    assert "fixture_observations_v1.collect(payload)" in worker
    assert worker.index("await fe11_postgame_corners_v1.collect(payload)") < worker.index(
        "fixture_observations_v1.collect(payload)"
    )


def test_no_overwrite_of_prior_scored_predictions_or_historical_truth():
    assert "ON CONFLICT (fixture_id, observed_at) DO NOTHING" in writer.CORNER_INSERT_SQL
    assert "b.captured_at < b.kickoff" in writer.CORNER_INSERT_SQL
    assert "x.observed_at_utc >= b.kickoff" in writer.CORNER_INSERT_SQL
    assert "o.final_status = 'FT'" in writer.CORNER_INSERT_SQL
    assert "bookmaker" not in writer.CORNER_INSERT_SQL.lower()
