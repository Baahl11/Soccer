"""Bounded display-only research backfill for recently persisted fixtures.

Run ONLY from the existing scheduler worker, after the canonical decision engine
has completed. Does not update sporting projections, model weights, market
prices, availability confidence, stake sizes, or existing rows.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import persistence

MAX_BACKFILL_FIXTURES_PER_TICK = max(
    0, min(2, int(os.getenv("SOCCER_EDGE_MAX_RESEARCH_BACKFILL_PER_TICK", "2")))
)

# Prioritize fixtures with saved model runs so their missing *input* evidence
# can be recovered, but never use model output to manufacture team statistics.
_PENDING_SQL = """
    SELECT f.fixture_id, f.league_id, f.league, f.country, f.season,
           f.round, f.kickoff, f.status, f.status_long,
           f.home_team_id, f.home_team, f.away_team_id, f.away_team,
           f.venue, f.city,
           COALESCE((
               SELECT sf.data_tier FROM soccer_feature_snapshots sf
               WHERE sf.fixture_id = f.fixture_id
               ORDER BY sf.captured_at DESC, sf.snapshot_id DESC LIMIT 1
           ), 'RESEARCH_ONLY') AS data_tier
    FROM soccer_fixtures f
    WHERE f.kickoff BETWEEN NOW() - interval '30 hours' AND NOW() + interval '2 hours'
      AND f.home_team_id > 0 AND f.away_team_id > 0
      AND f.league_id > 0 AND f.season > 0
      AND f.status IS DISTINCT FROM 'CANC'
      AND f.status IS DISTINCT FROM 'PST'
      AND NOT EXISTS (
         SELECT 1 FROM soccer_feature_snapshots s
         WHERE s.fixture_id = f.fixture_id
           AND s.payload #>> '{features,team_performance.home_played_split,value}' IS NOT NULL
           AND s.payload #>> '{features,team_performance.away_played_split,value}' IS NOT NULL
      )
      AND NOT EXISTS (
         SELECT 1 FROM soccer_feature_snapshots s
         WHERE s.fixture_id = f.fixture_id
           AND s.stage = 'RESEARCH_BACKFILL'
           AND s.captured_at > NOW() - interval '12 hours'
      )
    ORDER BY
        CASE WHEN EXISTS (
            SELECT 1 FROM soccer_model_runs m WHERE m.fixture_id = f.fixture_id
        ) THEN 0 ELSE 1 END,
        f.kickoff DESC, f.fixture_id
    LIMIT %s
"""


def pending_fixtures(limit: int = MAX_BACKFILL_FIXTURES_PER_TICK) -> list[dict[str, Any]]:
    """Read a small candidate batch from internal Render Postgres."""
    if limit <= 0 or not persistence.persistence_configured():
        return []
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_PENDING_SQL, (min(MAX_BACKFILL_FIXTURES_PER_TICK, limit),))
            records = cur.fetchall()
    fields = (
        "fixture_id", "league_id", "league", "country", "season",
        "round", "kickoff", "status", "status_long",
        "home_team_id", "home_team", "away_team_id", "away_team",
        "venue", "city", "data_tier",
    )
    return [dict(zip(fields, record)) for record in records]


async def collect(tick: dict[str, Any]) -> list[dict[str, Any]]:
    """Use at most four provider calls, only when the daily/tick budget allows."""
    if MAX_BACKFILL_FIXTURES_PER_TICK <= 0:
        return []
    calls_left = v2.MAX_API_CALLS_PER_TICK - v2._API_CALLS_THIS_TICK
    if calls_left < 6:
        return []
    if v2._LAST_DAILY_REMAINING is not None and v2._LAST_DAILY_REMAINING <= 70:
        return []
    limit = min(MAX_BACKFILL_FIXTURES_PER_TICK, (calls_left - 2) // 2)
    if limit <= 0:
        return []
    fixtures = pending_fixtures(limit)
    now = datetime.now(timezone.utc)
    events: list[dict[str, Any]] = []
    for fx in fixtures:
        if v2.MAX_API_CALLS_PER_TICK - v2._API_CALLS_THIS_TICK < 4:
            break
        if v2._LAST_DAILY_REMAINING is not None and v2._LAST_DAILY_REMAINING <= 70:
            break
        home_stats: dict[str, Any] = {}
        away_stats: dict[str, Any] = {}
        errors: list[str] = []
        try:
            home_stats = await base._team_stats(
                int(fx["home_team_id"]), int(fx["league_id"]), int(fx["season"]), now
            )
        except Exception as exc:
            errors.append("HOME_STATS_UNAVAILABLE:" + type(exc).__name__)
        try:
            away_stats = await base._team_stats(
                int(fx["away_team_id"]), int(fx["league_id"]), int(fx["season"]), now
            )
        except Exception as exc:
            errors.append("AWAY_STATS_UNAVAILABLE:" + type(exc).__name__)
        # Even a response without statistics is an explicit, time-bounded attempt.
        # No placeholder goals, injuries, xG, lineups or odds are invented.
        tier = fx.pop("data_tier")
        fx["kickoff"] = fx["kickoff"].isoformat() if hasattr(fx["kickoff"], "isoformat") else fx["kickoff"]
        events.append({
            "event_type": "SPORT_FEATURE_RESEARCH_BACKFILL",
            "stage": "RESEARCH_BACKFILL",
            "fixture": fx,
            "coverage": {"data_tier": tier},
            "sporting": {
                "home_stats": home_stats if isinstance(home_stats, dict) else {},
                "away_stats": away_stats if isinstance(away_stats, dict) else {},
                "sport_data": "RESEARCH_ONLY",
                "collection_scope": "HISTORICAL_TEAM_STATS_ONLY",
            },
            "classification": None,
            "availability_confidence": None,
            "bet_eligible": False,
            "raw_projection": None,
            "market_decision": None,
            "notes": ["Sport evidence only; historical inputs never promote a BET.", *errors],
        })
    return events
