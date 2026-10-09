"""Bounded provider observations for match-level facts, separate from sport/market projections.

Only active or completed registry fixtures are eligible. The pipeline never treats
unreported statistics as zero or uses observations to regrade pre-match decisions.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import automation_v2 as v2, persistence

MAX_FIXTURES = max(0, min(1, int(os.getenv("SOCCER_MATCH_OBSERVATION_FIXTURES_PER_TICK", "1"))))
OBSERVED_STATES = {"1H", "HT", "2H", "ET", "BT", "P", "FT", "AET", "PEN"}

_PENDING_SQL = """
    SELECT f.fixture_id, f.league_id, f.league, f.country, f.season,
           f.round, f.kickoff, f.status, f.status_long,
           f.home_team_id, f.home_team, f.away_team_id, f.away_team,
           f.venue, f.city
    FROM soccer_fixtures f
    WHERE f.kickoff >= NOW() - interval '48 hours'
      AND f.kickoff <= NOW()
      AND f.status IN ('1H','HT','2H','ET','BT','P','FT','AET','PEN')
      AND f.home_team_id > 0 AND f.away_team_id > 0
      AND NOT EXISTS (
          SELECT 1 FROM soccer_feature_snapshots s
          WHERE s.fixture_id = f.fixture_id
            AND s.stage = 'FIXTURE_OBSERVATION'
            AND s.captured_at > NOW() - CASE
                WHEN f.status IN ('FT','AET','PEN') THEN interval '48 hours'
                ELSE interval '20 minutes' END
      )
    ORDER BY CASE WHEN EXISTS (
                SELECT 1 FROM soccer_model_runs m WHERE m.fixture_id=f.fixture_id
             ) THEN 0 ELSE 1 END,
             CASE WHEN f.status IN ('FT','AET','PEN') THEN 0 ELSE 1 END,
             f.kickoff ASC, f.fixture_id
    LIMIT %s
"""

def pending_fixtures(limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or not persistence.persistence_configured():
        return []
    names = ("fixture_id","league_id","league","country","season",
             "round","kickoff","status","status_long","home_team_id",
             "home_team","away_team_id","away_team","venue","city")
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(_PENDING_SQL, (limit,))
            return [dict(zip(names,row)) for row in cur.fetchall()]


async def collect(tick: dict[str, Any]) -> list[dict[str, Any]]:
    if MAX_FIXTURES <= 0:
        tick["fixture_observation_status"] = "DISABLED"
        return []
    # The main scheduler and historical backfill always spend their budget first.
    if v2._API_CALLS_THIS_TICK > v2.MAX_API_CALLS_PER_TICK - 6:
        tick["fixture_observation_status"] = "DEFERRED_TICK_BUDGET"
        return []
    if v2._LAST_DAILY_REMAINING is not None and v2._LAST_DAILY_REMAINING <= 120:
        tick["fixture_observation_status"] = "DEFERRED_DAILY_QUOTA"
        return []

    fixtures = pending_fixtures(MAX_FIXTURES)
    tick["fixture_observation_selected_fixture_ids"] = [x["fixture_id"] for x in fixtures]
    events: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    for fx in fixtures:
        if v2._API_CALLS_THIS_TICK > v2.MAX_API_CALLS_PER_TICK - 4:
            attempts.append({"fixture_id": fx["fixture_id"], "status": "DEFERRED_TICK_BUDGET"})
            continue
        errors: list[str] = []
        payloads: dict[str, list[Any]] = {}
        for endpoint, kind in (
            ("fixtures/statistics","fixture_statistics"),
            ("fixtures/players","fixture_players"),
            ("fixtures/lineups","fixture_lineups"),
        ):
            try:
                result = await v2._budgeted_api_get(endpoint, {"fixture": int(fx["fixture_id"])})
                payloads[kind] = (result.get("response") or [])[:60 if kind == "fixture_players" else 2]
            except Exception as exc:
                payloads[kind] = []
                errors.append(f"{endpoint}: {type(exc).__name__}")
        scope = "POSTGAME_OBSERVATION" if str(fx["status"]).upper() in {"FT","AET","PEN"} else "LIVE_OBSERVATION"
        normalized = {
            **fx,
            "kickoff": fx["kickoff"].isoformat() if hasattr(fx["kickoff"],"isoformat") else fx["kickoff"],
        }
        counts = {kind: len(rows) for kind, rows in payloads.items()}
        attempts.append({"fixture_id":fx["fixture_id"],"status":"CAPTURED" if any(counts.values()) else
                         ("ERROR" if errors else "PROVIDER_EMPTY"),**counts,"errors":errors})
        events.append({
            "event_type": "FIXTURE_OBSERVATION_RESEARCH",
            "stage": "FIXTURE_OBSERVATION",
            "fixture": normalized,
            "coverage": {"data_tier":"RESEARCH_ONLY"},
            "sporting": {**payloads, "collection_scope": scope},
            "model_version": tick.get("model_version") or "SOCCER_EDGE_PROVIDER_OBSERVATION",
            "classification": None,
            "tier": None,
            "stake_units": 0.0,
            "bet_eligible": False,
            "raw_projection": None,
            "market_decision": None,
            "notes": ["Observed provider match data only; no market inference, pre-match model change or bet.", *errors],
        })
    tick["fixture_observation_attempts"] = attempts
    tick["fixture_observation_count"] = len(events)
    tick["fixture_observation_status"] = "CAPTURED" if events else ("NO_ELIGIBLE_FIXTURES" if not fixtures else "DEFERRED")
    return events
