"""FE11 universal postgresql writer, separate from canonical sporting decisions.

All SQL joins are exact fixture/kickoff/team matches. Prematch baseline must
be observed strictly before kickoff; postgame observations require that frozen
baseline. Bookmaker data, betting outputs and model weights cannot be stored.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from mcp_gateway import fe11_universal_lifecycle_v1 as lifecycle

PREGAME_INSERT_SQL = """
    WITH incoming AS (
        SELECT * FROM jsonb_to_recordset(%s::jsonb) AS x(
            fixture_id BIGINT, kickoff TIMESTAMPTZ, league_id BIGINT,
            home_team_id BIGINT, away_team_id BIGINT, status TEXT
        )
    )
    INSERT INTO soccer_fe11_pregame_baselines (
        fixture_id, captured_at, kickoff, league_id, home_team_id,
        away_team_id, source, source_status, snapshot,
        classification, bet_eligible
    )
    SELECT f.fixture_id, %s::timestamptz, f.kickoff,
           f.league_id, f.home_team_id, f.away_team_id,
           'API_FOOTBALL_FIXTURES_DATE', 'NS',
           jsonb_build_object(
               'source_observed_at_utc', %s::text,
               'capture_kind', 'ALL_ELIGIBLE_PREMATCH_IDENTITY_ONLY',
               'fixture', to_jsonb(x),
               'sporting_projection', 'NOT_VERIFIED',
               'availability', 'NOT_VERIFIED',
               'corners', 'NOT_VERIFIED',
               'bookmaker_price', 'NOT_VERIFIED'
           ),
           'RESEARCH_ONLY', FALSE
    FROM incoming x
    JOIN soccer_fixtures f ON f.fixture_id = x.fixture_id
    WHERE x.status = 'NS' AND f.status = 'NS'
      AND f.kickoff = x.kickoff AND f.kickoff > %s::timestamptz
      AND f.league_id = x.league_id
      AND f.home_team_id = x.home_team_id
      AND f.away_team_id = x.away_team_id
    ON CONFLICT (fixture_id) DO NOTHING
"""

POSTGAME_INSERT_SQL = """
    WITH incoming AS (
        SELECT * FROM jsonb_to_recordset(%s::jsonb) AS x(
            fixture_id BIGINT, kickoff TIMESTAMPTZ,
            home_team_id BIGINT, away_team_id BIGINT,
            status TEXT, goals JSONB, halftime JSONB,
            regulation_fulltime JSONB, second_half JSONB
        )
    )
    INSERT INTO soccer_fe11_official_outcome_snapshots (
        fixture_id, observed_at, kickoff, final_status,
        home_goals, away_goals, halftime_home_goals,
        halftime_away_goals, regulation_home_goals,
        regulation_away_goals, second_half_home_goals,
        second_half_away_goals, corners_home, corners_away,
        corners_verified, source, payload, bet_eligible
    )
    SELECT b.fixture_id, %s::timestamptz, b.kickoff, x.status,
           (x.goals->>'home')::integer,
           (x.goals->>'away')::integer,
           (x.halftime->>'home')::integer,
           (x.halftime->>'away')::integer,
           (x.regulation_fulltime->>'home')::integer,
           (x.regulation_fulltime->>'away')::integer,
           (x.second_half->>'home')::integer,
           (x.second_half->>'away')::integer,
           NULL::integer, NULL::integer, FALSE,
           'API_FOOTBALL_FIXTURES_DATE',
           jsonb_build_object(
               'source', 'API_FOOTBALL_FIXTURES_DATE',
               'observed_at_utc', %s::text,
               'score', to_jsonb(x),
               'corners_status', 'NOT_VERIFIED',
               'classification', 'RESULT_ONLY_NO_BET'
           ),
           FALSE
    FROM incoming x
    JOIN soccer_fe11_pregame_baselines b ON b.fixture_id = x.fixture_id
    JOIN soccer_fixtures f ON f.fixture_id = b.fixture_id
    WHERE b.kickoff = x.kickoff AND b.captured_at < b.kickoff
      AND b.home_team_id = x.home_team_id
      AND b.away_team_id = x.away_team_id
      AND f.kickoff = b.kickoff
      AND f.home_team_id = b.home_team_id
      AND f.away_team_id = b.away_team_id
      AND x.status IN ('FT','AET','PEN')
      AND %s::timestamptz >= b.kickoff
    ON CONFLICT (fixture_id, observed_at) DO NOTHING
"""

INVENTORY_SQL = """
    SELECT
        (SELECT COUNT(*) FROM soccer_fe11_pregame_baselines)::integer,
        (SELECT COUNT(DISTINCT fixture_id)
         FROM soccer_fe11_official_outcome_snapshots)::integer,
        (SELECT COUNT(*) FROM soccer_fe11_pregame_baselines b
         WHERE b.kickoff > NOW())::integer,
        (SELECT COUNT(DISTINCT fixture_id)
         FROM soccer_fe11_official_outcome_snapshots s
         WHERE s.halftime_home_goals IS NOT NULL
           AND s.halftime_away_goals IS NOT NULL
           AND s.second_half_home_goals IS NOT NULL
           AND s.second_half_away_goals IS NOT NULL)::integer,
        (SELECT COUNT(DISTINCT fixture_id)
         FROM soccer_fe11_verified_corner_snapshots)::integer,
        (SELECT COUNT(DISTINCT o.fixture_id)
         FROM soccer_fe11_official_outcome_snapshots o
         JOIN soccer_fe11_verified_corner_snapshots c
           ON c.fixture_id = o.fixture_id
         WHERE o.final_status = 'FT'
           AND o.halftime_home_goals IS NOT NULL
           AND o.halftime_away_goals IS NOT NULL
           AND o.second_half_home_goals IS NOT NULL
           AND o.second_half_away_goals IS NOT NULL)::integer
"""


def persist(cur: Any, tick: dict[str, Any], upsert_fixture: Callable) -> None:
    """No retrospective baseline or synthetic labels. Invalid payload: no writes."""
    prematch, outcomes = lifecycle.safe_tick_capture(
        tick.get("fe11_universal_lifecycle"),
        now_utc=str(tick.get("generated_at_utc") or ""),
    )
    if not prematch and not outcomes:
        return
    observed = tick["generated_at_utc"]
    # Existing discovery attaches at most 300 fixtures; here all verified
    # approved-league records must receive canonical fixture identities.
    for fx in prematch:
        upsert_fixture(cur, fx)
    if prematch:
        cur.execute(PREGAME_INSERT_SQL, (json.dumps(prematch), observed, observed, observed))
    if outcomes:
        cur.execute(POSTGAME_INSERT_SQL, (json.dumps(outcomes), observed, observed, observed))


def inventory(cur: Any) -> dict[str, Any]:
    """Aggregate-only read via already-open trusted internal DB connection."""
    cur.execute(INVENTORY_SQL)
    pre, done, future, period, corners_count, full_components = (
        cur.fetchone() or (0, 0, 0, 0, 0, 0)
    )
    return {
        "status": "VERIFIED_READ",
        "eligible_pregame_baselines": int(pre),
        "matched_official_final_fixtures": int(done),
        "current_future_baselines": int(future),
        "official_period_complete_fixtures": int(period),
        "official_ft_corners_verified_fixtures": int(corners_count),
        "official_period_and_corners_complete_fixtures": int(full_components),
        "joint_goals_corners_handicap_training_rows": 0,
        "joint_rows_status": "NOT_VERIFIED_JOINT_OOS_CALIBRATION",
        "provider_requests_added": 0,
        "production_bets_authorized": False,
    }
