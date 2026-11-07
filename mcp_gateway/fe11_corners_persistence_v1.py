"""Write independently verified FE11 FT corners, append-only and fail-closed."""
from __future__ import annotations

import json
from typing import Any

from mcp_gateway import fe11_universal_lifecycle_v1 as lifecycle
from mcp_gateway import fe11_postgame_corners_v1 as corners

ATTEMPT_SQL = """
    WITH incoming AS (
        SELECT * FROM jsonb_to_recordset(%s::jsonb) AS x(
            fixture_id BIGINT, kickoff TIMESTAMPTZ,
            observed_at_utc TIMESTAMPTZ, status TEXT,
            provider_requests_added INTEGER
        )
    )
    INSERT INTO soccer_fe11_corner_capture_attempts (
        fixture_id, last_attempt_at, attempt_count, last_status, provider_requests_added
    )
    SELECT b.fixture_id, x.observed_at_utc, 1, x.status, x.provider_requests_added
    FROM incoming x
    JOIN soccer_fe11_pregame_baselines b ON b.fixture_id = x.fixture_id
    WHERE b.kickoff = x.kickoff
      AND b.captured_at < b.kickoff
      AND x.observed_at_utc >= b.kickoff
      AND x.status IN (
          'VERIFIED_FT_CORNERS',
          'NOT_VERIFIED_PROVIDER_STATS',
          'NOT_VERIFIED_PROVIDER_ERROR'
      )
      AND x.provider_requests_added BETWEEN 0 AND 1
    ON CONFLICT (fixture_id) DO UPDATE SET
        last_attempt_at = EXCLUDED.last_attempt_at,
        attempt_count = LEAST(20, soccer_fe11_corner_capture_attempts.attempt_count + 1),
        last_status = EXCLUDED.last_status,
        provider_requests_added = EXCLUDED.provider_requests_added
"""

CORNER_INSERT_SQL = """
    WITH incoming AS (
        SELECT * FROM jsonb_to_recordset(%s::jsonb) AS x(
            fixture_id BIGINT, kickoff TIMESTAMPTZ,
            home_team_id BIGINT, away_team_id BIGINT,
            observed_at_utc TIMESTAMPTZ, home_corners INTEGER,
            away_corners INTEGER, period TEXT, source TEXT,
            source_response_verified BOOLEAN, bet_eligible BOOLEAN
        )
    )
    INSERT INTO soccer_fe11_verified_corner_snapshots (
        fixture_id, observed_at, kickoff, home_corners, away_corners,
        period, source, source_response_verified, bet_eligible
    )
    SELECT b.fixture_id, x.observed_at_utc, b.kickoff,
           x.home_corners, x.away_corners,
           'FT_90', 'API_FOOTBALL_FIXTURE_STATISTICS', TRUE, FALSE
    FROM incoming x
    JOIN soccer_fe11_pregame_baselines b ON b.fixture_id = x.fixture_id
    JOIN soccer_fixtures f ON f.fixture_id = b.fixture_id
    WHERE b.kickoff = x.kickoff
      AND f.kickoff = b.kickoff
      AND b.home_team_id = x.home_team_id
      AND b.away_team_id = x.away_team_id
      AND f.home_team_id = x.home_team_id
      AND f.away_team_id = x.away_team_id
      AND b.captured_at < b.kickoff
      AND x.observed_at_utc >= b.kickoff
      AND x.home_corners BETWEEN 0 AND 60
      AND x.away_corners BETWEEN 0 AND 60
      AND x.period = 'FT_90'
      AND x.source = 'API_FOOTBALL_FIXTURE_STATISTICS'
      AND x.source_response_verified IS TRUE
      AND x.bet_eligible IS FALSE
      AND EXISTS (
          SELECT 1 FROM soccer_fe11_official_outcome_snapshots o
          WHERE o.fixture_id = b.fixture_id AND o.final_status = 'FT'
            AND o.observed_at <= x.observed_at_utc
      )
    ON CONFLICT (fixture_id, observed_at) DO NOTHING
"""


def _valid_attempt(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    time = lifecycle._utc(row.get("observed_at_utc"))
    kickoff = lifecycle._utc(row.get("kickoff"))
    return (
        lifecycle._positive(row.get("fixture_id"))
        and time is not None and kickoff is not None and time >= kickoff
        and row.get("status") in {
            "VERIFIED_FT_CORNERS", "NOT_VERIFIED_PROVIDER_STATS",
            "NOT_VERIFIED_PROVIDER_ERROR"
        }
        and type(row.get("provider_requests_added")) is int
        and 0 <= row["provider_requests_added"] <= 1
    )


def _valid_corners(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    return (
        _valid_attempt({
            "fixture_id": row.get("fixture_id"),
            "kickoff": row.get("kickoff"),
            "observed_at_utc": row.get("observed_at_utc"),
            "status": "VERIFIED_FT_CORNERS",
            "provider_requests_added": 1,
        })
        and lifecycle._positive(row.get("home_team_id"))
        and lifecycle._positive(row.get("away_team_id"))
        and row["home_team_id"] != row["away_team_id"]
        and all(type(row.get(k)) is int and 0 <= row[k] <= 60
                for k in ("home_corners", "away_corners"))
        and row.get("period") == "FT_90"
        and row.get("source") == corners.SOURCE
        and row.get("source_response_verified") is True
        and row.get("bet_eligible") is False
    )


def persist(cur: Any, tick: dict[str, Any]) -> None:
    """Untrusted modified arrays are discarded completely, not partly written."""
    summary = tick.get("fe11_postgame_corner_summary")
    obs = tick.get("fe11_postgame_corner_observations")
    attempts = tick.get("fe11_postgame_corner_attempts")
    if (not isinstance(summary, dict) or summary.get("version") != corners.VERSION
            or summary.get("bet_eligible") is not False
            or summary.get("market_inputs_used") is not False
            or not isinstance(obs, list) or not isinstance(attempts, list)
            or len(obs) > corners.MAX_FIXTURES or len(attempts) > corners.MAX_FIXTURES
            or summary.get("captured") != len(obs)
            or any(not _valid_attempt(x) for x in attempts)
            or any(not _valid_corners(x) for x in obs)):
        return
    attempted_ids = {row["fixture_id"] for row in attempts}
    captured_ids = {row["fixture_id"] for row in obs}
    if (len(attempted_ids) != len(attempts)
            or len(captured_ids) != len(obs)
            or not captured_ids <= attempted_ids):
        return
    for row in obs:
        item = next(x for x in attempts if x["fixture_id"] == row["fixture_id"])
        if item["status"] != "VERIFIED_FT_CORNERS":
            return
        if item["observed_at_utc"] != row["observed_at_utc"]:
            return
    if attempts:
        cur.execute(ATTEMPT_SQL, (json.dumps(attempts),))
    if obs:
        cur.execute(CORNER_INSERT_SQL, (json.dumps(obs),))
