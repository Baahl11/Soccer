"""Bounded, authenticated, READ-ONLY export of genuine frozen soccer player models.

Only official fixture IDs explicitly supplied by the existing verified
postgame registry can be requested. No schema setup, Postgres writes, provider
calls, odds, retroactive forecasts, or Render subprocesses. Exact as-of
restrictions are enforced in SQL, again after fetching, and by the downstream
OOS join. Deployed only with a main-branch pinned GitHub OIDC route.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
STATUS = "VERIFIED_FROZEN_PLAYER_PREGAME_READONLY"
MAX_FIXTURES = 25
MAX_RESPONSE_BYTES = 1_500_000
MAX_DB_STATEMENT_MS = 2500
PREGAME_STAGES = ("T-10", "T-20", "T-30", "T-40")

_SQL = """
SELECT e.event_id, e.fixture_id, e.generated_at, e.stage,
       f.kickoff, f.home_team_id, f.away_team_id,
       e.payload->'player_shots_intelligence' AS shots,
       e.payload->'player_sot_intelligence' AS sot
FROM soccer_refresh_events e
JOIN soccer_fixtures f ON f.fixture_id = e.fixture_id
WHERE e.fixture_id = %s
  AND e.generated_at < f.kickoff
  AND e.generated_at <= NOW()
  AND f.kickoff < NOW()
  AND e.stage IN ('T-40','T-30','T-20','T-10')
  AND (e.payload ? 'player_shots_intelligence'
       OR e.payload ? 'player_sot_intelligence')
ORDER BY CASE e.stage
  WHEN 'T-10' THEN 4 WHEN 'T-20' THEN 3
  WHEN 'T-30' THEN 2 WHEN 'T-40' THEN 1 ELSE 0
END DESC, e.generated_at DESC
LIMIT 1
"""

def validate_fixture_ids(fixture_ids: Any) -> list[int]:
    if not isinstance(fixture_ids, list) or not 1 <= len(fixture_ids) <= MAX_FIXTURES:
        raise ValueError("FIXTURE_IDS_REQUIRED_LIMIT_25")
    ids: list[int] = []
    for raw in fixture_ids:
        if type(raw) is not int or not 0 < raw < 10**13:
            raise ValueError("INVALID_OFFICIAL_FIXTURE_ID")
        if raw in ids:
            raise ValueError("DUPLICATE_FIXTURE_ID")
        ids.append(raw)
    return ids


def _iso(value: Any) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("TIMESTAMP_NOT_VERIFIED")
    return value.astimezone(timezone.utc).isoformat()


def _compact(row: tuple[Any, ...]) -> dict[str, Any]:
    event_id, fixture_id, observed_at, stage, kickoff, home, away, shots, sot = row
    if type(event_id) is not int or event_id <= 0 or stage not in PREGAME_STAGES:
        raise ValueError("INVALID_FROZEN_EVENT")
    frozen, kick = _iso(observed_at), _iso(kickoff)
    if not (observed_at < kickoff <= datetime.now(timezone.utc)):
        raise ValueError("NOT_A_PREGAME_SNAPSHOT")
    if type(fixture_id) is not int or type(home) is not int or type(away) is not int:
        raise ValueError("INVALID_FIXTURE_IDENTITIES")
    if not (home > 0 and away > 0 and home != away):
        raise ValueError("INVALID_HOME_AWAY_TEAM_IDS")
    if not isinstance(shots, dict) and not isinstance(sot, dict):
        raise ValueError("NO_FROZEN_PLAYER_MODELS")
    return {
        "fixture_id": fixture_id,
        "generated_at": frozen,
        "stage": stage,
        "kickoff": kick,
        "home_team_id": home,
        "away_team_id": away,
        "source_event_id": event_id,
        "event_payload": {
            "fixture": {"fixture_id": fixture_id, "kickoff": kick},
            "player_shots_intelligence": shots if isinstance(shots, dict) else None,
            "player_sot_intelligence": sot if isinstance(sot, dict) else None,
        },
    }


def export_frozen(fixture_ids: Any) -> dict[str, Any]:
    """Single indexed LIMIT 1 per requested fixture; read-only transaction."""
    ids = validate_fixture_ids(fixture_ids)
    if not persistence.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "queried_fixtures": len(ids),
            "rows": [],
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }
    rows: list[dict[str, Any]] = []
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute("SET LOCAL statement_timeout = '2500ms'")
                for fid in ids:
                    cur.execute(_SQL, (fid,))
                    record = cur.fetchone()
                    if record is not None:
                        compact = _compact(record)
                        if compact["fixture_id"] != fid:
                            raise ValueError("WRONG_FIXTURE_FETCHED")
                        rows.append(compact)

    import json
    output: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "source": "POSTGRES_CANONICAL_SOCCER_REFRESH_EVENTS",
        "selection": "ONE_CANONICAL_PREGAME_EVENT_PER_FIXTURE_STAGE_T10_T20_T30_T40",
        "queried_fixtures": len(ids),
        "frozen_fixture_rows_found": len(rows),
        "source_event_ids": [x["source_event_id"] for x in rows],
        "rows": rows,
        "provider_requests_added": 0,
        "retroactive_predictions_created": False,
        "db_write_queries": 0,
        "production_promotion_allowed": False,
    }
    if len(json.dumps(output, default=str).encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("FROZEN_MODEL_RESPONSE_TOO_LARGE")
    return output
