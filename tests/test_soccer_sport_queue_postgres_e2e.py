"""Real PostgreSQL verification of the universal sports queue; disposable DB only."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from mcp_gateway import soccer_sport_queue_v1 as queue

pytestmark = pytest.mark.skipif(
    not os.environ.get("SOCCER_UNIVERSAL_TEST_POSTGRES_URL"),
    reason="Only runs against an explicitly named disposable test database.",
)


def test_real_sql_queue_idempotence_and_no_model_promotion():
    import psycopg

    url = os.environ["SOCCER_UNIVERSAL_TEST_POSTGRES_URL"]
    assert "universal_ci_disposable" in url, "Refuse to write to non-test DB."
    with psycopg.connect(url) as conn:
        with conn.cursor() as cur:
            cur.execute(Path("mcp_gateway/schema.sql").read_text(encoding="utf-8"))
            kickoff = datetime.now(timezone.utc) + timedelta(hours=4)
            cur.execute(
                """
                INSERT INTO soccer_fixtures
                    (fixture_id,kickoff,status,home_team_id,away_team_id,home_team,away_team)
                VALUES (991234567,%s,'NS',1,2,'Local','Visitante')
                """, (kickoff,),
            )
            queue.sync_pending(cur)
            queue.sync_pending(cur)
            cur.execute(
                "SELECT sport_status,attempt_count FROM soccer_sport_screen_queue WHERE fixture_id=991234567"
            )
            assert cur.fetchone() == ("QUEUED", 0)
            event = {
                "event_type": "SOCCER_REFRESH", "stage": "SPORT_PREVIEW",
                "fixture": {"fixture_id": 991234567, "kickoff": kickoff.isoformat()},
                "raw_sport_show_as_forecast": True,
                "raw_sport_screen_status": "PREMATCH_LIMITED_PROJECTION",
                "model_version": "TEST_ONLY",
            }
            at = datetime.now(timezone.utc)
            queue.record_event(cur, event, at=at.isoformat())
            # Later low-data refresh must not erase a genuine captured projection.
            event["raw_sport_show_as_forecast"] = False
            queue.record_event(cur, event, at=(at + timedelta(minutes=2)).isoformat())
            cur.execute(
                """SELECT sport_status,attempt_count,last_model_version
                   FROM soccer_sport_screen_queue WHERE fixture_id=991234567"""
            )
            assert cur.fetchone() == ("MODELED_LIMITED", 2, "TEST_ONLY")
            cur.execute("SELECT COUNT(*) FROM soccer_market_snapshots WHERE fixture_id=991234567")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT COUNT(*) FROM soccer_model_runs WHERE fixture_id=991234567")
            assert cur.fetchone()[0] == 0
