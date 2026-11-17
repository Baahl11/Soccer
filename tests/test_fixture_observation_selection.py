"""Execute the real selector with both live and final capture records."""
import os
import psycopg
import pytest

from mcp_gateway.fixture_observations_v1 import _PENDING_SQL


def test_live_capture_does_not_suppress_regulation_final():
    dsn = os.getenv("TEST_OBSERVATION_DSN")
    if not dsn:
        pytest.skip("Requires disposable PostgreSQL")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("""
            CREATE TEMP TABLE soccer_fixtures (
                fixture_id bigint, league_id bigint, league text, country text,
                season integer, round text, kickoff timestamptz, status text,
                status_long text, home_team_id bigint, home_team text,
                away_team_id bigint, away_team text, venue text, city text
            );
            CREATE TEMP TABLE soccer_model_runs (fixture_id bigint);
            CREATE TEMP TABLE soccer_feature_snapshots
                (fixture_id bigint, stage text, captured_at timestamptz);
            CREATE TEMP TABLE soccer_refresh_events (
                fixture_id bigint, stage text, event_type text,
                generated_at timestamptz, payload jsonb
            );
            INSERT INTO soccer_fixtures (fixture_id,kickoff,status,home_team_id,away_team_id)
            VALUES (1, now()-interval '3 hours','FT',10,20),
                   (2, now()-interval '3 hours','FT',11,21),
                   (3, now()-interval '1 hour','2H',12,22),
                   (4, now()-interval '3 hours','AET',13,23);
            INSERT INTO soccer_feature_snapshots
            SELECT i,'FIXTURE_OBSERVATION',now() FROM generate_series(1,4) i;
            INSERT INTO soccer_refresh_events VALUES
            (1,'FIXTURE_OBSERVATION','FIXTURE_OBSERVATION_RESEARCH',now(),
                '{"sporting":{"collection_scope":"LIVE_OBSERVATION"},"fixture":{"status":"2H"}}'),
            (2,'FIXTURE_OBSERVATION','FIXTURE_OBSERVATION_RESEARCH',now(),
                '{"sporting":{"collection_scope":"POSTGAME_OBSERVATION"},"fixture":{"status":"FT"}}'),
            (4,'FIXTURE_OBSERVATION','FIXTURE_OBSERVATION_RESEARCH',now(),
                '{"sporting":{"collection_scope":"POSTGAME_OBSERVATION"},"fixture":{"status":"FT"}}');
        """)
        cur.execute(_PENDING_SQL, (10,))
        assert [row[0] for row in cur.fetchall()] == [1, 4]
