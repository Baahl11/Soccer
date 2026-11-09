"""Offline safety checks: no Render/Postgres/prod network is contacted."""
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from mcp_gateway import player_frozen_oos_readonly_v1 as mod


def _event(fid=123, team_a=11, team_b=22, snapshot_minutes=30, stage="T-20"):
    kickoff = datetime.now(timezone.utc) - timedelta(hours=3)
    generated = kickoff - timedelta(minutes=snapshot_minutes)
    shots = {
        "status": "LIVE_RESEARCH_SHOTS_PROPS",
        "fixture_id": fid,
        "players": [{"player_id": 51, "team_id": team_a, "confirmed_starter": True,
                     "expected_shots": 1.4, "lines": [{"line": 0.5, "p_over": 0.7}]}]
    }
    return (900, fid, generated, stage, kickoff, team_a, team_b, shots, None)


def test_reject_injection_and_unbounded_inputs():
    for value in (None, [], list(range(1, 27)), ["123"], [True], [0], [-1],
                  [1, 1], [10**15], "123", {"ids": [123]}):
        with pytest.raises(ValueError):
            mod.validate_fixture_ids(value)
    assert mod.validate_fixture_ids([123, 456]) == [123, 456]


def test_exact_asof_provenance_and_fixture_ids():
    row = mod._compact(_event())
    assert row["fixture_id"] == 123
    assert row["source_event_id"] == 900
    assert row["generated_at"] < row["kickoff"]
    assert row["event_payload"]["player_shots_intelligence"]["players"][0]["player_id"] == 51
    assert row["event_payload"]["player_sot_intelligence"] is None
    with pytest.raises(ValueError):
        mod._compact(_event(snapshot_minutes=-1))
    with pytest.raises(ValueError):
        mod._compact(_event(team_b=11))
    with pytest.raises(ValueError):
        mod._compact(_event(stage="POSTGAME"))


class _Cursor:
    def __init__(self, rows):
        self.rows = iter(rows)
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, params=None):
        self.queries.append((query, params))

    def fetchone(self):
        return next(self.rows)


class _Conn:
    def __init__(self, rows):
        self.cur = _Cursor(rows)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def transaction(self):
        return self

    def cursor(self):
        return self.cur


def test_one_indexed_row_per_fixture_and_explicit_read_only():
    conn = _Conn([_event(fid=123), None])
    with patch.object(mod.persistence, "persistence_configured", return_value=True), \
         patch.object(mod.persistence, "_connect", return_value=conn):
        result = mod.export_frozen([123, 456])
    assert result["status"] == mod.STATUS
    assert result["queried_fixtures"] == 2
    assert result["frozen_fixture_rows_found"] == 1
    assert result["provider_requests_added"] == 0
    assert result["db_write_queries"] == 0
    assert result["production_promotion_allowed"] is False
    queries = conn.cur.queries
    assert any("SET TRANSACTION READ ONLY" in q for q, _ in queries)
    assert any("statement_timeout" in q for q, _ in queries)
    sql = [(q, p) for q, p in queries if "FROM soccer_refresh_events e" in q]
    assert [p for _, p in sql] == [(123,), (456,)]
    assert all("LIMIT 1" in q and "e.generated_at < f.kickoff" in q for q, _ in sql)


def test_no_broken_pg_connection_when_not_configured():
    with patch.object(mod.persistence, "persistence_configured", return_value=False):
        output = mod.export_frozen([123])
    assert output["status"] == "POSTGRES_NOT_CONFIGURED"
    assert output["rows"] == []
