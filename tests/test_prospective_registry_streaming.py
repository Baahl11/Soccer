"""Prove prospective SQL streaming does not materialize the whole JSON history.

Pure mocks; no live Postgres access, backfills, sporting rules or provider calls.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mcp_gateway import prospective_evaluation_registry_v1 as v


def _event(fixture_id: int, eligible: bool):
    return {
        "stage": "T-10",
        "classification": "BET",
        "tier": "B",
        "availability_confidence": 0.91,
        "coverage": {"data_tier": "A" if eligible else "C"},
        "fixture": {
            "fixture_id": fixture_id, "kickoff": "2026-10-13T17:00:00+00:00",
            "home_team": "Home FC", "away_team": "Away FC",
            "home_team_id": 91, "away_team_id": 92,
            "league_id": 10, "league": "Test League",
        },
        "lineups": {
            "both_xi_confirmed": True,
            "both_goalkeepers_confirmed": True,
        },
        "sporting_shortlist": {"shortlisted": True},
        "best_market": {
            "market": "Match Winner", "selection": "Home",
            "decimal_price": 2.10, "bookmaker": "Verified Book",
            "p_raw": 0.56, "p_market_fair": 0.50,
            "p_shrunk": 0.53, "prob_edge_pp": 3.0,
            "estimated_ev": 0.113, "tier": "B",
        },
    }


class FakeStreamingCursor:
    names = (
        "event_id", "fixture_id", "generated_at", "classification",
        "availability_confidence", "data_tier", "payload",
    )

    def __init__(self, rows):
        self.rows = rows
        self.index = 0
        self.sizes = []
        self.description = [SimpleNamespace(name=name) for name in self.names]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, args):
        assert "FROM soccer_refresh_events" in sql
        assert "ORDER BY generated_at ASC, event_id ASC" in sql
        assert args[1] == "T-10"

    def fetchmany(self, size):
        self.sizes.append(size)
        assert size <= 16, "batch must be small enough for large event JSON"
        part = self.rows[self.index:self.index + size]
        self.index += len(part)
        return part

    def fetchall(self):
        raise AssertionError("unbounded fetchall of event JSON would OOM production")


class FakeResultsCursor:
    def __init__(self):
        self.rows = [(9001, "FT", 2, 0), (9002, "FT", 0, 1)]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, args):
        assert "FROM soccer_results" in sql
        assert sorted(args[0]) == [9001, 9002]

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.streaming = FakeStreamingCursor(rows)
        self.named_called = False
        self.in_transaction = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    @contextmanager
    def transaction(self):
        self.in_transaction = True
        try:
            yield
        finally:
            self.in_transaction = False

    def cursor(self, name=None):
        if name is not None:
            assert name == "prospective_registry_v1_events"
            assert self.in_transaction, "autocommit=True requires explicit transaction"
            self.named_called = True
            return self.streaming
        return FakeResultsCursor()


def test_prospective_stream_preserves_first_eligible_selection_across_batches(monkeypatch):
    start = datetime(2026, 10, 7, 14, 30, tzinfo=timezone.utc)
    rows = []
    for idx in range(43):
        fid = 9002 if idx >= 27 else 9001
        # The first acceptable 9001 event is in batch two; 9002 in batch two.
        eligible = (idx >= 19)
        rows.append((
            idx + 101, fid, start + timedelta(minutes=idx),
            "BET", 0.91, "A" if eligible else "C",
            _event(fid, eligible),
        ))
    conn = FakeConnection(rows)
    monkeypatch.setattr(v.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(v.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(v.persistence, "_connect", lambda: conn)

    result = v.build_registry()
    assert conn.named_called
    assert len(conn.streaming.sizes) >= 4
    assert set(conn.streaming.sizes) == {16}
    assert result["status"] == "PROSPECTIVE_EVALUATION_REGISTRY_ACTIVE"
    assert result["summary"]["selection_rows"] == 2
    assert result["summary"]["selection_unique_fixtures"] == 2
    by_fixture = {row["fixture_id"]: row for row in result["selection_rows"]}
    assert by_fixture[9001]["event_id"] == 120
    assert by_fixture[9002]["event_id"] == 128
    assert len(result["settlement_rows"]) == 2
    assert result["summary"]["graded_bet_sample"]["settled"] == 2
    assert result["summary"]["material_recalibration_allowed"] is False
    assert result["historical_predictions_rewritten"] is False
    assert result["production_promotion_allowed"] is False
    assert result["model_weights_changed"] is False
