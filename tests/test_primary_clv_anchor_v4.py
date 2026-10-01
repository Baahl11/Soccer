import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from mcp_gateway import automation_v129
from mcp_gateway import price_resolver_v4 as price
from mcp_gateway import primary_clv_anchor_v4 as anchor


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.query = ""
        self.params = None
        self.description = [
            SimpleNamespace(name=name)
            for name in (
                "fixture_id",
                "market_family",
                "market",
                "signal_generated_at",
                "candidate_source",
                "league_id",
                "league",
                "country",
                "season",
                "round",
                "kickoff",
                "status",
                "status_long",
                "home_team_id",
                "home_team",
                "away_team_id",
                "away_team",
                "venue",
                "city",
            )
        ]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return self._rows


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self._cursor


def test_anchor_query_uses_oldest_unresolved_signal_and_preserves_strict_close(monkeypatch):
    signal_at = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 9, 30, 18, 45, tzinfo=timezone.utc)
    cursor = _FakeCursor(
        [
            (
                12345,
                "FT_TOTALS",
                "Goals Over/Under",
                signal_at,
                "MATCH_TABLE_PRICED_RESEARCH",
                39,
                "Premier League",
                "England",
                2026,
                "Regular Season - 1",
                kickoff,
                "NS",
                "Not Started",
                1,
                "Home",
                2,
                "Away",
                "Stadium",
                "City",
            )
        ]
    )
    connection = _FakeConnection(cursor)

    monkeypatch.setattr(price.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(price.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(price.persistence, "_connect", lambda: connection)

    report = anchor.load_primary_clv_maturation_backlog(
        lookback_days=30,
        lookahead_minutes=55,
        limit=80,
    )

    normalized_query = " ".join(cursor.query.split())
    assert "oldest_unresolved_signal AS" in normalized_query
    assert "cs.signal_generated_at ASC" in normalized_query
    assert "m.captured_at > cs.signal_generated_at" in normalized_query
    assert "m.provider_update > cs.signal_generated_at" in normalized_query
    assert report["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert report["source"] == "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V3_OLDEST_UNRESOLVED"
    assert report["candidate_count"] == 1
    assert report["candidate_family_counts"] == {"FT_TOTALS": 1}
    event = report["candidate_events"][0]
    meta = event["primary_clv_maturation"]
    assert meta["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert meta["signals"][0]["signal_generated_at"] == signal_at.isoformat()
    assert meta["requires_provider_update_after_signal"] is True
    assert meta["strict_close_semantics_changed"] is False
    assert meta["historical_signal_mutated"] is False


def test_v129_activates_anchor_only_during_upstream_tick_and_restores_loader(monkeypatch):
    original_loader = price._load_primary_clv_maturation_backlog
    sentinel_calls = []

    def sentinel_loader(*args, **kwargs):
        sentinel_calls.append((args, kwargs))
        return {"candidate_events": [], "candidate_count": 0}

    async def fake_v128_run_tick():
        assert price._load_primary_clv_maturation_backlog is sentinel_loader
        return {"events": [], "model_version": "SOCCER EDGE ENGINE v1.7"}

    monkeypatch.setattr(anchor, "load_primary_clv_maturation_backlog", sentinel_loader)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_v128_run_tick)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )

    payload = asyncio.run(automation_v129.run_tick())

    assert price._load_primary_clv_maturation_backlog is original_loader
    assert payload["version"] == "4.38.2-derivative-clv-anchor-repair"
    repair = payload["v213_primary_clv_anchor_repair"]
    assert repair["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert repair["strict_close_semantics_changed"] is False
    assert repair["frontend_changed"] is False
