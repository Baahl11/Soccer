import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mcp_gateway import automation_v129
from mcp_gateway import price_resolver_v4 as price
from mcp_gateway import team_totals_clv_anchor_v4 as anchor


COLUMNS = (
    "fixture_id", "market", "selection", "line", "signal_generated_at",
    "league_id", "league", "country", "season", "round", "kickoff",
    "status", "status_long", "home_team_id", "home_team", "away_team_id",
    "away_team", "venue", "city",
)


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""
        self.params = None
        self.description = [SimpleNamespace(name=name) for name in COLUMNS]
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def execute(self, query, params):
        self.query = query
        self.params = params
    def fetchall(self): return self.rows


class _Conn:
    def __init__(self, cursor): self._cursor = cursor
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): return False
    def cursor(self): return self._cursor


def _wire(monkeypatch, cursor):
    monkeypatch.setattr(price.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(price.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(price.persistence, "_connect", lambda: _Conn(cursor))


def test_team_totals_anchor_is_exact_oldest_unresolved_and_bounded(monkeypatch):
    signal_at = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
    cursor = _Cursor([(
        123, "Home Goals Over/Under", "Over", "1.5", signal_at,
        39, "League", "Country", 2026, "Round", kickoff, "NS", "Not Started",
        1, "Home", 2, "Away", "Stadium", "City",
    )])
    _wire(monkeypatch, cursor)
    before = datetime.now(timezone.utc)
    report = anchor.load_team_totals_maturation_backlog(limit=20)
    after = datetime.now(timezone.utc)

    normalized = " ".join(cursor.query.split())
    assert "upcoming_fixtures AS MATERIALIZED" in normalized
    assert "base_events AS MATERIALIZED" in normalized
    assert "oldest_unresolved_signal AS" in normalized
    assert "cs.signal_generated_at ASC" in normalized
    assert "m.captured_at > cs.signal_generated_at" in normalized
    assert "m.provider_update > cs.signal_generated_at" in normalized
    assert "(q.value ->> 'line')::NUMERIC - (cs.line)::NUMERIC" in normalized
    cutoff = cursor.params[0]
    assert before - timedelta(hours=anchor.DEFAULT_LOOKBACK_HOURS, minutes=1) <= cutoff
    assert cutoff <= after - timedelta(hours=anchor.DEFAULT_LOOKBACK_HOURS - 1)
    assert report["candidate_count"] == 1
    assert report["candidate_signal_count"] == 1
    meta = report["candidate_events"][0]["team_totals_clv_maturation"]
    assert meta["signal_generated_at"] == signal_at.isoformat()
    assert meta["signals"][0]["line"] == 1.5
    assert meta["requires_same_selection_and_line"] is True
    assert report["provider_budget_changed"] is False


def test_team_totals_anchor_groups_multiple_exact_signals_by_fixture(monkeypatch):
    kickoff = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)
    older = datetime(2026, 10, 1, 15, 50, tzinfo=timezone.utc)
    newer = datetime(2026, 10, 1, 16, 10, tzinfo=timezone.utc)
    cursor = _Cursor([
        (123, "Home Goals Over/Under", "Over", "1.5", newer, 39, "L", "C", 2026, "R", kickoff, "NS", "NS", 1, "H", 2, "A", "V", "City"),
        (123, "Away Goals Over/Under", "Under", "1.5", older, 39, "L", "C", 2026, "R", kickoff, "NS", "NS", 1, "H", 2, "A", "V", "City"),
    ])
    _wire(monkeypatch, cursor)
    report = anchor.load_team_totals_maturation_backlog(limit=20)
    assert report["candidate_count"] == 1
    meta = report["candidate_events"][0]["team_totals_clv_maturation"]
    assert meta["signal_generated_at"] == older.isoformat()
    assert len(meta["signals"]) == 2


def test_v129_activates_and_restores_team_totals_anchor(monkeypatch):
    original = price._load_team_totals_maturation_backlog
    def replacement(*args, **kwargs):
        return {"candidate_events": [], "candidate_count": 0}
    async def fake_upstream():
        assert price._load_team_totals_maturation_backlog is replacement
        return {"events": [], "model_version": "SOCCER EDGE ENGINE v1.7"}

    monkeypatch.setattr(anchor, "load_team_totals_maturation_backlog", replacement)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_upstream)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )
    payload = asyncio.run(automation_v129.run_tick())
    assert price._load_team_totals_maturation_backlog is original
    repair = payload["v215_7_team_totals_clv_anchor_repair"]
    assert repair["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert repair["provider_budget_changed"] is False
    assert repair["team_totals_maturation_max_calls_per_tick"] == price.TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK
    assert payload["version"] == "4.38.3-team-totals-clv-anchor-repair"
