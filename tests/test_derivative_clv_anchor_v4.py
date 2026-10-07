import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from mcp_gateway import automation_v129
from mcp_gateway import derivative_clv_anchor_v4 as anchor
from mcp_gateway import price_resolver_v4 as price


COLUMNS = (
    "fixture_id",
    "market_family",
    "market",
    "selection",
    "line",
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


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.query = ""
        self.params = None
        self.description = [SimpleNamespace(name=name) for name in COLUMNS]

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


def _wire_fake_db(monkeypatch, cursor):
    monkeypatch.setattr(price.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(price.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(price.persistence, "_connect", lambda: _FakeConnection(cursor))


def test_one_h_anchor_uses_oldest_unresolved_exact_signal(monkeypatch):
    signal_at = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 10, 1, 16, 45, tzinfo=timezone.utc)
    cursor = _FakeCursor([
        (
            12345,
            "1H",
            "Goals Over/Under - First Half",
            "Over",
            "1.5",
            signal_at,
            "DERIVATIVE_INTELLIGENCE:one_h_goals_intelligence",
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
    ])
    _wire_fake_db(monkeypatch, cursor)

    report = anchor.load_one_h_clv_maturation_backlog(
        lookback_days=30,
        lookahead_minutes=55,
        limit=80,
    )

    normalized = " ".join(cursor.query.split())
    assert "oldest_unresolved_signal AS" in normalized
    assert "cs.signal_generated_at ASC" in normalized
    assert "m.captured_at > cs.signal_generated_at" in normalized
    assert "m.provider_update > cs.signal_generated_at" in normalized
    assert "(q.value ->> 'line')::NUMERIC - (cs.line)::NUMERIC" in normalized
    assert report["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert report["candidate_family_counts"] == {"1H": 1}
    meta = report["candidate_events"][0]["primary_clv_maturation"]
    assert meta["signals"][0]["signal_generated_at"] == signal_at.isoformat()
    assert meta["requires_same_selection_and_line"] is True
    assert meta["strict_close_semantics_changed"] is False


def test_two_h_and_corners_use_expected_persisted_sources(monkeypatch):
    cursor = _FakeCursor([])
    _wire_fake_db(monkeypatch, cursor)
    anchor.load_two_h_clv_maturation_backlog(limit=10)
    assert "two_h_goals_intelligence" in cursor.query
    assert "'2H'::TEXT" in cursor.query

    cursor2 = _FakeCursor([])
    _wire_fake_db(monkeypatch, cursor2)
    anchor.load_corners_clv_maturation_backlog(limit=10)
    assert "corners_intelligence" in cursor2.query
    assert "team_corners_intelligence" in cursor2.query
    assert "'FT_CORNERS'::TEXT" in cursor2.query
    assert "'TEAM_CORNERS'::TEXT" in cursor2.query


def test_v129_activates_and_restores_derivative_anchor_loaders(monkeypatch):
    original_one_h = price._load_one_h_clv_maturation_backlog
    original_two_h = price._load_two_h_clv_maturation_backlog
    original_corners = price._load_corners_clv_maturation_backlog

    def one_h_loader(*args, **kwargs):
        return {"candidate_events": [], "candidate_count": 0}

    def two_h_loader(*args, **kwargs):
        return {"candidate_events": [], "candidate_count": 0}

    def corners_loader(*args, **kwargs):
        return {"candidate_events": [], "candidate_count": 0}

    async def fake_v128_run_tick():
        assert price._load_one_h_clv_maturation_backlog is one_h_loader
        assert price._load_two_h_clv_maturation_backlog is two_h_loader
        assert price._load_corners_clv_maturation_backlog is corners_loader
        return {"events": [], "model_version": "SOCCER EDGE ENGINE v1.7"}

    monkeypatch.setattr(anchor, "load_one_h_clv_maturation_backlog", one_h_loader)
    monkeypatch.setattr(anchor, "load_two_h_clv_maturation_backlog", two_h_loader)
    monkeypatch.setattr(anchor, "load_corners_clv_maturation_backlog", corners_loader)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_v128_run_tick)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )

    payload = asyncio.run(automation_v129.run_tick())

    assert price._load_one_h_clv_maturation_backlog is original_one_h
    assert price._load_two_h_clv_maturation_backlog is original_two_h
    assert price._load_corners_clv_maturation_backlog is original_corners
    repair = payload["v215_6_derivative_clv_anchor_repair"]
    assert repair["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert repair["scope"] == ["1H", "2H", "FT_CORNERS", "TEAM_CORNERS"]
    assert repair["provider_budget_changed"] is False
    assert payload["version"] == automation_v129.AUTOMATION_VERSION



def test_default_derivative_anchor_scan_is_bounded_to_one_day(monkeypatch):
    from datetime import datetime, timedelta, timezone

    cursor = _FakeCursor([])
    _wire_fake_db(monkeypatch, cursor)
    before = datetime.now(timezone.utc)
    anchor.load_one_h_clv_maturation_backlog(limit=10)
    after = datetime.now(timezone.utc)

    cutoff = cursor.params[0]
    assert timedelta(hours=23, minutes=59) <= (after - cutoff) <= timedelta(days=1, minutes=1)
    assert cutoff <= before - timedelta(hours=23, minutes=59)
    normalized = " ".join(cursor.query.split())
    assert "e.stage IN ('T-40','T-20','T-10')" in normalized
    assert anchor.DEFAULT_DERIVATIVE_ANCHOR_LOOKBACK_DAYS == 1
