import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from mcp_gateway import automation_v129
from mcp_gateway import price_resolver_v4 as price
from mcp_gateway import team_totals_clv_anchor_v4 as anchor
from mcp_gateway import team_totals_close_provenance_v4 as provenance


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


def _candidate(signal_at, kickoff, *, fixture_id=123, line=1.5):
    return {
        "stage": "T-40",
        "fixture": {"fixture_id": fixture_id, "kickoff": kickoff.isoformat()},
        "team_totals_clv_maturation": {
            "signals": [{
                "market_family": "TEAM_TOTALS",
                "market": "Home Goals Over/Under",
                "selection": "Over",
                "line": line,
                "signal_generated_at": signal_at.isoformat(),
            }]
        },
    }


def _resolved(provider_update, *, fixture_id=123, line=1.5):
    return {
        "fixture": {"fixture_id": fixture_id},
        "market": {
            "markets": [{
                "market": "Home Goals Over/Under",
                "bookmaker_id": 8,
                "bookmaker": "Book",
                "provider_update": provider_update.isoformat(),
                "values": [{"selection": "Over", "line": line, "decimal_price": 1.91}],
            }]
        },
    }


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
    assert "candidate_market AS MATERIALIZED" in normalized
    assert "exact_snapshot_resolution AS MATERIALIZED" in normalized
    assert "MAX(LEAST(m.captured_at, m.provider_update))" in normalized
    assert "LEFT JOIN exact_snapshot_resolution resolution" in normalized
    assert "resolution.latest_resolution_at <= cs.signal_generated_at" in normalized
    assert "ABS(resolution.line_num - cs.line_num) < 0.000001" in normalized
    assert "m.captured_at >= b.cutoff" in normalized
    assert "m.provider_update >= b.cutoff" in normalized
    assert "cs.signal_generated_at ASC" in normalized
    cutoff = cursor.params[0]
    assert before - timedelta(hours=anchor.DEFAULT_LOOKBACK_HOURS, minutes=1) <= cutoff
    assert cutoff <= after - timedelta(hours=anchor.DEFAULT_LOOKBACK_HOURS - 1)
    assert report["candidate_count"] == 1
    assert report["candidate_signal_count"] == 1
    assert report["query_strategy"] == anchor.QUERY_STRATEGY
    meta = report["candidate_events"][0]["team_totals_clv_maturation"]
    assert meta["signal_generated_at"] == signal_at.isoformat()
    assert meta["signals"][0]["line"] == 1.5
    assert meta["requires_same_selection_and_line"] is True
    assert meta["query_strategy"] == anchor.QUERY_STRATEGY
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


def test_close_provenance_requires_exact_side_line_and_strict_times():
    signal_at = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
    captured_at = datetime(2026, 10, 1, 16, 30, tzinfo=timezone.utc)
    kickoff = datetime(2026, 10, 1, 18, 0, tzinfo=timezone.utc)

    strict = provenance.build_report(
        candidate_events=[_candidate(signal_at, kickoff)],
        resolved_events=[_resolved(datetime(2026, 10, 1, 16, 20, tzinfo=timezone.utc))],
        captured_at=captured_at,
    )
    assert strict["strict_later_exact_quote_fixture_ids"] == [123]
    assert strict["reason_counts"] == {"STRICT_LATER_EXACT_QUOTE": 1}
    assert strict["samples"][0]["bookmaker_id"] == 8
    assert strict["provider_requests_added"] == 0
    assert strict["selection_logic_changed"] is False

    wrong_line = provenance.build_report(
        candidate_events=[_candidate(signal_at, kickoff, line=1.5)],
        resolved_events=[_resolved(datetime(2026, 10, 1, 16, 20, tzinfo=timezone.utc), line=2.5)],
        captured_at=captured_at,
    )
    assert wrong_line["reason_counts"] == {"NO_EXACT_SIDE_LINE_MATCH": 1}

    stale_provider = provenance.build_report(
        candidate_events=[_candidate(signal_at, kickoff)],
        resolved_events=[_resolved(datetime(2026, 10, 1, 15, 59, tzinfo=timezone.utc))],
        captured_at=captured_at,
    )
    assert stale_provider["reason_counts"] == {"PROVIDER_UPDATE_NOT_AFTER_SIGNAL": 1}


def test_v129_activates_restores_and_publishes_team_totals_audit(monkeypatch):
    original = price._load_team_totals_maturation_backlog
    replacement_calls = []

    def replacement(*args, **kwargs):
        replacement_calls.append(True)
        return {"candidate_events": [], "candidate_count": 0}

    async def fake_upstream():
        assert price._load_team_totals_maturation_backlog is not original
        price._load_team_totals_maturation_backlog()
        return {
            "events": [],
            "generated_at_utc": "2026-10-01T16:30:00+00:00",
            "model_version": "SOCCER EDGE ENGINE v1.7",
        }

    monkeypatch.setattr(anchor, "load_team_totals_maturation_backlog", replacement)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_upstream)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )
    payload = asyncio.run(automation_v129.run_tick())
    assert replacement_calls == [True]
    assert price._load_team_totals_maturation_backlog is original
    repair = payload["v215_7_team_totals_clv_anchor_repair"]
    assert repair["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert repair["provider_budget_changed"] is False
    assert repair["team_totals_maturation_max_calls_per_tick"] == price.TEAM_TOTALS_MATURATION_MAX_CALLS_PER_TICK
    audit = payload["v216_8_team_totals_close_provenance_audit"]
    assert audit["status"] == "OBSERVABILITY_ONLY"
    assert audit["provider_requests_added"] == 0
    assert audit["selection_logic_changed"] is False
    assert payload["version"] == "4.38.8-normalized-primary-clv-anchor-fallback"
