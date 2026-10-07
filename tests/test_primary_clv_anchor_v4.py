import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from mcp_gateway import automation_v129
from mcp_gateway import price_resolver_v4 as price
from mcp_gateway import primary_clv_anchor_v4 as anchor
from mcp_gateway import primary_clv_anchor_normalized_v4 as normalized


_MAIN_COLUMNS = (
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

_DIAGNOSTIC_COLUMNS = (
    "market_family",
    "priced_signal_rows",
    "priced_fixtures",
    "prekickoff_signal_fixtures",
    "future_active_fixtures",
    "within_lookahead_fixtures",
    "strict_later_quote_fixtures",
    "unresolved_within_lookahead_fixtures",
)


class _FakeCursor:
    def __init__(self, rows, diagnostic_rows=None):
        self._rows = rows
        self._diagnostic_rows = diagnostic_rows or []
        self._mode = "main"
        self.query = ""
        self.params = None
        self.description = [SimpleNamespace(name=name) for name in _MAIN_COLUMNS]
        self.execute_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params):
        self.execute_count += 1
        self.query = query
        self.params = params
        if "priced_signal_rows" in query and "WITH raw_signal AS" in query:
            self._mode = "diagnostic"
            self.description = [SimpleNamespace(name=name) for name in _DIAGNOSTIC_COLUMNS]
        else:
            self._mode = "main"
            self.description = [SimpleNamespace(name=name) for name in _MAIN_COLUMNS]

    def fetchall(self):
        return self._diagnostic_rows if self._mode == "diagnostic" else self._rows


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
        ],
        diagnostic_rows=[
            ("1X2", 393, 210, 205, 18, 3, 2, 1),
            ("BTTS", 168, 120, 118, 12, 4, 1, 3),
            ("FT_TOTALS", 586, 240, 235, 25, 5, 2, 3),
        ],
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

    assert report["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert report["source"] == "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V3_OLDEST_UNRESOLVED"
    assert report["candidate_count"] == 1
    assert report["candidate_family_counts"] == {"FT_TOTALS": 1}
    assert report["provider_requests_added"] == 0
    assert report["selection_logic_changed"] is False

    one_x_two = report["diagnostic_family_counts"]["1X2"]
    assert one_x_two["priced_signal_rows"] == 393
    assert one_x_two["future_active_fixtures"] == 18
    assert one_x_two["within_lookahead_fixtures"] == 3
    assert one_x_two["strict_later_quote_fixtures"] == 2
    assert one_x_two["unresolved_within_lookahead_fixtures"] == 1
    assert one_x_two["outside_lookahead_active_fixtures"] == 15

    diagnostic_query = " ".join(cursor.query.split())
    assert "m.captured_at > rs.signal_generated_at" in diagnostic_query
    assert "m.provider_update > rs.signal_generated_at" in diagnostic_query
    assert "COUNT(DISTINCT fixture_id)" in diagnostic_query

    event = report["candidate_events"][0]
    meta = event["primary_clv_maturation"]
    assert meta["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert meta["signals"][0]["signal_generated_at"] == signal_at.isoformat()
    assert meta["requires_provider_update_after_signal"] is True
    assert meta["strict_close_semantics_changed"] is False
    assert meta["historical_signal_mutated"] is False


def test_v129_captures_exclusion_audit_only_during_upstream_tick_and_restores_loader(monkeypatch):
    original_loader = price._load_primary_clv_maturation_backlog
    sentinel_calls = []
    sentinel_report = {
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_family_counts": {},
        "candidate_source_counts": {},
        "source": "TEST_PRIMARY_ANCHOR",
        "signal_anchor_policy": anchor.ANCHOR_POLICY,
        "diagnostic_schema_version": anchor.DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_family_counts": {
            "1X2": {
                "priced_signal_rows": 393,
                "priced_fixtures": 210,
                "prekickoff_signal_fixtures": 205,
                "future_active_fixtures": 18,
                "within_lookahead_fixtures": 0,
                "strict_later_quote_fixtures": 0,
                "unresolved_within_lookahead_fixtures": 0,
                "outside_lookahead_active_fixtures": 18,
            }
        },
        "diagnostic_window": {"lookahead_minutes": 55},
        "provider_requests_added": 0,
        "selection_logic_changed": False,
    }

    def sentinel_loader(*args, **kwargs):
        sentinel_calls.append((args, kwargs))
        return sentinel_report

    async def fake_v128_run_tick():
        assert price._load_primary_clv_maturation_backlog is not original_loader
        observed = price._load_primary_clv_maturation_backlog()
        assert observed is sentinel_report
        return {"events": [], "model_version": "SOCCER EDGE ENGINE v1.7"}

    monkeypatch.setattr(normalized, "load_primary_clv_maturation_backlog", sentinel_loader)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_v128_run_tick)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )

    payload = asyncio.run(automation_v129.run_tick())

    assert sentinel_calls
    assert sentinel_calls[0][1].get("include_diagnostics") is False
    assert price._load_primary_clv_maturation_backlog is original_loader
    assert payload["version"] == "4.38.8-normalized-primary-clv-anchor-fallback"
    repair = payload["v213_primary_clv_anchor_repair"]
    assert repair["signal_anchor_policy"] == anchor.ANCHOR_POLICY
    assert repair["strict_close_semantics_changed"] is False
    assert repair["frontend_changed"] is False

    audit = payload["v216_7_primary_clv_maturation_exclusion_audit"]
    assert audit["status"] == "OBSERVABILITY_ONLY"
    assert audit["provider_requests_added"] == 0
    assert audit["selection_logic_changed"] is False
    observation = audit["loader_observation"]
    assert observation["diagnostic_family_counts"]["1X2"]["priced_signal_rows"] == 393
    assert observation["diagnostic_family_counts"]["1X2"]["outside_lookahead_active_fixtures"] == 18



def test_live_primary_anchor_can_skip_heavy_exclusion_diagnostics(monkeypatch):
    signal_at = datetime(2026, 10, 6, 18, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 10, 6, 19, 0, tzinfo=timezone.utc)
    cursor = _FakeCursor(
        [
            (
                98765,
                "BTTS",
                "Both Teams Score",
                signal_at,
                "MATCH_TABLE_PRICED_RESEARCH",
                1,
                "League",
                "Country",
                2026,
                "Round",
                kickoff,
                "NS",
                "Not Started",
                1,
                "Home",
                2,
                "Away",
                "Venue",
                "City",
            )
        ],
        diagnostic_rows=[("BTTS", 100, 50, 50, 5, 1, 0, 1)],
    )
    connection = _FakeConnection(cursor)
    monkeypatch.setattr(price.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(price.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(price.persistence, "_connect", lambda: connection)

    report = anchor.load_primary_clv_maturation_backlog(
        lookback_days=30,
        lookahead_minutes=180,
        limit=80,
        include_diagnostics=False,
    )

    assert cursor.execute_count == 1
    assert report["candidate_count"] == 1
    assert report["diagnostic_status"] == "DEFERRED_OFFLINE"
    assert report["diagnostic_family_counts"] == {}


def test_v129_runtime_uses_current_tick_team_totals_audit_and_emits_health(monkeypatch):
    async def fake_v128_run_tick():
        return {
            "events": [],
            "model_version": "SOCCER EDGE ENGINE v1.7",
            "generated_at_utc": "2026-10-07T02:00:00+00:00",
            "price_resolution_v4": {
                "primary_clv_maturation_candidate_family_counts": {"BTTS": 2},
                "primary_clv_maturation_family_evaluation_counts": {"BTTS": 2},
                "primary_clv_maturation_family_refresh_counts": {},
                "primary_clv_maturation_not_matured_family_counts": {"BTTS": 2},
                "research_spillover_maturation_candidates": 1,
                "research_spillover_maturation_later_real_quote_refreshes": 0,
                "research_spillover_maturation_unchanged_provider_updates": 1,
            },
            "v4_018_btts_calibration_validation": {
                "status": "RESEARCH_HOLD",
                "true_clv": {"rows": 17, "unique_fixtures": 17, "minimum_rows": 50},
            },
            "v4_019_team_totals_oos_validation": {
                "status": "OOS_REVIEW_ELIGIBLE",
                "true_clv": {"rows": 341, "unique_fixtures": 43, "minimum_rows": 50},
            },
        }

    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_v128_run_tick)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )

    payload = asyncio.run(automation_v129.run_tick())
    audit = payload["v216_8_team_totals_close_provenance_audit"]
    assert audit["runtime_scope"] == "CURRENT_TICK_ONLY"
    assert audit["historical_audit_status"] == "DEFERRED_TO_OFFLINE_VALIDATION"
    health = payload["market_maturation_health"]
    assert health["status"] == "OBSERVABILITY_ONLY"
    assert health["families"]["BTTS"]["maturation_state"] == "WAITING_STRICT_LATER_QUOTE"
    assert health["families"]["TEAM_TOTALS"]["maturation_state"] == "CLV_GATE_MET"


def test_v129_uses_legacy_primary_loader_only_on_normalized_exception(monkeypatch):
    legacy_report = {
        "candidate_events": [],
        "candidate_count": 0,
        "candidate_family_counts": {},
        "candidate_source_counts": {},
        "source": "POSTGRES_PRIMARY_CLV_MATURATION_BACKLOG_V3_OLDEST_UNRESOLVED",
        "signal_anchor_policy": anchor.ANCHOR_POLICY,
        "diagnostic_schema_version": anchor.DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_status": "DEFERRED_OFFLINE",
        "diagnostic_family_counts": {},
        "diagnostic_window": {},
        "provider_requests_added": 0,
        "selection_logic_changed": False,
    }
    legacy_calls = []

    def broken_normalized(*args, **kwargs):
        raise RuntimeError("normalized store unavailable")

    def fake_legacy(*args, **kwargs):
        legacy_calls.append((args, kwargs))
        return legacy_report

    async def fake_v128_run_tick():
        observed = price._load_primary_clv_maturation_backlog()
        assert observed is legacy_report
        return {"events": [], "model_version": "SOCCER EDGE ENGINE v1.7"}

    monkeypatch.setattr(normalized, "load_primary_clv_maturation_backlog", broken_normalized)
    monkeypatch.setattr(anchor, "load_primary_clv_maturation_backlog", fake_legacy)
    monkeypatch.setattr(automation_v129.v128, "run_tick", fake_v128_run_tick)
    monkeypatch.setattr(
        automation_v129.dynamic_strength_challenger_v4,
        "build_report",
        lambda events: {"status": "TEST"},
    )

    payload = asyncio.run(automation_v129.run_tick())

    assert legacy_calls
    assert legacy_calls[0][1].get("include_diagnostics") is False
    repair = payload["v213_primary_clv_anchor_repair"]
    assert repair["legacy_pipeline_json_expansion_in_live_loader"] is True
    fallback = repair["normalized_loader_fallback"]
    assert fallback["used"] is True
    assert fallback["reason"] == "NORMALIZED_PRIMARY_CLV_LOADER_EXCEPTION"
    assert fallback["normalized_error_type"] == "RuntimeError"
    audit = payload["v216_7_primary_clv_maturation_exclusion_audit"]
    assert audit["loader_observation"]["fallback_used"] is True
    assert audit["loader_observation"]["fallback_source"] == legacy_report["source"]
