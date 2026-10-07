from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import inspect

from mcp_gateway import persistence
from mcp_gateway import primary_clv_anchor_normalized_v4 as normalized
from mcp_gateway import primary_clv_anchor_v4 as legacy
from mcp_gateway import primary_clv_signal_anchor_store_v4 as store
from mcp_gateway import price_resolver_v4 as price


def test_store_extracts_exact_legacy_primary_populations():
    tick = {
        "generated_at_utc": "2026-10-07T05:00:00+00:00",
        "market_mismatch_rows": [
            {
                "fixture_id": 1,
                "market_family": "BTTS",
                "market": "Both Teams Score",
                "selection": "Yes",
                "price": "1.90",
                "rankable": True,
            },
            {
                "fixture_id": 2,
                "market_family": "FT_TOTALS",
                "market": "Goals Over/Under",
                "selection": "Over 2.5",
                "price": "1.80",
                "rankable": False,
            },
        ],
        "match_table_rows": [
            {
                "fixture_id": 3,
                "market_family": "TOTAL",
                "market": "Goals Over/Under",
                "selection": "Over 2.5",
                "line": 2.5,
                "price": 1.85,
            },
            {
                "fixture_id": 4,
                "market_family": "FT_BTTS_RESEARCH",
                "market": "Both Teams Score",
                "selection": "Yes",
                "price": "1.95",
            },
        ],
    }

    rows = store.extract_tick_anchor_rows(tick)

    assert len(rows) == 2
    assert rows[0]["fixture_id"] == 1
    assert rows[0]["candidate_source"] == "PHASE16_RANKABLE"
    assert rows[0]["source_priority"] == 0
    assert rows[1]["fixture_id"] == 3
    assert rows[1]["market_family"] == "FT_TOTALS"
    assert rows[1]["candidate_source"] == "MATCH_TABLE_PRICED_RESEARCH"
    assert rows[1]["source_priority"] == 1


def test_schema_materializes_normalized_anchor_table_and_identity_index():
    schema = Path(persistence.__file__).with_name("schema.sql").read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS soccer_primary_clv_signal_anchors" in schema
    assert "idx_soccer_primary_clv_signal_anchor_identity" in schema
    assert "idx_soccer_primary_clv_signal_anchor_live" in schema


def test_persistence_dual_writes_anchor_rows():
    source = inspect.getsource(persistence.persist_tick)
    assert "primary_clv_signal_anchor_store_v4.persist_tick_anchor_rows(cur, tick)" in source


class _Cursor:
    def __init__(self, rows):
        self.rows = rows
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
        return self.rows


class _Conn:
    def __init__(self, cursor):
        self.c = cursor

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def cursor(self):
        return self.c


def test_normalized_loader_preserves_strict_close_without_json_expansion(monkeypatch):
    signal_at = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)
    kickoff = datetime(2026, 10, 7, 11, 0, tzinfo=timezone.utc)
    cursor = _Cursor([
        (
            123,
            "BTTS",
            "Both Teams Score",
            signal_at,
            "MATCH_TABLE_PRICED_RESEARCH",
            39,
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
    ])
    monkeypatch.setattr(price.persistence, "persistence_configured", lambda: True)
    monkeypatch.setattr(price.persistence, "ensure_schema", lambda: None)
    monkeypatch.setattr(price.persistence, "_connect", lambda: _Conn(cursor))

    report = normalized.load_primary_clv_maturation_backlog(
        lookback_days=30,
        lookahead_minutes=180,
        limit=80,
        include_diagnostics=False,
    )

    query = " ".join(cursor.query.split())
    assert "soccer_primary_clv_signal_anchors" in query
    assert "jsonb_array_elements" not in query
    assert "m.captured_at > cs.signal_generated_at" in query
    assert "m.provider_update > cs.signal_generated_at" in query
    assert report["candidate_count"] == 1
    assert report["candidate_family_counts"] == {"BTTS": 1}
    assert report["normalized_storage"] is True
    assert report["legacy_json_expansion_used"] is False


def test_equivalence_audit_compares_exact_candidate_signal_keys(monkeypatch):
    event = {
        "fixture": {"fixture_id": 99},
        "primary_clv_maturation": {
            "signals": [
                {
                    "market_family": "FT_TOTALS",
                    "market": "Goals Over/Under",
                    "signal_generated_at": "2026-10-07T10:00:00+00:00",
                    "candidate_source": "MATCH_TABLE_PRICED_RESEARCH",
                }
            ]
        },
    }
    monkeypatch.setattr(
        legacy,
        "load_primary_clv_maturation_backlog",
        lambda **kwargs: {"candidate_events": [event], "candidate_count": 1, "source": "LEGACY"},
    )
    monkeypatch.setattr(
        normalized,
        "load_primary_clv_maturation_backlog",
        lambda **kwargs: {"candidate_events": [event], "candidate_count": 1, "source": "NORMALIZED"},
    )

    report = normalized.compare_with_legacy()

    assert report["equivalent"] is True
    assert report["status"] == "EQUIVALENT"
    assert report["missing_from_normalized_count"] == 0
    assert report["extra_in_normalized_count"] == 0
    assert report["strict_close_semantics_changed"] is False
