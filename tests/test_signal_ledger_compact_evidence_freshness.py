from __future__ import annotations

import json
from pathlib import Path

from mcp_gateway import build_signal_ledger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_compact_persisted_tick_advances_evidence_without_synthetic_rows(tmp_path: Path) -> None:
    legacy = {
        "generated_at_utc": "2026-09-22T23:34:09.866728+00:00",
        "generated_at_local": "2026-09-22T17:34:09.866728-06:00",
        "timezone": "America/Mexico_City",
        "version": "legacy",
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "events": [
            {
                "fixture": {"fixture_id": 123, "kickoff": "2026-09-23T02:00:00+00:00"},
                "event_type": "SOCCER_REFRESH",
                "stage": "T-40",
                "classification": "WATCH",
                "bet_eligible": False,
            }
        ],
    }
    compact = {
        "schema_version": "2.1.0",
        "generated_at_utc": "2026-10-01T06:00:09.293854+00:00",
        "generated_at_local": "2026-10-01T00:00:09.293854-06:00",
        "version": "4.38.1-primary-clv-anchor-repair",
        "model_version": "SOCCER EDGE ENGINE v1.7",
        "event_count": 12,
        "database_persisted": True,
    }
    _write_jsonl(tmp_path / "history.jsonl", [legacy, compact])

    rows, summary = build_signal_ledger.build_rows(str(tmp_path))

    assert len(rows) == 1
    assert summary["last_generated_at_utc"] == legacy["generated_at_utc"]
    assert summary["last_materialized_ledger_row_at_utc"] == legacy["generated_at_utc"]
    assert summary["last_evidence_at_utc"] == compact["generated_at_utc"]
    assert summary["last_evidence_at_local"] == compact["generated_at_local"]
    assert summary["last_evidence_source"] == "COMPACT_HISTORY_EVENT_COUNT_WITH_POSTGRES_PERSISTED_TRUE"
    assert summary["compact_postgres_event_ticks"] == 1
    assert summary["compact_postgres_event_count_total"] == 12
    assert summary["synthetic_close_rows_added"] == 0
    assert summary["historical_probabilities_recomputed"] is False
    assert summary["historical_rows_recalibrated"] is False


def test_unpersisted_or_empty_compact_tick_does_not_advance_evidence(tmp_path: Path) -> None:
    compact_unpersisted = {
        "schema_version": "2.1.0",
        "generated_at_utc": "2026-10-01T06:00:09.293854+00:00",
        "generated_at_local": "2026-10-01T00:00:09.293854-06:00",
        "event_count": 12,
        "database_persisted": False,
    }
    compact_empty = {
        "schema_version": "2.1.0",
        "generated_at_utc": "2026-10-01T06:10:09.293854+00:00",
        "generated_at_local": "2026-10-01T00:10:09.293854-06:00",
        "event_count": 0,
        "database_persisted": True,
    }
    _write_jsonl(tmp_path / "history.jsonl", [compact_unpersisted, compact_empty])

    rows, summary = build_signal_ledger.build_rows(str(tmp_path))

    assert rows == []
    assert summary["last_evidence_at_utc"] is None
    assert summary["last_evidence_source"] is None
    assert summary["compact_postgres_event_ticks"] == 0
    assert summary["compact_postgres_event_count_total"] == 0
