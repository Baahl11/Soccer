from __future__ import annotations

import json
from pathlib import Path

from mcp_gateway import materialize_signal_ledger_postgres_v4 as materializer
from mcp_gateway import signal_ledger_postgres_v4


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _legacy_row(*, event_key: str = "legacy-a") -> dict:
    return {
        "event_key": event_key,
        "generated_at_local": "2026-09-22T18:34:09-06:00",
        "generated_at_utc": "2026-09-23T00:34:09+00:00",
        "fixture_id": 1,
        "stage": "T-90",
        "classification": "MONITORING",
        "sporting_shortlist": {"tracks": ["FT_TOTALS"]},
        "best_market": {"family": "FT_TOTALS", "selection": "OVER"},
    }


def _postgres_row(*, event_key: str = "postgres-b", fixture_id: int = 2) -> dict:
    return {
        "event_key": event_key,
        "generated_at_local": "2026-10-01T08:15:00-06:00",
        "generated_at_utc": "2026-10-01T14:15:00+00:00",
        "fixture_id": fixture_id,
        "stage": "CLOSE",
        "classification": "STRONG_EDGE",
        "sporting_shortlist": {"tracks": ["BTTS"]},
        "best_market": {"family": "BTTS", "selection": "YES"},
        "ledger_source": signal_ledger_postgres_v4.SOURCE,
        "postgres_event_id": 101,
        "pipeline_run_matched": True,
    }


def _page(rows: list[dict]) -> dict:
    return {
        "schema_version": "1.0.0",
        "model_version": materializer.DELTA_MODEL_VERSION,
        "status": "OK",
        "source": signal_ledger_postgres_v4.SOURCE,
        "raw_refresh_events_loaded": len(rows),
        "row_count": len(rows),
        "invalid_rows": 0,
        "rows_without_pipeline_run_match": 0,
        "last_event_id": 101,
        "next_event_id": 101,
        "has_more": False,
        "rows": rows,
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "production_promotion_allowed": False,
    }


def _summary() -> dict:
    return {
        "schema_version": "1.5.0",
        "ticks_read": 123,
        "bad_lines": 0,
        "compact_postgres_event_ticks": 5,
        "compact_postgres_event_count_total": 19,
        "last_evidence_at_local": "2026-10-01T07:26:04-06:00",
        "last_evidence_at_utc": "2026-10-01T13:26:04+00:00",
        "last_evidence_source": "COMPACT_HISTORY_EVENT_COUNT_WITH_POSTGRES_PERSISTED_TRUE",
        "last_evidence_event_count": 19,
        "last_evidence_database_persisted": True,
        "evidence_freshness_semantics": "verified compact evidence or materialized rows",
        "provider_requests_added": 0,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
    }


def test_materialize_adds_real_postgres_rows_and_advances_summary(tmp_path: Path) -> None:
    ledger = tmp_path / "signal_ledger.jsonl"
    summary = tmp_path / "signal_ledger_summary.json"
    pages = tmp_path / "delta_pages.jsonl"
    _write_jsonl(ledger, [_legacy_row()])
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    _write_jsonl(pages, [_page([_postgres_row()])])

    report = materializer.materialize(ledger_path=ledger, summary_path=summary, delta_pages_path=pages)
    merged = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    persisted = json.loads(summary.read_text(encoding="utf-8"))

    assert report["status"] == "OK"
    assert report["rows_added"] == 1
    assert report["rows_replaced"] == 0
    assert len(merged) == 2
    assert persisted["schema_version"] == "1.5.0"
    assert persisted["rows"] == 2
    assert persisted["last_materialized_ledger_row_at_utc"] == "2026-10-01T14:15:00+00:00"
    assert persisted["last_evidence_source"] == "MATERIALIZED_SIGNAL_LEDGER_ROW_POSTGRES"
    assert persisted["postgres_source_rows"] == 1
    assert persisted["provider_requests_added"] == 0
    assert persisted["historical_probabilities_recomputed"] is False
    assert persisted["historical_rows_recalibrated"] is False
    assert persisted["synthetic_close_rows_added"] == 0
    assert persisted["production_promotion_allowed"] is False


def test_materialize_replaces_same_event_key_without_duplicate(tmp_path: Path) -> None:
    ledger = tmp_path / "signal_ledger.jsonl"
    summary = tmp_path / "signal_ledger_summary.json"
    pages = tmp_path / "delta_pages.jsonl"
    _write_jsonl(ledger, [_legacy_row(event_key="same-key")])
    summary.write_text(json.dumps(_summary()), encoding="utf-8")
    _write_jsonl(pages, [_page([_postgres_row(event_key="same-key", fixture_id=1)])])

    report = materializer.materialize(ledger_path=ledger, summary_path=summary, delta_pages_path=pages)
    merged = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert report["rows_added"] == 0
    assert report["rows_replaced"] == 1
    assert report["rows_after"] == 1
    assert len(merged) == 1
    assert merged[0]["ledger_source"] == signal_ledger_postgres_v4.SOURCE
    assert merged[0]["postgres_event_id"] == 101
