from __future__ import annotations

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "mcp_gateway" / "refresh_signal_evidence_summary.py"
SPEC = importlib.util.spec_from_file_location("refresh_signal_evidence_summary_stdlib", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
refresh_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh_module)
refresh_summary = refresh_module.refresh_summary


def test_refresh_advances_only_from_real_persisted_compact_event_tick() -> None:
    summary = {
        "schema_version": "1.3.0",
        "last_generated_at_local": "2026-09-22T18:34:09-06:00",
        "last_generated_at_utc": "2026-09-23T00:34:09+00:00",
        "rows": 10758,
    }
    tick = {
        "schema_version": "2.1.0",
        "generated_at_utc": "2026-10-01T06:00:09.293854+00:00",
        "generated_at_local": "2026-10-01T00:00:09.293854-06:00",
        "event_count": 12,
        "database_persisted": True,
        "version": "4.38.1-primary-clv-anchor-repair",
        "model_version": "SOCCER EDGE ENGINE v1.7",
    }

    out = refresh_summary(summary, tick)

    assert out["rows"] == 10758
    assert out["last_generated_at_local"] == summary["last_generated_at_local"]
    assert out["last_materialized_ledger_row_at_local"] == summary["last_generated_at_local"]
    assert out["last_evidence_at_utc"] == tick["generated_at_utc"]
    assert out["last_evidence_at_local"] == tick["generated_at_local"]
    assert out["last_evidence_event_count"] == 12
    assert out["last_evidence_database_persisted"] is True
    assert out["synthetic_close_rows_added"] == 0
    assert out["historical_probabilities_recomputed"] is False
    assert out["historical_rows_recalibrated"] is False


def test_refresh_does_not_regress_evidence_timestamp() -> None:
    summary = {
        "last_generated_at_local": "2026-09-22T18:34:09-06:00",
        "last_evidence_at_utc": "2026-10-01T06:00:09+00:00",
        "last_evidence_at_local": "2026-10-01T00:00:09-06:00",
        "last_evidence_source": "COMPACT_HISTORY_EVENT_COUNT_WITH_POSTGRES_PERSISTED_TRUE",
    }
    older = {
        "generated_at_utc": "2026-09-30T23:00:00+00:00",
        "generated_at_local": "2026-09-30T17:00:00-06:00",
        "event_count": 5,
        "database_persisted": True,
    }

    out = refresh_summary(summary, older)

    assert out["last_evidence_at_utc"] == summary["last_evidence_at_utc"]
    assert out["last_evidence_at_local"] == summary["last_evidence_at_local"]
