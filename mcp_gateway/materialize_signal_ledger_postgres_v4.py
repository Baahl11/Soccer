from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp_gateway import signal_ledger_postgres_v4

DELTA_MODEL_VERSION = "SOCCER_SIGNAL_LEDGER_POSTGRES_DELTA_V4_1.1.0"
MATERIALIZER_VERSION = "SOCCER_SIGNAL_LEDGER_POSTGRES_MATERIALIZER_V4_1.0.0"
SUMMARY_COMPAT_SCHEMA_VERSION = "1.5.0"
MERGED_SOURCE = "MERGED_LEGACY_HISTORY_AND_POSTGRES_POINT_IN_TIME"


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        value = json.load(fh)
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    bad_lines = 0
    if not path.exists():
        return rows, bad_lines
    with path.open(encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(row, dict):
                bad_lines += 1
                continue
            rows.append(row)
    return rows, bad_lines


def _load_delta_pages(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pages = 0
    raw_refresh_events_loaded = 0
    invalid_rows = 0
    rows_without_pipeline_run_match = 0
    last_event_id = 0

    with path.open(encoding="utf-8", errors="strict") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            page = json.loads(raw)
            if not isinstance(page, dict):
                raise ValueError("delta page must be an object")
            if page.get("status") != "OK":
                raise ValueError(f"delta page status is not OK: {page.get('status')!r}")
            if page.get("model_version") != DELTA_MODEL_VERSION:
                raise ValueError(f"unexpected delta model_version: {page.get('model_version')!r}")
            if page.get("provider_requests_added") != 0:
                raise ValueError("delta page added provider requests")
            if page.get("canonical_bet_logic_changed") is not False:
                raise ValueError("delta page changed canonical bet logic")
            if page.get("model_weights_changed") is not False:
                raise ValueError("delta page changed model weights")
            if page.get("historical_probabilities_recomputed") is not False:
                raise ValueError("delta page recomputed historical probabilities")
            if page.get("historical_rows_recalibrated") is not False:
                raise ValueError("delta page recalibrated historical rows")
            if page.get("synthetic_close_rows_added") != 0:
                raise ValueError("delta page added synthetic CLOSE rows")
            if page.get("production_promotion_allowed") is not False:
                raise ValueError("delta page unexpectedly allows production promotion")

            page_rows = page.get("rows")
            if not isinstance(page_rows, list):
                raise ValueError("delta page rows must be an array")
            for row in page_rows:
                if not isinstance(row, dict):
                    raise ValueError("delta row must be an object")
                if not row.get("event_key"):
                    raise ValueError("delta row is missing event_key")
                rows.append(row)

            pages += 1
            raw_refresh_events_loaded += int(page.get("raw_refresh_events_loaded") or 0)
            invalid_rows += int(page.get("invalid_rows") or 0)
            rows_without_pipeline_run_match += int(page.get("rows_without_pipeline_run_match") or 0)
            last_event_id = max(last_event_id, int(page.get("last_event_id") or 0))

    return rows, {
        "pages": pages,
        "raw_refresh_events_loaded": raw_refresh_events_loaded,
        "rows_received": len(rows),
        "invalid_rows": invalid_rows,
        "rows_without_pipeline_run_match": rows_without_pipeline_run_match,
        "last_event_id": last_event_id,
    }


def _prefer_new_evidence(summary: dict[str, Any], previous: dict[str, Any]) -> None:
    materialized_local = summary.get("last_materialized_ledger_row_at_local")
    materialized_utc = summary.get("last_materialized_ledger_row_at_utc")
    materialized_dt = _parse_timestamp(materialized_utc or materialized_local)
    previous_local = previous.get("last_evidence_at_local")
    previous_utc = previous.get("last_evidence_at_utc")
    previous_dt = _parse_timestamp(previous_utc or previous_local)

    if materialized_dt is not None and (previous_dt is None or materialized_dt >= previous_dt):
        summary["last_evidence_at_local"] = materialized_local
        summary["last_evidence_at_utc"] = materialized_utc
        summary["last_evidence_source"] = "MATERIALIZED_SIGNAL_LEDGER_ROW_POSTGRES"
        summary["last_evidence_event_count"] = None
        summary["last_evidence_database_persisted"] = True
        return

    for key in (
        "last_evidence_at_local",
        "last_evidence_at_utc",
        "last_evidence_source",
        "last_evidence_event_count",
        "last_evidence_database_persisted",
    ):
        if key in previous:
            summary[key] = previous.get(key)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def materialize(*, ledger_path: Path, summary_path: Path, delta_pages_path: Path) -> dict[str, Any]:
    previous_summary = _load_json(summary_path)
    legacy_rows, bad_ledger_lines = _load_jsonl(ledger_path)
    delta_rows, delta_meta = _load_delta_pages(delta_pages_path)

    if not delta_rows:
        return {
            "schema_version": "1.0.0",
            "model_version": MATERIALIZER_VERSION,
            "status": "NO_NEW_ROWS",
            "legacy_rows_loaded": len(legacy_rows),
            "bad_ledger_lines": bad_ledger_lines,
            "rows_received": 0,
            "rows_added": 0,
            "rows_replaced": 0,
            "rows_after": len(legacy_rows),
            "provider_requests_added": 0,
            "canonical_bet_logic_changed": False,
            "model_weights_changed": False,
            "historical_probabilities_recomputed": False,
            "historical_rows_recalibrated": False,
            "synthetic_close_rows_added": 0,
            "production_promotion_allowed": False,
            **delta_meta,
        }

    merged: dict[str, dict[str, Any]] = {}
    anonymous_rows: list[dict[str, Any]] = []
    for row in legacy_rows:
        event_key = str(row.get("event_key") or "").strip()
        if event_key:
            merged.setdefault(event_key, row)
        else:
            anonymous_rows.append(row)

    rows_added = 0
    rows_replaced = 0
    for row in delta_rows:
        event_key = str(row.get("event_key") or "").strip()
        if event_key in merged:
            rows_replaced += 1
        else:
            rows_added += 1
        # The Postgres row is the canonical point-in-time representation for
        # this exact event_key. Replacement changes provenance, not the decision.
        merged[event_key] = row

    merged_rows = anonymous_rows + list(merged.values())
    merged_rows.sort(
        key=lambda row: (
            str(row.get("generated_at_local") or ""),
            int(row.get("fixture_id") or 0),
            str(row.get("stage") or ""),
            str(row.get("event_key") or ""),
        )
    )

    summary = signal_ledger_postgres_v4.summarize_rows(
        merged_rows,
        source=MERGED_SOURCE,
        postgres_rows=sum(str(row.get("ledger_source") or "") == signal_ledger_postgres_v4.SOURCE for row in merged_rows),
        legacy_rows_loaded=len(legacy_rows),
        duplicate_event_keys_replaced=rows_replaced,
    )
    # Keep the established Control Tower summary contract. v215 adds optional
    # provenance fields but does not force downstream consumers to migrate.
    summary["schema_version"] = str(previous_summary.get("schema_version") or SUMMARY_COMPAT_SCHEMA_VERSION)
    summary["ticks_read"] = int(previous_summary.get("ticks_read") or 0)
    summary["bad_lines"] = int(previous_summary.get("bad_lines") or 0) + bad_ledger_lines
    summary["compact_postgres_event_ticks"] = int(previous_summary.get("compact_postgres_event_ticks") or 0)
    summary["compact_postgres_event_count_total"] = int(previous_summary.get("compact_postgres_event_count_total") or 0)
    summary["evidence_freshness_semantics"] = previous_summary.get("evidence_freshness_semantics") or (
        "last_evidence_at advances from a materialized ledger row or verified compact-history evidence"
    )
    summary["materializer_model_version"] = MATERIALIZER_VERSION
    summary["postgres_delta_pages"] = delta_meta["pages"]
    summary["postgres_delta_rows_received"] = delta_meta["rows_received"]
    summary["postgres_delta_rows_added"] = rows_added
    summary["postgres_delta_rows_replaced"] = rows_replaced
    summary["postgres_delta_invalid_rows"] = delta_meta["invalid_rows"]
    summary["postgres_rows_without_pipeline_run_match"] = delta_meta["rows_without_pipeline_run_match"]
    summary["postgres_last_event_id"] = delta_meta["last_event_id"]
    summary["production_promotion_allowed"] = False
    _prefer_new_evidence(summary, previous_summary)

    _atomic_write_jsonl(ledger_path, merged_rows)
    _atomic_write_json(summary_path, summary)

    return {
        "schema_version": "1.0.0",
        "model_version": MATERIALIZER_VERSION,
        "status": "OK",
        "legacy_rows_loaded": len(legacy_rows),
        "bad_ledger_lines": bad_ledger_lines,
        "rows_received": delta_meta["rows_received"],
        "rows_added": rows_added,
        "rows_replaced": rows_replaced,
        "rows_after": len(merged_rows),
        "last_materialized_ledger_row_at_local": summary.get("last_materialized_ledger_row_at_local"),
        "last_materialized_ledger_row_at_utc": summary.get("last_materialized_ledger_row_at_utc"),
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "production_promotion_allowed": False,
        **delta_meta,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--delta-pages", required=True)
    args = parser.parse_args()

    report = materialize(
        ledger_path=Path(args.ledger),
        summary_path=Path(args.summary),
        delta_pages_path=Path(args.delta_pages),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
