from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

SOURCE = "HYBRID_LEGACY_HISTORY_PLUS_POSTGRES_REFRESH_EVENTS"
POSTGRES_SOURCE = "POSTGRES_SOCCER_REFRESH_EVENTS_POINT_IN_TIME"
SCHEMA_VERSION = "1.6.0"


def _parse_ts(value: Any) -> datetime | None:
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


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _sort_key(row: dict[str, Any]) -> tuple[datetime, int, str, str]:
    ts = _parse_ts(row.get("generated_at_utc") or row.get("generated_at_local"))
    if ts is None:
        ts = datetime.min.replace(tzinfo=timezone.utc)
    try:
        fixture_id = int(row.get("fixture_id") or 0)
    except (TypeError, ValueError):
        fixture_id = 0
    return ts, fixture_id, str(row.get("stage") or ""), str(row.get("event_key") or "")


def merge_rows(existing_rows: list[dict[str, Any]], postgres_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int]:
    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in existing_rows:
        key = str(row.get("event_key") or "").strip()
        if key:
            merged[key] = row
        else:
            anonymous.append(row)

    added = 0
    duplicates_ignored = 0
    for row in postgres_rows:
        key = str(row.get("event_key") or "").strip()
        if not key:
            anonymous.append(row)
            added += 1
            continue
        if key in merged:
            duplicates_ignored += 1
            continue
        merged[key] = row
        added += 1

    rows = list(merged.values()) + anonymous
    rows.sort(key=_sort_key)
    return rows, added, duplicates_ignored


def summarize_rows(rows: list[dict[str, Any]], previous: dict[str, Any], *, rows_added: int, duplicates_ignored: int) -> dict[str, Any]:
    rows = sorted(rows, key=_sort_key)
    stages = Counter(str(row.get("stage") or "UNKNOWN") for row in rows)
    classes = Counter(str(row.get("classification") or "UNKNOWN") for row in rows)
    signals: Counter[str] = Counter()
    fixtures: set[int] = set()
    finals: set[int] = set()
    with_market = with_raw = with_lineups = with_shadow = with_tactical = 0
    with_provenance = promotion_shadow_eligible = 0
    applied_calibration = frozen_artifact = 0
    postgres_source_rows = 0

    for row in rows:
        try:
            fixture_id = int(row.get("fixture_id"))
            fixtures.add(fixture_id)
            if row.get("result"):
                finals.add(fixture_id)
        except (TypeError, ValueError):
            pass
        with_market += bool(row.get("best_market"))
        with_raw += bool(row.get("raw_projection"))
        with_lineups += bool(row.get("lineups"))
        with_shadow += bool((row.get("raw_projection") or {}).get("relative_strength_shadow"))
        with_tactical += bool((row.get("result") or {}).get("tactical_stats"))
        postgres_source_rows += str(row.get("ledger_source") or "") == POSTGRES_SOURCE
        provenance = row.get("phase16_calibration_provenance")
        has_provenance = isinstance(provenance, dict) and bool(provenance)
        with_provenance += has_provenance
        promotion_shadow_eligible += has_provenance and provenance.get("promotion_shadow_eligible") is True
        calibrated = has_provenance and bool(provenance.get("calibrated_probability_fields"))
        applied_calibration += calibrated
        frozen_artifact += calibrated and bool((provenance.get("phase16_calibrator_artifact") or {}).get("fingerprint_sha256"))
        for track in (row.get("sporting_shortlist") or {}).get("tracks") or []:
            signals[str(track)] += 1

    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    last_row_utc = last.get("generated_at_utc")
    last_row_local = last.get("generated_at_local")
    last_row_dt = _parse_ts(last_row_utc or last_row_local)
    previous_evidence_utc = previous.get("last_evidence_at_utc")
    previous_evidence_local = previous.get("last_evidence_at_local")
    previous_evidence_dt = _parse_ts(previous_evidence_utc or previous_evidence_local)

    if previous_evidence_dt is not None and (last_row_dt is None or previous_evidence_dt > last_row_dt):
        evidence_utc = previous_evidence_utc
        evidence_local = previous_evidence_local
        evidence_source = previous.get("last_evidence_source") or "PREVIOUS_SUMMARY_EVIDENCE"
    else:
        evidence_utc = last_row_utc
        evidence_local = last_row_local
        evidence_source = "MATERIALIZED_SIGNAL_LEDGER_ROW" if rows else previous.get("last_evidence_source")

    return {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "timezone_basis": previous.get("timezone_basis") or "America/Mexico_City",
        "ticks_read": previous.get("ticks_read"),
        "bad_lines": previous.get("bad_lines", 0),
        "rows": len(rows),
        "unique_fixtures": len(fixtures),
        "fixtures_with_final_result": len(finals),
        "rows_with_raw_projection": with_raw,
        "rows_with_relative_strength_shadow": with_shadow,
        "rows_with_market": with_market,
        "rows_with_lineups": with_lineups,
        "rows_with_tactical_stats": with_tactical,
        "rows_with_phase16_calibration_provenance": with_provenance,
        "rows_phase16_promotion_shadow_eligible": promotion_shadow_eligible,
        "rows_with_applied_phase16_calibration": applied_calibration,
        "rows_with_frozen_phase16_calibrator_artifact": frozen_artifact,
        "stage_counts": dict(sorted(stages.items())),
        "classification_counts": dict(sorted(classes.items())),
        "sport_signal_counts": dict(sorted(signals.items())),
        "first_generated_at_local": first.get("generated_at_local") if rows else None,
        "first_generated_at_utc": first.get("generated_at_utc") if rows else None,
        "last_generated_at_local": last_row_local,
        "last_generated_at_utc": last_row_utc,
        "last_materialized_ledger_row_at_local": last_row_local,
        "last_materialized_ledger_row_at_utc": last_row_utc,
        "last_evidence_at_local": evidence_local,
        "last_evidence_at_utc": evidence_utc,
        "last_evidence_source": evidence_source,
        "legacy_rows_loaded": len(rows) - postgres_source_rows,
        "postgres_source_rows": postgres_source_rows,
        "postgres_rows_added_this_merge": rows_added,
        "duplicate_event_keys_ignored_this_merge": duplicates_ignored,
        "merge_policy": "EVENT_KEY_DEDUP; EXISTING_HISTORY_IS_IMMUTABLE; APPEND_ONLY_POSTGRES_POINT_IN_TIME_ROWS",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
        "synthetic_close_rows_added": 0,
        "point_in_time_payloads_reused": True,
        "production_promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True)
    parser.add_argument("--postgres-jsonl", required=True)
    parser.add_argument("--previous-summary", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    existing_rows = _load_jsonl(args.existing)
    postgres_rows = _load_jsonl(args.postgres_jsonl)
    previous_summary = json.loads(Path(args.previous_summary).read_text(encoding="utf-8"))
    merged_rows, added, duplicates = merge_rows(existing_rows, postgres_rows)
    summary = summarize_rows(merged_rows, previous_summary, rows_added=added, duplicates_ignored=duplicates)

    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in merged_rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
