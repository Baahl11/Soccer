from __future__ import annotations

import argparse
import json
import os
from typing import Any

from mcp_gateway import signal_ledger_postgres_v4


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def merge_rows(
    legacy_rows: list[dict[str, Any]],
    postgres_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    merged: dict[str, dict[str, Any]] = {}
    anonymous = 0
    for row in legacy_rows:
        key = str(row.get("event_key") or "").strip()
        if not key:
            key = f"LEGACY_ANON_{anonymous}"
            anonymous += 1
        merged[key] = row

    replaced = 0
    for row in postgres_rows:
        key = str(row.get("event_key") or "").strip()
        if not key:
            continue
        if key in merged:
            replaced += 1
        merged[key] = row

    rows = sorted(
        merged.values(),
        key=lambda r: (
            str(r.get("generated_at_local") or ""),
            int(r.get("fixture_id") or 0),
            str(r.get("stage") or ""),
        ),
    )
    return rows, replaced


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing", required=True)
    parser.add_argument("--postgres-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    legacy_rows = _load_jsonl(args.existing)
    with open(args.postgres_json, encoding="utf-8") as fh:
        payload = json.load(fh)
    postgres_rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(postgres_rows, list):
        raise SystemExit("Postgres payload rows is not a list")
    postgres_rows = [row for row in postgres_rows if isinstance(row, dict)]

    merged_rows, replaced = merge_rows(legacy_rows, postgres_rows)
    summary = signal_ledger_postgres_v4.summarize_rows(
        merged_rows,
        source="LEGACY_HISTORY_PLUS_POSTGRES_POINT_IN_TIME_REFRESH_EVENTS",
        postgres_rows=len(postgres_rows),
        legacy_rows_loaded=len(legacy_rows),
        duplicate_event_keys_replaced=replaced,
    )
    summary.update({
        "postgres_export_summary": payload.get("summary") if isinstance(payload, dict) else None,
        "merge_policy": "EVENT_KEY_DEDUP; POSTGRES_POINT_IN_TIME_ROW_WINS_ON_DUPLICATE",
        "provider_requests_added": 0,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "historical_probabilities_recomputed": False,
        "historical_rows_recalibrated": False,
    })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in merged_rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
