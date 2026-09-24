from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Iterable

from mcp_gateway.settlement_postgres_v4 import _market_performance_summary


def _parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return out if out.tzinfo is not None else out.replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _num_key(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return ""


def decision_key(row: dict[str, Any]) -> tuple[Any, ...]:
    try:
        fixture_id = int(row.get("fixture_id"))
    except (TypeError, ValueError):
        fixture_id = row.get("fixture_id")
    return (
        fixture_id,
        str(row.get("classification") or "").upper(),
        str(row.get("market_family") or "").upper(),
        _norm(row.get("market")),
        _norm(row.get("selection")),
        _num_key(row.get("line")),
    )


def merge_rows(*collections: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[Any, ...], tuple[datetime, int, dict[str, Any]]] = {}
    source_rank = 0
    for rows in collections:
        source_rank += 1
        for raw in rows:
            if not isinstance(raw, dict) or raw.get("fixture_id") is None:
                continue
            row = dict(raw)
            key = decision_key(row)
            captured_at = _parse_dt(row.get("generated_at_local") or row.get("generated_at"))
            prior = latest.get(key)
            # Later timestamp wins. On exact timestamp ties, later collection
            # (Postgres) wins because it is the canonical persisted event source.
            if prior is None or captured_at > prior[0] or (
                captured_at == prior[0] and source_rank >= prior[1]
            ):
                latest[key] = (captured_at, source_rank, row)
    return [
        value[2]
        for _, value in sorted(
            latest.items(),
            key=lambda item: (
                item[1][0],
                str(item[0]),
            ),
        )
    ]


def build_merge_report(
    legacy_rows: Iterable[dict[str, Any]],
    postgres_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    legacy = [row for row in legacy_rows if isinstance(row, dict)]
    postgres = [row for row in postgres_rows if isinstance(row, dict)]
    merged = merge_rows(legacy, postgres)
    summary = _market_performance_summary(merged)
    return {
        "schema_version": "1.0.0",
        "model_version": "SOCCER_SETTLEMENT_MERGE_V4_1.0.0",
        "status": "LEGACY_POSTGRES_UNION_ACTIVE",
        "legacy_rows": len(legacy),
        "postgres_rows": len(postgres),
        "merged_rows": len(merged),
        "deduped_overlap_rows": len(legacy) + len(postgres) - len(merged),
        "market_performance_summary": summary,
        "rows": merged,
        "provider_requests_added": 0,
        "runtime_logic_changed": False,
        "notes": [
            "Legacy Git-history settlements are retained for the pre-compact-history period.",
            "Postgres settlements extend the ledger after detailed events moved out of Git history.",
            "The latest timestamp wins for an overlapping decision identity; Postgres wins exact timestamp ties.",
            "This merge changes evidence storage only and does not reclassify historical picks.",
        ],
    }


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_postgres_report(path: str) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    rows = value.get("rows") if isinstance(value, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge legacy and Postgres settlement ledgers.")
    parser.add_argument("--legacy-ledger", required=True)
    parser.add_argument("--postgres-report", required=True)
    parser.add_argument("--output-ledger", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    report = build_merge_report(
        _load_jsonl(args.legacy_ledger),
        _load_postgres_report(args.postgres_report),
    )
    os.makedirs(os.path.dirname(args.output_ledger) or ".", exist_ok=True)
    with open(args.output_ledger, "w", encoding="utf-8") as handle:
        for row in report["rows"]:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.output_summary, "w", encoding="utf-8") as handle:
        json.dump(report["market_performance_summary"], handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    with open(args.output_report, "w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in report.items() if key != "rows"}, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
