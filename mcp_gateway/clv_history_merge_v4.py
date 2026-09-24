from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MERGE_VERSION = "SOCCER_TRUE_CLV_HISTORY_MERGE_V4_1.0.0"
MIN_TRUE_CLOSE_ROWS = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return out if out.tzinfo is not None else None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _family(row: dict[str, Any]) -> str | None:
    existing = str(row.get("market_family") or "").strip()
    if existing:
        return existing
    market = _norm(row.get("market"))
    if "winner" in market:
        return "1X2"
    if "both teams" in market or "btts" in market:
        return "BTTS"
    if "over/under" in market or "over under" in market:
        return "FT_TOTALS"
    return None


def _current_fixture_family_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        family = _family(row)
        if family:
            keys.add((fixture_id, family))
    return keys


def normalize_history_rows(
    history_rows: Iterable[dict[str, Any]],
    current_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    covered = _current_fixture_family_keys(current_rows)
    latest: dict[tuple[int, str], tuple[datetime, dict[str, Any]]] = {}

    for row in history_rows:
        if not isinstance(row, dict) or row.get("is_true_closing_line") is not True:
            continue
        try:
            fixture_id = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        family = _family(row)
        if family is None or (fixture_id, family) in covered:
            continue

        signal_ts = _parse_dt(row.get("signal_timestamp_local") or row.get("entry_timestamp"))
        close_ts = _parse_dt(row.get("close_timestamp_local") or row.get("closing_timestamp"))
        kickoff = _parse_dt(row.get("kickoff_local") or row.get("kickoff"))
        if signal_ts is None or close_ts is None or kickoff is None:
            continue
        if signal_ts >= close_ts or close_ts >= kickoff:
            continue

        clv = _num(row.get("clv_probability_pp"))
        if clv is None:
            clv = _num(row.get("probability_clv"))
        if clv is None:
            continue

        key = (fixture_id, family)
        prior = latest.get(key)
        if prior is None or signal_ts > prior[0]:
            latest[key] = (signal_ts, row)

    normalized: list[dict[str, Any]] = []
    for (fixture_id, family), (signal_ts, row) in sorted(latest.items()):
        close_ts = _parse_dt(row.get("close_timestamp_local") or row.get("closing_timestamp"))
        kickoff = _parse_dt(row.get("kickoff_local") or row.get("kickoff"))
        entry_price = _num(row.get("signal_price") or row.get("entry_price"))
        closing_price = _num(row.get("close_price") or row.get("closing_price"))
        entry_fair = _num(row.get("signal_fair_probability") or row.get("entry_fair_probability"))
        closing_fair = _num(row.get("close_fair_probability") or row.get("closing_fair_probability"))
        clv = _num(row.get("clv_probability_pp"))
        if clv is None:
            clv = _num(row.get("probability_clv"))

        normalized.append({
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture_id,
            "league": row.get("league"),
            "home_team": row.get("home_team"),
            "away_team": row.get("away_team"),
            "kickoff": kickoff.isoformat() if kickoff else row.get("kickoff_local"),
            "stage": row.get("stage"),
            "classification": row.get("classification"),
            "market_family": family,
            "market": row.get("market"),
            "selection": row.get("selection"),
            "tier": row.get("tier"),
            "confidence": None,
            "model_version": row.get("model_version"),
            "bookmaker": row.get("bookmaker_at_signal") or row.get("bookmaker"),
            "signal_source": "HISTORICAL_DEDICATED_CLOSE",
            "entry_timestamp": signal_ts.isoformat(),
            "entry_line": _num(row.get("line")),
            "line": _num(row.get("line")),
            "entry_price": entry_price,
            "signal_price": entry_price,
            "entry_fair_probability": entry_fair,
            "signal_fair_probability": entry_fair,
            "closing_timestamp": close_ts.isoformat() if close_ts else None,
            "closing_line": _num(row.get("line")),
            "closing_price": closing_price,
            "close_price": closing_price,
            "closing_fair_probability": closing_fair,
            "close_fair_probability": closing_fair,
            "probability_clv": clv,
            "clv_probability_pp": clv,
            "price_clv": None,
            "clv_price_pct": None,
            "line_movement": 0.0 if row.get("line") is not None else None,
            "bookmaker_at_signal": row.get("bookmaker_at_signal"),
            "is_true_closing_line": True,
            "closing_line_status": row.get("closing_line_status") or "HISTORICAL_DEDICATED_PREKICKOFF_CLOSE",
            "probability_comparable_same_line": True,
            "closing_source": "HISTORICAL_DEDICATED_PREKICKOFF_CLOSE",
            "same_book_preferred": "SAME_BOOK" in str(row.get("closing_line_status") or ""),
        })

    return normalized


def merge_report(
    postgres_report: dict[str, Any],
    history_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    current_rows = postgres_report.get("rows") if isinstance(postgres_report.get("rows"), list) else []
    historical = normalize_history_rows(history_rows, current_rows)
    merged_rows = list(current_rows) + historical

    family_counts = Counter()
    source_counts = Counter()
    unique_by_family: dict[str, set[int]] = {}
    for row in merged_rows:
        family = _family(row)
        if family:
            family_counts[family] += 1
            unique_by_family.setdefault(family, set()).add(int(row["fixture_id"]))
        source_counts[str(row.get("signal_source") or "UNKNOWN")] += 1

    comparable = [row for row in merged_rows if row.get("probability_comparable_same_line") is True]
    out = dict(postgres_report)
    out["rows"] = merged_rows
    out["tracked_rows"] = len(merged_rows)
    out["comparable_true_clv_rows"] = len(comparable)
    out["status"] = "ACTIVE_TRUE_CLV_SAMPLE" if len(comparable) >= MIN_TRUE_CLOSE_ROWS else "COLLECTING_TRUE_CLV"
    out["family_counts"] = dict(sorted(family_counts.items()))
    out["signal_source_counts"] = dict(sorted(source_counts.items()))
    out["canonical_merge_version"] = MERGE_VERSION
    out["historical_backfill_rows_added"] = len(historical)
    out["historical_backfill_unique_fixtures"] = len({row["fixture_id"] for row in historical})
    out["unique_fixtures_by_family"] = {
        family: len(fixtures) for family, fixtures in sorted(unique_by_family.items())
    }
    out["provider_requests_added"] = 0

    notes = list(out.get("notes") or [])
    notes.append(
        "Historical dedicated-close backfill adds at most one latest pre-kickoff row per fixture/family and is skipped whenever Postgres already covers that fixture/family."
    )
    out["notes"] = notes
    return out


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge strict historical dedicated-close CLV into the canonical Postgres CLV ledger.")
    parser.add_argument("--postgres-report", required=True)
    parser.add_argument("--history-true-clv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = merge_report(_load_json(args.postgres_report), _load_jsonl(args.history_true_clv))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(json.dumps({
        "canonical_merge_version": report["canonical_merge_version"],
        "tracked_rows": report["tracked_rows"],
        "comparable_true_clv_rows": report["comparable_true_clv_rows"],
        "historical_backfill_rows_added": report["historical_backfill_rows_added"],
        "historical_backfill_unique_fixtures": report["historical_backfill_unique_fixtures"],
        "unique_fixtures_by_family": report["unique_fixtures_by_family"],
        "family_counts": report["family_counts"],
        "provider_requests_added": report["provider_requests_added"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
