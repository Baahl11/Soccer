from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MERGE_VERSION = "SOCCER_TRUE_CLV_HISTORY_MERGE_V4_1.2.0"
MIN_TRUE_CLOSE_ROWS = 50
TEAM_TOTAL_FAMILIES = {"TEAM_TOTALS", "HOME_TT", "AWAY_TT"}


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


def _fixture_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _current_fixture_family_keys(rows: Iterable[dict[str, Any]]) -> set[tuple[int, str]]:
    keys: set[tuple[int, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixture_id = _fixture_id(row.get("fixture_id"))
        if fixture_id is None:
            continue
        family = _family(row)
        if family:
            keys.add((fixture_id, family))
    return keys


def normalize_history_rows(history_rows: Iterable[dict[str, Any]], current_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    covered = _current_fixture_family_keys(current_rows)
    latest: dict[tuple[int, str], tuple[datetime, dict[str, Any]]] = {}
    for row in history_rows:
        if not isinstance(row, dict) or row.get("is_true_closing_line") is not True:
            continue
        fixture_id = _fixture_id(row.get("fixture_id"))
        if fixture_id is None:
            continue
        family = _family(row)
        if family is None or (fixture_id, family) in covered:
            continue
        signal_ts = _parse_dt(row.get("signal_timestamp_local") or row.get("entry_timestamp"))
        close_ts = _parse_dt(row.get("close_timestamp_local") or row.get("closing_timestamp"))
        kickoff = _parse_dt(row.get("kickoff_local") or row.get("kickoff"))
        if signal_ts is None or close_ts is None or kickoff is None or signal_ts >= close_ts or close_ts >= kickoff:
            continue
        provider_close_ts = _parse_dt(row.get("closing_provider_update") or row.get("close_provider_update"))
        if str(family).upper() in TEAM_TOTAL_FAMILIES and (provider_close_ts is None or provider_close_ts <= signal_ts or provider_close_ts >= kickoff):
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
        provider_close_ts = _parse_dt(row.get("closing_provider_update") or row.get("close_provider_update"))
        entry_price = _num(row.get("signal_price") or row.get("entry_price"))
        closing_price = _num(row.get("close_price") or row.get("closing_price"))
        entry_fair = _num(row.get("signal_fair_probability") or row.get("entry_fair_probability"))
        closing_fair = _num(row.get("close_fair_probability") or row.get("closing_fair_probability"))
        clv = _num(row.get("clv_probability_pp"))
        if clv is None:
            clv = _num(row.get("probability_clv"))
        normalized.append({"schema_version": SCHEMA_VERSION, "fixture_id": fixture_id, "league": row.get("league"), "home_team": row.get("home_team"), "away_team": row.get("away_team"), "kickoff": kickoff.isoformat() if kickoff else row.get("kickoff_local"), "stage": row.get("stage"), "classification": row.get("classification"), "market_family": family, "market": row.get("market"), "selection": row.get("selection"), "tier": row.get("tier"), "confidence": None, "model_version": row.get("model_version"), "bookmaker": row.get("bookmaker_at_signal") or row.get("bookmaker"), "signal_source": "HISTORICAL_DEDICATED_CLOSE", "entry_timestamp": signal_ts.isoformat(), "entry_line": _num(row.get("line")), "line": _num(row.get("line")), "entry_price": entry_price, "signal_price": entry_price, "entry_fair_probability": entry_fair, "signal_fair_probability": entry_fair, "closing_timestamp": close_ts.isoformat() if close_ts else None, "closing_provider_update": provider_close_ts.isoformat() if provider_close_ts is not None else None, "closing_line": _num(row.get("line")), "closing_price": closing_price, "close_price": closing_price, "closing_fair_probability": closing_fair, "close_fair_probability": closing_fair, "probability_clv": clv, "clv_probability_pp": clv, "price_clv": None, "clv_price_pct": None, "line_movement": 0.0 if row.get("line") is not None else None, "bookmaker_at_signal": row.get("bookmaker_at_signal"), "is_true_closing_line": True, "closing_line_status": row.get("closing_line_status") or "HISTORICAL_DEDICATED_PREKICKOFF_CLOSE", "probability_comparable_same_line": True, "closing_source": "HISTORICAL_DEDICATED_PREKICKOFF_CLOSE", "same_book_preferred": "SAME_BOOK" in str(row.get("closing_line_status") or "")})
    return normalized


def _canonical_exact_key(row: dict[str, Any]) -> tuple[Any, ...] | None:
    fixture_id = _fixture_id(row.get("fixture_id"))
    family = _family(row)
    entry_ts = _parse_dt(row.get("entry_timestamp") or row.get("signal_timestamp_local"))
    if fixture_id is None or family is None or entry_ts is None:
        return None
    line = _num(row.get("entry_line"))
    if line is None:
        line = _num(row.get("line"))
    return (fixture_id, family, _norm(row.get("market")), _norm(row.get("selection")), line, _norm(row.get("bookmaker") or row.get("bookmaker_at_signal")), entry_ts.isoformat())


def _strict_canonical_row(row: dict[str, Any]) -> bool:
    if not isinstance(row, dict) or row.get("probability_comparable_same_line") is not True:
        return False
    family = _family(row)
    entry_ts = _parse_dt(row.get("entry_timestamp") or row.get("signal_timestamp_local"))
    close_ts = _parse_dt(row.get("closing_timestamp") or row.get("close_timestamp_local"))
    kickoff = _parse_dt(row.get("kickoff") or row.get("kickoff_local"))
    if family is None or entry_ts is None or close_ts is None or kickoff is None or not (entry_ts < close_ts < kickoff):
        return False
    if str(family).upper() in TEAM_TOTAL_FAMILIES:
        provider_close_ts = _parse_dt(row.get("closing_provider_update") or row.get("close_provider_update"))
        if provider_close_ts is None or not (entry_ts < provider_close_ts < kickoff):
            return False
    return _canonical_exact_key(row) is not None


def preserve_previous_canonical_rows(previous_rows: Iterable[dict[str, Any]], current_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    current_keys = {_canonical_exact_key(row) for row in current_rows if isinstance(row, dict)}
    preserved: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in previous_rows:
        if not _strict_canonical_row(row):
            continue
        key = _canonical_exact_key(row)
        if key is None or key in current_keys:
            continue
        prior = preserved.get(key)
        if prior is None:
            preserved[key] = dict(row)
            continue
        prior_close = _parse_dt(prior.get("closing_timestamp") or prior.get("close_timestamp_local"))
        row_close = _parse_dt(row.get("closing_timestamp") or row.get("close_timestamp_local"))
        if row_close is not None and (prior_close is None or row_close > prior_close):
            preserved[key] = dict(row)
    return list(preserved.values())


def merge_report(postgres_report: dict[str, Any], history_rows: Iterable[dict[str, Any]], previous_canonical_rows: Iterable[dict[str, Any]] = ()) -> dict[str, Any]:
    current_rows = postgres_report.get("rows") if isinstance(postgres_report.get("rows"), list) else []
    preserved_canonical = preserve_previous_canonical_rows(previous_canonical_rows, current_rows)
    current_plus_preserved = list(current_rows) + preserved_canonical
    historical = normalize_history_rows(history_rows, current_plus_preserved)
    merged_rows = current_plus_preserved + historical
    family_counts = Counter(); source_counts = Counter(); unique_by_family: dict[str, set[int]] = {}
    malformed_fixture_rows = 0
    for row in merged_rows:
        if not isinstance(row, dict):
            continue
        family = _family(row)
        fixture_id = _fixture_id(row.get("fixture_id"))
        if family:
            family_counts[family] += 1
            if fixture_id is not None:
                unique_by_family.setdefault(family, set()).add(fixture_id)
            else:
                malformed_fixture_rows += 1
        source_counts[str(row.get("signal_source") or "UNKNOWN")] += 1
    comparable = [row for row in merged_rows if isinstance(row, dict) and row.get("probability_comparable_same_line") is True]
    out = dict(postgres_report)
    out.update({"rows": merged_rows, "tracked_rows": len(merged_rows), "comparable_true_clv_rows": len(comparable), "status": "ACTIVE_TRUE_CLV_SAMPLE" if len(comparable) >= MIN_TRUE_CLOSE_ROWS else "COLLECTING_TRUE_CLV", "family_counts": dict(sorted(family_counts.items())), "signal_source_counts": dict(sorted(source_counts.items())), "canonical_merge_version": MERGE_VERSION, "historical_backfill_rows_added": len(historical), "previous_canonical_rows_preserved": len(preserved_canonical), "historical_backfill_unique_fixtures": len({row["fixture_id"] for row in historical}), "unique_fixtures_by_family": {family: len(fixtures) for family, fixtures in sorted(unique_by_family.items())}, "malformed_fixture_rows_ignored_for_unique_counts": malformed_fixture_rows, "provider_requests_added": 0})
    notes = list(out.get("notes") or [])
    notes.append("Prior canonical strict True CLV rows are preserved by exact signal key so a bounded current Postgres window cannot silently delete previously validated evidence.")
    notes.append("Historical dedicated-close backfill adds at most one latest pre-kickoff row per fixture/family and is skipped whenever Postgres already covers that fixture/family.")
    notes.append("Historical Team Totals backfill additionally requires a provider close update strictly after the signal and before kickoff; legacy rows without this provenance cannot re-enter canonical true CLV.")
    out["notes"] = notes
    return out


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as handle: value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows=[]
    if not path or not os.path.exists(path): return rows
    with open(path,"r",encoding="utf-8") as handle:
        for line in handle:
            line=line.strip()
            if not line: continue
            try: value=json.loads(line)
            except json.JSONDecodeError: continue
            if isinstance(value,dict): rows.append(value)
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(description="Merge strict historical dedicated-close CLV into the canonical Postgres CLV ledger.")
    parser.add_argument("--postgres-report", required=True); parser.add_argument("--history-true-clv", required=True); parser.add_argument("--previous-canonical"); parser.add_argument("--output", required=True); args=parser.parse_args()
    report=merge_report(_load_json(args.postgres_report), _load_jsonl(args.history_true_clv), _load_jsonl(args.previous_canonical) if args.previous_canonical else [])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as handle: json.dump(report,handle,ensure_ascii=False,indent=2,sort_keys=True); handle.write("\n")
    print(json.dumps({"canonical_merge_version":report["canonical_merge_version"],"tracked_rows":report["tracked_rows"],"comparable_true_clv_rows":report["comparable_true_clv_rows"],"historical_backfill_rows_added":report["historical_backfill_rows_added"],"historical_backfill_unique_fixtures":report["historical_backfill_unique_fixtures"],"previous_canonical_rows_preserved":report.get("previous_canonical_rows_preserved",0),"unique_fixtures_by_family":report["unique_fixtures_by_family"],"family_counts":report["family_counts"],"malformed_fixture_rows_ignored_for_unique_counts":report["malformed_fixture_rows_ignored_for_unique_counts"],"provider_requests_added":report["provider_requests_added"]},indent=2,sort_keys=True))

if __name__ == "__main__": main()
