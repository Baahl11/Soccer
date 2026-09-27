from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base
from mcp_gateway.ft_totals_validation_v4 import SUPPORTED_LINES, split_asian_line

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FT_TOTALS_LINE_COVERAGE_AUDIT_V4_1.0.0"

_CANONICAL_MARKETS = {
    "goals over/under",
    "over/under",
    "goals over under",
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _payload(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return [row for row in parsed if isinstance(row, dict)] if isinstance(parsed, list) else []
    return []


def is_canonical_ft_total_market(value: Any) -> bool:
    return _norm(value) in _CANONICAL_MARKETS


def value_side_line(value: dict[str, Any]) -> tuple[str | None, float | None]:
    raw = value.get("raw_selection")
    if raw is None:
        raw = value.get("selection")
    if raw is None:
        raw = value.get("value")
    text = _norm(raw)

    side = None
    if text.startswith("over"):
        side = "OVER"
    elif text.startswith("under"):
        side = "UNDER"

    line = None
    for key in ("line", "handicap", "parsed_line"):
        line = _num(value.get(key))
        if line is not None:
            break
    if line is None:
        match = re.search(r"\b(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)\b", text, flags=re.IGNORECASE)
        if match:
            line = _num(match.group(1))

    if line is not None:
        line = round(float(line), 2)
    return side, line


def value_price(value: dict[str, Any]) -> float | None:
    for key in ("decimal_price", "price", "odd"):
        parsed = _num(value.get(key))
        if parsed is not None and 1.0 < parsed <= 1000.0:
            return parsed
    return None


def _line_key(value: float) -> str:
    rounded = round(float(value), 2)
    return f"{rounded:.2f}".rstrip("0").rstrip(".")


def _supported_line(value: float) -> bool:
    return any(math.isclose(float(value), float(line), abs_tol=1e-8) for line in SUPPORTED_LINES)


def _is_quarter_line(value: float) -> bool:
    return len(split_asian_line(float(value))) == 2


def summarize_rows(rows: Iterable[dict[str, Any]], *, lookback_days: int) -> dict[str, Any]:
    rows_seen = 0
    canonical_rows = 0
    canonical_pre_kickoff_rows = 0
    value_rows = 0
    priced_value_rows = 0
    exact_line_value_rows = 0
    supported_line_value_rows = 0
    unsupported_line_value_rows = 0
    invalid_side_or_line_rows = 0

    all_fixtures: set[int] = set()
    pre_kickoff_fixtures: set[int] = set()
    provider_update_fixtures: set[int] = set()
    bookmakers: set[str] = set()
    stages: Counter[str] = Counter()
    market_names: Counter[str] = Counter()
    observed_lines: set[float] = set()
    pre_kickoff_lines: set[float] = set()
    quarter_line_fixtures: set[int] = set()
    pre_kickoff_quarter_line_fixtures: set[int] = set()

    line_stats: dict[float, dict[str, Any]] = defaultdict(lambda: {
        "market_snapshot_rows": 0,
        "pre_kickoff_market_snapshot_rows": 0,
        "value_rows": 0,
        "priced_value_rows": 0,
        "provider_update_value_rows": 0,
        "fixtures": set(),
        "pre_kickoff_fixtures": set(),
        "bookmakers": set(),
        "stages": Counter(),
        "side_counts": Counter(),
    })

    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        rows_seen += 1
        if not is_canonical_ft_total_market(row.get("market")):
            continue
        canonical_rows += 1
        market_names[str(row.get("market") or "UNKNOWN")] += 1
        pre_kickoff = bool(row.get("pre_kickoff"))
        if pre_kickoff:
            canonical_pre_kickoff_rows += 1
        stage = str(row.get("stage") or "UNKNOWN")
        stages[stage] += 1
        bookmaker = str(row.get("bookmaker") or "").strip()
        if bookmaker:
            bookmakers.add(bookmaker)

        fixture_id = row.get("fixture_id")
        fixture_int = None
        try:
            fixture_int = int(fixture_id) if fixture_id is not None else None
        except (TypeError, ValueError):
            fixture_int = None
        if fixture_int is not None:
            all_fixtures.add(fixture_int)
            if pre_kickoff:
                pre_kickoff_fixtures.add(fixture_int)
            if row.get("provider_update") is not None:
                provider_update_fixtures.add(fixture_int)

        row_lines: set[float] = set()
        values = _payload(row.get("values"))
        value_rows += len(values)
        for value in values:
            side, line = value_side_line(value)
            if side is None or line is None:
                invalid_side_or_line_rows += 1
                continue
            exact_line_value_rows += 1
            price = value_price(value)
            if price is not None:
                priced_value_rows += 1
            if _supported_line(line):
                supported_line_value_rows += 1
            else:
                unsupported_line_value_rows += 1

            observed_lines.add(line)
            if pre_kickoff:
                pre_kickoff_lines.add(line)
            row_lines.add(line)
            stats = line_stats[line]
            stats["value_rows"] += 1
            stats["priced_value_rows"] += int(price is not None)
            stats["provider_update_value_rows"] += int(row.get("provider_update") is not None)
            stats["side_counts"][side] += 1
            stats["stages"][stage] += 1
            if bookmaker:
                stats["bookmakers"].add(bookmaker)
            if fixture_int is not None:
                stats["fixtures"].add(fixture_int)
                if pre_kickoff:
                    stats["pre_kickoff_fixtures"].add(fixture_int)
                if _is_quarter_line(line):
                    quarter_line_fixtures.add(fixture_int)
                    if pre_kickoff:
                        pre_kickoff_quarter_line_fixtures.add(fixture_int)

        for line in row_lines:
            line_stats[line]["market_snapshot_rows"] += 1
            if pre_kickoff:
                line_stats[line]["pre_kickoff_market_snapshot_rows"] += 1

        if len(sample_rows) < 50:
            sample_rows.append({
                "fixture_id": fixture_int,
                "captured_at": row.get("captured_at"),
                "stage": stage,
                "bookmaker": bookmaker or None,
                "market": row.get("market"),
                "provider_update": row.get("provider_update"),
                "pre_kickoff": pre_kickoff,
                "observed_lines": sorted(row_lines),
                "values": values[:8],
            })

    by_line: dict[str, Any] = {}
    for line in sorted(line_stats):
        stats = line_stats[line]
        by_line[_line_key(line)] = {
            "line": line,
            "supported_by_v4_settlement": _supported_line(line),
            "quarter_line": _is_quarter_line(line),
            "market_snapshot_rows": int(stats["market_snapshot_rows"]),
            "pre_kickoff_market_snapshot_rows": int(stats["pre_kickoff_market_snapshot_rows"]),
            "value_rows": int(stats["value_rows"]),
            "priced_value_rows": int(stats["priced_value_rows"]),
            "provider_update_value_rows": int(stats["provider_update_value_rows"]),
            "unique_fixtures": len(stats["fixtures"]),
            "pre_kickoff_unique_fixtures": len(stats["pre_kickoff_fixtures"]),
            "bookmaker_count": len(stats["bookmakers"]),
            "side_counts": dict(sorted(stats["side_counts"].items())),
            "stages": dict(sorted(stats["stages"].items())),
        }

    supported_line_coverage: dict[str, Any] = {}
    missing_supported_lines: list[float] = []
    missing_supported_pre_kickoff_lines: list[float] = []
    for supported in SUPPORTED_LINES:
        line = float(supported)
        stats = line_stats.get(line)
        observed = stats is not None and int(stats["value_rows"]) > 0
        pre_kickoff_observed = stats is not None and bool(stats["pre_kickoff_fixtures"])
        if not observed:
            missing_supported_lines.append(line)
        if not pre_kickoff_observed:
            missing_supported_pre_kickoff_lines.append(line)
        supported_line_coverage[_line_key(line)] = {
            "line": line,
            "observed": observed,
            "observed_pre_kickoff": pre_kickoff_observed,
            "unique_fixtures": len(stats["fixtures"]) if stats is not None else 0,
            "pre_kickoff_unique_fixtures": len(stats["pre_kickoff_fixtures"]) if stats is not None else 0,
            "priced_value_rows": int(stats["priced_value_rows"]) if stats is not None else 0,
        }

    quarter_lines_observed = sorted(line for line in observed_lines if _is_quarter_line(line))
    quarter_lines_pre_kickoff = sorted(line for line in pre_kickoff_lines if _is_quarter_line(line))

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "FT_TOTALS_OBSERVED_LINE_COVERAGE_AUDIT",
        "lookback_days": int(lookback_days),
        "source": "POSTGRES_SOCCER_MARKET_SNAPSHOTS",
        "source_semantics": "FRESH_PROVIDER_MARKET_SNAPSHOTS_PERSISTED_BY_RUNTIME; CACHE_REPLAYS_EXCLUDED_UPSTREAM",
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_settled_sample_changed": False,
        "actionable_sample_changed": False,
        "rows_seen": rows_seen,
        "canonical_market_snapshot_rows": canonical_rows,
        "canonical_pre_kickoff_market_snapshot_rows": canonical_pre_kickoff_rows,
        "unique_fixtures": len(all_fixtures),
        "pre_kickoff_unique_fixtures": len(pre_kickoff_fixtures),
        "provider_update_unique_fixtures": len(provider_update_fixtures),
        "bookmaker_count": len(bookmakers),
        "value_rows": value_rows,
        "priced_value_rows": priced_value_rows,
        "exact_line_value_rows": exact_line_value_rows,
        "supported_line_value_rows": supported_line_value_rows,
        "unsupported_line_value_rows": unsupported_line_value_rows,
        "invalid_side_or_line_rows": invalid_side_or_line_rows,
        "observed_lines": sorted(observed_lines),
        "pre_kickoff_observed_lines": sorted(pre_kickoff_lines),
        "quarter_lines_observed": quarter_lines_observed,
        "quarter_lines_pre_kickoff": quarter_lines_pre_kickoff,
        "quarter_line_unique_fixtures": len(quarter_line_fixtures),
        "pre_kickoff_quarter_line_unique_fixtures": len(pre_kickoff_quarter_line_fixtures),
        "supported_lines": [float(line) for line in SUPPORTED_LINES],
        "missing_supported_lines": missing_supported_lines,
        "missing_supported_pre_kickoff_lines": missing_supported_pre_kickoff_lines,
        "supported_line_coverage": supported_line_coverage,
        "by_line": by_line,
        "stages": dict(sorted(stages.items())),
        "market_name_counts": dict(market_names.most_common()),
        "sample_rows": sample_rows,
        "policy": (
            "POSTGRES_READ_ONLY; STRICT_CANONICAL_FULL_MATCH_TOTALS_ONLY; FRESH_PERSISTED_MARKET_SNAPSHOTS; "
            "OBSERVED_LINE_AVAILABILITY_IS_DIAGNOSTIC_ONLY; PRE_KICKOFF_FIXTURE_COUNTS_ARE_DEDUPED_BY_LINE; "
            "NO PROVIDER CALLS; NO BET/TIER/STAKE/THRESHOLD CHANGE; OBSERVED MARKET COVERAGE DOES NOT SATISFY "
            "V4_016 MODEL_SETTLED, ACTIONABLE, OOS, CLV OR PRODUCTION_PROMOTION GATES"
        ),
    }


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 100000) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_rows = max(100, min(int(max_rows), 300000))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "lookback_days": lookback_days,
            "provider_requests_added": 0,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
            "canonical_bet_logic_changed": False,
            "model_settled_sample_changed": False,
            "actionable_sample_changed": False,
            "by_line": {},
            "supported_line_coverage": {},
        }

    persistence_base.ensure_schema()
    with persistence_base._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    m.fixture_id,
                    m.captured_at,
                    m.stage,
                    m.bookmaker_id,
                    m.bookmaker,
                    m.market_id,
                    m.market,
                    m.values,
                    m.provider_update,
                    f.kickoff,
                    CASE
                        WHEN f.kickoff IS NOT NULL AND m.captured_at < f.kickoff THEN TRUE
                        ELSE FALSE
                    END AS pre_kickoff
                FROM soccer_market_snapshots m
                LEFT JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                WHERE m.captured_at >= NOW() - (%s * INTERVAL '1 day')
                  AND LOWER(TRIM(COALESCE(m.market, ''))) IN (
                    'goals over/under',
                    'over/under',
                    'goals over under'
                  )
                ORDER BY m.captured_at DESC
                LIMIT %s
                """,
                (lookback_days, max_rows),
            )
            columns = [desc.name for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    for row in rows:
        for key in ("captured_at", "provider_update", "kickoff"):
            value = row.get(key)
            if isinstance(value, datetime):
                row[key] = value.astimezone(timezone.utc).isoformat()

    report = summarize_rows(rows, lookback_days=lookback_days)
    report["rows_scanned_from_postgres"] = len(rows)
    report["max_rows"] = max_rows
    return report
