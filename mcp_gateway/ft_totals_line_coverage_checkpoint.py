from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import persistence

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_FT_TOTALS_LINE_COVERAGE_CHECKPOINT_V1.0.0"
CACHE_TTL_SECONDS = 1800
LOOKBACK_DAYS = 180
SUPPORTED_VALIDATION_LINES = (1.5, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5)

_CACHE: dict[str, Any] | None = None
_CACHE_MONOTONIC: float | None = None


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _is_quarter_line(line: float) -> bool:
    quarter_units = round(float(line) * 4.0)
    half_units = float(line) * 2.0
    return (
        math.isclose(float(line) * 4.0, quarter_units, abs_tol=1e-8)
        and not math.isclose(half_units, round(half_units), abs_tol=1e-8)
    )


def _build_report(
    line_rows: list[dict[str, Any]],
    overall: dict[str, Any],
    *,
    lookback_days: int,
) -> dict[str, Any]:
    lines: list[dict[str, Any]] = []
    quarter_fixture_ids_total = 0
    supported_fixture_ids_total = 0
    for row in line_rows:
        line = _num(row.get("line"))
        if line is None:
            continue
        line = round(line, 2)
        unique_fixtures = int(row.get("unique_fixtures") or 0)
        item = {
            "line": line,
            "value_rows": int(row.get("value_rows") or 0),
            "unique_fixtures": unique_fixtures,
            "snapshot_rows": int(row.get("snapshot_rows") or 0),
            "bookmaker_count": int(row.get("bookmaker_count") or 0),
            "first_captured_at": row.get("first_captured_at"),
            "last_captured_at": row.get("last_captured_at"),
            "quarter_line": _is_quarter_line(line),
            "validation_supported": line in SUPPORTED_VALIDATION_LINES,
        }
        lines.append(item)
        if item["quarter_line"]:
            quarter_fixture_ids_total += unique_fixtures
        if item["validation_supported"]:
            supported_fixture_ids_total += unique_fixtures

    lines.sort(key=lambda item: float(item["line"]))
    by_line = {f"{float(item['line']):.2f}": item for item in lines}
    focus = {
        key: by_line.get(key, {
            "line": float(key),
            "value_rows": 0,
            "unique_fixtures": 0,
            "snapshot_rows": 0,
            "bookmaker_count": 0,
            "first_captured_at": None,
            "last_captured_at": None,
            "quarter_line": True,
            "validation_supported": True,
        })
        for key in ("2.25", "2.75", "3.25")
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_READY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "lookback_days": int(lookback_days),
        "strict_market_policy": "FULL_MATCH_GOALS_OVER_UNDER_ONLY",
        "pre_kickoff_only": True,
        "market_snapshot_rows": int(overall.get("market_snapshot_rows") or 0),
        "unique_fixtures": int(overall.get("unique_fixtures") or 0),
        "unique_bookmakers": int(overall.get("unique_bookmakers") or 0),
        "line_count": len(lines),
        "lines": lines,
        "focus_quarter_lines": focus,
        "quarter_line_fixture_sum_non_deduped": quarter_fixture_ids_total,
        "supported_line_fixture_sum_non_deduped": supported_fixture_ids_total,
        "provider_requests_added": 0,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "counts_as_model_settled": False,
        "counts_as_oos": False,
        "counts_as_true_clv": False,
        "policy": (
            "POSTGRES_READ_ONLY; STRICT_PREKICKOFF_FT_TOTAL_MARKET_SNAPSHOTS; "
            "UNIQUE_FIXTURES_DEDUPED_PER_LINE; CROSS_LINE_SUMS_ARE_DIAGNOSTIC_ONLY; "
            "NO_PROVIDER_CALLS; NO_MODEL_GATE_OR_PROMOTION_EFFECT"
        ),
    }


def _query_postgres(*, lookback_days: int) -> dict[str, Any]:
    if not persistence.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "lookback_days": int(lookback_days),
            "provider_requests_added": 0,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        }

    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH expanded AS (
                    SELECT
                        m.fixture_id,
                        m.captured_at,
                        m.bookmaker_id,
                        m.market_id,
                        value,
                        ROUND((value->>'line')::numeric, 2) AS line
                    FROM soccer_market_snapshots m
                    JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(m.values) = 'array' THEN m.values
                            ELSE '[]'::jsonb
                        END
                    ) AS value
                    WHERE m.captured_at >= NOW() - (%s * INTERVAL '1 day')
                      AND m.captured_at < f.kickoff
                      AND LOWER(TRIM(COALESCE(m.market, ''))) IN (
                            'goals over/under',
                            'over/under',
                            'goals over under'
                      )
                      AND jsonb_typeof(value) = 'object'
                      AND value ? 'line'
                      AND COALESCE(value->>'line', '') ~ '^[+-]?[0-9]+([.][0-9]+)?$'
                )
                SELECT
                    line::text AS line,
                    COUNT(*) AS value_rows,
                    COUNT(DISTINCT fixture_id) AS unique_fixtures,
                    COUNT(DISTINCT (fixture_id, captured_at, bookmaker_id, market_id)) AS snapshot_rows,
                    COUNT(DISTINCT bookmaker_id) AS bookmaker_count,
                    MIN(captured_at) AS first_captured_at,
                    MAX(captured_at) AS last_captured_at
                FROM expanded
                GROUP BY line
                ORDER BY line
                """,
                (max(1, min(int(lookback_days), 730)),),
            )
            columns = [desc.name for desc in cur.description]
            line_rows = [dict(zip(columns, row)) for row in cur.fetchall()]

            cur.execute(
                """
                SELECT
                    COUNT(*) AS market_snapshot_rows,
                    COUNT(DISTINCT m.fixture_id) AS unique_fixtures,
                    COUNT(DISTINCT m.bookmaker_id) AS unique_bookmakers
                FROM soccer_market_snapshots m
                JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                WHERE m.captured_at >= NOW() - (%s * INTERVAL '1 day')
                  AND m.captured_at < f.kickoff
                  AND LOWER(TRIM(COALESCE(m.market, ''))) IN (
                        'goals over/under',
                        'over/under',
                        'goals over under'
                  )
                """,
                (max(1, min(int(lookback_days), 730)),),
            )
            row = cur.fetchone()
            overall_columns = [desc.name for desc in cur.description]
            overall = dict(zip(overall_columns, row)) if row else {}

    for row in line_rows:
        for key in ("first_captured_at", "last_captured_at"):
            value = row.get(key)
            if isinstance(value, datetime):
                row[key] = value.astimezone(timezone.utc).isoformat()

    return _build_report(line_rows, overall, lookback_days=lookback_days)


def build(*, lookback_days: int = LOOKBACK_DAYS, force_refresh: bool = False) -> dict[str, Any]:
    global _CACHE, _CACHE_MONOTONIC
    now = time.monotonic()
    if (
        not force_refresh
        and isinstance(_CACHE, dict)
        and _CACHE_MONOTONIC is not None
        and (now - _CACHE_MONOTONIC) < CACHE_TTL_SECONDS
    ):
        cached = dict(_CACHE)
        cached["cache_status"] = "MEMORY_CACHE_HIT"
        cached["cache_ttl_seconds"] = CACHE_TTL_SECONDS
        return cached

    try:
        report = _query_postgres(lookback_days=max(1, min(int(lookback_days), 730)))
    except Exception as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "HISTORICAL_FT_TOTALS_LINE_COVERAGE_ERROR",
            "detail": str(exc)[:300],
            "lookback_days": max(1, min(int(lookback_days), 730)),
            "provider_requests_added": 0,
            "decision_weight": 0.0,
            "production_promotion_allowed": False,
        }

    _CACHE = dict(report)
    _CACHE_MONOTONIC = now
    report = dict(report)
    report["cache_status"] = "POSTGRES_REFRESH"
    report["cache_ttl_seconds"] = CACHE_TTL_SECONDS
    return report
