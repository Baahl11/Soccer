from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

MODEL_VERSION = "SOCCER_PRIMARY_CLV_SIGNAL_ANCHOR_STORE_V4_1.0.0"
SUPPORTED_FAMILIES = {"1X2", "FT_TOTALS", "BTTS"}


def _family(value: Any) -> str:
    raw = str(value or "").strip().upper()
    if raw in {"TOTAL", "FT_TOTALS_RESEARCH"}:
        return "FT_TOTALS"
    if raw in {"FT_BTTS", "FT_BTTS_RESEARCH"}:
        return "BTTS"
    if raw in {"FT_1X2", "FT_1X2_RESEARCH", "MATCH_WINNER"}:
        return "1X2"
    return raw


def _truthy_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "t", "1", "yes", "y"}


def _is_json_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _text(value: Any) -> str | None:
    if value is None:
        return None
    out = str(value).strip()
    return out or None


def extract_tick_anchor_rows(tick: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract exactly the primary signal populations used by the legacy loader.

    This is storage normalization only. It does not change model probabilities,
    rankings, CLV chronology, market matching, provider budgets, or bet logic.
    """
    generated_at = tick.get("generated_at_utc")
    if not generated_at:
        return []

    out: list[dict[str, Any]] = []

    mismatch_rows = (
        tick.get("market_mismatch_rows")
        if isinstance(tick.get("market_mismatch_rows"), list)
        else []
    )
    for row in mismatch_rows:
        if not isinstance(row, dict):
            continue
        family = _family(row.get("market_family"))
        market = _text(row.get("market"))
        selection = _text(row.get("selection"))
        price = _text(row.get("price"))
        fixture_id = row.get("fixture_id")
        if (
            family not in SUPPORTED_FAMILIES
            or not _truthy_bool(row.get("rankable"))
            or fixture_id is None
            or market is None
            or selection is None
            or price is None
        ):
            continue
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "fixture_id": fixture_id,
                "market_family": family,
                "market": market,
                "selection": selection,
                "line_text": _text(row.get("line")),
                "price_text": price,
                "signal_generated_at": generated_at,
                "candidate_source": "PHASE16_RANKABLE",
                "source_priority": 0,
                "source_row": row,
            }
        )

    match_rows = (
        tick.get("match_table_rows")
        if isinstance(tick.get("match_table_rows"), list)
        else []
    )
    for row in match_rows:
        if not isinstance(row, dict):
            continue
        family = _family(row.get("market_family"))
        market = _text(row.get("market"))
        selection = _text(row.get("selection"))
        fixture_id = row.get("fixture_id")
        price_value = row.get("price")
        if (
            family not in SUPPORTED_FAMILIES
            or fixture_id is None
            or market is None
            or selection is None
            or not _is_json_number(price_value)
            or float(price_value) <= 1.0
        ):
            continue
        try:
            fixture_id = int(fixture_id)
        except (TypeError, ValueError):
            continue
        out.append(
            {
                "fixture_id": fixture_id,
                "market_family": family,
                "market": market,
                "selection": selection,
                "line_text": _text(row.get("line")),
                "price_text": str(price_value),
                "signal_generated_at": generated_at,
                "candidate_source": "MATCH_TABLE_PRICED_RESEARCH",
                "source_priority": 1,
                "source_row": row,
            }
        )

    return out


def persist_tick_anchor_rows(cur: Any, tick: dict[str, Any]) -> int:
    rows = extract_tick_anchor_rows(tick)
    inserted = 0
    for row in rows:
        cur.execute(
            """
            INSERT INTO soccer_primary_clv_signal_anchors (
                fixture_id, market_family, market, selection, line_text, price_text,
                signal_generated_at, candidate_source, source_priority, source_row
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT DO NOTHING
            """,
            (
                row["fixture_id"],
                row["market_family"],
                row["market"],
                row["selection"],
                row.get("line_text"),
                row.get("price_text"),
                row["signal_generated_at"],
                row["candidate_source"],
                row["source_priority"],
                json.dumps(row.get("source_row") or {}),
            ),
        )
        try:
            inserted += max(0, int(cur.rowcount or 0))
        except (TypeError, ValueError):
            pass
    return inserted


def materialize_historical_anchors(
    *,
    lookback_days: int = 180,
    max_runs: int = 100000,
) -> dict[str, Any]:
    """One-time/offline backfill from compact historical pipeline envelopes."""
    from mcp_gateway import persistence

    lookback_days = max(1, min(int(lookback_days), 730))
    max_runs = max(100, min(int(max_runs), 200000))
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

    if not persistence.persistence_configured():
        return {
            "status": "POSTGRES_NOT_CONFIGURED",
            "model_version": MODEL_VERSION,
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
        }

    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)::BIGINT
                FROM soccer_pipeline_runs
                WHERE generated_at_utc >= %s
                """,
                (cutoff,),
            )
            runs_in_window = int((cur.fetchone() or [0])[0] or 0)

            cur.execute(
                """
                WITH recent_runs AS (
                    SELECT generated_at_utc, payload
                    FROM soccer_pipeline_runs
                    WHERE generated_at_utc >= %s
                    ORDER BY generated_at_utc ASC, run_id ASC
                    LIMIT %s
                ),
                candidates AS (
                    SELECT
                        (mm.row ->> 'fixture_id')::BIGINT AS fixture_id,
                        UPPER(mm.row ->> 'market_family') AS market_family,
                        mm.row ->> 'market' AS market,
                        mm.row ->> 'selection' AS selection,
                        NULLIF(mm.row ->> 'line', '') AS line_text,
                        mm.row ->> 'price' AS price_text,
                        p.generated_at_utc AS signal_generated_at,
                        'PHASE16_RANKABLE'::TEXT AS candidate_source,
                        0::INT AS source_priority,
                        mm.row AS source_row
                    FROM recent_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)) = 'array'
                            THEN COALESCE(p.payload -> 'market_mismatch_rows', '[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) AS mm(row)
                    WHERE UPPER(mm.row ->> 'market_family') IN ('1X2','FT_TOTALS','BTTS')
                      AND COALESCE((mm.row ->> 'rankable')::boolean, false) = true
                      AND NULLIF(mm.row ->> 'fixture_id', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'market', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'selection', '') IS NOT NULL
                      AND NULLIF(mm.row ->> 'price', '') IS NOT NULL

                    UNION ALL

                    SELECT
                        (mt.row ->> 'fixture_id')::BIGINT AS fixture_id,
                        CASE
                            WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row ->> 'market_family')
                        END AS market_family,
                        mt.row ->> 'market' AS market,
                        mt.row ->> 'selection' AS selection,
                        NULLIF(mt.row ->> 'line', '') AS line_text,
                        mt.row ->> 'price' AS price_text,
                        p.generated_at_utc AS signal_generated_at,
                        'MATCH_TABLE_PRICED_RESEARCH'::TEXT AS candidate_source,
                        1::INT AS source_priority,
                        mt.row AS source_row
                    FROM recent_runs p
                    CROSS JOIN LATERAL jsonb_array_elements(
                        CASE
                            WHEN jsonb_typeof(COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)) = 'array'
                            THEN COALESCE(p.payload -> 'match_table_rows', '[]'::jsonb)
                            ELSE '[]'::jsonb
                        END
                    ) AS mt(row)
                    WHERE CASE
                            WHEN UPPER(mt.row ->> 'market_family') IN ('TOTAL','FT_TOTALS_RESEARCH') THEN 'FT_TOTALS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_BTTS','FT_BTTS_RESEARCH') THEN 'BTTS'
                            WHEN UPPER(mt.row ->> 'market_family') IN ('FT_1X2','FT_1X2_RESEARCH','MATCH_WINNER') THEN '1X2'
                            ELSE UPPER(mt.row ->> 'market_family')
                          END IN ('1X2','FT_TOTALS','BTTS')
                      AND NULLIF(mt.row ->> 'fixture_id', '') IS NOT NULL
                      AND NULLIF(mt.row ->> 'market', '') IS NOT NULL
                      AND NULLIF(mt.row ->> 'selection', '') IS NOT NULL
                      AND NULLIF(mt.row ->> 'price', '') IS NOT NULL
                      AND jsonb_typeof(mt.row -> 'price') = 'number'
                      AND (mt.row ->> 'price')::DOUBLE PRECISION > 1.0
                ),
                inserted AS (
                    INSERT INTO soccer_primary_clv_signal_anchors (
                        fixture_id, market_family, market, selection, line_text, price_text,
                        signal_generated_at, candidate_source, source_priority, source_row
                    )
                    SELECT
                        fixture_id, market_family, market, selection, line_text, price_text,
                        signal_generated_at, candidate_source, source_priority, source_row
                    FROM candidates
                    ON CONFLICT DO NOTHING
                    RETURNING 1
                )
                SELECT COUNT(*)::BIGINT FROM inserted
                """,
                (cutoff, max_runs),
            )
            inserted_rows = int((cur.fetchone() or [0])[0] or 0)

            cur.execute(
                """
                SELECT
                    COUNT(*)::BIGINT AS rows,
                    COUNT(DISTINCT fixture_id)::BIGINT AS fixtures,
                    market_family
                FROM soccer_primary_clv_signal_anchors
                WHERE signal_generated_at >= %s
                GROUP BY market_family
                ORDER BY market_family
                """,
                (cutoff,),
            )
            family_counts = {
                str(family): {"rows": int(rows or 0), "fixtures": int(fixtures or 0)}
                for rows, fixtures, family in cur.fetchall()
            }

    return {
        "status": "OK",
        "model_version": MODEL_VERSION,
        "lookback_days": lookback_days,
        "max_runs": max_runs,
        "runs_in_window": runs_in_window,
        "run_limit_saturated": runs_in_window > max_runs,
        "inserted_rows": inserted_rows,
        "family_counts": family_counts,
        "provider_requests_added": 0,
        "strict_close_semantics_changed": False,
        "models_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "provider_budget_changed": False,
        "canonical_bet_logic_changed": False,
        "historical_probabilities_recomputed": False,
        "production_promotion_allowed": False,
        "policy": (
            "OFFLINE_STORAGE_NORMALIZATION_ONLY; EXACT_LEGACY_PRIMARY_SIGNAL_POPULATIONS; "
            "NO_PROVIDER_CALLS; NO_CLV_SEMANTIC_CHANGE"
        ),
    }
