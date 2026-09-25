from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from mcp_gateway import persistence as persistence_base

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_RESEARCH_DERIVATIVE_MARKET_AUDIT_V4_1.0.0"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def classify_market(market: Any) -> str | None:
    name = _norm(market)
    if not name:
        return None

    player_card_tokens = (
        "player cards",
        "player card",
        "player booked",
        "player booking",
        "to be booked",
    )
    if any(token in name for token in player_card_tokens):
        return "PLAYER_CARDS"
    if "shots on target" in name or "shot on target" in name:
        return "SOT"
    if "goalkeeper saves" in name or "keeper saves" in name or "gk saves" in name:
        return "GK_SAVES"
    if "goalscorer" in name or "anytime scorer" in name or "player to score" in name:
        return "GOALSCORER"
    if "assist" in name:
        return "ASSISTS"
    if "player shots" in name or "player shot" in name:
        return "SHOTS"

    card_tokens = (
        "cards over/under",
        "card over/under",
        "total cards",
        "total yellow cards",
        "yellow cards",
        "red card",
        "team cards",
        "booking points",
        "bookings",
    )
    if any(token in name for token in card_tokens):
        return "CARDS"
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


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def value_line(value: dict[str, Any]) -> float | None:
    for key in ("line", "parsed_line", "handicap"):
        parsed = _num(value.get(key))
        if parsed is not None:
            return parsed
    raw = value.get("raw_selection")
    if raw is None:
        raw = value.get("selection")
    if raw is None:
        raw = value.get("value")
    match = re.search(
        r"\b(?:over|under)\s+([+-]?\d+(?:\.\d+)?)\b",
        str(raw or ""),
        flags=re.IGNORECASE,
    )
    return _num(match.group(1)) if match else None


def _price(value: dict[str, Any]) -> float | None:
    for key in ("decimal_price", "price", "odd"):
        parsed = _num(value.get(key))
        if parsed is not None and parsed > 1.0:
            return parsed
    return None


def summarize_rows(rows: Iterable[dict[str, Any]], *, lookback_days: int) -> dict[str, Any]:
    family_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    samples: list[dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        family = classify_market(row.get("market"))
        if family is None:
            continue
        values = _payload(row.get("values"))
        line_values = sum(1 for value in values if value_line(value) is not None)
        priced_values = sum(1 for value in values if _price(value) is not None)
        normalized = {
            **row,
            "family": family,
            "value_count": len(values),
            "line_value_count": line_values,
            "priced_value_count": priced_values,
        }
        family_rows[family].append(normalized)
        if len(samples) < 50:
            samples.append({
                "fixture_id": row.get("fixture_id"),
                "captured_at": row.get("captured_at"),
                "stage": row.get("stage"),
                "bookmaker": row.get("bookmaker"),
                "market": row.get("market"),
                "family": family,
                "provider_update": row.get("provider_update"),
                "confirmed_xi_before_market": bool(row.get("confirmed_xi_before_market")),
                "pre_kickoff": bool(row.get("pre_kickoff")),
                "value_count": len(values),
                "line_value_count": line_values,
            })

    summaries: dict[str, Any] = {}
    all_fixtures: set[int] = set()
    all_confirmed: set[int] = set()
    all_rows = 0
    all_line_values = 0

    for family in ("CARDS", "PLAYER_CARDS", "SHOTS", "SOT", "GOALSCORER", "ASSISTS", "GK_SAVES"):
        items = family_rows.get(family, [])
        fixtures = {int(row["fixture_id"]) for row in items if row.get("fixture_id") is not None}
        prekickoff_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None and bool(row.get("pre_kickoff"))
        }
        confirmed_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None
            and bool(row.get("confirmed_xi_before_market"))
            and bool(row.get("pre_kickoff"))
        }
        provider_fixtures = {
            int(row["fixture_id"])
            for row in items
            if row.get("fixture_id") is not None and row.get("provider_update") is not None
        }
        stages = Counter(str(row.get("stage") or "UNKNOWN") for row in items)
        bookmakers = {str(row.get("bookmaker")) for row in items if row.get("bookmaker")}
        line_values = sum(int(row.get("line_value_count") or 0) for row in items)
        priced_values = sum(int(row.get("priced_value_count") or 0) for row in items)

        summaries[family] = {
            "market_snapshot_rows": len(items),
            "unique_fixtures": len(fixtures),
            "pre_kickoff_unique_fixtures": len(prekickoff_fixtures),
            "provider_update_unique_fixtures": len(provider_fixtures),
            "confirmed_xi_pre_kickoff_unique_fixtures": len(confirmed_fixtures),
            "bookmaker_count": len(bookmakers),
            "value_rows": sum(int(row.get("value_count") or 0) for row in items),
            "priced_value_rows": priced_values,
            "exact_line_value_rows": line_values,
            "stages": dict(sorted(stages.items())),
            "exact_observed_market_history_materialized": bool(items),
            "confirmed_xi_overlap_materialized": bool(confirmed_fixtures),
        }
        all_fixtures.update(fixtures)
        all_confirmed.update(confirmed_fixtures)
        all_rows += len(items)
        all_line_values += line_values

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_DERIVATIVE_MARKET_AUDIT",
        "lookback_days": int(lookback_days),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "decision_weight": 0.0,
        "market_snapshot_rows": all_rows,
        "unique_fixtures": len(all_fixtures),
        "confirmed_xi_pre_kickoff_unique_fixtures": len(all_confirmed),
        "exact_line_value_rows": all_line_values,
        "families": summaries,
        "sample_rows": samples,
        "policy": (
            "POSTGRES_READ_ONLY; PREMATCH MARKET SNAPSHOTS ONLY FOR OOS READINESS; "
            "CONFIRMED_XI MUST EXIST AT_OR_BEFORE MARKET CAPTURE; NO PROVIDER CALLS; "
            "MARKET HISTORY DOES NOT IMPLY MODEL VALIDATION OR PRODUCTION PROMOTION"
        ),
    }


def build_from_postgres(*, lookback_days: int = 180, max_rows: int = 50000) -> dict[str, Any]:
    lookback_days = max(1, min(int(lookback_days), 730))
    max_rows = max(100, min(int(max_rows), 200000))
    if not persistence_base.persistence_configured():
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "POSTGRES_NOT_CONFIGURED",
            "lookback_days": lookback_days,
            "provider_requests_added": 0,
            "production_promotion_allowed": False,
            "families": {},
            "sample_rows": [],
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
                    END AS pre_kickoff,
                    EXISTS (
                        SELECT 1
                        FROM soccer_lineup_snapshots l
                        WHERE l.fixture_id = m.fixture_id
                          AND l.captured_at <= m.captured_at
                          AND l.both_xi_confirmed IS TRUE
                    ) AS confirmed_xi_before_market
                FROM soccer_market_snapshots m
                LEFT JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
                WHERE m.captured_at >= NOW() - (%s * INTERVAL '1 day')
                  AND (
                    LOWER(COALESCE(m.market, '')) LIKE '%%card%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%booking%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%shot%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%goalkeeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%keeper save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%gk save%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%goalscorer%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%scorer%%'
                    OR LOWER(COALESCE(m.market, '')) LIKE '%%assist%%'
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
