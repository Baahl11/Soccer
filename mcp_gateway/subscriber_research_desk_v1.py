"""Owner-only research price bridge. Never turns WATCH into a canonical BET."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp_gateway import persistence, subscriber_contract_v2

SQL = """
SELECT fixture_id, home_team, away_team, league_name, kickoff_utc,
       market_family, selection, line, bookmaker_id, bookmaker_name,
       price_decimal, provider_update_at, captured_at, p_raw, p_shrunk,
       p_market_fair, p_breakeven, hypothetical_ev_pct,
       edge_vs_breakeven_pp, value_diagnostic, classification, reasons,
       independent_model_validated, lineup_verified, availability_confidence
FROM soccer_prospective.pro_price_desk_v1
WHERE slate_date = (CURRENT_TIMESTAMP AT TIME ZONE 'America/Mexico_City')::date
  AND kickoff_utc > CURRENT_TIMESTAMP
  AND fixture_status = 'NS'
  AND provider_update_at >= CURRENT_TIMESTAMP - INTERVAL '90 minutes'
  AND captured_at >= CURRENT_TIMESTAMP - INTERVAL '30 minutes'
  AND bookmaker_id = 8
  AND classification IN ('WATCH','LEAN','BET')
ORDER BY provider_update_at DESC, kickoff_utc, fixture_id
LIMIT 60
"""

def _query_rows() -> list[dict[str, Any]]:
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(SQL)
                fields = [col.name for col in cur.description]
                return [dict(zip(fields, row)) for row in cur.fetchall()]

def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(value)
        return n if n == n and abs(n) != float("inf") else None
    except (TypeError, ValueError, OverflowError):
        return None

def _time(value: Any) -> str | None:
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).isoformat()
    return None

def make_board(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Strictly research. A source 'BET' never becomes a new product BET."""
    result = []
    for row in rows:
        price = _number(row.get("price_decimal"))
        if price is None or price <= 1:
            continue
        result.append({
            "fixture_id": row["fixture_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "league": row["league_name"],
            "kickoff_utc": _time(row["kickoff_utc"]),
            "market_family": row["market_family"],
            "selection": row["selection"],
            "line": _number(row["line"]),
            "bookmaker": row["bookmaker_name"],
            "bookmaker_id": row["bookmaker_id"],
            "price_decimal": price,
            "quote_provider_updated_at": _time(row["provider_update_at"]),
            "quote_captured_at": _time(row["captured_at"]),
            "raw_sport_probability": _number(row["p_raw"]),
            "market_shrunk_probability": _number(row["p_shrunk"]),
            "fair_market_probability": _number(row["p_market_fair"]),
            "breakeven_probability": _number(row["p_breakeven"]),
            "hypothetical_ev_pct": _number(row["hypothetical_ev_pct"]),
            "edge_vs_breakeven_pp": _number(row["edge_vs_breakeven_pp"]),
            "value_diagnostic": row["value_diagnostic"],
            "classification": "WATCH",
            "source_classification": row["classification"],
            "price_bucket": ("EDGE_MINUS110_OR_BETTER"
                             if Decimal(str(price)) >= Decimal(21)/Decimal(11)
                             else "EDGE_REGULAR"),
            "lineup_verified": row["lineup_verified"] is True,
            "independent_model_validated": row["independent_model_validated"] is True,
            "availability_confidence": _number(row["availability_confidence"]),
            "blockers": list(row["reasons"] or []),
            "bet_eligible": False,
            "actionable": False,
        })
    return {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_FRESH_QUOTE_PREVIEW" if result else "NO_FRESH_RESEARCH_QUOTES",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_use": "OWNER_ONLY_PROSPECTIVE_RESEARCH",
        "bets": 0,
        "notification_delivery_enabled": False,
        "canonical_bet_logic_changed": False,
        "prices_reconfirmation_required": True,
        "provider_requests_added": 0,
        "rows": result,
    }

async def priced_research(request: Request) -> JSONResponse:
    entitlement, error = await subscriber_contract_v2._resolve_entitlement(request)
    if error is not None:
        return error
    access = subscriber_contract_v2._access(entitlement or {})
    if not access.get("authenticated") or not (access.get("owner") or access.get("admin")):
        return JSONResponse({"error": "OWNER_ONLY_RESEARCH"}, status_code=403,
                            headers={"Cache-Control": "no-store"})
    try:
        rows = await asyncio.to_thread(_query_rows)
    except Exception:
        return JSONResponse({"error": "RESEARCH_SOURCE_UNAVAILABLE", "status": "NOT_VERIFIED"},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return JSONResponse(make_board(rows), headers={"Cache-Control": "no-store"})
