"""Owner-only research price bridge. Never turns WATCH into a canonical BET."""
from __future__ import annotations

import asyncio
import os
import httpx
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

def _query_research_feed() -> dict[str, Any]:
    """Prefer restricted owner-only Supabase bridge, never browser credentials.

    The canonical subscriber pipeline remains on Render Postgres. The separate
    prospective *research* schema lives in a dedicated Supabase project.
    A read-only bearer capability authenticates only this internal server hop.
    """
    bridge_url = os.getenv("SOCCER_RESEARCH_BRIDGE_URL", "").strip()
    bridge_token = os.getenv("SOCCER_RESEARCH_BRIDGE_TOKEN", "").strip()
    if bridge_url or bridge_token:
        allowed = ("https://pqoobjacewxigngzhbhp.supabase.co/"
                   "functions/v1/soccer-owner-price-bridge")
        if bridge_url != allowed or len(bridge_token) != 96:
            raise RuntimeError("Research bridge configuration invalid")
        response = httpx.get(
            bridge_url,
            headers={"x-soccer-bridge-token": bridge_token, "Accept":"application/json"},
            timeout=8.0,
            follow_redirects=False,
        )
        response.raise_for_status()
        payload = response.json()
        if (not isinstance(payload, dict)
            or payload.get("status") != "CONNECTED_RESEARCH_READ_ONLY"
            or not isinstance(payload.get("rows"), list)
            or not isinstance(payload.get("sgp_rows", []), list)
            or payload.get("production_bets_authorized") is not False):
            raise RuntimeError("Research bridge returned invalid provenance")
        return {"rows": payload["rows"], "sgp_rows": payload.get("sgp_rows", [])}

    # Legacy development fallback only; never infer that Render Postgres
    # contains the private Supabase schema from a successful Render deployment.
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(SQL)
                fields = [col.name for col in cur.description]
                return {"rows": [dict(zip(fields, row)) for row in cur.fetchall()],
                        "sgp_rows": []}


def _query_rows() -> list[dict[str, Any]]:
    """Backward compatible for private connectivity health and offline tests."""
    return _query_research_feed()["rows"]


def _sgp_market_rows(sgp_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Book-offered combo != custom correlated player bet builder ticket."""
    combos: list[dict[str, Any]] = []
    players: list[dict[str, Any]] = []
    for row in sgp_rows:
        if (row.get("quote_freshness") != "CURRENT_QUOTE"
            or row.get("classification") != "WATCH"
            or row.get("bet_eligible") is not False
            or row.get("bookmaker_id") != 8):
            continue
        price = _number(row.get("offered_price_decimal"))
        if price is None or price < 21/11:
            continue
        offer_type = row.get("offer_type")
        if offer_type not in {"BOOK_QUOTED_COMPOSITE", "PLAYER_COMPONENT_QUOTE"}:
            continue
        is_direct = offer_type == "BOOK_QUOTED_COMPOSITE"
        if is_direct and row.get("bookmaker_direct_combo_quote") is not True:
            continue
        if not is_direct and row.get("bookmaker_direct_combo_quote") is not False:
            continue
        item = {
            "fixture_id": row["fixture_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "league": row["league_name"],
            "kickoff_utc": _time(row.get("kickoff_utc")),
            "bookmaker": "Bet365",
            "bookmaker_id": 8,
            "provider_market": row["provider_market"],
            "provider_market_id": row["provider_market_id"],
            "selection": row["selection"],
            "price_decimal": price,
            "quote_updated_at": _time(row.get("provider_updated_at")),
            "captured_at": _time(row.get("captured_at")),
            "offer_type": offer_type,
            "price_bucket": ("PLUS110_OR_BETTER" if price >= 2.10
                             else "MINUS110_TO_PLUS110"),
            "player_label": row.get("player_label"),
            "observed_team_side": row.get("observed_team_side"),
            "raw_joint_probability": _number(row.get("raw_joint_probability")) if is_direct else None,
            "raw_hypothetical_ev_pct": _number(row.get("raw_hypothetical_ev_pct")) if is_direct else None,
            "joint_probability_validated": False,
            "model_validated": False,
            "sgp_betslip_price_verified": False,
            "classification": "WATCH",
            "bet_eligible": False,
            "actionable": False,
            "research_flags": list(row.get("research_flags") or []),
            "price_meaning": ("BOOK_QUOTED_SINGLE_COMPOSITE_MARKET" if is_direct
                              else "INDIVIDUAL_PLAYER_LEG_NOT_SGP_TICKET"),
        }
        (combos if is_direct else players).append(item)
    combos.sort(key=lambda x: (
        -(x["raw_hypothetical_ev_pct"] if x["raw_hypothetical_ev_pct"] is not None else -999),
        abs(x["price_decimal"]-2.10),x["fixture_id"]
    ))
    players.sort(key=lambda x: (abs(x["price_decimal"]-2.10), x["fixture_id"],str(x["selection"])))
    return {"sgp_composites": combos[:20], "sgp_player_components": players[:50]}


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        n = float(value)
        return n if n == n and abs(n) != float("inf") else None
    except (TypeError, ValueError, OverflowError):
        return None

def _time(value: Any) -> str | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc).isoformat()
    return None

def make_board(rows: list[dict[str, Any]], sgp_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
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
    sgp = _sgp_market_rows(sgp_rows or [])
    return {
        "schema_version": "1.1.0",
        "status": ("RESEARCH_ONLY_FRESH_QUOTE_PREVIEW" if result or sgp["sgp_composites"] or sgp["sgp_player_components"]
                   else "NO_FRESH_RESEARCH_QUOTES"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_use": "OWNER_ONLY_PROSPECTIVE_RESEARCH",
        "bets": 0,
        "notification_delivery_enabled": False,
        "canonical_bet_logic_changed": False,
        "prices_reconfirmation_required": True,
        "provider_requests_added": 0,
        "rows": result,
        **sgp,
        "sgp_custom_player_ticket_price_verified": False,
        "sgp_player_builder_enabled": False,
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
        feed = await asyncio.to_thread(_query_research_feed)
    except Exception:
        return JSONResponse({"error": "RESEARCH_SOURCE_UNAVAILABLE", "status": "NOT_VERIFIED"},
                            status_code=503, headers={"Cache-Control": "no-store"})
    return JSONResponse(make_board(feed["rows"], feed["sgp_rows"]), headers={"Cache-Control": "no-store"})
