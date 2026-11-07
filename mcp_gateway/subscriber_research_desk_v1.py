"""Owner-only research price bridge. Never turns WATCH into a canonical BET."""
from __future__ import annotations

import asyncio
import os
import re
import httpx
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any
from starlette.requests import Request
from starlette.responses import JSONResponse
from mcp_gateway import persistence, subscriber_contract_v2
from mcp_gateway import player_prop_sport_first_bridge_v1 as player_sport_bridge

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
        if (row.get("quote_freshness") not in {"CURRENT_QUOTE", "NEEDS_RECONFIRMATION"}
            or row.get("classification") != "WATCH"
            or row.get("bet_eligible") is not False
            or row.get("bookmaker_id") != 8):
            continue
        offer_type = row.get("offer_type")
        if offer_type not in {"BOOK_QUOTED_COMPOSITE", "PLAYER_COMPONENT_QUOTE"}:
            continue
        is_direct = offer_type == "BOOK_QUOTED_COMPOSITE"
        price = _number(row.get("offered_price_decimal"))
        if price is None or price <= 1:
            continue
        # Existing composite catalog may use its own research price floor.
        # Player quotes have NO price-based selection threshold; a price
        # is neither a player statistic nor a sport-first model projection.
        if is_direct and price < 21/11:
            continue
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
            "quote_freshness": row.get("quote_freshness"),
            "quote_reconfirmation_required": True,
            "source_quote_fresh": row.get("quote_freshness") == "CURRENT_QUOTE",
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
            "research_flags": list(row.get("research_flags") or []) +
                 (["PRICE_SOURCE_OUTDATED_RECONFIRM_BEFORE_ACTION"]
                  if row.get("quote_freshness") != "CURRENT_QUOTE" else []),
            "quote_source": row.get("research_source") or "SOCCER_PROSPECTIVE_PRIVATE",
            "price_meaning": ("BOOK_QUOTED_SINGLE_COMPOSITE_MARKET" if is_direct
                              else "INDIVIDUAL_PLAYER_LEG_NOT_SGP_TICKET"),
        }
        (combos if is_direct else players).append(item)
    # An available bookmaker market is *not* a model candidate.
    # All market prices remain accessible only in a separate, explicitly
    # non-recommendation catalog. Never display opposite outcomes as picks.
    combos.sort(key=lambda x: (
        not x["source_quote_fresh"],
        -(x["raw_hypothetical_ev_pct"] if x["raw_hypothetical_ev_pct"] is not None else -999),
        abs(x["price_decimal"]-2.10),x["fixture_id"]
    ))
    positive_raw = [
        item for item in combos
        if item["raw_joint_probability"] is not None
        and item["raw_hypothetical_ev_pct"] is not None
        and item["raw_hypothetical_ev_pct"] > 0
        and item["raw_joint_probability"] > 1/item["price_decimal"]
    ]
    # Conservative fixture-level portfolio suppression: max ONE hypothesis
    # per match, even when markets are not strictly complementary. This
    # prevents contradictory or correlated options being sold as a pick set.
    selected: list[dict[str, Any]] = []
    used_fixture_ids: set[int] = set()
    for item in positive_raw:
        fixture_id = item["fixture_id"]
        if fixture_id in used_fixture_ids:
            continue
        used_fixture_ids.add(fixture_id)
        selected.append({
            **item,
            "signal_scope": "ONE_UNVALIDATED_RAW_HYPOTHESIS_PER_FIXTURE",
            "signal_label": "RAW_EXPERIMENTAL_NOT_BET",
        })
    chosen_keys = {
        (x["fixture_id"], x["provider_market_id"], x["selection"])
        for x in selected
    }
    market_catalog: list[dict[str, Any]] = []
    for item in combos[:50]:
        key = (item["fixture_id"], item["provider_market_id"], item["selection"])
        ev = item["raw_hypothetical_ev_pct"]
        if key in chosen_keys:
            reason = "RAW_EXPERIMENTAL_SIGNAL_SHOWN_SEPARATELY"
        elif ev is None or item["raw_joint_probability"] is None:
            reason = "NO_JOINT_PROBABILITY_MODEL"
        elif ev <= 0:
            reason = "NON_POSITIVE_RAW_HYPOTHETICAL_EV"
        else:
            reason = "SUPPRESSED_ALTERNATIVE_SAME_FIXTURE"
        market_catalog.append({
            **item,
            "catalog_reason": reason,
            "catalog_only": True,
            "not_a_pick": True,
        })
    # HARD SPORT-FIRST GATE:
    # A bookmaker quote alone (even when fresh) never identifies a genuine
    # player prop candidate. These source rows contain no canonical player ID,
    # verified player-level historical sample, opponent-adjusted rate,
    # projected minutes/XI, exact settlement operator or calibrated model P.
    # Do not sort by closeness to decimal 2.10 (+110), display these as
    # statistical picks, or use the bookmaker price to infer probabilities.
    # Preserve ONLY aggregate audit counts until an independent canonical
    # player sport model supplies, verifies and joins all required evidence.
    verified_player_candidates: list[dict[str, Any]] = []
    return {
        "sgp_composites": selected[:20],
        "sgp_market_catalog": market_catalog,
        "sgp_catalog_count": len(combos),
        "sgp_composites_suppressed": max(0,len(combos)-len(selected)),
        "sgp_player_components": verified_player_candidates,
        "sgp_player_quote_catalog_count": len(players),
        "sgp_player_statistically_validated_count": 0,
        "sgp_player_selection_policy": "SPORT_FIRST_STATS_AND_OOS_BEFORE_MARKET",
        "sgp_player_selection_status": "BLOCKED_NO_VERIFIED_PLAYER_SPORT_MODEL",
        "sgp_player_quote_price_used_for_ranking": False,
        "sgp_player_prices_are_picks": False,
        "sgp_player_verification_needed": [
            "CANONICAL_PLAYER_ID_AND_VERIFIED_RECENT_SAMPLE",
            "PLAYER_SHOTS_OR_SOT_RATE_AND_EXPECTED_MINUTES",
            "OPPONENT_ADJUSTED_PREMATCH_PROJECTION",
            "CONFIRMED_STARTER_XI_AVAILABILITY",
            "EXACT_MARKET_OPERATOR_AND_SETTLEMENT",
            "INDEPENDENT_OOS_PROBABILITY_VALIDATION",
            "FRESH_BOOK_PRICE_FOR_VALUE_COMPARISON_ONLY",
        ],
    }



RENDER_CACHED_SGP_SQL = """
WITH latest AS (
  SELECT DISTINCT ON (m.fixture_id, m.market_id)
    m.fixture_id, f.league AS league_name, f.home_team, f.away_team,
    f.kickoff AS kickoff_utc, m.bookmaker_id, m.bookmaker,
    m.market_id AS provider_market_id, m.market AS provider_market,
    m.values, m.provider_update AS provider_updated_at,
    m.captured_at
  FROM soccer_market_snapshots m
  JOIN soccer_fixtures f ON f.fixture_id = m.fixture_id
  WHERE m.bookmaker_id = 8
    AND m.market_id IN (7, 24, 25, 49, 92, 240, 241, 269, 270)
    AND f.status = 'NS'
    AND f.kickoff > now() AND f.kickoff < now() + interval '36 hours'
    AND m.captured_at >= now() - interval '12 hours'
    AND m.provider_update >= now() - interval '12 hours'
    AND m.provider_update <= now() + interval '5 minutes'
  ORDER BY m.fixture_id, m.market_id, m.captured_at DESC, m.snapshot_id DESC
)
SELECT * FROM latest ORDER BY captured_at DESC LIMIT 80
"""

_RENDER_COMPOSITE_MARKETS = {7, 24, 25, 49}
_RENDER_PLAYER_MARKETS = {92, 240, 241, 269, 270}


def _render_cache_sgp_rows(market_rows: list[dict[str, Any]],
                           *, now: datetime | None = None) -> list[dict[str, Any]]:
    """Convert cached Bet365 ID8 sportsbook rows into research-only SGP evidence.

    This reads EXISTING canonical sport-shortlist price snapshots. It never
    calls API-Football, manufactures a multi-player joint price, or certifies EV.
    """
    now = now or datetime.now(timezone.utc)
    out: list[dict[str, Any]] = []
    for market in market_rows:
        try:
            market_id = int(market.get("provider_market_id"))
            bookmaker_id = int(market.get("bookmaker_id"))
        except (TypeError, ValueError):
            continue
        if bookmaker_id != 8 or market_id not in _RENDER_COMPOSITE_MARKETS | _RENDER_PLAYER_MARKETS:
            continue
        kickoff = _as_utc_datetime(market.get("kickoff_utc"))
        updated = _as_utc_datetime(market.get("provider_updated_at"))
        captured = _as_utc_datetime(market.get("captured_at"))
        if not kickoff or not updated or not captured:
            continue
        if not (now < kickoff < now + timedelta(hours=36)):
            continue
        if not (now-timedelta(hours=12) <= updated <= now+timedelta(minutes=5)):
            continue
        if not (now-timedelta(hours=12) <= captured <= now+timedelta(minutes=5)):
            continue
        freshness = ("CURRENT_QUOTE" if updated>=now-timedelta(minutes=90)
                     and captured>=now-timedelta(minutes=60)
                     else "NEEDS_RECONFIRMATION")
        is_direct = market_id in _RENDER_COMPOSITE_MARKETS
        source_values = market.get("values")
        if not isinstance(source_values, list):
            continue
        count = 0
        for item in source_values:
            if not isinstance(item, dict):
                continue
            price = _number(item.get("decimal_price", item.get("odd")))
            if price is None or price <= 1:
                continue
            if is_direct and price < 21/11:
                continue
            selection = str(item.get("raw_selection") or item.get("selection") or "").strip()
            if not selection:
                continue
            # Player quote inventory is not a shortlist: no filtering by
            # +110, -110 or any other price. Sports model selection is
            # independent and happens before optional exact source matching.
            player = (re.sub(r"\s*-\s*\d+(?:\.\d+)?\s*$","",selection).strip()
                      if not is_direct else None)
            side = ("HOME" if market_id in {240,269} else
                    "AWAY" if market_id in {241,270} else None)
            out.append({
                "fixture_id": market.get("fixture_id"),
                "home_team": market.get("home_team"),
                "away_team": market.get("away_team"),
                "league_name": market.get("league_name"),
                "kickoff_utc": kickoff,
                "bookmaker_id": 8,
                "provider_market_id": market_id,
                "provider_market": market.get("provider_market"),
                "selection": selection,
                "offered_price_decimal": price,
                "provider_updated_at": updated,
                "captured_at": captured,
                "offer_type": ("BOOK_QUOTED_COMPOSITE" if is_direct else "PLAYER_COMPONENT_QUOTE"),
                "player_label": player or None,
                "observed_team_side": side,
                # Only IDs derived from an official confirmed XI can attach
                # to frozen player-sport model snapshots. Never infer an ID
                # from name resemblance or a bookmaker decimal price.
                "player_id": item.get("player_id"),
                "team_id": item.get("team_id"),
                "xi_alignment_status": item.get("xi_alignment_status"),
                "line_basis": item.get("line_basis"),
                "parsed_line": item.get("parsed_line"),
                "player_market_family": (
                    "SHOTS" if market_id in (240,241)
                    and "shot" in str(market.get("provider_market") or "").lower()
                    and "target" not in str(market.get("provider_market") or "").lower()
                    else "SOT" if market_id in (269,270)
                    and "target" in str(market.get("provider_market") or "").lower()
                    else None
                ),
                "raw_joint_probability": None,
                "raw_hypothetical_ev_pct": None,
                "quote_freshness": freshness,
                "classification": "WATCH",
                "bet_eligible": False,
                "bookmaker_direct_combo_quote": is_direct,
                "research_flags": [
                    "RENDER_SNAPSHOT_RESEARCH_ONLY", "NO_PROVEN_CORRELATED_JOINT_MODEL",
                    "BOOKMAKER_OFFER_RECONFIRMATION_REQUIRED", "XI_AND_PLAYER_MINUTES_NOT_VERIFIED",
                ],
                "research_source": "RENDER_POSTGRES_PERSISTED_API_FOOTBALL_ID8",
            })
            count += 1
            if count >= 16:
                break
    return out


def _as_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z","+00:00"))
        except ValueError:
            return None
    if isinstance(value, datetime) and value.tzinfo is not None:
        return value.astimezone(timezone.utc)
    return None


def _query_render_cached_sgp() -> list[dict[str, Any]]:
    if not persistence.persistence_configured():
        return []
    with persistence._connect() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(RENDER_CACHED_SGP_SQL)
                fields = [col.name for col in cur.description]
                rows = [dict(zip(fields,row)) for row in cur.fetchall()]
    return _render_cache_sgp_rows(rows)


def _dedup_sgp_quotes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the freshest matching source quote without hiding provenance."""
    order = sorted(rows, key=lambda r: (
        r.get("quote_freshness") == "CURRENT_QUOTE",
        _as_utc_datetime(r.get("captured_at")) or datetime.min.replace(tzinfo=timezone.utc),
    ), reverse=True)
    seen: set[tuple[Any, ...]] = set()
    out = []
    for row in order:
        key = (row.get("fixture_id"),row.get("bookmaker_id"),
               row.get("provider_market_id"),row.get("selection"),
               _time(row.get("provider_updated_at")) or str(row.get("provider_updated_at")))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


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
        "schema_version": "1.2.0",
        "status": ("RESEARCH_ONLY_FRESH_QUOTE_PREVIEW" if result or sgp["sgp_composites"]
                   else "PLAYER_QUOTES_ONLY_NO_VERIFIED_SPORTING_EDGE"
                   if sgp["sgp_player_quote_catalog_count"] else "NO_FRESH_RESEARCH_QUOTES"),
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
    # PostgreSQL is the canonical scanner cache; Supabase is the isolated pilot.
    # A missing cache must not cause an otherwise functioning owner preview to fail.
    try:
        cached = await asyncio.to_thread(_query_render_cached_sgp)
    except Exception:
        cached = []
    all_sgp = _dedup_sgp_quotes(feed["sgp_rows"] + cached)
    board = make_board(feed["rows"], all_sgp)
    board["render_cached_sgp_sources"] = len(cached)
    # Bounded, read-only: canonical frozen pregame player-model snapshots are
    # selected by fixture kickoff/stage, never by quote price. Matched offers
    # require official player ID + team ID + exact operator/line + as-of time.
    try:
        board["player_sport_research"] = await asyncio.to_thread(
            player_sport_bridge.build_live_research, all_sgp
        )
    except Exception:
        board["player_sport_research"] = {
            "status": "NOT_VERIFIED_SPORT_MODEL_SOURCE_UNAVAILABLE",
            "sport": {"rows": [], "verified_pregame_sport_models": 0},
            "market_alignment": {"raw_model_exact_matches": 0,
                                  "unverified_offer_count": len(all_sgp)},
            "oos": {"status": "NOT_VERIFIED_CURRENT_OOS_REPORT"},
            "bet_eligible": False, "production_promotion_allowed": False,
            "provider_requests_added": 0,
        }
    return JSONResponse(board, headers={"Cache-Control": "no-store"})
