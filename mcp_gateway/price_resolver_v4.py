from __future__ import annotations

import asyncio
import json
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from mcp_gateway import persistence

MODEL_VERSION = "SOCCER_PRICE_RESOLVER_V4_1.0.0"
API_BASE_URL = os.getenv("API_BASE_URL", "https://v3.football.api-sports.io").rstrip("/")
DEFAULT_MAX_API_CALLS = int(os.getenv("SOCCER_PRICE_RESOLVER_MAX_API_CALLS", "25"))
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("SOCCER_PRICE_RESOLVER_TIMEOUT_SECONDS", "12"))

FRESHNESS_MINUTES = {
    "EARLY_RESEARCH": 180,
    "T-90": 60,
    "T-60": 45,
    "T-40": 30,
    "T-30": 20,
    "T-20": 15,
    "T-10": 10,
    "CLOSE": 5,
}

ELIGIBLE_STATUSES = {"WAIT_PRICE", "WAIT_FRESH_QUOTE", "STALE_QUOTE"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _parse_value(value: Any) -> tuple[str, float | None]:
    text = " ".join(str(value or "").strip().split())
    low = text.lower()
    for prefix in ("over ", "under "):
        if low.startswith(prefix):
            try:
                return text.split()[0].title(), float(text.split()[-1])
            except (ValueError, IndexError):
                return text.split()[0].title(), None
    return text, None


def _fair_probs(prices: list[float]) -> list[float | None]:
    implied = [(1.0 / p) if p and p > 1.0 else None for p in prices]
    total = sum(p for p in implied if p is not None)
    if total <= 0:
        return [None for _ in prices]
    return [(p / total) if p is not None else None for p in implied]


def normalize_api_response(payload: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for response_row in payload.get("response") or []:
        fixture = response_row.get("fixture") or {}
        fixture_id = fixture.get("id")
        update = response_row.get("update")
        for bookmaker in response_row.get("bookmakers") or []:
            bookmaker_id = bookmaker.get("id")
            bookmaker_name = bookmaker.get("name")
            for bet in bookmaker.get("bets") or []:
                market_id = bet.get("id")
                market_name = str(bet.get("name") or "")
                values = bet.get("values") or []
                parsed: list[dict[str, Any]] = []
                for item in values:
                    selection, line = _parse_value(item.get("value"))
                    price = _num(item.get("odd"))
                    if price is None or price <= 1.0:
                        continue
                    parsed.append({
                        "selection": selection,
                        "line": line,
                        "decimal_price": price,
                    })

                # De-vig complete mutually-exclusive groups only.
                if market_name.lower() in {"match winner", "both teams score", "both teams to score"}:
                    fairs = _fair_probs([float(v["decimal_price"]) for v in parsed])
                    for value, fair in zip(parsed, fairs):
                        value["fair_probability"] = fair
                elif "goals over/under" in market_name.lower() or market_name.lower() in {"goals over/under", "over/under"}:
                    by_line: dict[float, list[dict[str, Any]]] = defaultdict(list)
                    for value in parsed:
                        if value.get("line") is not None:
                            by_line[float(value["line"])].append(value)
                    for _, group in by_line.items():
                        if len(group) < 2:
                            continue
                        fairs = _fair_probs([float(v["decimal_price"]) for v in group])
                        for value, fair in zip(group, fairs):
                            value["fair_probability"] = fair

                if parsed:
                    normalized.append({
                        "fixture_id": fixture_id,
                        "bookmaker_id": bookmaker_id,
                        "bookmaker": bookmaker_name,
                        "market_id": market_id,
                        "market": market_name,
                        "values": parsed,
                        "provider_update": update,
                        "source": "API_FOOTBALL_ODDS_V3",
                    })
    return normalized


def _market_kind(market: str) -> str | None:
    name = _norm(market)
    if name == "match winner":
        return "1X2"
    if name in {"both teams score", "both teams to score"} or "both teams" in name:
        return "BTTS"
    if "goals over/under" in name or name == "over/under":
        return "FT_TOTALS"
    return None


def _event_projection(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("raw_projection")
    return raw if isinstance(raw, dict) else {}


def _desired_offer(row: dict[str, Any], event: dict[str, Any]) -> tuple[str | None, str | None, float | None, float | None]:
    family = str(row.get("market_family") or "").upper()
    selection = _norm(row.get("selection"))
    raw = _event_projection(event)

    if family in {"FT_TOTALS_RESEARCH", "FT_TOTALS", "TOTAL"}:
        side = "Over" if "over" in selection else "Under" if "under" in selection else None
        if side is None:
            return None, None, None, None
        p_over = _num(raw.get("raw_over_2_5_prob"))
        p_raw = p_over if side == "Over" else (1.0 - p_over if p_over is not None else None)
        return "FT_TOTALS", side, 2.5, p_raw

    if family in {"FT_BTTS_RESEARCH", "BTTS", "FT_BTTS"}:
        p_raw = _num(raw.get("raw_btts_yes_prob"))
        return "BTTS", "Yes", None, p_raw

    if family in {"FT_1X2_RESEARCH", "1X2", "FT_1X2"}:
        home = _num(raw.get("raw_home_win_prob"))
        away = _num(raw.get("raw_away_win_prob"))
        if home is None and away is None:
            return None, None, None, None
        if (home or 0.0) >= (away or 0.0):
            return "1X2", "Home", None, home
        return "1X2", "Away", None, away

    return None, None, None, None


def _selection_matches(value: dict[str, Any], desired_selection: str, desired_line: float | None) -> bool:
    selection = _norm(value.get("selection"))
    wanted = _norm(desired_selection)
    aliases = {
        "home": {"home", "1"},
        "away": {"away", "2"},
        "draw": {"draw", "x"},
        "yes": {"yes"},
        "no": {"no"},
        "over": {"over"},
        "under": {"under"},
    }
    if selection not in aliases.get(wanted, {wanted}):
        return False
    if desired_line is None:
        return True
    line = _num(value.get("line"))
    return line is not None and abs(line - desired_line) < 1e-9


def choose_reference_offer(
    markets: list[dict[str, Any]],
    *,
    family: str,
    selection: str,
    line: float | None,
) -> dict[str, Any] | None:
    offers: list[dict[str, Any]] = []
    for market in markets:
        if _market_kind(str(market.get("market") or "")) != family:
            continue
        for value in market.get("values") or []:
            if not _selection_matches(value, selection, line):
                continue
            price = _num(value.get("decimal_price"))
            fair = _num(value.get("fair_probability"))
            if price is None or price <= 1.0 or fair is None:
                continue
            offers.append({
                "fixture_id": market.get("fixture_id"),
                "bookmaker_id": market.get("bookmaker_id"),
                "bookmaker": market.get("bookmaker"),
                "market_id": market.get("market_id"),
                "market": market.get("market"),
                "selection": selection,
                "line": line,
                "decimal_price": price,
                "fair_probability": fair,
                "provider_update": market.get("provider_update"),
                "source": market.get("source") or "API_FOOTBALL_ODDS_V3",
            })
    if not offers:
        return None

    median_price = statistics.median(float(o["decimal_price"]) for o in offers)
    offers.sort(key=lambda o: (abs(float(o["decimal_price"]) - median_price), str(o.get("bookmaker") or "")))
    chosen = dict(offers[0])
    chosen["bookmaker_count"] = len(offers)
    chosen["reference_policy"] = "MEDIAN_PRICE_NEAREST_BOOKMAKER"
    return chosen


def _freshness_minutes(stage: Any) -> int:
    return int(FRESHNESS_MINUTES.get(str(stage or "").upper(), 30))


def _load_cached_markets(fixture_id: int, stage: Any) -> list[dict[str, Any]]:
    if not persistence.persistence_configured():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_freshness_minutes(stage))
    persistence.ensure_schema()
    with persistence._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fixture_id, bookmaker_id, bookmaker, market_id, market, values, provider_update
                FROM soccer_market_snapshots
                WHERE fixture_id = %s
                  AND captured_at >= %s
                ORDER BY captured_at DESC, snapshot_id DESC
                """,
                (fixture_id, cutoff),
            )
            rows = cur.fetchall()
            columns = [desc.name for desc in cur.description]
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(zip(columns, raw))
        provider_update = row.get("provider_update")
        if isinstance(provider_update, datetime):
            row["provider_update"] = provider_update.isoformat()
        key = (row.get("bookmaker_id"), row.get("market_id"), str(row.get("values")))
        if key in seen:
            continue
        seen.add(key)
        row["source"] = "POSTGRES_MARKET_SNAPSHOT_CACHE"
        out.append(row)
    return out


def _header_int(response: httpx.Response, name: str) -> int | None:
    value = response.headers.get(name)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _apply_quota_accounting(payload: dict[str, Any], calls: int, daily_remaining: int | None) -> None:
    payload["api_calls_this_tick"] = int(payload.get("api_calls_this_tick") or 0) + int(calls or 0)
    if daily_remaining is not None:
        current = payload.get("last_daily_remaining")
        try:
            current_int = int(current) if current is not None else None
        except (TypeError, ValueError):
            current_int = None
        payload["last_daily_remaining"] = daily_remaining if current_int is None else min(current_int, daily_remaining)
        quota = payload.get("quota")
        if isinstance(quota, dict):
            quota["daily_remaining"] = payload["last_daily_remaining"]


async def _fetch_fixture_odds(
    client: httpx.AsyncClient,
    fixture_id: int,
    *,
    api_key: str,
    remaining_calls: int,
) -> tuple[list[dict[str, Any]], int, str, int | None]:
    if remaining_calls <= 0:
        return [], 0, "PRICE_BUDGET_EXHAUSTED", None
    markets: list[dict[str, Any]] = []
    page = 1
    calls = 0
    daily_remaining: int | None = None
    while calls < remaining_calls:
        response = await client.get(
            f"{API_BASE_URL}/odds",
            params={"fixture": fixture_id, "page": page},
            headers={"x-apisports-key": api_key},
        )
        calls += 1
        response.raise_for_status()
        observed_remaining = _header_int(response, "x-ratelimit-requests-remaining")
        if observed_remaining is not None:
            daily_remaining = observed_remaining if daily_remaining is None else min(daily_remaining, observed_remaining)
        payload = response.json()
        markets.extend(normalize_api_response(payload))
        paging = payload.get("paging") or {}
        current = int(paging.get("current") or page)
        total = int(paging.get("total") or current)
        if current >= total:
            break
        page = current + 1
    return markets, calls, "PRICE_API_RESOLVED" if markets else "PRICE_API_NO_FIXTURE_OR_MARKET", daily_remaining


def _attach_market_to_event(event: dict[str, Any], markets: list[dict[str, Any]], source_status: str) -> None:
    if not markets:
        return
    event["market"] = {
        "source": "API_FOOTBALL_ODDS_V3" if source_status == "PRICE_API_RESOLVED" else "POSTGRES_MARKET_SNAPSHOT_CACHE",
        "resolution_status": source_status,
        "markets": markets,
    }


def _enrich_row(row: dict[str, Any], event: dict[str, Any], markets: list[dict[str, Any]], source_status: str) -> str:
    family, selection, line, p_raw = _desired_offer(row, event)
    if family is None or selection is None:
        row["price_resolution_status"] = "PRICE_API_NO_EXACT_MARKET_MAPPING"
        return "PRICE_API_NO_EXACT_MARKET_MAPPING"

    offer = choose_reference_offer(markets, family=family, selection=selection, line=line)
    if offer is None:
        family_markets = [market for market in markets if _market_kind(str(market.get("market") or "")) == family]
        available_selection_rows = [
            value
            for market in family_markets
            for value in (market.get("values") or [])
            if _selection_matches(value, selection, None)
        ]
        available_lines = sorted({
            float(value["line"])
            for value in available_selection_rows
            if _num(value.get("line")) is not None
        })
        if not family_markets:
            status = "PRICE_API_NO_MARKET"
        elif not available_selection_rows:
            status = "PRICE_API_NO_SELECTION"
        elif line is not None and available_lines:
            status = "PRICE_API_NO_EXACT_LINE"
        else:
            status = "PRICE_API_NO_MARKET"
        row["price_resolution_status"] = status
        row["price_resolution_family"] = family
        row["price_resolution_selection"] = selection
        row["price_resolution_line"] = line
        row["price_resolution_available_lines"] = available_lines[:20]
        return status

    row["market_family"] = family
    row["market"] = offer.get("market")
    row["selection"] = selection
    row["line"] = line
    row["price"] = round(float(offer["decimal_price"]), 3)
    row["bookmaker"] = offer.get("bookmaker")
    row["p_market_fair"] = round(float(offer["fair_probability"]), 6)
    row["p_raw"] = round(float(p_raw), 6) if p_raw is not None else None
    row["prob_edge_pp"] = round((float(p_raw) - float(offer["fair_probability"])) * 100.0, 4) if p_raw is not None else None
    row["price_resolution_status"] = source_status
    row["price_resolution_source"] = offer.get("source")
    row["price_resolution_provider_update"] = offer.get("provider_update")
    row["price_resolution_bookmaker_count"] = offer.get("bookmaker_count")
    row["price_resolution_reference_policy"] = offer.get("reference_policy")
    row["price_resolution_calibrated_probability_added"] = False
    return source_status


async def resolve_payload(payload: dict[str, Any], *, max_api_calls: int | None = None) -> dict[str, Any]:
    rows = payload.get("match_table_rows") if isinstance(payload.get("match_table_rows"), list) else []
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    api_key = os.getenv("API_FOOTBALL_KEY", "").strip()
    budget = max(0, int(DEFAULT_MAX_API_CALLS if max_api_calls is None else max_api_calls))
    calls = 0
    provider_daily_remaining: int | None = None
    counts: dict[str, int] = defaultdict(int)
    fixture_cache: dict[int, tuple[list[dict[str, Any]], str]] = {}

    targets = [
        row for row in rows
        if isinstance(row, dict)
        and str(row.get("execution_status") or "").upper() in ELIGIBLE_STATUSES
        and str(row.get("stage") or "").upper() != "POSTGAME"
        and row.get("fixture_id") is not None
    ]

    async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS, follow_redirects=True) as client:
        for row in targets:
            fixture_id = int(row["fixture_id"])
            event_index = row.get("row_index")
            event = events[event_index] if isinstance(event_index, int) and 0 <= event_index < len(events) and isinstance(events[event_index], dict) else {}

            if fixture_id not in fixture_cache:
                cached = await asyncio.to_thread(_load_cached_markets, fixture_id, row.get("stage"))
                if cached:
                    fixture_cache[fixture_id] = (cached, "PRICE_CACHE_HIT")
                elif not api_key:
                    fixture_cache[fixture_id] = ([], "PRICE_API_KEY_MISSING")
                elif calls >= budget:
                    fixture_cache[fixture_id] = ([], "PRICE_BUDGET_EXHAUSTED")
                else:
                    try:
                        markets, used, status, observed_remaining = await _fetch_fixture_odds(
                            client,
                            fixture_id,
                            api_key=api_key,
                            remaining_calls=budget - calls,
                        )
                        calls += used
                        if observed_remaining is not None:
                            provider_daily_remaining = (
                                observed_remaining
                                if provider_daily_remaining is None
                                else min(provider_daily_remaining, observed_remaining)
                            )
                        fixture_cache[fixture_id] = (markets, status)
                    except Exception as exc:
                        fixture_cache[fixture_id] = ([], "PRICE_API_ERROR")
                        row["price_resolution_error"] = str(exc)[:180]

            markets, status = fixture_cache[fixture_id]
            if markets:
                _attach_market_to_event(event, markets, status)
                resolved_status = _enrich_row(row, event, markets, status)
            else:
                row["price_resolution_status"] = status
                resolved_status = status
            counts[resolved_status] += 1

    _apply_quota_accounting(payload, calls, provider_daily_remaining)

    payload["price_resolution_v4"] = {
        "schema_version": "1.0.0",
        "model_version": MODEL_VERSION,
        "status": "ACTIVE_PRICE_RESOLVER",
        "candidate_rows": len(targets),
        "unique_candidate_fixtures": len({int(row["fixture_id"]) for row in targets}),
        "api_calls_added": calls,
        "max_api_calls": budget,
        "resolution_counts": dict(sorted(counts.items())),
        "provider": "API_FOOTBALL",
        "endpoint": "/odds?fixture=<id>",
        "catalog_policy": "/odds/bookmakers and /odds/bets are metadata catalogs; fixture /odds response is authoritative for prices.",
        "simulated_odds_allowed": False,
        "calibrated_probability_fabricated": False,
        "provider_requests_added": calls,
        "provider_daily_remaining_observed": provider_daily_remaining,
        "quota_accounting_included_in_api_calls_this_tick": True,
    }
    payload["price_resolution_provider_requests_added"] = calls
    return payload["price_resolution_v4"]
