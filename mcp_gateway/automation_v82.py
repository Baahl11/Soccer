from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v37 as v37
from mcp_gateway import automation_v81 as v81
from mcp_gateway import galaxy_builder_v4 as v4

MODEL_VERSION = v81.MODEL_VERSION
AUTOMATION_VERSION = "3.58.0"

PRIMARY_BOOKMAKER = os.getenv("SOCCER_EDGE_PRIMARY_BOOKMAKER", "Bet365").strip() or "Bet365"
BOOKMAKER_CACHE_TTL = timedelta(hours=24)
BOOKMAKER_CACHE_NAMESPACE = "odds_reference"
BOOKMAKER_CACHE_KEY = "prematch_bookmakers_v1"

_ORIGINAL_COMMON_BOOK_REFERENCE = v4._common_book_reference
_ORIGINAL_ROLLING_COMMON_BOOK = v37._common_book


def _norm_book(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


def _is_primary(value: Any) -> bool:
    return _norm_book(value) == _norm_book(PRIMARY_BOOKMAKER)


def _generated_at(payload: dict[str, Any]) -> datetime:
    raw = payload.get("generated_at_utc")
    try:
        now = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        now = datetime.now(dt_timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt_timezone.utc)
    return now.astimezone(dt_timezone.utc)


def _price_maps_for_legs(legs: list[dict[str, Any]]) -> tuple[list[dict[str, dict[str, Any]]], set[str]]:
    by_leg: list[dict[str, dict[str, Any]]] = []
    common: set[str] | None = None
    for leg in legs:
        prices: dict[str, dict[str, Any]] = {}
        for quote in v4._verified_quotes(leg):
            bookmaker = str(quote.get("bookmaker") or "").strip()
            price = v4._num(quote.get("price"))
            if not bookmaker or price is None or price <= 1:
                continue
            current = prices.get(bookmaker)
            if current is None or price > float(current.get("price") or 0.0):
                prices[bookmaker] = dict(quote)
        if not prices:
            return [], set()
        by_leg.append(prices)
        common = set(prices) if common is None else common & set(prices)
    return by_leg, common or set()


def _book_product(book: str, by_leg: list[dict[str, dict[str, Any]]]) -> float:
    return math.prod(float(prices[book]["price"]) for prices in by_leg)


def _primary_common_book_reference(legs: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_leg, common = _price_maps_for_legs(legs)
    if not by_leg or not common:
        return None

    products = {book: _book_product(book, by_leg) for book in common}
    primary_books = [book for book in common if _is_primary(book)]
    best_book = max(common, key=lambda book: products[book])
    best_decimal = products[best_book]

    # Prefer Bet365 when it clears the builder's minimum component-product target.
    # If it does not, preserve the best viable fallback so a real edge is not hidden.
    primary_book = max(primary_books, key=lambda book: products[book]) if primary_books else None
    primary_decimal = products.get(primary_book) if primary_book else None
    primary_gate = bool(primary_book and primary_decimal is not None and primary_decimal >= v4.TARGET_DECIMAL)
    chosen = primary_book if primary_gate else best_book
    chosen_decimal = products[chosen]

    components = [prices[chosen] for prices in by_leg]
    best_components = [prices[best_book] for prices in by_leg]
    primary_components = [prices[primary_book] for prices in by_leg] if primary_book else []

    return {
        "bookmaker": chosen,
        "reference_bookmaker_policy": "BET365_PRIMARY_THEN_BEST_VERIFIED_FALLBACK",
        "primary_bookmaker": PRIMARY_BOOKMAKER,
        "primary_bookmaker_available_all_legs": bool(primary_book),
        "primary_bookmaker_used": bool(primary_book and chosen == primary_book),
        "fallback_used": not bool(primary_book and chosen == primary_book),
        "primary_component_product_decimal_reference": round(primary_decimal, 4) if primary_decimal else None,
        "primary_component_product_american_reference": v2._american(primary_decimal) if primary_decimal else None,
        "primary_components": primary_components,
        "best_available_bookmaker": best_book,
        "best_available_component_product_decimal_reference": round(best_decimal, 4),
        "best_available_component_product_american_reference": v2._american(best_decimal),
        "best_available_components": best_components,
        "component_product_decimal_reference": round(chosen_decimal, 4),
        "component_product_american_reference": v2._american(chosen_decimal),
        "components": components,
        "target_decimal": v4.TARGET_DECIMAL,
        "target_american": 110,
        "target_component_reference_gate_passed": chosen_decimal >= v4.TARGET_DECIMAL,
        "warning": (
            "REFERENCE_ONLY_NOT_FINAL_SGP_QUOTE; SAME_GAME COMPONENTS ARE CORRELATED; "
            "EXACT_CORRELATION_ADJUSTED_SGP_PRICE_REQUIRED"
        ),
    }


def _primary_rolling_common_book(combo: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], float] | None:
    maps = [v37._quote_map(list(row.get("quotes") or [])) for row in combo]
    if any(not mapping for mapping in maps):
        return None
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    if not common:
        return None

    viable: list[tuple[float, str, list[dict[str, Any]]]] = []
    for book in common:
        quotes = [mapping[book] for mapping in maps]
        component_edges = []
        for row, quote in zip(combo, quotes):
            probability = float(row.get("probability") or 0.0)
            price = float(quote.get("price") or 0.0)
            component_edges.append((probability - 1.0 / price) * 100.0)
        if min(component_edges) < v37.MIN_COMPONENT_EDGE_PP:
            continue
        product = math.prod(float(quote["price"]) for quote in quotes)
        viable.append((product, book, quotes))
    if not viable:
        return None

    primary = [row for row in viable if _is_primary(row[1])]
    if primary:
        product, book, quotes = max(primary, key=lambda row: row[0])
        return book, quotes, product

    product, book, quotes = max(viable, key=lambda row: row[0])
    return book, quotes, product


def _clean_bookmaker_rows(response: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response or []:
        if not isinstance(item, dict):
            continue
        bookmaker_id = item.get("id")
        name = str(item.get("name") or "").strip()
        if bookmaker_id is None or not name:
            continue
        rows.append({"id": bookmaker_id, "name": name})
    rows.sort(key=lambda row: (str(row.get("name") or "").lower(), str(row.get("id"))))
    return rows


async def _audit_bookmaker_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    now = _generated_at(payload)
    cached = base._cache_get(
        BOOKMAKER_CACHE_NAMESPACE,
        BOOKMAKER_CACHE_KEY,
        BOOKMAKER_CACHE_TTL,
        now,
    )
    rows = cached if isinstance(cached, list) else None
    source = "CACHE_24H" if rows is not None else "NOT_CHECKED"
    provider_requests_added = 0

    if rows is None:
        used = int(v2._API_CALLS_THIS_TICK or 0)
        cap = int(v2.MAX_API_CALLS_PER_TICK or 0)
        if cap - used <= 1:
            return {
                "schema_version": "1.0.0",
                "status": "DEFERRED_PROVIDER_BUDGET",
                "source": "NOT_CHECKED",
                "primary_bookmaker": PRIMARY_BOOKMAKER,
                "primary_bookmaker_found": None,
                "primary_bookmaker_matches": [],
                "provider_requests_added": 0,
            }
        response = await v6._adaptive_paced_api_get("odds/bookmakers", {})
        rows = _clean_bookmaker_rows(response.get("response"))
        base._cache_set(
            BOOKMAKER_CACHE_NAMESPACE,
            BOOKMAKER_CACHE_KEY,
            rows,
            now,
        )
        source = "API_FOOTBALL_ODDS_BOOKMAKERS"
        provider_requests_added = 1

    matches = [row for row in rows if _is_primary(row.get("name"))]
    return {
        "schema_version": "1.0.0",
        "status": "PRIMARY_BOOKMAKER_FOUND" if matches else "PRIMARY_BOOKMAKER_NOT_FOUND_IN_PROVIDER_CATALOG",
        "source": source,
        "catalog_count": len(rows),
        "primary_bookmaker": PRIMARY_BOOKMAKER,
        "primary_bookmaker_found": bool(matches),
        "primary_bookmaker_matches": matches,
        "provider_requests_added": provider_requests_added,
        "policy": "BET365_PRIMARY_BY_VERIFIED_PROVIDER_NAME; NEVER_INVENT_BOOKMAKER_ID",
    }


def _annotate_builder(payload: dict[str, Any]) -> dict[str, Any]:
    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    builder = dict(builder)

    bet365_sgp = 0
    fallback_sgp = 0
    for row in builder.get("same_game_candidates") or []:
        if not isinstance(row, dict):
            continue
        ref = row.get("component_price_reference") if isinstance(row.get("component_price_reference"), dict) else {}
        book = ref.get("bookmaker")
        row["reference_bookmaker_policy"] = "BET365_PRIMARY_THEN_BEST_VERIFIED_FALLBACK"
        row["primary_bookmaker"] = PRIMARY_BOOKMAKER
        row["primary_bookmaker_used"] = _is_primary(book)
        row["executable_sgp_quote_required"] = True
        row["internal_fair_price_source"] = "DIRECT_SCORE_MATRIX_JOINT_PROBABILITY"
        if _is_primary(book):
            bet365_sgp += 1
        else:
            fallback_sgp += 1

    bet365_multi = 0
    fallback_multi = 0
    for row in builder.get("multi_match_candidates") or []:
        if not isinstance(row, dict):
            continue
        book = row.get("bookmaker")
        row["reference_bookmaker_policy"] = "BET365_PRIMARY_THEN_BEST_VERIFIED_FALLBACK"
        row["primary_bookmaker"] = PRIMARY_BOOKMAKER
        row["primary_bookmaker_used"] = _is_primary(book)
        row["calculated_same_book_combined_decimal"] = row.get("component_product_decimal_reference")
        row["calculated_same_book_combined_american"] = row.get("component_product_american_reference")
        row["calculated_combined_price_method"] = "PRODUCT_OF_DISTINCT_FIXTURE_DECIMAL_ODDS_FROM_SAME_BOOK"
        row["calculated_combined_price_is_sgp_price"] = False
        if _is_primary(book):
            bet365_multi += 1
        else:
            fallback_multi += 1

    builder["bookmaker_reference_policy"] = {
        "primary": PRIMARY_BOOKMAKER,
        "selection": "PRIMARY_IF_COMMON_AND_VIABLE; OTHERWISE_BEST_VERIFIED_COMMON_BOOK_FALLBACK",
        "same_book_required_for_all_legs": True,
        "multi_match_combined_price": "PRODUCT_OF_DISTINCT_FIXTURE_DECIMAL_ODDS",
        "same_game_combined_price": "NEVER_PRODUCT; USE_JOINT_MODEL_FOR_FAIR_PRICE_AND_REQUIRE_EXECUTABLE_BOOK_QUOTE",
        "bet365_same_game_candidates_this_tick": bet365_sgp,
        "fallback_same_game_candidates_this_tick": fallback_sgp,
        "bet365_multi_match_candidates_active": bet365_multi,
        "fallback_multi_match_candidates_active": fallback_multi,
    }
    payload["galaxy_builder"] = builder
    return builder["bookmaker_reference_policy"]


async def run_tick() -> dict[str, Any]:
    previous_sgp = v4._common_book_reference
    previous_multi = v37._common_book
    v4._common_book_reference = _primary_common_book_reference
    v37._common_book = _primary_rolling_common_book
    try:
        payload = await v81.run_tick()
    finally:
        v4._common_book_reference = previous_sgp
        v37._common_book = previous_multi

    audit = await _audit_bookmaker_catalog(payload)
    policy = _annotate_builder(payload)

    payload["bookmaker_catalog_audit"] = audit
    payload["primary_bookmaker_policy"] = policy
    payload["v358_provider_requests_added"] = int(audit.get("provider_requests_added") or 0)
    payload["v358_model_weights_changed"] = False
    payload["v358_canonical_bet_logic_changed"] = False
    payload["v358_bookmaker_checkpoint"] = (
        "BET365_PRIMARY_REFERENCE_ATTACHED; DISTINCT_FIXTURE_PARLAY_ODDS_CALCULATED_FROM_SAME_BOOK; "
        "SGP_INTERNAL_FAIR_PRICE_REMAINS_CORRELATION_AWARE_AND_EXECUTABLE_COMBINED_QUOTE_IS_SEPARATE"
    )
    payload["api_calls_this_tick"] = max(
        int(payload.get("api_calls_this_tick") or 0),
        int(v2._API_CALLS_THIS_TICK or 0),
    )
    if v2._LAST_DAILY_REMAINING is not None:
        payload["last_daily_remaining"] = v2._LAST_DAILY_REMAINING
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
