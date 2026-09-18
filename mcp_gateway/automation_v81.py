from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v80 as v80

MODEL_VERSION = v80.MODEL_VERSION
AUTOMATION_VERSION = "3.57.0"
CATALOG_CACHE_TTL = timedelta(hours=24)
CATALOG_CACHE_NAMESPACE = "odds_reference"
CATALOG_CACHE_KEY = "prematch_bets_v1"
_COMBINATION_TERMS = (
    "same game parlay",
    "same-game parlay",
    "same game",
    "sgp",
    "bet builder",
    "bet-builder",
    "builder",
    "parlay",
    "accumulator",
    "acca",
    "combination",
    "combo",
)


def _clean_catalog_rows(response: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in response or []:
        if not isinstance(item, dict):
            continue
        bet_id = item.get("id")
        name = str(item.get("name") or "").strip()
        if bet_id is None or not name:
            continue
        rows.append({"id": bet_id, "name": name})
    rows.sort(key=lambda row: (str(row.get("name") or "").lower(), str(row.get("id"))))
    return rows


def _combination_like(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name") or "")
        normalized = " ".join(name.lower().split())
        if any(term in normalized for term in _COMBINATION_TERMS):
            out.append(dict(row))
    return out


def _generated_at(payload: dict[str, Any]) -> datetime:
    raw = payload.get("generated_at_utc")
    try:
        now = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        now = datetime.now(dt_timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt_timezone.utc)
    return now.astimezone(dt_timezone.utc)


async def _audit_catalog(payload: dict[str, Any]) -> dict[str, Any]:
    now = _generated_at(payload)
    cached = base._cache_get(
        CATALOG_CACHE_NAMESPACE,
        CATALOG_CACHE_KEY,
        CATALOG_CACHE_TTL,
        now,
    )
    rows = cached if isinstance(cached, list) else None
    source = "CACHE_24H" if rows is not None else "NOT_CHECKED"
    provider_requests_added = 0

    if rows is None:
        used = int(v2._API_CALLS_THIS_TICK or 0)
        cap = int(v2.MAX_API_CALLS_PER_TICK or 0)
        # Preserve at least one call of headroom for higher-priority lifecycle work.
        if cap - used <= 1:
            return {
                "schema_version": "1.0.0",
                "status": "DEFERRED_PROVIDER_BUDGET",
                "source": "NOT_CHECKED",
                "catalog_count": None,
                "explicit_sgp_like_count": None,
                "explicit_sgp_like_bet_types": [],
                "provider_requests_added": 0,
                "model_weights_changed": False,
                "canonical_bet_logic_changed": False,
                "policy": "REFERENCE_CATALOG_AUDIT_ONLY; NO BET PROMOTION",
            }
        response = await v6._adaptive_paced_api_get("odds/bets", {})
        rows = _clean_catalog_rows(response.get("response"))
        base._cache_set(
            CATALOG_CACHE_NAMESPACE,
            CATALOG_CACHE_KEY,
            rows,
            now,
        )
        source = "API_FOOTBALL_ODDS_BETS"
        provider_requests_added = 1

    combo = _combination_like(rows)
    names = [row["name"] for row in rows]
    status = (
        "EXPLICIT_SGP_LIKE_BET_TYPE_FOUND"
        if combo
        else "NO_EXPLICIT_SGP_LIKE_BET_TYPE_FOUND_BY_NAME_AUDIT"
    )
    return {
        "schema_version": "1.0.0",
        "status": status,
        "source": source,
        "catalog_count": len(rows),
        "bet_types": rows,
        "bet_type_names": names,
        "explicit_sgp_like_count": len(combo),
        "explicit_sgp_like_bet_types": combo,
        "provider_requests_added": provider_requests_added,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "internal_sgp_fair_price_status": (
            "ALREADY_COMPUTED_FROM_DIRECT_SCORE_MATRIX_JOINT_PROBABILITY"
        ),
        "executable_sgp_quote_status": (
            "REQUIRES_EXPLICIT_PROVIDER_OR_SPORTSBOOK_COMBINED_QUOTE"
        ),
        "interpretation": (
            "Catalog name audit can confirm an explicit SGP/Bet Builder-style bet type when present. "
            "Absence of a matching name means no explicit combined-price type was found in the current "
            "prematch bet catalog; it does not convert component-product odds into an executable SGP quote."
        ),
        "policy": (
            "SPORT FIRST -> MARKET SECOND -> COMBINATION THIRD; INTERNAL FAIR PRICE MAY BE CALCULATED; "
            "FINAL BET STILL REQUIRES A VERIFIED EXECUTABLE COMBINED PRICE"
        ),
    }


async def run_tick() -> dict[str, Any]:
    payload = await v80.run_tick()
    audit = await _audit_catalog(payload)
    payload["odds_catalog_audit"] = audit
    payload["v357_provider_requests_added"] = int(audit.get("provider_requests_added") or 0)
    payload["v357_model_weights_changed"] = False
    payload["v357_canonical_bet_logic_changed"] = False
    payload["v357_odds_catalog_checkpoint"] = (
        "PREMATCH_ODDS_BET_CATALOG_AUDIT_ATTACHED; INTERNAL_SGP_FAIR_PRICE_CONFIRMED; "
        "EXECUTABLE_COMBINED_QUOTE_REMAINS_SEPARATE"
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
