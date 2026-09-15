from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v23 as v23

MODEL_VERSION = v23.MODEL_VERSION
AUTOMATION_VERSION = "3.1.1"
IDENTITY_CACHE_TTL = timedelta(days=7)


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _identity_key(league_id: Any, season: Any) -> str:
    return f"{league_id}:{season}"


def _needs_identity(row: dict[str, Any]) -> bool:
    country = _clean(row.get("country"))
    competition = _clean(row.get("competition"))
    return country in {None, "NOT VERIFIED"} or not competition or competition.startswith("League #")


async def _identity_map(now: datetime, payload: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], str]:
    cached = base._cache_get("league_identity_bulk", "v1", IDENTITY_CACHE_TTL, now)
    if isinstance(cached, dict) and cached:
        return cached, "CACHE"

    used = int(payload.get("api_calls_this_tick") or 0)
    cap = int(payload.get("max_api_calls_per_tick") or 20)
    if used >= cap:
        return {}, "SKIPPED_PROVIDER_BUDGET"

    response = await base._api_get("leagues", {})
    mapping: dict[str, dict[str, Any]] = {}
    for row in response.get("response", []) or []:
        league = row.get("league") or {}
        league_id = league.get("id")
        if league_id is None:
            continue
        name = _clean(league.get("name"))
        country = _clean(league.get("country"))
        for season_row in row.get("seasons") or []:
            season = season_row.get("year")
            if season is None:
                continue
            mapping[_identity_key(league_id, season)] = {
                "league_id": league_id,
                "season": season,
                "competition": name,
                "country": country,
            }

    if mapping:
        base._cache_set("league_identity_bulk", "v1", mapping, now)
    payload["api_calls_this_tick"] = max(
        int(payload.get("api_calls_this_tick") or 0), int(v2._API_CALLS_THIS_TICK or 0)
    )
    if v2._LAST_DAILY_REMAINING is not None:
        payload["last_daily_remaining"] = v2._LAST_DAILY_REMAINING
    return mapping, "API_FOOTBALL_BULK_LEAGUES"


def _apply_identity(payload: dict[str, Any], mapping: dict[str, dict[str, Any]]) -> tuple[int, int]:
    registry = payload.get("league_coverage_registry")
    competitions = registry.get("competitions") if isinstance(registry, dict) else None
    if not isinstance(competitions, list):
        competitions = []

    resolved = 0
    unresolved = 0
    by_key: dict[str, dict[str, Any]] = {}
    for row in competitions:
        if not isinstance(row, dict):
            continue
        key = _identity_key(row.get("league_id"), row.get("season"))
        identity = mapping.get(key) or {}
        if _needs_identity(row):
            if _clean(identity.get("country")):
                row["country"] = identity.get("country")
            if _clean(identity.get("competition")):
                row["competition"] = identity.get("competition")
        if _needs_identity(row):
            unresolved += 1
        else:
            resolved += 1
        by_key[key] = row

    fixture_identity: dict[Any, tuple[str | None, str | None]] = {}
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        fx = event.get("fixture")
        if not isinstance(fx, dict):
            continue
        key = _identity_key(fx.get("league_id"), fx.get("season"))
        identity = mapping.get(key) or by_key.get(key) or {}
        country = _clean(fx.get("country"))
        competition = _clean(fx.get("league"))
        if country in {None, "NOT VERIFIED"} and _clean(identity.get("country")):
            fx["country"] = identity.get("country")
        if (not competition or competition.startswith("League #")) and _clean(identity.get("competition")):
            fx["league"] = identity.get("competition")
        fixture_identity[fx.get("fixture_id")] = (_clean(fx.get("country")), _clean(fx.get("league")))

    for row in payload.get("match_table_rows") or []:
        if not isinstance(row, dict):
            continue
        identity = fixture_identity.get(row.get("fixture_id"))
        if not identity:
            continue
        country, competition = identity
        if (_clean(row.get("country")) in {None, "NOT VERIFIED"}) and country:
            row["country"] = country
        current_comp = _clean(row.get("competition"))
        if (not current_comp or current_comp.startswith("League #")) and competition:
            row["competition"] = competition

    return resolved, unresolved


async def run_tick() -> dict[str, Any]:
    payload = await v23.run_tick()
    now_raw = payload.get("generated_at_utc")
    try:
        now = datetime.fromisoformat(str(now_raw).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        now = datetime.now(dt_timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt_timezone.utc)
    now = now.astimezone(dt_timezone.utc)

    registry = payload.get("league_coverage_registry")
    competitions = registry.get("competitions") if isinstance(registry, dict) else []
    missing_before = sum(1 for row in competitions or [] if isinstance(row, dict) and _needs_identity(row))

    mapping: dict[str, dict[str, Any]] = {}
    source = "NOT_NEEDED"
    if missing_before:
        mapping, source = await _identity_map(now, payload)
    resolved, unresolved = _apply_identity(payload, mapping)

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["league_identity_enrichment"] = {
        "policy": "BULK_GLOBAL_LEAGUE_MAP_ONLY; ZERO_PER_LEAGUE_PROVIDER_CALLS",
        "source": source,
        "cache_ttl_days": int(IDENTITY_CACHE_TTL.total_seconds() // 86400),
        "missing_before": missing_before,
        "verified_identity_rows": resolved,
        "unresolved_identity_rows": unresolved,
        "provider_calls_per_league": 0,
        "galaxy_activation_required": False,
    }
    return payload
