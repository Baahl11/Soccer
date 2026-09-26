from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any, Awaitable, Callable

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v88 as v88

MODEL_VERSION = v88.MODEL_VERSION
AUTOMATION_VERSION = "3.63.0-weekend-horizon"

SLATE_FLOOR_MIN_FIXTURES = int(os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES", "12"))
SLATE_FLOOR_MIN_DAILY_REMAINING = int(
    os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_DAILY_REMAINING", "4000")
)
FUTURE_SLATE_HORIZON_DAYS = max(
    0, min(2, int(os.getenv("SOCCER_EDGE_FUTURE_SLATE_HORIZON_DAYS", "2")))
)
FUTURE_SLATE_TTL_HOURS = max(
    1, int(os.getenv("SOCCER_EDGE_FUTURE_SLATE_TTL_HOURS", "4"))
)
FUTURE_SLATE_DAY1_MIN_DAILY_REMAINING = int(
    os.getenv("SOCCER_EDGE_FUTURE_SLATE_DAY1_MIN_DAILY_REMAINING", "4500")
)
FUTURE_SLATE_DAY2_MIN_DAILY_REMAINING = int(
    os.getenv("SOCCER_EDGE_FUTURE_SLATE_DAY2_MIN_DAILY_REMAINING", "6000")
)

ApiGet = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


def _endpoint_name(endpoint: str) -> str:
    return str(endpoint or "").strip().lstrip("/")


def _is_raw_date_fixture_slate(endpoint: str, params: dict[str, Any]) -> bool:
    """Match only the scheduler's raw fixtures?date slate request.

    Team recent-form calls also hit /fixtures, but they include `team`/`last` and
    must never be expanded into a global date slate.
    """
    if _endpoint_name(endpoint) != "fixtures":
        return False
    if not isinstance(params, dict) or not params.get("date"):
        return False
    disallowed = {
        "id",
        "ids",
        "live",
        "league",
        "season",
        "team",
        "last",
        "next",
        "from",
        "to",
        "round",
        "venue",
        "status",
    }
    return not any(key in params for key in disallowed)


def _raw_fixture_id(row: Any) -> int | None:
    if not isinstance(row, dict):
        return None
    fixture = row.get("fixture") if isinstance(row.get("fixture"), dict) else {}
    value = fixture.get("id")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _payload_fixture_count(payload: dict[str, Any]) -> int:
    return sum(1 for row in payload.get("response") or [] if _raw_fixture_id(row) is not None)


def _merge_fixture_payloads(primary: dict[str, Any], reconciliation: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(primary)
    rows: list[Any] = []
    seen: set[int] = set()

    for payload in (primary, reconciliation):
        for row in payload.get("response") or []:
            fixture_id = _raw_fixture_id(row)
            if fixture_id is None or fixture_id in seen:
                continue
            seen.add(fixture_id)
            rows.append(row)

    merged["response"] = rows
    merged["results"] = len(rows)
    if reconciliation.get("quota"):
        merged["quota"] = reconciliation.get("quota")
    merged["slate_reconciliation"] = {
        "schema_version": "1.0.0",
        "primary_results": _payload_fixture_count(primary),
        "reconciliation_results": _payload_fixture_count(reconciliation),
        "merged_results": len(rows),
        "dedupe_key": "fixture.id",
    }
    return merged


def _future_prefetch_offsets(local_now: datetime, daily_remaining: Any) -> list[int]:
    try:
        remaining = int(daily_remaining) if daily_remaining is not None else None
    except (TypeError, ValueError):
        remaining = None

    offsets: list[int] = []
    if FUTURE_SLATE_HORIZON_DAYS >= 1 and (
        local_now.hour >= 22
        or (remaining is not None and remaining > FUTURE_SLATE_DAY1_MIN_DAILY_REMAINING)
    ):
        offsets.append(1)
    if (
        FUTURE_SLATE_HORIZON_DAYS >= 2
        and remaining is not None
        and remaining > FUTURE_SLATE_DAY2_MIN_DAILY_REMAINING
    ):
        offsets.append(2)
    return offsets


async def _future_date_payload(
    api_get: ApiGet,
    endpoint: str,
    params: dict[str, Any],
    future_date: str,
    now_utc: datetime,
) -> tuple[dict[str, Any], bool]:
    cached = base._cache_get(
        "future_raw_fixture_slate",
        future_date,
        timedelta(hours=FUTURE_SLATE_TTL_HOURS),
        now_utc,
    )
    if isinstance(cached, dict) and isinstance(cached.get("response"), list):
        replay = deepcopy(cached)
        # Quota headers must always come from a live request in this tick.
        replay.pop("quota", None)
        return replay, True

    future_params = dict(params)
    future_params["date"] = future_date
    payload = await api_get(endpoint, future_params)
    cache_value = deepcopy(payload)
    cache_value.pop("quota", None)
    base._cache_set("future_raw_fixture_slate", future_date, cache_value, now_utc)
    return payload, False


def _slate_dimensions(payload: dict[str, Any]) -> tuple[int, int]:
    leagues: set[str] = set()
    countries: set[str] = set()
    for row in payload.get("response") or []:
        if not isinstance(row, dict):
            continue
        league = row.get("league") if isinstance(row.get("league"), dict) else {}
        league_id = league.get("id")
        league_name = str(league.get("name") or "").strip()
        country = str(league.get("country") or "").strip()
        if league_id is not None:
            leagues.add(f"id:{league_id}")
        elif league_name:
            leagues.add(f"name:{country}|{league_name}")
        if country:
            countries.add(country)
    return len(leagues), len(countries)


def _daily_remaining_allows(value: Any) -> bool:
    try:
        if value is None:
            return True
        return int(value) > SLATE_FLOOR_MIN_DAILY_REMAINING
    except (TypeError, ValueError):
        return True


def _tick_budget_allows_extra_call() -> bool:
    try:
        cap = int(v2.MAX_API_CALLS_PER_TICK or 0)
        used = int(v2._API_CALLS_THIS_TICK or 0)
    except (TypeError, ValueError):
        return False
    return cap <= 0 or used < cap


def _new_metrics() -> dict[str, Any]:
    return {
        "schema_version": "1.1.0",
        "policy": "RAW_API_FOOTBALL_ROLLING_DATE_SLATE_WITH_CACHED_FUTURE_PREFETCH",
        "min_fixture_count": SLATE_FLOOR_MIN_FIXTURES,
        "min_daily_remaining": SLATE_FLOOR_MIN_DAILY_REMAINING,
        "triggered": False,
        "reason": None,
        "primary_date": None,
        "reconciliation_date": None,
        "primary_slate_count": None,
        "reconciliation_slate_count": 0,
        "merged_slate_count": None,
        "api_slate_reconciliation_calls": 0,
        "provider_requests_added_max": 2,
        "future_prefetch_horizon_days": FUTURE_SLATE_HORIZON_DAYS,
        "future_prefetch_ttl_hours": FUTURE_SLATE_TTL_HOURS,
        "future_prefetch_cache_hits": 0,
        "future_prefetch_cache_misses": 0,
        "future_prefetch_provider_requests_added": 0,
        "future_prefetch_fixture_count": 0,
        "scan_dates": [],
        "scan_date_counts": {},
        "unique_leagues_scanned": 0,
        "unique_countries_scanned": 0,
        "league_allowlist_applied": False,
        "scope": "all fixtures returned by API-Football for selected dates; no league allowlist; no odds/model/tier/stake threshold change",
    }


def _should_reconcile(
    *,
    payload: dict[str, Any],
    params: dict[str, Any],
    now_utc: datetime,
    metrics: dict[str, Any],
) -> tuple[bool, str]:
    if metrics.get("api_slate_reconciliation_calls"):
        return False, "ALREADY_RECONCILED_THIS_TICK"
    if not _is_raw_date_fixture_slate("fixtures", params):
        return False, "NOT_RAW_DATE_SLATE"

    local_now = now_utc.astimezone(base.TIMEZONE)
    today = local_now.date().isoformat()
    if str(params.get("date")) != today:
        return False, "NOT_TODAY_PRIMARY_SLATE"
    if local_now.hour >= 22:
        return False, "LEGACY_LATE_DAY_NEXT_DATE_ALREADY_ENABLED"

    primary_count = _payload_fixture_count(payload)
    if primary_count >= SLATE_FLOOR_MIN_FIXTURES:
        return False, "PRIMARY_SLATE_MEETS_FLOOR"
    if not _tick_budget_allows_extra_call():
        return False, "TICK_BUDGET_BLOCKED"
    if not _daily_remaining_allows(v2._LAST_DAILY_REMAINING):
        return False, "DAILY_BUDGET_BLOCKED"
    return True, "PRIMARY_SLATE_BELOW_FLOOR"


def _attach_top_level_metrics(payload: dict[str, Any], metrics: dict[str, Any]) -> None:
    payload["slate_floor_reconciliation"] = metrics
    payload["slate_floor_triggered"] = bool(metrics.get("triggered"))
    payload["slate_floor_reason"] = metrics.get("reason")
    payload["slate_floor_min_fixture_count"] = metrics.get("min_fixture_count")
    payload["api_slate_reconciliation_calls"] = int(metrics.get("api_slate_reconciliation_calls") or 0)
    payload["api_raw_slate_count"] = metrics.get("primary_slate_count")
    payload["api_raw_reconciliation_slate_count"] = metrics.get("reconciliation_slate_count")
    payload["merged_slate_count"] = metrics.get("merged_slate_count")
    payload["slate_source_policy"] = (
        "RAW_API_FOOTBALL_TODAY_FIRST; HIGH_QUOTA_CACHED_48H_PREFETCH; NO_LEAGUE_ALLOWLIST; LOW_SLATE_FALLBACK_TOMORROW"
    )


async def run_tick() -> dict[str, Any]:
    metrics = _new_metrics()
    previous_original_paced = v6._ORIGINAL_PACED_API_GET

    async def paced_with_slate_floor(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = await previous_original_paced(endpoint, params)
        if not _is_raw_date_fixture_slate(endpoint, params):
            return payload

        now_utc = datetime.now(dt_timezone.utc)
        local_now = now_utc.astimezone(base.TIMEZONE)
        primary_date = str(params.get("date"))
        metrics["primary_date"] = primary_date
        metrics["primary_slate_count"] = _payload_fixture_count(payload)
        metrics["merged_slate_count"] = metrics["primary_slate_count"]
        metrics["scan_dates"] = [primary_date]
        metrics["scan_date_counts"] = {primary_date: metrics["primary_slate_count"]}

        today = local_now.date().isoformat()
        if primary_date == today:
            merged = payload
            for offset in _future_prefetch_offsets(local_now, v2._LAST_DAILY_REMAINING):
                if not _tick_budget_allows_extra_call():
                    break
                future_date = (local_now.date() + timedelta(days=offset)).isoformat()
                future_payload, cache_hit = await _future_date_payload(
                    previous_original_paced,
                    endpoint,
                    params,
                    future_date,
                    now_utc,
                )
                future_count = _payload_fixture_count(future_payload)
                merged = _merge_fixture_payloads(merged, future_payload)
                metrics["scan_dates"].append(future_date)
                metrics["scan_date_counts"][future_date] = future_count
                metrics["future_prefetch_fixture_count"] += future_count
                if cache_hit:
                    metrics["future_prefetch_cache_hits"] += 1
                else:
                    metrics["future_prefetch_cache_misses"] += 1
                    metrics["future_prefetch_provider_requests_added"] += 1
                    metrics["api_slate_reconciliation_calls"] += 1

            if len(metrics["scan_dates"]) > 1:
                metrics["triggered"] = True
                metrics["merged_slate_count"] = _payload_fixture_count(merged)
                metrics["reason"] = (
                    "PRIMARY_SLATE_BELOW_FLOOR_FUTURE_PREFETCH_INCLUDED"
                    if metrics["primary_slate_count"] < SLATE_FLOOR_MIN_FIXTURES
                    else "PRIMARY_SLATE_MEETS_FLOOR_WITH_FUTURE_PREFETCH"
                )
                leagues, countries = _slate_dimensions(merged)
                metrics["unique_leagues_scanned"] = leagues
                metrics["unique_countries_scanned"] = countries
                return merged

        should_reconcile, reason = _should_reconcile(
            payload=payload,
            params=params,
            now_utc=now_utc,
            metrics=metrics,
        )
        metrics["reason"] = reason
        if not should_reconcile:
            leagues, countries = _slate_dimensions(payload)
            metrics["unique_leagues_scanned"] = leagues
            metrics["unique_countries_scanned"] = countries
            return payload

        tomorrow = (local_now.date() + timedelta(days=1)).isoformat()
        reconciliation, cache_hit = await _future_date_payload(
            previous_original_paced,
            endpoint,
            params,
            tomorrow,
            now_utc,
        )
        merged = _merge_fixture_payloads(payload, reconciliation)

        metrics["triggered"] = True
        metrics["reason"] = "PRIMARY_SLATE_BELOW_FLOOR_RECONCILED_WITH_NEXT_DATE"
        metrics["reconciliation_date"] = tomorrow
        metrics["reconciliation_slate_count"] = _payload_fixture_count(reconciliation)
        metrics["merged_slate_count"] = _payload_fixture_count(merged)
        metrics["scan_dates"].append(tomorrow)
        metrics["scan_date_counts"][tomorrow] = metrics["reconciliation_slate_count"]
        if not cache_hit:
            metrics["api_slate_reconciliation_calls"] += 1
        leagues, countries = _slate_dimensions(merged)
        metrics["unique_leagues_scanned"] = leagues
        metrics["unique_countries_scanned"] = countries
        return merged

    v6._ORIGINAL_PACED_API_GET = paced_with_slate_floor
    try:
        payload = await v88.run_tick()
    finally:
        v6._ORIGINAL_PACED_API_GET = previous_original_paced

    _attach_top_level_metrics(payload, metrics)
    payload["v362_provider_requests_added"] = int(metrics.get("api_slate_reconciliation_calls") or 0)
    payload["v362_provider_request_scope"] = "fixtures?date=tomorrow only when primary date slate is below floor"
    payload["v362_model_weights_changed"] = False
    payload["v362_canonical_bet_logic_changed"] = False
    payload["v362_runtime_promotion_added"] = False
    payload["v362_stake_or_tier_change"] = False
    payload["v362_slate_floor_checkpoint"] = (
        "SLATE FLOOR RECONCILIATION: A TINY TODAY SLATE NO LONGER PREVENTS THE SCHEDULER FROM "
        "ADDING TOMORROW'S RAW API-FOOTBALL DATE SLATE WHEN BUDGET ALLOWS; SPORT SCREEN AND MARKET "
        "GATES REMAIN UNCHANGED."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
