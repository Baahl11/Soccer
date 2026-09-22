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
AUTOMATION_VERSION = "3.62.0"

SLATE_FLOOR_MIN_FIXTURES = int(os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES", "12"))
SLATE_FLOOR_MIN_DAILY_REMAINING = int(
    os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_DAILY_REMAINING", "4000")
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
        "schema_version": "1.0.0",
        "policy": "RAW_API_FOOTBALL_DATE_SLATE_FLOOR_RECONCILIATION",
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
        "provider_requests_added_max": 1,
        "scope": "fixtures?date slate only; no odds request; no market/tier/stake/model-weight change",
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
        "RAW_API_FOOTBALL_TODAY_FIRST; IF_FIXTURE_COUNT_BELOW_FLOOR_AND_BUDGET_ALLOWS_ADD_TOMORROW"
    )


async def run_tick() -> dict[str, Any]:
    metrics = _new_metrics()
    previous_original_paced = v6._ORIGINAL_PACED_API_GET

    async def paced_with_slate_floor(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        payload = await previous_original_paced(endpoint, params)
        if not _is_raw_date_fixture_slate(endpoint, params):
            return payload

        now_utc = datetime.now(dt_timezone.utc)
        metrics["primary_date"] = str(params.get("date"))
        metrics["primary_slate_count"] = _payload_fixture_count(payload)
        metrics["merged_slate_count"] = metrics["primary_slate_count"]

        should_reconcile, reason = _should_reconcile(
            payload=payload,
            params=params,
            now_utc=now_utc,
            metrics=metrics,
        )
        metrics["reason"] = reason
        if not should_reconcile:
            return payload

        tomorrow = (now_utc.astimezone(base.TIMEZONE).date() + timedelta(days=1)).isoformat()
        recon_params = dict(params)
        recon_params["date"] = tomorrow
        reconciliation = await previous_original_paced(endpoint, recon_params)
        merged = _merge_fixture_payloads(payload, reconciliation)

        metrics["triggered"] = True
        metrics["reason"] = "PRIMARY_SLATE_BELOW_FLOOR_RECONCILED_WITH_NEXT_DATE"
        metrics["reconciliation_date"] = tomorrow
        metrics["reconciliation_slate_count"] = _payload_fixture_count(reconciliation)
        metrics["merged_slate_count"] = _payload_fixture_count(merged)
        metrics["api_slate_reconciliation_calls"] = 1
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
