from __future__ import annotations

import os
from typing import Any, Awaitable, Callable

from mcp_gateway import automation_v12 as v12
from mcp_gateway import automation_v90 as v90

MODEL_VERSION = v90.MODEL_VERSION
AUTOMATION_VERSION = "3.62.2"

GALAXY_SLATE_FLOOR_MIN_FIXTURES = int(
    os.getenv("SOCCER_EDGE_SLATE_FLOOR_MIN_FIXTURES", "12")
)

GalaxySlatePayload = Callable[[str, Any], Awaitable[dict[str, Any] | None]]


def _metric_value(key: str) -> int:
    try:
        return int(v12._SLATE_METRICS.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _restore_metric(key: str, value: int) -> None:
    v12._SLATE_METRICS[key] = int(value)


def _bump_metric(key: str, amount: int = 1) -> None:
    v12._SLATE_METRICS[key] = _metric_value(key) + int(amount)


def _payload_results(payload: dict[str, Any] | None) -> int:
    if not isinstance(payload, dict):
        return 0
    try:
        return int(payload.get("results") or len(payload.get("response") or []))
    except (TypeError, ValueError):
        return 0


def _annotate_payload(payload: dict[str, Any]) -> None:
    metrics = dict(payload.get("galaxy_first_metrics") or {})
    floor_rejections = _metric_value("galaxy_slate_floor_rejections")
    floor_last_count = _metric_value("galaxy_slate_floor_last_rejected_count")

    metrics["galaxy_slate_floor_min_fixture_count"] = GALAXY_SLATE_FLOOR_MIN_FIXTURES
    metrics["galaxy_slate_floor_rejections"] = floor_rejections
    metrics["galaxy_slate_floor_last_rejected_count"] = floor_last_count or None
    metrics["tiny_galaxy_slate_can_avoid_api"] = False
    metrics["galaxy_slate_floor_policy"] = (
        "FRESH_GALAXY_SLATE_REQUIRES_MIN_FIXTURE_COUNT; OTHERWISE FALL BACK TO API_FOOTBALL DATE SLATE"
    )
    payload["galaxy_first_metrics"] = metrics

    payload["v3622_provider_requests_added"] = 0
    payload["v3622_provider_request_scope"] = (
        "No direct request here; tiny Galaxy slate is rejected so existing API fallback/floor path can run."
    )
    payload["v3622_galaxy_slate_floor_min_fixture_count"] = GALAXY_SLATE_FLOOR_MIN_FIXTURES
    payload["v3622_galaxy_slate_floor_rejections"] = floor_rejections
    payload["v3622_model_weights_changed"] = False
    payload["v3622_canonical_bet_logic_changed"] = False
    payload["v3622_runtime_promotion_added"] = False
    payload["v3622_stake_or_tier_change"] = False
    payload["v3622_checkpoint"] = (
        "GALAXY-FIRST SLATE FLOOR: a fresh Galaxy persisted slate below the fixture floor is not allowed "
        "to avoid the API-Football date slate. This corrects the tiny 2-fixture universe without changing picks, "
        "model weights, thresholds, tiers or stakes."
    )


async def run_tick() -> dict[str, Any]:
    original_galaxy_slate_payload: GalaxySlatePayload = v12._galaxy_slate_payload

    async def floor_guarded_galaxy_slate_payload(match_date: str, now: Any) -> dict[str, Any] | None:
        before_hits = _metric_value("galaxy_slate_hits")
        before_avoided = _metric_value("api_slate_calls_avoided")
        payload = await original_galaxy_slate_payload(match_date, now)
        count = _payload_results(payload)
        if payload is not None and count < GALAXY_SLATE_FLOOR_MIN_FIXTURES:
            # v12 increments these counters inside _galaxy_slate_payload before returning.
            # Restore them so a rejected tiny Galaxy slate is not counted as a hit or an avoided API call.
            _restore_metric("galaxy_slate_hits", before_hits)
            _restore_metric("api_slate_calls_avoided", before_avoided)
            _bump_metric("galaxy_slate_floor_rejections")
            v12._SLATE_METRICS["galaxy_slate_floor_last_rejected_count"] = count
            return None
        return payload

    v12._galaxy_slate_payload = floor_guarded_galaxy_slate_payload
    try:
        payload = await v90.run_tick()
    finally:
        v12._galaxy_slate_payload = original_galaxy_slate_payload

    _annotate_payload(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
