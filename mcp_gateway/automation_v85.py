from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v84 as v84
from mcp_gateway import quote_freshness

MODEL_VERSION = v84.MODEL_VERSION
AUTOMATION_VERSION = "3.58.3"


async def run_tick() -> dict[str, Any]:
    payload = await v84.run_tick()
    metrics = payload.get("galaxy_first_metrics") if isinstance(payload.get("galaxy_first_metrics"), dict) else {}
    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    pool = builder.get("rolling_leg_pool") if isinstance(builder.get("rolling_leg_pool"), dict) else {}

    payload["v3583_provider_requests_added"] = 0
    payload["v3583_model_weights_changed"] = False
    payload["v3583_canonical_bet_logic_changed"] = False
    payload["v3583_quote_freshness"] = {
        "schema_version": "1.0.0",
        "status": "LIVE_CODE_PENDING_NATURAL_VALIDATION",
        "freshness_anchor": "PROVIDER_UPDATE",
        "capture_time_refreshes_quote_age": False,
        "max_age_minutes": quote_freshness.DEFAULT_MAX_AGE_MINUTES,
        "max_future_skew_minutes": quote_freshness.MAX_FUTURE_SKEW_MINUTES,
        "missing_or_unparseable_timestamp_is_fresh": False,
        "api_fallback_current_run_auto_fresh": False,
        "canonical_stale_quote_decisions_downgraded": int(
            metrics.get("stale_provider_quote_decisions_downgraded") or 0
        ),
        "stale_verified_market_safety_blocks": int(
            metrics.get("stale_verified_market_safety_blocks") or 0
        ),
        "rolling_pool_active_legs": int(pool.get("active_leg_count") or 0),
        "provider_requests_added": 0,
    }
    payload["v3583_quote_freshness_checkpoint"] = (
        "QUOTE AGE IS ANCHORED TO PROVIDER_UPDATE FOR GALAXY AND API FALLBACK; "
        "CURRENT WORKER CAPTURE TIME NEVER REJUVENATES STALE ODDS; CANONICAL BET/LEAN "
        "AND GALAXY ROLLING CANDIDATES FAIL CLOSED ON STALE OR MISSING PROVIDER TIMESTAMPS"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
