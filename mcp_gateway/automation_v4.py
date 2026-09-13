from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v3 as v3

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
AUTOMATION_VERSION = "1.4.0"

# Scheduler-side pacing. API-Football exposes a high minute ceiling, but bursts
# can still trigger provider-side rate limiting. Keep one request in flight at a
# time and space starts far enough apart to leave headroom for interactive MCP
# calls and provider jitter.
MIN_REQUEST_INTERVAL_SECONDS = float(os.getenv("SOCCER_EDGE_MIN_REQUEST_INTERVAL_SECONDS", "0.75"))
RATE_LIMIT_MAX_RETRIES = int(os.getenv("SOCCER_EDGE_RATE_LIMIT_MAX_RETRIES", "2"))
RATE_LIMIT_BACKOFF_SECONDS = float(os.getenv("SOCCER_EDGE_RATE_LIMIT_BACKOFF_SECONDS", "4"))

_REQUEST_LOCK = asyncio.Lock()
_LAST_REQUEST_STARTED = 0.0
_ORIGINAL_EVALUATE_MARKET = v2.evaluate_market


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "too many requests" in text or "ratelimit" in text or "rate limit" in text


async def _paced_api_get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Apply hard daily/tick budgets plus scheduler-side pacing and 429 retry."""
    global _LAST_REQUEST_STARTED

    attempt = 0
    while True:
        async with _REQUEST_LOCK:
            if v2._API_CALLS_THIS_TICK >= v2.MAX_API_CALLS_PER_TICK:
                raise v2.TickBudgetExceeded(
                    f"Per-tick API budget reached ({v2.MAX_API_CALLS_PER_TICK}); lower-priority work deferred."
                )
            if v2._LAST_DAILY_REMAINING is not None and v2._LAST_DAILY_REMAINING <= 50:
                raise v2.TickBudgetExceeded("Daily API reserve guard reached; lower-priority work deferred.")

            now = time.monotonic()
            wait_for = MIN_REQUEST_INTERVAL_SECONDS - (now - _LAST_REQUEST_STARTED)
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            # Count every real provider request, including retries.
            v2._API_CALLS_THIS_TICK += 1
            _LAST_REQUEST_STARTED = time.monotonic()

            try:
                payload = await v2._ORIGINAL_API_GET(endpoint, params)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= RATE_LIMIT_MAX_RETRIES:
                    raise
                attempt += 1
                backoff = RATE_LIMIT_BACKOFF_SECONDS * (2 ** (attempt - 1))
            else:
                remaining = (payload.get("quota") or {}).get("daily_remaining")
                try:
                    if remaining is not None:
                        v2._LAST_DAILY_REMAINING = int(remaining)
                except (TypeError, ValueError):
                    pass
                return payload

        # Sleep outside the request lock so a future implementation can allow
        # other low-risk work while this endpoint cools down.
        await asyncio.sleep(backoff)


def _is_first_half_market(name: str) -> bool:
    n = (name or "").strip().lower()
    return (
        "first half" in n
        or "1st half" in n
        or "1st-half" in n
        or n.startswith("1h ")
        or " 1h " in n
    )


def _safe_evaluate_market(
    raw: dict[str, Any],
    market: Any,
    coverage: dict[str, Any],
    availability: float | None,
    stage: str,
    lineup: Any,
) -> dict[str, Any]:
    """Prevent full-match goal probabilities from being applied to 1H markets.

    The current automated projection is full-match only. First-half markets may
    still be stored as market snapshots, but they are not eligible for model
    comparison until a dedicated 1H projection exists.
    """
    if not isinstance(market, dict):
        return _ORIGINAL_EVALUATE_MARKET(raw, market, coverage, availability, stage, lineup)

    rows = list(market.get("markets") or [])
    first_half_rows = [r for r in rows if _is_first_half_market(r.get("market") or "")]
    supported_rows = [r for r in rows if not _is_first_half_market(r.get("market") or "")]

    if first_half_rows and not supported_rows:
        return {
            "status": "WATCH",
            "reason": "FIRST_HALF_MODEL_NOT_IMPLEMENTED",
            "decisions": [],
            "unsupported_market_count": len(first_half_rows),
        }

    filtered = dict(market)
    filtered["markets"] = supported_rows
    decision = _ORIGINAL_EVALUATE_MARKET(raw, filtered, coverage, availability, stage, lineup)
    if first_half_rows:
        decision = dict(decision)
        decision["first_half_markets_ignored"] = len(first_half_rows)
        decision["first_half_reason"] = "Dedicated 1H model required; full-match probabilities cannot be reused."
    return decision


async def run_tick() -> dict[str, Any]:
    # v2/v3 perform all provider work through the base module attribute, so one
    # patch covers fixtures, team stats, injuries, lineups, odds and postgame.
    base._api_get = _paced_api_get
    v2.evaluate_market = _safe_evaluate_market

    payload = await v3.run_tick()
    payload["version"] = AUTOMATION_VERSION
    payload["request_pacing_seconds"] = MIN_REQUEST_INTERVAL_SECONDS
    payload["rate_limit_max_retries"] = RATE_LIMIT_MAX_RETRIES
    payload["first_half_market_model"] = "BLOCKED_PENDING_EXPLICIT_1H_MODEL"
    return payload
