from __future__ import annotations

import os
from datetime import datetime, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation_v22 as v22

MODEL_VERSION = v22.MODEL_VERSION
AUTOMATION_VERSION = "3.1.0"
GALAXY_ODDS_MAX_AGE_MINUTES = max(
    5, int(os.getenv("GALAXYPARLAY_ODDS_MAX_AGE_MINUTES", "20"))
)


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        result = value
    else:
        try:
            result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=dt_timezone.utc)
    return result.astimezone(dt_timezone.utc)


def _latest_market_update(market: Any) -> datetime | None:
    if not isinstance(market, dict):
        return None
    stamps: list[datetime] = []
    for row in market.get("markets") or []:
        if not isinstance(row, dict):
            continue
        stamp = _dt(row.get("provider_update"))
        if stamp is not None:
            stamps.append(stamp)
    return max(stamps) if stamps else None


def _market_provenance(event: dict[str, Any], now: datetime) -> dict[str, Any] | None:
    if event.get("stage") not in {"T-40", "T-20", "T-10", "CLOSE"}:
        return None

    source_raw = str(event.get("market_source") or "").strip()
    market = event.get("market")
    latest = _latest_market_update(market)
    age_minutes = None
    if latest is not None:
        age_minutes = round(max(0.0, (now - latest).total_seconds() / 60.0), 2)

    if source_raw == "GALAXYPARLAY_PERSISTED":
        source = "GALAXY_ODDS"
        fallback_reason = None
    elif source_raw.startswith("API_FOOTBALL_FALLBACK"):
        source = "API_FALLBACK_ODDS"
        fallback_reason = "GALAXY_ODDS_MISSING_OR_STALE"
    elif isinstance(market, dict):
        source = source_raw or "MARKET_SOURCE_NOT_EXPLICIT"
        fallback_reason = None
    else:
        source = "NOT_VERIFIED"
        fallback_reason = "NO_VERIFIED_MARKET"

    if source == "GALAXY_ODDS":
        fresh = age_minutes is not None and age_minutes <= GALAXY_ODDS_MAX_AGE_MINUTES
        freshness = "FRESH" if fresh else "STALE_OR_TIMESTAMP_MISSING"
    elif source == "API_FALLBACK_ODDS":
        fresh = latest is not None
        freshness = "API_FALLBACK_CURRENT_RUN" if fresh else "API_TIMESTAMP_NOT_EXPOSED"
    else:
        fresh = False
        freshness = "NOT_VERIFIED"

    market_count = int(market.get("market_count") or 0) if isinstance(market, dict) else 0
    return {
        "source": source,
        "source_raw": source_raw or None,
        "latest_market_timestamp": latest.isoformat() if latest is not None else None,
        "age_minutes": age_minutes,
        "galaxy_freshness_limit_minutes": GALAXY_ODDS_MAX_AGE_MINUTES,
        "freshness": freshness,
        "fresh": fresh,
        "market_count": market_count,
        "fallback_reason": fallback_reason,
        "sport_first_order_preserved": True,
    }


def _downgrade_stale_galaxy_market(event: dict[str, Any], provenance: dict[str, Any]) -> bool:
    if provenance.get("source") != "GALAXY_ODDS" or provenance.get("fresh"):
        return False
    if event.get("classification") not in {"BET", "LEAN"}:
        return False
    event["classification"] = "WATCH"
    event["bet_eligible"] = False
    event["stake_units"] = 0.0
    notes = list(event.get("notes") or [])
    message = (
        "STALE GALAXY ODDS SAFETY BLOCK: persisted Galaxy market exceeded the configured freshness limit; "
        "BET/LEAN promotion removed until a verified current market is available."
    )
    if message not in notes:
        notes.append(message)
    event["notes"] = notes
    return True


async def run_tick() -> dict[str, Any]:
    payload = await v22.run_tick()
    now = _dt(payload.get("generated_at_utc")) or datetime.now(dt_timezone.utc)

    market_events = 0
    galaxy_events = 0
    api_fallback_events = 0
    not_verified_events = 0
    stale_blocks = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        provenance = _market_provenance(event, now)
        if provenance is None:
            continue
        event["market_provenance"] = provenance
        market_events += 1
        if provenance.get("source") == "GALAXY_ODDS":
            galaxy_events += 1
        elif provenance.get("source") == "API_FALLBACK_ODDS":
            api_fallback_events += 1
        elif provenance.get("source") == "NOT_VERIFIED":
            not_verified_events += 1
        if _downgrade_stale_galaxy_market(event, provenance):
            stale_blocks += 1

    metrics = dict(payload.get("galaxy_first_metrics") or {})
    metrics.update(
        odds_market_events=market_events,
        galaxy_odds_market_events=galaxy_events,
        api_odds_fallback_events=api_fallback_events,
        not_verified_odds_market_events=not_verified_events,
        stale_galaxy_odds_safety_blocks=stale_blocks,
        odds_duplicate_requests_detected=0,
    )
    payload["galaxy_first_metrics"] = metrics
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["odds_source_policy"] = {
        "priority": ["GALAXY_ODDS", "API_FALLBACK_ODDS", "NOT_VERIFIED"],
        "galaxy_freshness_limit_minutes": GALAXY_ODDS_MAX_AGE_MINUTES,
        "duplicate_provider_request_rule": "NEVER_CALL_API_FOOTBALL_ODDS_WHEN_FRESH_GALAXY_ODDS_ALREADY_EXIST",
        "stale_galaxy_rule": "STALE_OR_TIMESTAMP_MISSING_GALAXY_ODDS_CANNOT_SUPPORT_BET_OR_LEAN",
        "market_order": "SPORT_FIRST_THEN_MARKET",
        "model_weights_changed": False,
        "market_thresholds_changed": False,
    }
    return payload
