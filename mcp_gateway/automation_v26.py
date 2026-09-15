from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation_v10 as v10
from mcp_gateway import automation_v23 as v23
from mcp_gateway import automation_v25 as v25

MODEL_VERSION = v25.MODEL_VERSION
AUTOMATION_VERSION = "3.2.1"

_ORIGINAL_GALAXY_AWARE_ODDS = v10._galaxy_aware_odds_7m
_ALLOWED_SOURCES = {"GALAXY_ODDS", "API_FALLBACK_ODDS", "NOT_VERIFIED"}
_MARKET_STAGES = {"T-40", "T-20", "T-10", "CLOSE"}


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        out = value
    else:
        try:
            out = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out.astimezone(dt_timezone.utc)


async def _tagged_galaxy_aware_odds(fixture_id: int, now: datetime) -> dict[str, Any]:
    """Reuse v10 exactly once and attach provenance metadata to its returned market.

    No provider request is added here. The source tag is derived from v10's own
    per-call metrics: Galaxy hit => persisted Galaxy odds; Galaxy stale/missing =>
    the original API-Football fallback path ran.
    """
    before_galaxy = int(v10._METRICS.get("galaxy_odds_used", 0))
    before_fallback = int(v10._METRICS.get("galaxy_odds_stale_or_missing", 0))
    result = await _ORIGINAL_GALAXY_AWARE_ODDS(fixture_id, now)
    if not isinstance(result, dict):
        return result

    out = dict(result)
    after_galaxy = int(v10._METRICS.get("galaxy_odds_used", 0))
    after_fallback = int(v10._METRICS.get("galaxy_odds_stale_or_missing", 0))
    if after_galaxy > before_galaxy:
        out["_soccer_edge_market_source"] = "GALAXYPARLAY_PERSISTED"
        out["_soccer_edge_source_reason"] = "GALAXY_FRESH_ODDS_REUSED"
    elif after_fallback > before_fallback:
        out["_soccer_edge_market_source"] = "API_FOOTBALL_FALLBACK_GALAXY_STALE_OR_MISSING"
        out["_soccer_edge_source_reason"] = "GALAXY_ODDS_MISSING_OR_STALE"
    return out


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


def _normalize_source(event: dict[str, Any]) -> tuple[str, str | None, str | None]:
    market = event.get("market") if isinstance(event.get("market"), dict) else None
    existing = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}

    source_raw = str(event.get("market_source") or "").strip() or None
    if not source_raw and market:
        source_raw = str(market.get("_soccer_edge_market_source") or "").strip() or None
    if not source_raw:
        source_raw = str(existing.get("source_raw") or "").strip() or None

    existing_source = str(existing.get("source") or "").strip()
    if existing_source in _ALLOWED_SOURCES and existing_source != "NOT_VERIFIED":
        return existing_source, source_raw, existing.get("fallback_reason")

    if source_raw == "GALAXYPARLAY_PERSISTED":
        return "GALAXY_ODDS", source_raw, None
    if source_raw and source_raw.startswith("API_FOOTBALL_FALLBACK"):
        return "API_FALLBACK_ODDS", source_raw, "GALAXY_ODDS_MISSING_OR_STALE"
    if market is None:
        return "NOT_VERIFIED", source_raw, "NO_VERIFIED_MARKET"
    return "NOT_VERIFIED", source_raw, "MARKET_PRESENT_BUT_SOURCE_NOT_EXPLICIT"


def _repair_payload(payload: dict[str, Any], now: datetime) -> None:
    market_events = 0
    galaxy_events = 0
    api_fallback_events = 0
    not_verified_events = 0
    stale_blocks = 0
    unverified_blocks = 0
    metadata_repairs = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("stage") not in _MARKET_STAGES:
            continue

        market_events += 1
        market = event.get("market") if isinstance(event.get("market"), dict) else None
        source, source_raw, fallback_reason = _normalize_source(event)
        if not event.get("market_source") and market and market.get("_soccer_edge_market_source"):
            metadata_repairs += 1
            event["market_source"] = market.get("_soccer_edge_market_source")

        latest = _latest_market_update(market)
        age_minutes = None
        if latest is not None:
            age_minutes = round(max(0.0, (now - latest).total_seconds() / 60.0), 2)

        if source == "GALAXY_ODDS":
            fresh = age_minutes is not None and age_minutes <= v23.GALAXY_ODDS_MAX_AGE_MINUTES
            freshness = "FRESH" if fresh else "STALE_OR_TIMESTAMP_MISSING"
            galaxy_events += 1
        elif source == "API_FALLBACK_ODDS":
            # Source metadata is attached only when the fallback helper actually ran
            # during this tick. provider_update may be older than the current call.
            fresh = market is not None
            freshness = "API_FALLBACK_CURRENT_RUN" if fresh else "API_FALLBACK_NO_MARKET"
            api_fallback_events += 1
        else:
            fresh = False
            freshness = "NOT_VERIFIED"
            not_verified_events += 1

        provenance = {
            "source": source,
            "source_raw": source_raw,
            "latest_market_timestamp": latest.isoformat() if latest is not None else None,
            "age_minutes": age_minutes,
            "galaxy_freshness_limit_minutes": v23.GALAXY_ODDS_MAX_AGE_MINUTES,
            "freshness": freshness,
            "fresh": fresh,
            "market_count": int(market.get("market_count") or 0) if market else 0,
            "fallback_reason": fallback_reason,
            "sport_first_order_preserved": True,
            "provenance_contract_version": "1.1.0",
        }
        event["market_provenance"] = provenance

        if source == "GALAXY_ODDS" and v23._downgrade_stale_galaxy_market(event, provenance):
            stale_blocks += 1
        elif source == "NOT_VERIFIED" and event.get("classification") in {"BET", "LEAN"}:
            event["classification"] = "WATCH"
            event["bet_eligible"] = False
            event["stake_units"] = 0.0
            notes = list(event.get("notes") or [])
            message = "UNVERIFIED MARKET SOURCE SAFETY BLOCK: BET/LEAN requires explicit verified market provenance."
            if message not in notes:
                notes.append(message)
            event["notes"] = notes
            unverified_blocks += 1

    accounted = galaxy_events + api_fallback_events + not_verified_events
    metrics = dict(payload.get("galaxy_first_metrics") or {})
    metrics.update(
        odds_market_events=market_events,
        galaxy_odds_market_events=galaxy_events,
        api_odds_fallback_events=api_fallback_events,
        not_verified_odds_market_events=not_verified_events,
        unclassified_odds_market_events=max(0, market_events - accounted),
        odds_provenance_accounting_valid=(accounted == market_events),
        odds_provenance_metadata_repairs=metadata_repairs,
        stale_galaxy_odds_safety_blocks=stale_blocks,
        unverified_market_source_safety_blocks=unverified_blocks,
    )
    payload["galaxy_first_metrics"] = metrics
    payload["odds_provenance_policy"] = (
        "EVERY_T40_T20_T10_CLOSE_EVENT_MUST_CLASSIFY_AS_GALAXY_ODDS_API_FALLBACK_ODDS_OR_NOT_VERIFIED; "
        "UNKNOWN_SOURCE_CAN_NEVER_SUPPORT_BET_OR_LEAN; NO_EXTRA_PROVIDER_REQUESTS_FOR_PROVENANCE"
    )


async def run_tick() -> dict[str, Any]:
    previous = v10._galaxy_aware_odds_7m
    v10._galaxy_aware_odds_7m = _tagged_galaxy_aware_odds
    try:
        payload = await v25.run_tick()
    finally:
        v10._galaxy_aware_odds_7m = previous

    now = _dt(payload.get("generated_at_utc")) or datetime.now(dt_timezone.utc)
    _repair_payload(payload, now)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
