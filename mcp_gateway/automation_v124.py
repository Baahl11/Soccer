from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from mcp_gateway import automation_v123 as v123
from mcp_gateway import price_resolver_v4

MODEL_VERSION = v123.MODEL_VERSION
AUTOMATION_VERSION = "4.33.0-primary-clv-priority"

# Engineering priority only. This never changes a model score, market gate,
# threshold, stake, tier, or the number of provider calls available to the tick.
PRIMARY_MATURATION_FAMILY_ORDER = ("1X2", "BTTS", "FT_TOTALS")
_FAMILY_PRIORITY = {family: index for index, family in enumerate(PRIMARY_MATURATION_FAMILY_ORDER)}


def _canonical_family(value: Any) -> str | None:
    family = str(value or "").strip().upper()
    aliases = {
        "1X2": "1X2",
        "FT_1X2": "1X2",
        "FT_1X2_RESEARCH": "1X2",
        "BTTS": "BTTS",
        "FT_BTTS": "BTTS",
        "FT_BTTS_RESEARCH": "BTTS",
        "FT_TOTALS": "FT_TOTALS",
        "TOTAL": "FT_TOTALS",
        "FT_TOTALS_RESEARCH": "FT_TOTALS",
    }
    return aliases.get(family)


def _event_families(event: dict[str, Any]) -> list[str]:
    meta = event.get("primary_clv_maturation")
    if not isinstance(meta, dict):
        return []
    families: list[str] = []
    for signal in meta.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        family = _canonical_family(signal.get("market_family") or signal.get("family"))
        if family and family not in families:
            families.append(family)
    return families


def _event_fixture_id(event: dict[str, Any]) -> int:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    try:
        return int(fixture.get("fixture_id") or 0)
    except (TypeError, ValueError):
        return 0


def _event_kickoff(event: dict[str, Any]) -> str:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return str(fixture.get("kickoff") or "")


def _event_priority(event: dict[str, Any]) -> tuple[int, str, int]:
    families = _event_families(event)
    family_priority = min((_FAMILY_PRIORITY.get(family, 99) for family in families), default=99)
    return family_priority, _event_kickoff(event), _event_fixture_id(event)


def _prioritize_primary_backlog_result(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(result, dict):
        return result, {
            "candidate_count": 0,
            "reordered": False,
            "first_family_before": None,
            "first_family_after": None,
        }

    events = [event for event in (result.get("candidate_events") or []) if isinstance(event, dict)]
    before_ids = [_event_fixture_id(event) for event in events]
    before_family = (_event_families(events[0]) or [None])[0] if events else None
    ordered = sorted(events, key=_event_priority)
    after_ids = [_event_fixture_id(event) for event in ordered]
    after_family = (_event_families(ordered[0]) or [None])[0] if ordered else None

    prioritized = dict(result)
    prioritized["candidate_events"] = ordered
    telemetry = {
        "candidate_count": len(ordered),
        "reordered": before_ids != after_ids,
        "first_family_before": before_family,
        "first_family_after": after_family,
        "family_order": list(PRIMARY_MATURATION_FAMILY_ORDER),
    }
    return prioritized, telemetry


async def run_tick() -> dict[str, Any]:
    original_loader: Callable[..., dict[str, Any]] = price_resolver_v4._load_primary_clv_maturation_backlog
    priority_telemetry: dict[str, Any] = {
        "candidate_count": 0,
        "reordered": False,
        "first_family_before": None,
        "first_family_after": None,
        "family_order": list(PRIMARY_MATURATION_FAMILY_ORDER),
    }

    def prioritized_loader(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal priority_telemetry
        raw = original_loader(*args, **kwargs)
        prioritized, priority_telemetry = _prioritize_primary_backlog_result(raw)
        return prioritized

    price_resolver_v4._load_primary_clv_maturation_backlog = prioritized_loader
    try:
        payload = await v123.run_tick()
    finally:
        price_resolver_v4._load_primary_clv_maturation_backlog = original_loader

    payload["v206_primary_clv_maturation_priority"] = {
        "schema_version": "1.0.0",
        "status": "ACTIVE",
        **deepcopy(priority_telemetry),
        "provider_requests_added": 0,
        "provider_budget_changed": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "gates_changed": False,
        "canonical_bet_logic_changed": False,
        "policy": (
            "REORDER_EXISTING_PRIMARY_CLV_MATURATION_CANDIDATES_ONLY; "
            "1X2_THEN_BTTS_THEN_FT_TOTALS; SAME_ELIGIBILITY; SAME_LOOKAHEAD; "
            "SAME_PROVIDER_CALL_CAP; NO_SYNTHETIC_CLOSE"
        ),
    }
    payload["v206_checkpoint"] = (
        "PRIMARY CLV MATURATION PRIORITY: when existing eligible primary-market candidates compete "
        "for the same reserved price budget, 1X2 is attempted before BTTS and FT Totals. "
        "No candidate is created, no strict-close rule is relaxed, and no provider budget is increased."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
