from __future__ import annotations

from typing import Any

from mcp_gateway import cards_rate_registry, red_cards_rate_registry

SCHEMA_VERSION = "1.0.0"


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build(event: dict[str, Any], yellow_registry: dict[str, Any] | None, red_registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    referee = str(fixture.get("referee") or "").strip()
    if not referee:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "referee": None,
            "reason": "CURRENT_FIXTURE_REFEREE_NOT_AVAILABLE",
            "actionable": False,
            "decision_weight": 0.0,
        }

    yrefs = yellow_registry.get("referees") if isinstance(yellow_registry, dict) and isinstance(yellow_registry.get("referees"), dict) else {}
    rrefs = red_registry.get("referees") if isinstance(red_registry, dict) and isinstance(red_registry.get("referees"), dict) else {}
    yrow = yrefs.get(referee) if isinstance(yrefs.get(referee), dict) else {}
    rrow = rrefs.get(referee) if isinstance(rrefs.get(referee), dict) else {}
    yn = int(yrow.get("n") or 0)
    rn = int(rrow.get("n") or 0)
    yellow_total = _num(yrow.get("total_yellow")) or 0.0
    red_events = _num(rrow.get("any_red_event")) or 0.0
    total_red = _num(rrow.get("total_red")) or 0.0
    yellow_avg = yellow_total / yn if yn > 0 else None
    red_match_rate = red_events / rn if rn > 0 else None
    red_avg = total_red / rn if rn > 0 else None

    sample = max(yn, rn)
    sample_band = "HIGH" if sample >= 30 else "MEDIUM" if sample >= 12 else "LOW"
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_PROFILE",
        "referee": referee,
        "assignment_source": "API_FIXTURE_ASSIGNMENT",
        "official_independent_verification": False,
        "historical_profile": {
            "yellow_sample_n": yn,
            "yellow_cards_per_match": round(yellow_avg, 4) if yellow_avg is not None else None,
            "red_sample_n": rn,
            "matches_with_any_red_rate": round(red_match_rate, 6) if red_match_rate is not None else None,
            "red_cards_per_match": round(red_avg, 6) if red_avg is not None else None,
            "sample_band": sample_band,
        },
        "feature_gates": {
            "yellow_cards_adjustment_eligible": yn >= 8,
            "red_cards_adjustment_eligible": rn >= 20,
            "standalone_bet_signal_allowed": False,
        },
        "missing": [
            "FOULS_PER_MATCH_REFEREE_HISTORY_NOT_IN_REGISTRY",
            "PENALTY_RATE_REFEREE_HISTORY_NOT_IN_REGISTRY",
            "INDEPENDENT_OFFICIAL_ASSIGNMENT_VERIFICATION_NOT_AUTOMATED",
        ],
        "actionable": False,
        "decision_weight": 0.0,
        "policy": "REFEREE IS A CONDITIONING FEATURE ONLY; NEVER A STANDALONE BET SIGNAL; USE ONLY WHEN SAMPLE GATE IS MET",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    yellow_registry = cards_rate_registry.load_registry()
    red_registry = red_cards_rate_registry.load_registry()
    assigned = profiled = yellow_eligible = red_eligible = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, yellow_registry, red_registry)
        event["referee_intelligence"] = intel
        if intel.get("referee"):
            assigned += 1
        if intel.get("status") == "LIVE_RESEARCH_PROFILE":
            profiled += 1
        gates = intel.get("feature_gates") if isinstance(intel.get("feature_gates"), dict) else {}
        if gates.get("yellow_cards_adjustment_eligible"):
            yellow_eligible += 1
        if gates.get("red_cards_adjustment_eligible"):
            red_eligible += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["referee"] = intel
    return {
        "yellow_registry_loaded": bool(yellow_registry),
        "red_registry_loaded": bool(red_registry),
        "events_with_referee_assignment": assigned,
        "profiled_events": profiled,
        "yellow_adjustment_eligible_events": yellow_eligible,
        "red_adjustment_eligible_events": red_eligible,
        "provider_requests_added": 0,
    }
