from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
APPROVED_SOURCES = ("API-Football v3", "GalaxyParlay persisted contract")


def _fixture(event: dict[str, Any]) -> dict[str, Any]:
    row = event.get("fixture")
    return row if isinstance(row, dict) else {}


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "APPROVED_SOURCES_NPXG_NOT_EXPOSED",
        "approved_sources": list(APPROVED_SOURCES),
        "api_football": {
            "status": "NPXG_NOT_EXPOSED_IN_CURRENT_APPROVED_CONTRACT",
            "penalty_events_available": True,
            "penalty_events_are_sufficient_to_compute_npxg": False,
        },
        "galaxyparlay": {
            "status": "NPXG_NOT_EXPOSED_IN_CURRENT_INTEGRATION_CONTRACT",
            "sport_source": raw.get("sport_source"),
            "contract_version": raw.get("galaxy_contract_version"),
        },
        "dependency": {
            "verified_xg_required": True,
            "verified_penalty_xg_or_native_npxg_required": True,
            "current_xg_status": (
                (event.get("xg_xga_intelligence") or {}).get("status")
                if isinstance(event.get("xg_xga_intelligence"), dict)
                else None
            ),
        },
        "penalty_exclusion": {
            "fixed_penalty_xg_subtraction_allowed": False,
            "goal_event_count_subtraction_allowed": False,
            "reason": (
                "NPXG REQUIRES NATIVE NON-PENALTY xG OR SHOT-LEVEL/PENALTY xG PROVENANCE; "
                "PENALTY GOAL/MISS EVENTS ALONE DO NOT IDENTIFY THE xG MASS TO REMOVE"
            ),
        },
        "npxga_definition": "Opponent verified npxG in the same fixture.",
        "canonical_goal_lambda_adjustment": 0.0,
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": "API_FOOTBALL_AND_CURRENT_GALAXYPARLAY_CONTRACT_DO_NOT_EXPOSE_VERIFIED_NPXG",
        "policy": (
            "APPROVED SOURCES ONLY: API-FOOTBALL + GALAXYPARLAY. "
            "NEVER DERIVE npxG BY SUBTRACTING A FIXED PENALTY VALUE OR BY RELABELING GOAL LAMBDAS."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | str]:
    blocked = 0
    for event in payload.get("events") or []:
        if (
            not isinstance(event, dict)
            or event.get("event_type") != "SOCCER_REFRESH"
            or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}
        ):
            continue
        intel = build(event)
        event["npxg_npxga_intelligence"] = intel
        blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["npxg_npxga"] = intel
    return {
        "approved_source_policy": "API_FOOTBALL_PLUS_GALAXYPARLAY_ONLY",
        "source_blocked_events": blocked,
        "provider_requests_added": 0,
    }
