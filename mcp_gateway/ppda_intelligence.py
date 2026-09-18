from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
APPROVED_SOURCES = ("API-Football v3", "GalaxyParlay persisted contract")


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "APPROVED_SOURCES_PPDA_NOT_EXPOSED",
        "approved_sources": list(APPROVED_SOURCES),
        "api_football": {
            "status": "NO_NORMALIZED_PPDA_IN_CURRENT_FIXTURE_STATISTICS_CONTRACT",
            "available_inputs_are_sufficient_to_reconstruct_ppda": False,
        },
        "galaxyparlay": {
            "status": "PPDA_NOT_EXPOSED_IN_CURRENT_PERSISTED_INTEGRATION_CONTRACT",
        },
        "definition_gate": {
            "normalized_provider_definition_required": True,
            "spatial_zone_definition_required": True,
            "defensive_action_denominator_definition_required": True,
            "whole_match_passes_divided_by_tackles_allowed": False,
        },
        "legacy_code_warning": (
            "Legacy tactical_analysis files read a ppda key when supplied by callers, "
            "but no approved ingestion path currently populates verified PPDA. "
            "That placeholder must not be treated as data."
        ),
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": "API_FOOTBALL_AND_CURRENT_GALAXYPARLAY_CONTRACT_DO_NOT_EXPOSE_NORMALIZED_PPDA",
        "policy": (
            "DO NOT INVENT PPDA FROM TOTAL PASSES, POSSESSION, TACKLES OR FOULS. "
            "ACTIVATE ONLY WITH AN APPROVED SOURCE THAT EXPOSES A NORMALIZED, AUDITABLE PPDA DEFINITION."
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
        event["ppda_intelligence"] = intel
        blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["ppda"] = intel
    return {
        "approved_source_policy": "API_FOOTBALL_PLUS_GALAXYPARLAY_ONLY",
        "source_blocked_events": blocked,
        "provider_requests_added": 0,
    }
