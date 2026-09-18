from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
APPROVED_SOURCES = ("API-Football v3", "GalaxyParlay persisted contract")


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "APPROVED_SOURCES_FIELD_TILT_NOT_EXPOSED",
        "approved_sources": list(APPROVED_SOURCES),
        "api_football": {
            "status": "TOTAL_POSSESSION_AVAILABLE_BUT_FIELD_TILT_INPUTS_NOT_EXPOSED",
            "total_ball_possession_may_be_relabeled_as_field_tilt": False,
        },
        "galaxyparlay": {
            "status": "FIELD_TILT_NOT_EXPOSED_IN_CURRENT_PERSISTED_INTEGRATION_CONTRACT",
        },
        "definition_gate": {
            "territorial_zone_definition_required": True,
            "touch_or_pass_denominator_definition_required": True,
            "home_away_symmetric_measurement_required": True,
            "plain_possession_percentage_is_sufficient": False,
        },
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": "APPROVED_SOURCES_DO_NOT_EXPOSE_A_CONSISTENT_FIELD_TILT_DEFINITION_OR_INPUTS",
        "policy": (
            "DO NOT RELABEL TOTAL POSSESSION AS FIELD TILT. ACTIVATE ONLY WITH AN APPROVED "
            "SOURCE THAT EXPOSES CONSISTENT TERRITORIAL POSSESSION/TOUCH/PASS INPUTS AND AN AUDITABLE DEFINITION."
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
        event["field_tilt_intelligence"] = intel
        blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["field_tilt"] = intel
    return {
        "approved_source_policy": "API_FOOTBALL_PLUS_GALAXYPARLAY_ONLY",
        "source_blocked_events": blocked,
        "provider_requests_added": 0,
    }
