from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"
APPROVED_SOURCES = ("API-Football v3", "GalaxyParlay persisted contract")


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "APPROVED_SOURCES_BOX_ENTRIES_NOT_EXPOSED",
        "approved_sources": list(APPROVED_SOURCES),
        "api_football": {
            "status": "SHOTS_INSIDE_BOX_MAY_EXIST_BUT_BOX_ENTRIES_NOT_EXPOSED",
            "shots_inside_box_may_be_relabeled_as_box_entries": False,
        },
        "galaxyparlay": {
            "status": "BOX_ENTRIES_NOT_EXPOSED_IN_CURRENT_PERSISTED_INTEGRATION_CONTRACT",
        },
        "definition_gate": {
            "explicit_penalty_area_entry_event_required": True,
            "entry_definition_required": True,
            "repeat_entry_counting_rule_required": True,
            "shots_inside_box_are_sufficient": False,
        },
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": "APPROVED_SOURCES_DO_NOT_EXPOSE_EXPLICIT_BOX_ENTRY_EVENTS_OR_METRIC",
        "policy": (
            "DO NOT RELABEL SHOTS INSIDE THE BOX, CROSSES, CORNERS OR POSSESSION AS BOX ENTRIES. "
            "ACTIVATE ONLY WITH AN APPROVED PROVIDER-DEFINED PENALTY-AREA ENTRY METRIC OR EVENT FEED."
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
        event["box_entries_intelligence"] = intel
        blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["box_entries"] = intel
    return {
        "approved_source_policy": "API_FOOTBALL_PLUS_GALAXYPARLAY_ONLY",
        "source_blocked_events": blocked,
        "provider_requests_added": 0,
    }
