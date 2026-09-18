from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.1.0"
APPROVED_SOURCES = ("API-Football v3", "GalaxyParlay persisted contract")


def _fixture(event: dict[str, Any]) -> dict[str, Any]:
    row = event.get("fixture")
    return row if isinstance(row, dict) else {}


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(event)
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    galaxy_source = str(raw.get("sport_source") or "")
    galaxy_xg = {
        "home": raw.get("raw_home_xg"),
        "away": raw.get("raw_away_xg"),
    }

    # Current GalaxyParlay integration explicitly returns these values as
    # "NOT EXPOSED BY INTEGRATION CONTRACT". Even if Soccer Edge has goal
    # lambdas, those are expected-goal-count model parameters, not shot-quality xG.
    galaxy_explicit = (
        galaxy_source == "GALAXYPARLAY_PERSISTED"
        and isinstance(galaxy_xg["home"], (int, float))
        and isinstance(galaxy_xg["away"], (int, float))
        and bool(raw.get("galaxy_xg_verified_source"))
    )

    if galaxy_explicit:
        status = "VERIFIED_GALAXYPARLAY_XG_AVAILABLE"
        source_status = "GALAXYPARLAY_EXPLICIT_VERIFIED_XG"
    else:
        status = "APPROVED_SOURCES_XG_NOT_EXPOSED"
        source_status = "DATA_BLOCKED"

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": status,
        "approved_sources": list(APPROVED_SOURCES),
        "api_football": {
            "status": "XG_NOT_EXPOSED_IN_CURRENT_APPROVED_FIXTURE_STATISTICS_CONTRACT",
            "normal_scheduler_source": True,
        },
        "galaxyparlay": {
            "status": source_status,
            "sport_source": galaxy_source or None,
            "contract_version": raw.get("galaxy_contract_version"),
            "reported_home_xg": galaxy_xg["home"],
            "reported_away_xg": galaxy_xg["away"],
            "requires_explicit_verified_xg_flag": True,
        },
        "internal_goal_model": {
            "raw_home_goal_rate": raw.get("raw_home_goal_rate"),
            "raw_away_goal_rate": raw.get("raw_away_goal_rate"),
            "may_be_relabeled_as_xg": False,
            "reason": "GOAL_LAMBDA_IS_NOT_SHOT_QUALITY_XG",
        },
        "xga_definition": "Opponent verified xG in the same fixture; never goals conceded or model lambda.",
        "canonical_goal_lambda_adjustment": 0.0,
        "canonical_model_weights_changed": False,
        "actionable": False,
        "decision_weight": 0.0,
        "block_reason": None if galaxy_explicit else "API_FOOTBALL_AND_CURRENT_GALAXYPARLAY_CONTRACT_DO_NOT_EXPOSE_VERIFIED_XG",
        "policy": (
            "APPROVED SOURCES ONLY: API-FOOTBALL + GALAXYPARLAY. "
            "DO NOT ADD A THIRD PROVIDER WITHOUT EXPLICIT PROJECT APPROVAL. "
            "INTERNAL GOAL LAMBDAS/GOALS/SHOTS MUST NEVER BE RELABELED AS xG."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool | str]:
    live = blocked = 0
    for event in payload.get("events") or []:
        if (
            not isinstance(event, dict)
            or event.get("event_type") != "SOCCER_REFRESH"
            or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}
        ):
            continue
        intel = build(event)
        event["xg_xga_intelligence"] = intel
        if intel.get("status") == "VERIFIED_GALAXYPARLAY_XG_AVAILABLE":
            live += 1
        else:
            blocked += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["xg_xga"] = intel
    return {
        "approved_source_policy": "API_FOOTBALL_PLUS_GALAXYPARLAY_ONLY",
        "live_research_events": live,
        "source_blocked_events": blocked,
        "provider_requests_added": 0,
    }
