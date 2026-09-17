from __future__ import annotations

from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v51 as v51
from mcp_gateway import halftime_2h_intelligence

MODEL_VERSION = v51.MODEL_VERSION
AUTOMATION_VERSION = "3.28.0"

_ORIGINAL_STAGE_FOR = base._stage_for
_ORIGINAL_EVENT_FOR_FIXTURE = base._event_for_fixture


def _stage_for_with_halftime(minutes_to_kickoff: float, status: str) -> str | None:
    if str(status or "").upper() == "HT":
        return "HT"
    return _ORIGINAL_STAGE_FOR(minutes_to_kickoff, status)


async def _event_for_fixture_with_halftime(fx: dict[str, Any], stage: str, now) -> dict[str, Any]:
    if stage != "HT":
        return await _ORIGINAL_EVENT_FOR_FIXTURE(fx, stage, now)
    # Deliberately use only the already-fetched daily fixture row. This avoids
    # new provider requests and prevents a pregame bundle from being relabeled
    # as halftime intelligence.
    return {
        "event_type": "SOCCER_REFRESH",
        "stage": "HT",
        "fixture": fx,
        "coverage": {"known": False, "data_tier": "C", "source": "CURRENT_SLATE_FIXTURE_ONLY"},
        "classification": "WATCH",
        "bet_eligible": False,
        "availability_confidence": None,
        "market": "NOT VERIFIED",
        "lineups": "NOT VERIFIED_IN_HT_PATH",
        "injuries": "NOT VERIFIED_IN_HT_PATH",
        "notes": [
            "Halftime research path uses the verified current fixture status/score already present in the slate fetch.",
            "Red cards, halftime shots/SOT and live 2H market price are not fetched in v3.28 and remain explicit blockers.",
        ],
    }


async def run_tick() -> dict[str, Any]:
    previous_stage = base._stage_for
    previous_event = base._event_for_fixture
    base._stage_for = _stage_for_with_halftime
    base._event_for_fixture = _event_for_fixture_with_halftime
    try:
        payload = await v51.run_tick()
    finally:
        base._stage_for = previous_stage
        base._event_for_fixture = previous_event

    metrics = halftime_2h_intelligence.attach(payload)
    payload["two_h_halftime_intelligence"] = {
        "schema_version": halftime_2h_intelligence.SCHEMA_VERSION,
        **metrics,
        "provider_requests_added": 0,
        "canonical_model_weights_changed": False,
        "canonical_thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "galaxy_promotion_allowed": False,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "model": "PREGAME_2H_BASELINE_X_HALFTIME_STATE_MULTIPLIER_v0.1",
        "scope": "CURRENT_HT_SCORE_STATE_ONLY_V1; RED_CARDS_SHOTS_SOT_PENDING",
        "policy": "DEDICATED_HT_STAGE; NEVER RELABEL PREGAME_2H; ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY",
    }
    payload["v328_provider_requests_added"] = 0
    payload["v328_model_weights_changed"] = False
    payload["v328_canonical_bet_logic_changed"] = False
    payload["v328_halftime_2h_checkpoint"] = "DEDICATED_HT_LIVE_RESEARCH_PATH_ADDED; PRODUCTION_UNCHANGED"
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
