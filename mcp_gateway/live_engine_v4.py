from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_LIVE_ENGINE_V4_1.0.0"

REQUIRED_LIVE_FIELDS = (
    "fixture_id",
    "minute",
    "score",
    "red_cards",
    "shots",
    "shots_on_target",
    "corners",
    "possession",
    "game_state",
    "pregame_prior",
)

LIVE_TARGETS = (
    "live_ft_goals",
    "live_2h_goals",
    "next_goal_later",
    "live_corners_later",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _pair(container: Any, home_key: str = "home", away_key: str = "away") -> tuple[float, float] | None:
    if not isinstance(container, dict):
        return None
    home = _num(container.get(home_key))
    away = _num(container.get(away_key))
    if home is None or away is None:
        return None
    return home, away


def validate_live_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []

    for field in REQUIRED_LIVE_FIELDS:
        if snapshot.get(field) is None:
            blockers.append(f"MISSING_{field.upper()}")

    minute = _num(snapshot.get("minute"))
    if minute is None or minute < 0 or minute > 130:
        blockers.append("INVALID_LIVE_MINUTE")

    score = _pair(snapshot.get("score"))
    if score is None or min(score) < 0:
        blockers.append("INVALID_LIVE_SCORE")

    for field in ("red_cards", "shots", "shots_on_target", "corners"):
        values = _pair(snapshot.get(field))
        if values is None or min(values) < 0:
            blockers.append(f"INVALID_{field.upper()}")

    possession = _pair(snapshot.get("possession"))
    if possession is None or min(possession) < 0 or max(possession) > 100:
        blockers.append("INVALID_POSSESSION")
    elif abs(sum(possession) - 100.0) > 5.0:
        warnings.append("POSSESSION_SUM_NOT_NEAR_100")

    xg = snapshot.get("xg")
    if xg is not None:
        if not bool(snapshot.get("xg_verified")):
            blockers.append("LIVE_XG_PRESENT_BUT_NOT_VERIFIED")
        values = _pair(xg)
        if values is None or min(values) < 0:
            blockers.append("INVALID_LIVE_XG")

    prior = snapshot.get("pregame_prior")
    if not isinstance(prior, dict) or not prior:
        blockers.append("PREGAME_PRIOR_REQUIRED")

    return {
        "valid": not blockers,
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
    }


def build_live_state(snapshot: dict[str, Any]) -> dict[str, Any]:
    validation = validate_live_snapshot(snapshot)
    if not validation["valid"]:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "LIVE_INPUT_BLOCKED",
            "actionable": False,
            "decision_weight": 0.0,
            "validation": validation,
            "production_promotion_allowed": False,
        }

    score = _pair(snapshot["score"])
    red = _pair(snapshot["red_cards"])
    shots = _pair(snapshot["shots"])
    sot = _pair(snapshot["shots_on_target"])
    corners = _pair(snapshot["corners"])
    possession = _pair(snapshot["possession"])
    assert score and red and shots and sot and corners and possession

    xg_pair = _pair(snapshot.get("xg")) if snapshot.get("xg") is not None else None
    minute = float(snapshot["minute"])

    derived = {
        "score_diff_home_minus_away": score[0] - score[1],
        "total_goals": score[0] + score[1],
        "red_card_diff_home_minus_away": red[0] - red[1],
        "shot_diff_home_minus_away": shots[0] - shots[1],
        "sot_diff_home_minus_away": sot[0] - sot[1],
        "corner_diff_home_minus_away": corners[0] - corners[1],
        "possession_diff_home_minus_away": possession[0] - possession[1],
        "xg_diff_home_minus_away": (xg_pair[0] - xg_pair[1]) if xg_pair else None,
    }

    target_status = {
        "live_ft_goals": "RESEARCH_INPUT_READY",
        "live_2h_goals": "RESEARCH_INPUT_READY" if minute >= 45 else "WAIT_HALFTIME_OR_LATER",
        "next_goal_later": "NOT_IMPLEMENTED_LATER",
        "live_corners_later": "NOT_IMPLEMENTED_LATER",
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "LIVE_RESEARCH_STATE_READY",
        "fixture_id": snapshot.get("fixture_id"),
        "minute": minute,
        "score": {"home": score[0], "away": score[1]},
        "game_state": snapshot.get("game_state"),
        "pregame_prior": snapshot.get("pregame_prior"),
        "derived": derived,
        "xg_used": bool(xg_pair and snapshot.get("xg_verified")),
        "target_status": target_status,
        "posterior_probability_update": None,
        "posterior_update_status": "BLOCKED_UNTIL_LIVE_MODEL_OOS_CALIBRATED",
        "actionable": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "provider_requests_added": 0,
        "policy": (
            "VERIFIED_LIVE_INPUTS_ONLY; PREGAME_PRIOR_MUST_BE_EXPLICIT; "
            "NO NUMERIC POSTERIOR UPDATE UNTIL LIVE OOS CALIBRATION EXISTS; "
            "NO LIVE BET_LEAN_PROMOTION"
        ),
        "validation": validation,
    }
