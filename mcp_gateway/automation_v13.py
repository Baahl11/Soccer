from __future__ import annotations

from datetime import datetime
from typing import Any

from mcp_gateway import automation_v10 as v10
from mcp_gateway import automation_v12 as v12

MODEL_VERSION = "SOCCER EDGE ENGINE v1.2"
AUTOMATION_VERSION = "2.2.0"

_ORIGINAL_GALAXY_RAW_PROJECTION = v10._galaxy_raw_projection


def _valid_probability(value: Any) -> float | None:
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    if 0.0 <= p <= 1.0:
        return p
    return None


def _verified_galaxy_1x2(shadow: dict[str, Any], now: datetime) -> tuple[bool, str]:
    predictions = v10._latest_predictions(shadow, now)
    row = predictions.get("match_winner_v2_shadow")
    if not isinstance(row, dict):
        return False, "MISSING_MATCH_WINNER_V2_SHADOW"

    payload = v10._prediction_json(row)
    probs = [
        _valid_probability(payload.get("home_win")),
        _valid_probability(payload.get("draw")),
        _valid_probability(payload.get("away_win")),
    ]
    if any(value is None for value in probs):
        return False, "INVALID_MATCH_WINNER_V2_SHADOW_PROBABILITIES"
    if sum(probs) <= 0.0:
        return False, "ZERO_MASS_MATCH_WINNER_V2_SHADOW"
    return True, "VERIFIED_MATCH_WINNER_V2_SHADOW"


def _no_fabricated_1x2_projection(
    shadow: dict[str, Any], fx: dict[str, Any], now: datetime
) -> dict[str, Any] | None:
    raw = _ORIGINAL_GALAXY_RAW_PROJECTION(shadow, fx, now)
    if not isinstance(raw, dict):
        return raw

    valid, reason = _verified_galaxy_1x2(shadow, now)
    out = dict(raw)
    out["side_model_status"] = reason

    if valid:
        out["side_model_verified"] = True
        return out

    # Preserve independently modeled FT totals/BTTS, but never fabricate a side
    # probability vector when Galaxy has no verified 1X2 prediction.
    out["side_model_verified"] = False
    out["raw_home_win_prob"] = None
    out["raw_draw_prob"] = None
    out["raw_away_win_prob"] = None

    scores = dict(out.get("screen_scores") or {})
    scores["side_edge_score"] = 0.0
    scores["side_score_status"] = "NOT_MODELED_MISSING_VERIFIED_1X2"
    out["screen_scores"] = scores

    limitations = list(out.get("model_limitations") or [])
    message = (
        "1X2 NOT MODELED: Galaxy match_winner_v2_shadow is missing/invalid; "
        "the previous neutral 0.365/0.27/0.365 fallback is explicitly prohibited."
    )
    if message not in limitations:
        limitations.append(message)
    out["model_limitations"] = limitations
    return out


async def run_tick() -> dict[str, Any]:
    previous = v10._galaxy_raw_projection
    v10._galaxy_raw_projection = _no_fabricated_1x2_projection
    try:
        payload = await v12.run_tick()
    finally:
        v10._galaxy_raw_projection = previous

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["one_x_two_missing_prediction_policy"] = (
        "NO_FABRICATED_NEUTRAL_PRIOR; MISSING_OR_INVALID_GALAXY_1X2 => NOT_MODELED"
    )
    payload["one_x_two_mode"] = "RESEARCH_ONLY_PENDING_ADVANCED_VALIDATION"
    return payload
