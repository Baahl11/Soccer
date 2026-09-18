from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v83 as v83

MODEL_VERSION = v83.MODEL_VERSION
AUTOMATION_VERSION = "3.58.2"


async def run_tick() -> dict[str, Any]:
    payload = await v83.run_tick()

    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    policy = (
        builder.get("market_backed_sgp_policy")
        if isinstance(builder.get("market_backed_sgp_policy"), dict)
        else {}
    )

    payload["v3582_provider_requests_added"] = 0
    payload["v3582_model_weights_changed"] = False
    payload["v3582_canonical_bet_logic_changed"] = False
    payload["v3582_ft_goals_galaxy_alignment"] = {
        "schema_version": "1.0.0",
        "status": "LIVE_CODE_PENDING_NATURAL_VALIDATION",
        "ft_goals_leg_source": policy.get("ft_goals_leg_source")
        or "CANONICAL_MARKET_DECISION_LADDER",
        "ft_goals_probability_source": policy.get("ft_goals_probability_source")
        or "P_SHRUNK_MATCHED_TO_THE_SAME_OBSERVED_BOOKMAKER_QUOTE",
        "ft_goals_score_matrix_role": policy.get("ft_goals_score_matrix_role")
        or "JOINT_SGP_CORRELATION_ONLY",
        "rolling_pool_uses_same_leg_source": True,
        "synthetic_fixed_ft_goals_ladder_allowed": False,
        "provider_requests_added": 0,
    }
    payload["v3582_ft_goals_galaxy_checkpoint"] = (
        "GALAXY FT_GOALS LEGS NOW COME FROM THE CANONICAL OBSERVED MARKET_DECISION LADDER "
        "(EXACT LINE + BOOKMAKER + P_SHRUNK); DIRECT SCORE MATRIX REMAINS THE CORRELATION-AWARE "
        "JOINT SGP ENGINE, NOT A SECOND FT_GOALS MARGINAL LADDER"
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
