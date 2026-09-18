from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v86 as v86
from mcp_gateway import galaxy_score_matrix_expansion

MODEL_VERSION = v86.MODEL_VERSION
AUTOMATION_VERSION = "3.60.0"


async def run_tick() -> dict[str, Any]:
    payload = await v86.run_tick()
    metrics = galaxy_score_matrix_expansion.attach(payload)

    payload["v360_provider_requests_added"] = 0
    payload["v360_model_weights_changed"] = False
    payload["v360_canonical_thresholds_changed"] = False
    payload["v360_canonical_bet_logic_changed"] = False
    payload["v360_primary_galaxy_feed_promotion"] = False
    payload["v360_score_matrix_expansion"] = {
        "schema_version": galaxy_score_matrix_expansion.SCHEMA_VERSION,
        **metrics,
        "status": "LIVE_RESEARCH_CODE_PENDING_NATURAL_VALIDATION",
        "binary_score_matrix_families_added": [
            "ONE_X_TWO",
            "TEAM_TOTAL_HOME",
            "TEAM_TOTAL_AWAY",
            "CORRECT_SCORE",
            "ASIAN_HANDICAP_HALF",
        ],
        "settlement_aware_families_added": [
            "DNB",
            "ASIAN_HANDICAP_INTEGER_OR_QUARTER",
        ],
        "same_game_marginal_multiplication_allowed": False,
        "exact_observed_fresh_quote_required": True,
        "candidate_merge_into_primary_galaxy_feed": False,
        "production_promotion_allowed": False,
        "provider_requests_added": 0,
    }
    payload["v360_score_matrix_checkpoint"] = (
        "#44 SCORE-MATRIX EXPANSION: 1X2, TEAM TOTALS, CORRECT SCORE AND HALF-GOAL "
        "ASIAN HANDICAP USE DIRECT SCORE-MATRIX INTERSECTION; DNB AND INTEGER/QUARTER "
        "ASIAN HANDICAP USE SETTLEMENT-AWARE SCORE-MATRIX DIAGNOSTICS. NO SAME-GAME "
        "MARGINAL MULTIPLICATION, NO NEW PROVIDER CALLS, NO PRIMARY GALAXY PROMOTION."
    )
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
