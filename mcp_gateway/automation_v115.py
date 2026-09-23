from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v114 as v114
from mcp_gateway import promotion_framework_v4

MODEL_VERSION = v114.MODEL_VERSION
AUTOMATION_VERSION = "4.24.0-phase19.promotion_framework"


def _annotate_phase19(payload: dict[str, Any]) -> None:
    payload["phase19_promotion_framework"] = {
        "schema_version": "1.0.0",
        "status": "FRAMEWORK_IMPLEMENTED",
        "model_version": promotion_framework_v4.MODEL_VERSION,
        "states": list(promotion_framework_v4.STATES),
        "sample_policy": {
            "directional_read": promotion_framework_v4.DIRECTIONAL_READ_MIN,
            "tier_b_review_preferred": promotion_framework_v4.TIER_B_REVIEW_MIN,
            "tier_a_review": promotion_framework_v4.TIER_A_REVIEW_MIN,
            "tier_s_review": promotion_framework_v4.TIER_S_REVIEW_MIN,
            "model_weight_change": promotion_framework_v4.MODEL_WEIGHT_CHANGE_MIN,
        },
        "automatic_report": True,
        "manual_approval_required": True,
        "automatic_promotion_allowed": False,
        "automatic_demotion_allowed_under_safety_policy": True,
        "runtime_state_mutation_enabled": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 19 formalizes DORMANT/RESEARCH/SHADOW/LEAN_ELIGIBLE/TIER_B/TIER_A/TIER_S/DEMOTED "
            "with roadmap sample thresholds. Production promotion always requires manual approval; "
            "only safety demotion can be flagged automatically after sufficient negative evidence."
        ),
    }
    payload["phase19_provider_requests_added"] = 0
    payload["phase19_model_weights_changed"] = False
    payload["phase19_canonical_bet_logic_changed"] = False
    payload["phase19_checkpoint"] = (
        "PROMOTION FRAMEWORK IMPLEMENTED. Sample thresholds follow the master roadmap; no automatic "
        "promotion is allowed, manual approval is mandatory, and safety demotion can only be flagged "
        "for already-production markets with sufficient negative ROI and family CLV evidence."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v114.run_tick()
    _annotate_phase19(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
