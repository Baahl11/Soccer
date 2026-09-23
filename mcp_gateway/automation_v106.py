from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v105 as v105
from mcp_gateway import team_totals_oos_v4

MODEL_VERSION = v105.MODEL_VERSION
AUTOMATION_VERSION = "4.15.0-v4.019"


def _annotate_v4_019(payload: dict[str, Any]) -> None:
    payload["v4_019_team_totals_oos_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": team_totals_oos_v4.MODEL_VERSION,
        "minimum_research_fixtures": team_totals_oos_v4.MIN_RESEARCH_FIXTURES,
        "minimum_actionable_review_fixtures": team_totals_oos_v4.MIN_ACTIONABLE_REVIEW_FIXTURES,
        "minimum_family_specific_true_clv_rows": team_totals_oos_v4.MIN_TRUE_CLV_ROWS,
        "required_lines": list(team_totals_oos_v4.REQUIRED_LINES),
        "family_specific_true_clv_required": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-019 validates Team Totals OOS evidence separately by team role and line. "
            "Existing probability sample can satisfy research sample size while promotion remains "
            "blocked until exact Team Total market history/true CLV and the source promotion gate pass."
        ),
    }
    payload["v4_019_provider_requests_added"] = 0
    payload["v4_019_model_weights_changed"] = False
    payload["v4_019_canonical_bet_logic_changed"] = False
    payload["v4_019_checkpoint"] = (
        "TEAM TOTALS OOS VALIDATION GATE IMPLEMENTED. Home/Away 0.5/1.5/2.5 research rows, "
        "role-line calibration and family-specific true CLV are reviewed conservatively. "
        "No automatic promotion, tier, stake, model-weight or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v105.run_tick()
    _annotate_v4_019(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
