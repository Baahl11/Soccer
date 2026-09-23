from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v107 as v107
from mcp_gateway import two_h_oos_v4

MODEL_VERSION = v107.MODEL_VERSION
AUTOMATION_VERSION = "4.17.0-v4.021"


def _annotate_v4_021(payload: dict[str, Any]) -> None:
    payload["v4_021_2h_oos_validation"] = {
        "schema_version": "1.0.0",
        "status": "VALIDATION_GATE_IMPLEMENTED",
        "model_version": two_h_oos_v4.MODEL_VERSION,
        "minimum_oos": two_h_oos_v4.MIN_OOS,
        "minimum_family_specific_true_clv_rows": two_h_oos_v4.MIN_TRUE_CLV_ROWS,
        "dedicated_ht_research_stage_required": True,
        "family_specific_true_clv_required": True,
        "challenger_must_beat_baseline_on": ["brier", "log_loss", "mae_2h_lambda"],
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "V4-021 validates the halftime-conditioned 2H challenger against the pregame baseline. "
            "Promotion is blocked unless Brier/log-loss/MAE improve, a dedicated halftime research "
            "scheduler exists and 2H-specific true CLV evidence is available."
        ),
    }
    payload["v4_021_provider_requests_added"] = 0
    payload["v4_021_model_weights_changed"] = False
    payload["v4_021_canonical_bet_logic_changed"] = False
    payload["v4_021_checkpoint"] = (
        "2H OOS VALIDATION GATE IMPLEMENTED. Walk-forward challenger quality, dedicated halftime "
        "research path, live context and 2H-specific true CLV are required. No automatic promotion, "
        "tier, stake, model-weight or canonical BET change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v107.run_tick()
    _annotate_v4_021(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
