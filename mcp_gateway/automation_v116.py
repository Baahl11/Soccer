from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v115 as v115
from mcp_gateway import risk_engine_v4

MODEL_VERSION = v115.MODEL_VERSION
AUTOMATION_VERSION = "4.25.0-phase20.risk_engine"


def _annotate_phase20(payload: dict[str, Any]) -> None:
    payload["phase20_bankroll_risk_engine"] = {
        "schema_version": "1.0.0",
        "status": "ENGINE_IMPLEMENTED_LOCKED_UNTIL_PRODUCTION_VALIDATION",
        "model_version": risk_engine_v4.MODEL_VERSION,
        "risk_controls": [
            "unit_sizing",
            "fractional_kelly",
            "max_stake",
            "max_daily_exposure",
            "max_correlated_exposure",
            "league_exposure",
            "market_exposure",
            "drawdown_control",
            "stop_conditions",
        ],
        "hard_principles": {
            "no_chase": True,
            "no_martingale": True,
            "no_short_streak_stake_increase": True,
        },
        "manual_policy_configuration_required": True,
        "production_execution_enabled": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 20 implements fractional Kelly and exposure/drawdown controls but keeps "
            "staking locked until Phase 19 has a manually approved production market. No default "
            "production limits are invented; reviewed caps are required before execution."
        ),
    }
    payload["phase20_provider_requests_added"] = 0
    payload["phase20_model_weights_changed"] = False
    payload["phase20_canonical_bet_logic_changed"] = False
    payload["phase20_checkpoint"] = (
        "BANKROLL/RISK ENGINE IMPLEMENTED AND LOCKED. Fractional Kelly and hard exposure caps "
        "exist, but no live stake can be produced without a production-validated market and an "
        "explicit reviewed risk policy. No chase, martingale or short-streak stake increase."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v115.run_tick()
    _annotate_phase20(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
