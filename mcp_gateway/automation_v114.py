from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v113 as v113
from mcp_gateway import oos_backtest_v4

MODEL_VERSION = v113.MODEL_VERSION
AUTOMATION_VERSION = "4.23.0-phase18.oos_backtest"


def _annotate_phase18(payload: dict[str, Any]) -> None:
    payload["phase18_oos_backtest_framework"] = {
        "schema_version": "1.0.0",
        "status": "FRAMEWORK_IMPLEMENTED",
        "model_version": oos_backtest_v4.MODEL_VERSION,
        "chronological_splits_supported": True,
        "rolling_windows": list(oos_backtest_v4.ROLLING_WINDOWS),
        "walk_forward_validation_supported": True,
        "no_training_performance_promotion": True,
        "timestamp_disciplines_required": [
            "bookmaker",
            "lineup",
            "feature",
        ],
        "metrics": [
            "brier",
            "log_loss",
            "calibration",
            "mae",
            "rmse",
            "hit_rate",
            "roi",
            "clv",
            "drawdown",
            "max_losing_streak",
            "volatility",
        ],
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 18 formalizes chronological splits, rolling windows, walk-forward evaluation "
            "and timestamp discipline. Settlement data can measure hit rate/ROI/drawdown/streak/"
            "volatility; probability metrics require frozen point-in-time prediction snapshots."
        ),
    }
    payload["phase18_provider_requests_added"] = 0
    payload["phase18_model_weights_changed"] = False
    payload["phase18_canonical_bet_logic_changed"] = False
    payload["phase18_checkpoint"] = (
        "OOS/BACKTEST FRAMEWORK IMPLEMENTED. Chronological/rolling/walk-forward evaluation and "
        "anti-leakage timestamp discipline are explicit; realized risk metrics are separated from "
        "probability calibration metrics. No production promotion or BET logic change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v113.run_tick()
    _annotate_phase18(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
