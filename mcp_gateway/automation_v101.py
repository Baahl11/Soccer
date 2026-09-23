from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v100 as v100
from mcp_gateway import model_disagreement_v4

MODEL_VERSION = v100.MODEL_VERSION
AUTOMATION_VERSION = "4.10.0-v4.014"


def _annotate_v4_014(payload: dict[str, Any]) -> None:
    payload["v4_014_model_disagreement_engine"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_FRAMEWORK_IMPLEMENTED",
        "model_version": model_disagreement_v4.MODEL_VERSION,
        "required_components": list(model_disagreement_v4.REQUIRED_COMPONENTS),
        "probability_targets": list(model_disagreement_v4.PROBABILITY_KEYS),
        "classification_thresholds": {
            "low_max_range": model_disagreement_v4.LOW_MAX_RANGE,
            "moderate_max_range": model_disagreement_v4.MODERATE_MAX_RANGE,
        },
        "same_fixture_component_predictions_required": True,
        "market_fields_used": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Disagreement is measured only from same-fixture component probabilities. "
            "The engine reports dispersion and worst-target divergence; it does not "
            "change picks, confidence, prices, thresholds or production weights."
        ),
    }
    payload["v4_014_provider_requests_added"] = 0
    payload["v4_014_model_weights_changed"] = False
    payload["v4_014_canonical_bet_logic_changed"] = False
    payload["v4_014_checkpoint"] = (
        "MODEL DISAGREEMENT ENGINE IMPLEMENTED: cross-model probability dispersion "
        "is measured per target with LOW/MODERATE/HIGH research classifications. "
        "No market input, provider calls, model weights, tiers, stakes, thresholds "
        "or canonical BET logic changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v100.run_tick()
    _annotate_v4_014(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
