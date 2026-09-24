from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v121 as v121
from mcp_gateway import automation_v92 as v92
from mcp_gateway import automation_v112 as v112
from mcp_gateway import price_resolver_v4

MODEL_VERSION = v121.MODEL_VERSION
AUTOMATION_VERSION = "4.31.0-price-resolver-v4"


def _annotate_checkpoint(payload: dict[str, Any]) -> None:
    resolution = payload.get("price_resolution_v4") if isinstance(payload.get("price_resolution_v4"), dict) else {}
    payload["price_resolution_checkpoint"] = {
        "schema_version": "1.0.0",
        "status": resolution.get("status") or "NOT_RUN",
        "model_version": price_resolver_v4.MODEL_VERSION,
        "candidate_rows": int(resolution.get("candidate_rows") or 0),
        "unique_candidate_fixtures": int(resolution.get("unique_candidate_fixtures") or 0),
        "api_calls_added": int(resolution.get("api_calls_added") or 0),
        "resolution_counts": dict(resolution.get("resolution_counts") or {}),
        "simulated_odds_allowed": False,
        "calibrated_probability_fabricated": False,
        "phase16_recomputed_after_price_resolution": True,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "production_promotion_allowed": False,
        "note": (
            "Price resolver uses real API-Football /odds fixture quotes or fresh Postgres market snapshots. "
            "Resolved rows are re-evaluated by execution-status separation and Phase16, but no calibrated "
            "probability is fabricated and no BET/tier/stake/model threshold is changed."
        ),
    }


async def run_tick() -> dict[str, Any]:
    payload = await v121.run_tick()

    await price_resolver_v4.resolve_payload(payload)

    # Price enrichment changes exact market/selection/price/fair probability on
    # research visibility rows. Recompute readiness and Phase16 against that
    # enriched point-in-time view. Both functions remain research-only.
    v92._annotate_decision_separation(payload)
    v112._annotate_phase16(payload)

    _annotate_checkpoint(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
