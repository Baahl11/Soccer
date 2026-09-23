from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v92 as v92

MODEL_VERSION = v92.MODEL_VERSION
AUTOMATION_VERSION = "4.2.0-v4.006"


def _annotate_v4_006(payload: dict[str, Any]) -> None:
    fair = payload.get("fair_scheduler")
    if not isinstance(fair, dict):
        fair = {}
    payload["v4_006_fair_scheduler"] = {
        "schema_version": fair.get("schema_version") or "1.0.0",
        "policy": fair.get("policy") or "WEIGHTED_FAIR_60_ACTIONABLE_25_UNSEEN_15_EXPLORATORY",
        "weights_pct": fair.get("weights_pct") or {
            "actionable": 60,
            "unseen": 25,
            "exploratory": 15,
        },
        "eligible_queue_counts": fair.get("eligible_queue_counts") or {},
        "planned_slot_counts": fair.get("planned_slot_counts") or {},
        "processed_category_counts": fair.get("processed_category_counts") or {},
        "eligible_tier_ab_due_count": int(fair.get("eligible_tier_ab_due_count") or 0),
        "due_with_any_deep_dive_count": int(fair.get("due_with_any_deep_dive_count") or 0),
        "due_analyzed_pct": fair.get("due_analyzed_pct"),
        "analyzed_by_t90_pct": fair.get("analyzed_by_t90_pct"),
        "analyzed_by_t40_pct": fair.get("analyzed_by_t40_pct"),
        "starvation_count": int(fair.get("starvation_count") or 0),
        "avg_deep_dives_per_due_fixture": fair.get("avg_deep_dives_per_due_fixture"),
        "provider_calls_per_processed_fixture": fair.get("provider_calls_per_processed_fixture"),
        "actionable_due_count": int(fair.get("actionable_due_count") or 0),
        "actionable_processed_count": int(fair.get("actionable_processed_count") or 0),
        "actionable_refresh_pct": fair.get("actionable_refresh_pct"),
        "tier_ab_due_coverage_target_pct": 90.0,
        "shortlist_actionable_refresh_target_pct": 80.0,
        "tier_ab_due_coverage_target_met": fair.get("tier_ab_due_coverage_target_met"),
        "shortlist_actionable_refresh_target_met": fair.get("shortlist_actionable_refresh_target_met"),
        "state_count": int(fair.get("state_count") or 0),
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "status": "LIVE_VALIDATION",
    }
    payload["v4_006_provider_requests_added"] = 0
    payload["v4_006_model_weights_changed"] = False
    payload["v4_006_canonical_bet_logic_changed"] = False
    payload["v4_006_checkpoint"] = (
        "FAIR SCHEDULER LIVE: 60% actionable/shortlist, 25% unseen fixtures, "
        "15% exploratory repeats using durable fixture aging/deep-dive history. "
        "No model weights, canonical BET logic, tiers, stakes, or extra provider calls changed."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v92.run_tick()
    _annotate_v4_006(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
