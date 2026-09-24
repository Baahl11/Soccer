from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v111 as v111
from mcp_gateway import market_mismatch_v4

MODEL_VERSION = v111.MODEL_VERSION
AUTOMATION_VERSION = "4.21.0-phase16.market_mismatch"


def _annotate_phase16(payload: dict[str, Any]) -> None:
    rows = payload.get("match_table_rows") if isinstance(payload.get("match_table_rows"), list) else []
    scan = market_mismatch_v4.find_mismatches(rows, top_n=20)

    payload["phase16_market_mismatch_finder"] = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ENGINE_IMPLEMENTED",
        "model_version": market_mismatch_v4.MODEL_VERSION,
        "rows_analyzed": scan["rows_analyzed"],
        "fixtures_analyzed": scan["fixtures_analyzed"],
        "rankable_rows": scan["rankable_rows"],
        "non_rankable_rows": scan.get("non_rankable_rows", 0),
        "non_rankable_reason_counts": scan.get("non_rankable_reason_counts", {}),
        "non_rankable_reason_counts_by_family": scan.get("non_rankable_reason_counts_by_family", {}),
        "market_family_coverage": scan["market_family_coverage"],
        "rankable_market_family_coverage": scan["rankable_market_family_coverage"],
        "ranking_weights": scan["ranking_weights"],
        "calibrated_probability_required_for_ranking": True,
        "raw_legacy_edge_used_for_diagnostic_only": True,
        "correlation_suppression_enabled": True,
        "primary_candidate_count": len(scan["primary_candidates"]),
        "correlated_candidate_count": len(scan["correlated_candidates_suppressed"]),
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "runtime_prediction_weight": 0.0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "note": (
            "Phase 16 compares market families per fixture using sport confidence, calibrated edge, "
            "data quality, price quality and uncertainty. Rows without an explicit calibrated model "
            "probability remain diagnostic-only; legacy p_shrunk/prob_edge does not qualify for ranking. "
            "Correlated market families are suppressed from independent treatment."
        ),
    }
    payload["market_mismatch_rows"] = scan["primary_candidates"]
    payload["market_mismatch_correlated_suppressed"] = scan["correlated_candidates_suppressed"]
    payload["phase16_provider_requests_added"] = 0
    payload["phase16_model_weights_changed"] = False
    payload["phase16_canonical_bet_logic_changed"] = False
    payload["phase16_checkpoint"] = (
        "MARKET MISMATCH FINDER IMPLEMENTED. Ranking requires calibrated model probability plus "
        "fresh/fair market probability; sport confidence, data quality, price quality and uncertainty "
        "are combined and correlated markets are suppressed. No production promotion or BET logic change."
    )


async def run_tick() -> dict[str, Any]:
    payload = await v111.run_tick()
    _annotate_phase16(payload)
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
