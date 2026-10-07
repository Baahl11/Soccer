from __future__ import annotations

import argparse
import json
import os
from typing import Any

from mcp_gateway import formation_fm5_raw_projection_v1 as fm5_raw
from mcp_gateway import formation_fm5_readiness_gate_v1 as fm5_gate
from mcp_gateway import formation_fm6_market_validation_v1 as fm6
from mcp_gateway import formation_fm7_promotion_review_v1 as fm7
from mcp_gateway import formation_matchup_fm4_style_ablation_v1 as fm4

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_LIFECYCLE_HEALTH_V1.0.0"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _load(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def build_health(
    *,
    fm4_report: dict[str, Any],
    fm5_readiness: dict[str, Any],
    prospective: dict[str, Any],
    personnel_backfill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    f4 = _dict(fm4_report)
    f5 = _dict(fm5_readiness)
    pros = _dict(prospective)
    personnel = _dict(personnel_backfill)

    style = _dict(f4.get("style_profile"))
    density = _dict(style.get("prior_density"))
    overlay = _dict(f4.get("personnel_overlay"))
    coverage = _dict(overlay.get("coverage"))
    endpoint = _dict(pros.get("endpoint_summary"))
    graded = _dict(endpoint.get("graded_bet_sample"))

    style_n = _int(density.get("rows_with_both_min_field_n_ge_3"))
    style_target = _int(f5.get("fm4", {}).get("minimum_prior_style_rows")) or 100
    graded_rows = _int(graded.get("rows"))
    graded_target = _int(endpoint.get("graded_bet_target")) or fm7.MIN_GRADED_BETS
    ready_components = [
        str(value) for value in (f5.get("ready_components") or []) if value
    ]

    personnel_artifact = bool(personnel)
    personnel_captured = _int(personnel.get("captured"))
    personnel_history_fixtures = _int(
        personnel.get("materialized_personnel_history_fixture_count")
    )

    fm6_status = (
        "READY_FOR_EXACT_MARKET_RESEARCH"
        if ready_components and f5.get("integration_review_allowed") is True
        else "BLOCKED_WAITING_FM5_SPORTING_COMPONENT"
    )
    fm7_status = (
        "AWAITING_FM6_VALIDATION_AND_PROSPECTIVE_SAMPLE"
        if ready_components
        else "BLOCKED_UPSTREAM_FM5"
    )

    engineering_contracts = {
        "FM4_STYLE_PERSONNEL_ABLATION": {
            "model_version": fm4.MODEL_VERSION,
            "materialized": f4.get("model_version") == fm4.MODEL_VERSION,
        },
        "FM5_READINESS_GATE": {
            "model_version": fm5_gate.MODEL_VERSION,
            "materialized": f5.get("model_version") == fm5_gate.MODEL_VERSION,
        },
        "FM5_RAW_SPORT_PROJECTION": {
            "model_version": fm5_raw.MODEL_VERSION,
            "materialized": True,
            "production_weight": 0.0,
        },
        "FM6_EXACT_MARKET_VALIDATION": {
            "model_version": fm6.MODEL_VERSION,
            "materialized": True,
            "production_weight": 0.0,
        },
        "FM7_PROMOTION_REVIEW": {
            "model_version": fm7.MODEL_VERSION,
            "materialized": True,
            "automatic_activation_allowed": False,
        },
    }

    software_ready = all(
        row.get("materialized") is True for row in engineering_contracts.values()
    )

    evidence_blockers: list[str] = []
    if style_n < style_target:
        evidence_blockers.append(f"FM4_STYLE_HISTORY_{style_n}_LT_{style_target}")
    evidence_blockers.extend(str(v) for v in (f5.get("blockers") or []) if v)
    if not personnel_artifact:
        evidence_blockers.append("FM4_PERSONNEL_BACKFILL_ARTIFACT_NOT_MATERIALIZED")
    if graded_rows < graded_target:
        evidence_blockers.append(
            f"PROSPECTIVE_GRADED_BETS_{graded_rows}_LT_{graded_target}"
        )
    if endpoint.get("material_recalibration_allowed") is not True:
        evidence_blockers.append("MATERIAL_RECALIBRATION_NOT_ALLOWED")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": (
            "SOFTWARE_CONTRACTS_COMPLETE_EVIDENCE_ACCUMULATING"
            if software_ready
            else "SOFTWARE_CONTRACTS_INCOMPLETE"
        ),
        "governing_policy": "SPORT_FIRST_MARKET_SECOND",
        "software_contracts_complete": software_ready,
        "engineering_contracts": engineering_contracts,
        "phases": {
            "FM4": {
                "status": f4.get("status") or "NOT_MATERIALIZED",
                "style_history_rows_both_teams_n_ge_3": style_n,
                "style_history_target": style_target,
                "personnel_current_both_xi_rows": _int(
                    coverage.get("current_both_xi_confirmed_rows")
                ),
                "personnel_both_prior_xi_rows": _int(
                    coverage.get("rows_with_both_prior_confirmed_xi")
                ),
                "personnel_outcome_ablation_ready": (
                    overlay.get("outcome_ablation_ready") is True
                ),
                "personnel_backfill_artifact_materialized": personnel_artifact,
                "personnel_backfill_last_run_captured": personnel_captured,
                "personnel_history_fixture_count": personnel_history_fixtures,
                "production_enabled": False,
                "decision_weight": 0.0,
            },
            "FM5": {
                "status": f5.get("status") or "NOT_MATERIALIZED",
                "ready_components": ready_components,
                "integration_review_allowed": (
                    f5.get("integration_review_allowed") is True
                ),
                "automatic_integration_allowed": False,
                "raw_projection_contract_materialized": True,
                "production_enabled": False,
                "decision_weight": 0.0,
            },
            "FM6": {
                "status": fm6_status,
                "contract_materialized": True,
                "exact_market_only": True,
                "market_can_create_sporting_thesis": False,
                "production_enabled": False,
                "decision_weight": 0.0,
            },
            "FM7": {
                "status": fm7_status,
                "contract_materialized": True,
                "prospective_graded_bet_rows": graded_rows,
                "prospective_graded_bet_target": graded_target,
                "material_recalibration_allowed": (
                    endpoint.get("material_recalibration_allowed") is True
                ),
                "manual_approval_required": True,
                "automatic_activation_allowed": False,
                "production_enabled": False,
            },
        },
        "evidence_blockers": sorted(set(evidence_blockers)),
        "provider_requests_added": 0,
        "market_prices_consumed_to_create_sporting_projection": False,
        "historical_predictions_rewritten": False,
        "model_weights_changed": False,
        "thresholds_changed": False,
        "canonical_bet_logic_changed": False,
        "production_promotion_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize formation lifecycle software/evidence health."
    )
    parser.add_argument("--fm4-report", required=True)
    parser.add_argument("--fm5-readiness", required=True)
    parser.add_argument("--prospective", required=True)
    parser.add_argument("--personnel-backfill")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_health(
        fm4_report=_load(args.fm4_report),
        fm5_readiness=_load(args.fm5_readiness),
        prospective=_load(args.prospective),
        personnel_backfill=_load(args.personnel_backfill),
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
