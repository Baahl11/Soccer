from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v10 as v10

AUTOMATION_VERSION = "2.0.1"
MODEL_VERSION = "SOCCER EDGE ENGINE v1.1"

_ORIGINAL_V10_EVALUATE = v10._research_only_side_btts_and_galaxy_gate
_SPLIT_SAMPLE_REASON = "Minimum home/away split sample <5."
_GATE_SUBSTITUTION_REASON = (
    "GALAXY_SAMPLE_GATE: Galaxy publication/calibration gate substitutes only for the "
    "Soccer baseline split-sample gate; Galaxy source sample is NOT EXPOSED by the "
    "integration contract and is not fabricated."
)
_BLOCKING_REASON_FRAGMENTS = (
    "Data Tier A/B required",
    "Raw soccer projection unavailable",
    "Availability Confidence <0.85",
    "Material XI/goalkeeper verification incomplete",
    "requires RECHECK",
    "Discrepancy recheck required",
)


def _galaxy_gate_aware_evaluate(
    raw: dict[str, Any],
    market: Any,
    coverage: dict[str, Any],
    availability: float | None,
    stage: str,
    lineup: Any,
) -> dict[str, Any]:
    decision = _ORIGINAL_V10_EVALUATE(
        raw, market, coverage, availability, stage, lineup
    )
    if not isinstance(decision, dict):
        return decision
    if raw.get("sport_source") != "GALAXYPARLAY_PERSISTED":
        return decision
    if not raw.get("galaxy_actionable_total_model"):
        return decision

    out = dict(decision)
    rows = [dict(row) for row in (out.get("decisions") or []) if isinstance(row, dict)]
    promoted = 0

    for row in rows:
        if row.get("family") != "TOTAL":
            continue
        reasons = [str(reason) for reason in (row.get("reasons") or [])]
        if _SPLIT_SAMPLE_REASON not in reasons:
            continue

        reasons = [reason for reason in reasons if reason != _SPLIT_SAMPLE_REASON]
        if _GATE_SUBSTITUTION_REASON not in reasons:
            reasons.append(_GATE_SUBSTITUTION_REASON)
        row["reasons"] = reasons
        row["galaxy_sample_gate"] = "UPSTREAM_PUBLICATION_CALIBRATION_GATE"

        blocked = bool(row.get("discrepancy_recheck")) or any(
            fragment.lower() in reason.lower()
            for reason in reasons
            for fragment in _BLOCKING_REASON_FRAGMENTS
        )
        if blocked or row.get("classification") != "WATCH":
            continue

        edge = row.get("prob_edge_pp")
        tier = row.get("tier")
        if tier == "B":
            row["classification"] = "BET"
            row["stake_units"] = round(0.40 * float(availability or 0.0), 2)
            promoted += 1
        elif tier in {"A", "S"}:
            reasons.append(
                "Galaxy Tier A/S remains WATCH until Soccer-vs-Galaxy out-of-sample validation is sufficient."
            )
            row["reasons"] = reasons
        elif isinstance(edge, (int, float)) and edge > 0:
            row["classification"] = "LEAN"
            row["stake_units"] = 0.0
            promoted += 1

    rank = {"BET": 4, "WATCH": 3, "LEAN": 2, "PASS": 1}
    rows.sort(
        key=lambda item: (
            rank.get(str(item.get("classification")), 0),
            item.get("prob_edge_pp") if isinstance(item.get("prob_edge_pp"), (int, float)) else -999,
            item.get("estimated_ev") if isinstance(item.get("estimated_ev"), (int, float)) else -999,
        ),
        reverse=True,
    )
    best = rows[0] if rows else out.get("best_decision")
    out["decisions"] = rows[:20]
    out["best_decision"] = best
    out["status"] = best.get("classification") if isinstance(best, dict) else out.get("status", "WATCH")
    out["galaxy_sample_gate_substitutions"] = promoted
    return out


async def run_tick() -> dict[str, Any]:
    previous = v10._research_only_side_btts_and_galaxy_gate
    v10._research_only_side_btts_and_galaxy_gate = _galaxy_gate_aware_evaluate
    try:
        payload = await v10.run_tick()
    finally:
        v10._research_only_side_btts_and_galaxy_gate = previous

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["galaxy_sample_policy"] = (
        "DO_NOT_INVENT_SOURCE_SAMPLE; USE_GALAXY_PUBLICATION_CALIBRATION_GATE_ONLY_AS_"
        "SPLIT_SAMPLE_GATE_SUBSTITUTE_FOR_FT_TOTALS"
    )
    payload["tier_policy"] = (
        "GALAXY_FT_TOTALS_TIER_B_OR_LEAN_ACTIONABLE_IF_ALL_OTHER_GATES_PASS; "
        "A_S_WATCH_PENDING_CROSS_MODEL_VALIDATION"
    )
    return payload
