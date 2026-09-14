from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v7 as v7

MODEL_VERSION = "SOCCER EDGE ENGINE v1.0"
AUTOMATION_VERSION = "1.8.0"

_ORIGINAL_SAFE_MARKET_EVALUATE = v6._safe_market_evaluate
_SIDE_RESEARCH_REASON = (
    "1X2_RESEARCH_ONLY: the automated Soccer Edge side model is a limited goal-rate "
    "Poisson baseline and is not eligible for BET/LEAN until an advanced side model "
    "with verified out-of-sample validation is integrated."
)


def _research_only_side_evaluate(
    raw: dict[str, Any],
    market: Any,
    coverage: dict[str, Any],
    availability: float | None,
    stage: str,
    lineup: Any,
) -> dict[str, Any]:
    decision = _ORIGINAL_SAFE_MARKET_EVALUATE(
        raw, market, coverage, availability, stage, lineup
    )
    if not isinstance(decision, dict):
        return decision

    out = dict(decision)
    rows = [dict(row) for row in (out.get("decisions") or []) if isinstance(row, dict)]
    changed = 0
    for row in rows:
        if row.get("family") != "1X2":
            continue
        row["side_model_status"] = "RESEARCH_ONLY_PENDING_ADVANCED_VALIDATION"
        reasons = list(row.get("reasons") or [])
        if _SIDE_RESEARCH_REASON not in reasons:
            reasons.append(_SIDE_RESEARCH_REASON)
        row["reasons"] = reasons
        if row.get("classification") in {"BET", "LEAN"}:
            row["classification"] = "WATCH"
            row["stake_units"] = 0.0
            changed += 1

    rank = {"BET": 4, "WATCH": 3, "LEAN": 2, "PASS": 1}
    rows.sort(
        key=lambda d: (
            rank.get(str(d.get("classification")), 0),
            d.get("prob_edge_pp") if isinstance(d.get("prob_edge_pp"), (int, float)) else -999,
            d.get("estimated_ev") if isinstance(d.get("estimated_ev"), (int, float)) else -999,
        ),
        reverse=True,
    )
    best = rows[0] if rows else out.get("best_decision")
    out["decisions"] = rows[:20]
    out["best_decision"] = best
    out["status"] = best.get("classification") if isinstance(best, dict) else out.get("status", "WATCH")
    out["side_market_mode"] = "RESEARCH_ONLY_PENDING_ADVANCED_VALIDATION"
    out["side_decisions_downgraded"] = changed
    return out


async def run_tick() -> dict[str, Any]:
    previous = v6._safe_market_evaluate
    v6._safe_market_evaluate = _research_only_side_evaluate
    try:
        payload = await v7.run_tick()
    finally:
        v6._safe_market_evaluate = previous

    payload["version"] = AUTOMATION_VERSION
    payload["side_market_mode"] = "RESEARCH_ONLY_PENDING_ADVANCED_VALIDATION"
    payload["side_market_reason"] = _SIDE_RESEARCH_REASON
    payload["actionable_model_scope"] = "FT_TOTALS_BTTS_ONLY_UNTIL_ADVANCED_1X2_VALIDATED"
    return payload
