from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_PHASE15_V4_1.0.0"
MIN_PROP_TRUE_CLV = 50
MIN_GK_PROFILES = 100

PROP_KEYS = (
    "shots",
    "sot",
    "goalscorer",
    "assists",
    "cards",
    "gk_saves",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_player_prop_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    selection = _norm(row.get("selection"))
    combined = f"{market} {selection}"
    tokens = (
        "player",
        "shots on target",
        "shot on target",
        "shots",
        "goalscorer",
        "anytime scorer",
        "assist",
        "goalkeeper saves",
        "gk saves",
        "player cards",
        "player booked",
    )
    return any(token in combined for token in tokens)


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_player_prop_market(row)]
    values = [_num(row.get("clv_probability_pp")) for row in selected]
    valid = [value for value in values if value is not None]
    return {
        "rows": len(selected),
        "unique_fixtures": len({row.get("fixture_id") for row in selected if row.get("fixture_id") is not None}),
        "avg_probability_clv_pp": round(sum(valid) / len(valid), 6) if valid else None,
        "positive_rows": sum(1 for value in valid if value > 0),
        "negative_rows": sum(1 for value in valid if value < 0),
        "flat_rows": sum(1 for value in valid if math.isclose(value, 0.0, abs_tol=1e-12)),
    }


def _prop_summary(payload: dict[str, Any]) -> dict[str, Any]:
    failures = payload.get("failures") if isinstance(payload.get("failures"), list) else []
    return {
        "status": payload.get("status"),
        "structural_pass": str(payload.get("status") or "").upper() == "PASS" and not failures,
        "profiles_checked": int(payload.get("profiles_checked") or 0),
        "profiles_valid": int(payload.get("profiles_valid") or 0),
        "oos_validation_complete": bool(payload.get("oos_validation_complete")),
        "actionable": bool(payload.get("actionable")),
        "decision_weight": _num(payload.get("decision_weight")) or 0.0,
        "failures": failures,
        "validation_scope": payload.get("validation_scope"),
    }


def build_report(
    shots: dict[str, Any],
    sot: dict[str, Any],
    goalscorer: dict[str, Any],
    assists: dict[str, Any],
    cards: dict[str, Any],
    gk_saves: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    props = {
        "shots": _prop_summary(shots),
        "sot": _prop_summary(sot),
        "goalscorer": _prop_summary(goalscorer),
        "assists": _prop_summary(assists),
        "cards": _prop_summary(cards),
        "gk_saves": _prop_summary(gk_saves),
    }

    clv = summarize_true_clv(true_clv_rows)
    blockers: list[str] = []
    warnings: list[str] = []

    for name, summary in props.items():
        if not summary["structural_pass"]:
            blockers.append(f"{name.upper()}_STRUCTURAL_SANITY_FAILED")
        if not summary["oos_validation_complete"]:
            blockers.append(f"{name.upper()}_OOS_VALIDATION_INCOMPLETE")
        if summary["actionable"] or summary["decision_weight"] != 0:
            warnings.append(f"{name.upper()}_ACTIONABLE_OR_WEIGHT_NONZERO_BEFORE_GATE")

    if props["gk_saves"]["profiles_valid"] < MIN_GK_PROFILES:
        blockers.append(f"GK_SAVES_VALID_PROFILES_{props['gk_saves']['profiles_valid']}_LT_{MIN_GK_PROFILES}")
    if clv["rows"] < MIN_PROP_TRUE_CLV:
        blockers.append(f"PLAYER_PROP_TRUE_CLV_{clv['rows']}_LT_{MIN_PROP_TRUE_CLV}")

    blockers.extend([
        "CONFIRMED_XI_OOS_COVERAGE_NOT_MATERIALIZED",
        "EXPECTED_MINUTES_OOS_VALIDATION_NOT_MATERIALIZED",
        "EXACT_OBSERVED_PROP_LINE_HISTORY_NOT_MATERIALIZED",
        "PROP_SPECIFIC_CALIBRATION_NOT_MATERIALIZED",
    ])

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_15_PLAYER_PROPS",
        "status": "REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "prop_families": props,
        "shared_engine_requirements": {
            "confirmed_lineup_required": True,
            "expected_minutes_required": True,
            "starter_probability_required": True,
            "role_required": True,
            "set_piece_role_required_where_relevant": True,
            "penalty_role_required_where_relevant": True,
            "opponent_adjustment_required": True,
            "recent_and_season_rate_required": True,
            "shrinkage_required": True,
        },
        "true_clv": {
            **clv,
            "minimum_rows": MIN_PROP_TRUE_CLV,
            "family_specific": True,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "All six prop modules currently pass structural sanity, but structural sanity is not OOS performance.",
            "No player prop may become actionable without confirmed XI/role/minutes and an exact observed sportsbook line.",
            "Goalkeeper saves currently has a much smaller validated profile pool than outfield prop families.",
            "Prop-specific calibration and true CLV must be tracked independently by market family.",
        ],
    }


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 15 Player Props validation gate.")
    parser.add_argument("--shots", required=True)
    parser.add_argument("--sot", required=True)
    parser.add_argument("--goalscorer", required=True)
    parser.add_argument("--assists", required=True)
    parser.add_argument("--cards", required=True)
    parser.add_argument("--gk-saves", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.shots),
        _load_json(args.sot),
        _load_json(args.goalscorer),
        _load_json(args.assists),
        _load_json(args.cards),
        _load_json(args.gk_saves),
        _load_jsonl(args.true_clv_tracking),
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
