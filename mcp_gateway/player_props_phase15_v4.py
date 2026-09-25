from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.4.0"
MODEL_VERSION = "SOCCER_PLAYER_PROPS_PHASE15_V4_1.4.0"
MIN_PROP_TRUE_CLV = 50
MIN_PROP_TRUE_CLV_FIXTURES = 20
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
        "goal scorer",
        "anytime scorer",
        "anytime goal scorer",
        "assist",
        "goalkeeper saves",
        "gk saves",
        "player cards",
        "player booked",
    )
    return any(token in combined for token in tokens)


CLV_FAMILY_TO_PROP = {
    "SHOTS": "shots",
    "SOT": "sot",
    "GOALSCORER_ANYTIME": "goalscorer",
    "ASSISTS": "assists",
    "PLAYER_CARDS": "cards",
    "GK_SAVES": "gk_saves",
}


def _clv_prop_name(row: dict[str, Any]) -> str | None:
    family = str(row.get("market_family") or "").upper().strip()
    if family in CLV_FAMILY_TO_PROP:
        return CLV_FAMILY_TO_PROP[family]

    market = _norm(row.get("market"))
    selection = _norm(row.get("selection"))
    combined = f"{market} {selection}"
    if "goalkeeper save" in combined or "gk save" in combined or "keeper save" in combined:
        return "gk_saves"
    if "shots on target" in combined or "shot on target" in combined:
        return "sot"
    if "shot" in combined and "team" not in market:
        return "shots"
    if "anytime" in combined and ("scorer" in combined or "goal scorer" in combined):
        return "goalscorer"
    if "assist" in combined:
        return "assists"
    if "player card" in combined or "player booked" in combined or "to be booked" in combined:
        return "cards"
    return None


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        row for row in rows
        if isinstance(row, dict)
        and row.get("is_true_closing_line") is not False
        and _clv_prop_name(row) is not None
    ]
    values = [_num(row.get("clv_probability_pp")) for row in selected]
    valid = [value for value in values if value is not None]
    by_family: dict[str, dict[str, Any]] = {}

    for prop_name in PROP_KEYS:
        items = [row for row in selected if _clv_prop_name(row) == prop_name]
        probabilities = [_num(row.get("clv_probability_pp")) for row in items]
        valid_probabilities = [value for value in probabilities if value is not None]
        fixtures = {
            row.get("fixture_id")
            for row in items
            if row.get("fixture_id") is not None
        }
        player_fixtures = {
            (row.get("fixture_id"), row.get("player_id"))
            for row in items
            if row.get("fixture_id") is not None and row.get("player_id") is not None
        }
        by_family[prop_name] = {
            "rows": len(items),
            "unique_fixtures": len(fixtures),
            "unique_player_fixtures": len(player_fixtures),
            "probability_clv_rows": len(valid_probabilities),
            "price_clv_rows": sum(1 for row in items if _num(row.get("price_clv_pct")) is not None),
            "avg_probability_clv_pp": (
                round(sum(valid_probabilities) / len(valid_probabilities), 6)
                if valid_probabilities else None
            ),
            "minimum_rows": MIN_PROP_TRUE_CLV,
            "minimum_unique_fixtures": MIN_PROP_TRUE_CLV_FIXTURES,
            "row_target_met": len(items) >= MIN_PROP_TRUE_CLV,
            "fixture_diversity_target_met": len(fixtures) >= MIN_PROP_TRUE_CLV_FIXTURES,
        }

    return {
        "rows": len(selected),
        "unique_fixtures": len({row.get("fixture_id") for row in selected if row.get("fixture_id") is not None}),
        "avg_probability_clv_pp": round(sum(valid) / len(valid), 6) if valid else None,
        "positive_rows": sum(1 for value in valid if value > 0),
        "negative_rows": sum(1 for value in valid if value < 0),
        "flat_rows": sum(1 for value in valid if math.isclose(value, 0.0, abs_tol=1e-12)),
        "by_family": by_family,
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


PROP_AUDIT_FAMILY = {
    "shots": "SHOTS",
    "sot": "SOT",
    "goalscorer": "GOALSCORER_ANYTIME",
    "assists": "ASSISTS",
    "cards": "PLAYER_CARDS",
    "gk_saves": "GK_SAVES",
}
LINE_REQUIRED_PROPS = {"shots", "sot", "gk_saves"}


OOS_FAMILY_TO_PROP = {
    "SHOTS": "shots",
    "SOT": "sot",
    "GOALSCORER_ANYTIME": "goalscorer",
    "ASSISTS": "assists",
    "PLAYER_CARDS": "cards",
    "GK_SAVES": "gk_saves",
}


def _oos_evidence(report: dict[str, Any] | None, family: str) -> dict[str, Any]:
    families = report.get("families") if isinstance(report, dict) and isinstance(report.get("families"), dict) else {}
    row = families.get(family) if isinstance(families.get(family), dict) else {}
    return {
        "audit_family": family,
        "status": row.get("status"),
        "player_game_rows": int(row.get("player_game_rows") or 0),
        "binary_probability_rows": int(row.get("binary_probability_rows") or 0),
        "unique_fixtures": int(row.get("unique_fixtures") or 0),
        "unique_player_fixtures": int(row.get("unique_player_fixtures") or 0),
        "minimum_player_games_for_review": int(row.get("minimum_player_games_for_review") or 0),
        "sample_target_met": bool(row.get("sample_target_met")),
        "oos_validation_complete": bool(row.get("oos_validation_complete")),
        "brier_score": _num(row.get("brier_score")),
        "log_loss": _num(row.get("log_loss")),
        "expected_count_mae": _num(row.get("expected_count_mae")),
        "expected_count_rmse": _num(row.get("expected_count_rmse")),
        "expected_minutes_mae": _num(row.get("expected_minutes_mae")),
        "calibration_ece": _num(row.get("calibration_ece")),
        "calibration_bins": row.get("calibration_bins") if isinstance(row.get("calibration_bins"), list) else [],
    }


def _market_evidence(audit: dict[str, Any] | None, family: str) -> dict[str, Any]:
    families = audit.get("families") if isinstance(audit, dict) and isinstance(audit.get("families"), dict) else {}
    row = families.get(family) if isinstance(families.get(family), dict) else {}
    return {
        "audit_family": family,
        "market_snapshot_rows": int(row.get("market_snapshot_rows") or 0),
        "unique_fixtures": int(row.get("unique_fixtures") or 0),
        "pre_kickoff_unique_fixtures": int(row.get("pre_kickoff_unique_fixtures") or 0),
        "provider_update_unique_fixtures": int(row.get("provider_update_unique_fixtures") or 0),
        "confirmed_xi_pre_kickoff_unique_fixtures": int(row.get("confirmed_xi_pre_kickoff_unique_fixtures") or 0),
        "bookmaker_count": int(row.get("bookmaker_count") or 0),
        "priced_value_rows": int(row.get("priced_value_rows") or 0),
        "exact_line_value_rows": int(row.get("exact_line_value_rows") or 0),
        "xi_aligned_value_rows": int(row.get("xi_aligned_value_rows") or 0),
        "xi_aligned_priced_value_rows": int(row.get("xi_aligned_priced_value_rows") or 0),
        "xi_aligned_exact_line_value_rows": int(row.get("xi_aligned_exact_line_value_rows") or 0),
        "confirmed_xi_player_aligned_unique_fixtures": int(row.get("confirmed_xi_player_aligned_unique_fixtures") or 0),
        "exact_observed_market_history_materialized": bool(row.get("exact_observed_market_history_materialized")),
        "confirmed_xi_overlap_materialized": bool(row.get("confirmed_xi_overlap_materialized")),
        "player_xi_alignment_materialized": bool(row.get("player_xi_alignment_materialized")),
    }


def build_report(
    shots: dict[str, Any],
    sot: dict[str, Any],
    goalscorer: dict[str, Any],
    assists: dict[str, Any],
    cards: dict[str, Any],
    gk_saves: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
    market_audit: dict[str, Any] | None = None,
    oos_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    props = {
        "shots": _prop_summary(shots),
        "sot": _prop_summary(sot),
        "goalscorer": _prop_summary(goalscorer),
        "assists": _prop_summary(assists),
        "cards": _prop_summary(cards),
        "gk_saves": _prop_summary(gk_saves),
    }
    for prop_name, audit_family in PROP_AUDIT_FAMILY.items():
        props[prop_name]["market_evidence"] = _market_evidence(market_audit, audit_family)
        props[prop_name]["oos_evidence"] = _oos_evidence(oos_report, audit_family)
        if isinstance(oos_report, dict) and isinstance(oos_report.get("families"), dict):
            props[prop_name]["oos_validation_complete"] = props[prop_name]["oos_evidence"]["oos_validation_complete"]

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

    for prop_name in PROP_KEYS:
        family_clv = clv["by_family"][prop_name]
        prefix = prop_name.upper()
        if family_clv["rows"] < MIN_PROP_TRUE_CLV:
            blockers.append(
                f"{prefix}_TRUE_CLV_{family_clv['rows']}_LT_{MIN_PROP_TRUE_CLV}"
            )
        if family_clv["unique_fixtures"] < MIN_PROP_TRUE_CLV_FIXTURES:
            blockers.append(
                f"{prefix}_TRUE_CLV_FIXTURES_{family_clv['unique_fixtures']}_LT_{MIN_PROP_TRUE_CLV_FIXTURES}"
            )

    for prop_name, summary in props.items():
        evidence = summary["market_evidence"]
        blocker_prefix = prop_name.upper()
        if evidence["market_snapshot_rows"] <= 0 or evidence["priced_value_rows"] <= 0:
            blockers.append(f"{blocker_prefix}_OBSERVED_MARKET_PRICE_HISTORY_MISSING")
        if evidence["confirmed_xi_pre_kickoff_unique_fixtures"] <= 0:
            blockers.append(f"{blocker_prefix}_CONFIRMED_XI_MARKET_OVERLAP_MISSING")
        if (
            evidence["confirmed_xi_player_aligned_unique_fixtures"] <= 0
            or evidence["xi_aligned_priced_value_rows"] <= 0
        ):
            blockers.append(f"{blocker_prefix}_CONFIRMED_XI_PLAYER_PRICE_OVERLAP_MISSING")
        if prop_name in LINE_REQUIRED_PROPS and evidence["exact_line_value_rows"] <= 0:
            blockers.append(f"{blocker_prefix}_EXACT_LINE_HISTORY_MISSING")
        if prop_name in LINE_REQUIRED_PROPS and evidence["xi_aligned_exact_line_value_rows"] <= 0:
            blockers.append(f"{blocker_prefix}_XI_ALIGNED_EXACT_LINE_HISTORY_MISSING")

    if not (isinstance(oos_report, dict) and isinstance(oos_report.get("families"), dict)):
        blockers.extend([
            "PLAYER_PROP_OOS_LEDGER_NOT_MATERIALIZED",
            "EXPECTED_MINUTES_OOS_VALIDATION_NOT_MATERIALIZED",
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
            "player_level_xi_alignment_required": True,
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
            "minimum_unique_fixtures_per_family": MIN_PROP_TRUE_CLV_FIXTURES,
            "family_specific": True,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "All six prop modules currently pass structural sanity, but structural sanity is not OOS performance.",
            "No player prop may become actionable without confirmed XI/role/minutes and an exact observed sportsbook market price aligned to the quoted confirmed starter; numeric exact lines are additionally required for line-based props such as shots, SOT and goalkeeper saves.",
            "Fixture-level XI overlap alone is insufficient: the priced player selection itself must resolve unambiguously to the confirmed XI at or before quote capture.",
            "Goalkeeper saves currently has a much smaller validated profile pool than outfield prop families.",
            "The goalscorer model is an anytime-scorer model; First Goal Scorer and Last Goal Scorer market history are captured separately and cannot satisfy the anytime evidence gate.",
            "Prop-specific calibration and true CLV must be tracked independently by market family.",
            "Phase15 true-CLV review requires both per-family row volume and fixture diversity; many player prices from a tiny fixture set cannot satisfy maturity.",
            "One-way player markets may contribute exact-instrument price CLV while probability CLV remains explicitly non-de-vigged/unavailable.",
            "OOS completion is sourced from the dedicated finalized-result ledger by prop family; structural sanity files do not self-promote OOS readiness.",
            "OOS calibration uses one canonical pregame snapshot per fixture and joins finalized player outcomes by player_id, with Brier/log-loss, count error, minutes error, and calibration bins reported separately.",
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
    parser.add_argument("--market-audit", required=False)
    parser.add_argument("--oos-report", required=False)
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
        _load_json(args.market_audit) if args.market_audit else {},
        _load_json(args.oos_report) if args.oos_report else {},
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
