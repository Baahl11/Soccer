from __future__ import annotations

import argparse
import json
import math
import os
import re
from typing import Any, Iterable

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_CARDS_REFEREE_PHASE14_V4_1.1.0"
MIN_YELLOW_OOS = 200
MIN_REFEREE_ADJUSTED = 100
MIN_RED_MARKET_REVIEW = 500
MIN_RED_ACTIONABLE = 1000
MIN_CARD_TRUE_CLV = 50


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def is_card_market(row: dict[str, Any]) -> bool:
    market = _norm(row.get("market"))
    return any(token in market for token in ("card", "booking", "yellow", "red card"))


def summarize_true_clv(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if isinstance(row, dict) and is_card_market(row)]
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
        "exact_observed_market_history_materialized": bool(row.get("exact_observed_market_history_materialized")),
        "confirmed_xi_overlap_materialized": bool(row.get("confirmed_xi_overlap_materialized")),
    }


def build_report(
    yellow_cards: dict[str, Any],
    red_cards: dict[str, Any],
    player_cards: dict[str, Any],
    true_clv_rows: Iterable[dict[str, Any]],
    market_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    yellow_metrics = yellow_cards.get("metrics") if isinstance(yellow_cards.get("metrics"), dict) else {}
    yellow_gate = yellow_cards.get("promotion_gate") if isinstance(yellow_cards.get("promotion_gate"), dict) else {}
    yellow_n = int(yellow_metrics.get("n") or 0)
    yellow_ref_n = int(yellow_metrics.get("referee_adjusted_n") or 0)

    red_overall = red_cards.get("overall") if isinstance(red_cards.get("overall"), dict) else {}
    red_gate = red_cards.get("promotion_gate") if isinstance(red_cards.get("promotion_gate"), dict) else {}
    red_n = int(red_cards.get("walk_forward_evaluated") or red_overall.get("n") or 0)
    red_ref = red_cards.get("referee_adjusted") if isinstance(red_cards.get("referee_adjusted"), dict) else {}
    red_ref_n = int(red_ref.get("n") or 0)

    player_profiles = int(player_cards.get("profiles_checked") or 0)
    player_structural_pass = str(player_cards.get("status") or "").upper() == "PASS" and not bool(player_cards.get("failures"))
    player_oos_complete = bool(player_cards.get("oos_validation_complete"))
    player_actionable = bool(player_cards.get("actionable"))

    clv = summarize_true_clv(true_clv_rows)
    match_cards_market_evidence = _market_evidence(market_audit, "CARDS")
    player_cards_market_evidence = _market_evidence(market_audit, "PLAYER_CARDS")
    blockers: list[str] = []
    warnings: list[str] = []

    if yellow_n < MIN_YELLOW_OOS:
        blockers.append(f"YELLOW_OOS_{yellow_n}_LT_{MIN_YELLOW_OOS}")
    if yellow_ref_n < MIN_REFEREE_ADJUSTED:
        blockers.append(f"YELLOW_REFEREE_ADJUSTED_{yellow_ref_n}_LT_{MIN_REFEREE_ADJUSTED}")
    if yellow_gate.get("enabled") is not True:
        blockers.append("YELLOW_CARDS_PROMOTION_GATE_DISABLED")

    if red_n < MIN_RED_MARKET_REVIEW:
        blockers.append(f"RED_CARD_OOS_{red_n}_LT_MARKET_REVIEW_{MIN_RED_MARKET_REVIEW}")
    if red_n < MIN_RED_ACTIONABLE:
        blockers.append(f"RED_CARD_OOS_{red_n}_LT_ACTIONABLE_{MIN_RED_ACTIONABLE}")
    if red_ref_n < int(red_gate.get("minimum_referee_adjusted_oos") or 200):
        blockers.append("RED_CARD_REFEREE_ADJUSTED_SAMPLE_INSUFFICIENT")
    if red_gate.get("enabled") is not True:
        blockers.append("RED_CARDS_PROMOTION_GATE_DISABLED")

    if not player_structural_pass:
        blockers.append("PLAYER_CARDS_STRUCTURAL_SANITY_FAILED")
    if not player_oos_complete:
        blockers.append("PLAYER_CARDS_OOS_VALIDATION_INCOMPLETE")
    if player_actionable:
        warnings.append("PLAYER_CARDS_ACTIONABLE_FLAG_UNEXPECTED_BEFORE_OOS_GATE")

    if clv["rows"] < MIN_CARD_TRUE_CLV:
        blockers.append(f"CARD_TRUE_CLV_{clv['rows']}_LT_{MIN_CARD_TRUE_CLV}")
    if match_cards_market_evidence["priced_value_rows"] <= 0:
        blockers.append("MATCH_CARD_OBSERVED_MARKET_PRICE_HISTORY_MISSING")
    if player_cards_market_evidence["priced_value_rows"] <= 0:
        blockers.append("PLAYER_CARD_OBSERVED_MARKET_PRICE_HISTORY_MISSING")
    if bool(player_cards.get("bookmaker_card_scoring_rule_assumed")):
        blockers.append("BOOKMAKER_CARD_SCORING_RULE_ASSUMPTION_NOT_ALLOWED")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_14_CARDS_REFEREE",
        "status": "REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD",
        "production_promotion_allowed": False,
        "manual_review_required": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "yellow_cards": {
            "oos_n": yellow_n,
            "minimum_oos": MIN_YELLOW_OOS,
            "referee_adjusted_n": yellow_ref_n,
            "minimum_referee_adjusted": MIN_REFEREE_ADJUSTED,
            "mae_total_yellow": _num(yellow_metrics.get("mae_total_yellow")),
            "lines": yellow_metrics.get("lines") if isinstance(yellow_metrics.get("lines"), dict) else {},
            "market_scope": yellow_cards.get("market_scope"),
            "promotion_gate": yellow_gate,
        },
        "red_cards": {
            "oos_n": red_n,
            "minimum_market_review": MIN_RED_MARKET_REVIEW,
            "minimum_actionable": MIN_RED_ACTIONABLE,
            "overall": red_overall,
            "referee_adjusted_n": red_ref_n,
            "promotion_gate": red_gate,
        },
        "market_evidence": {
            "match_cards": match_cards_market_evidence,
            "player_cards": player_cards_market_evidence,
            "audit_model_version": market_audit.get("model_version") if isinstance(market_audit, dict) else None,
        },
        "player_cards": {
            "profiles_checked": player_profiles,
            "structural_pass": player_structural_pass,
            "oos_validation_complete": player_oos_complete,
            "actionable": player_actionable,
            "decision_weight": player_cards.get("decision_weight"),
            "bookmaker_card_scoring_rule_assumed": player_cards.get("bookmaker_card_scoring_rule_assumed"),
        },
        "true_clv": {
            **clv,
            "minimum_rows": MIN_CARD_TRUE_CLV,
            "family_specific": True,
        },
        "blockers": blockers,
        "warnings": warnings,
        "notes": [
            "Yellow-card counts, red-card occurrence and player-card props remain separate targets.",
            "Referee effects cannot be promoted with zero verified referee-adjusted OOS observations.",
            "Sportsbook card-settlement/scoring rules must be mapped explicitly; generic card points are not assumed.",
            "Observed match-card market history and player-card prop history are audited separately; generic card market coverage cannot satisfy player-card evidence gates.",
            "Player-card structural sanity is necessary but not equivalent to OOS performance validation.",
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
    parser = argparse.ArgumentParser(description="Phase 14 Cards/Referee validation gate.")
    parser.add_argument("--yellow-cards", required=True)
    parser.add_argument("--red-cards", required=True)
    parser.add_argument("--player-cards", required=True)
    parser.add_argument("--true-clv-tracking", required=True)
    parser.add_argument("--market-audit", required=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(
        _load_json(args.yellow_cards),
        _load_json(args.red_cards),
        _load_json(args.player_cards),
        _load_jsonl(args.true_clv_tracking),
        _load_json(args.market_audit) if args.market_audit else {},
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
