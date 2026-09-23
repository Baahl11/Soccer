from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_RISK_ENGINE_V4_1.0.0"

PRODUCTION_STATES = {"TIER_B", "TIER_A", "TIER_S"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def kelly_fraction(probability: Any, decimal_price: Any) -> float | None:
    p = _num(probability)
    price = _num(decimal_price)
    if p is None or price is None or not (0.0 < p < 1.0) or price <= 1.0:
        return None
    b = price - 1.0
    q = 1.0 - p
    edge = (b * p - q) / b
    return max(edge, 0.0)


def validate_policy(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    fractional = _num(policy.get("fractional_kelly_multiplier"))
    if fractional is None or not (0.0 < fractional <= 1.0):
        errors.append("INVALID_FRACTIONAL_KELLY_MULTIPLIER")

    for key in (
        "max_stake_units",
        "max_daily_exposure_units",
        "max_correlated_exposure_units",
        "max_league_exposure_units",
        "max_market_exposure_units",
        "max_drawdown_units",
    ):
        value = _num(policy.get(key))
        if value is None or value <= 0:
            errors.append(f"INVALID_{key.upper()}")
    return errors


def propose_stake(
    *,
    probability: Any,
    decimal_price: Any,
    bankroll_units: Any,
    production_validated: bool,
    policy: dict[str, Any],
    daily_exposure_units: Any = 0.0,
    correlated_exposure_units: Any = 0.0,
    league_exposure_units: Any = 0.0,
    market_exposure_units: Any = 0.0,
    current_drawdown_units: Any = 0.0,
    recent_results: list[str] | None = None,
) -> dict[str, Any]:
    errors = validate_policy(policy)
    if errors:
        return {
            "stake_units": 0.0,
            "status": "BLOCKED_INVALID_POLICY",
            "blockers": errors,
            "kelly_fraction": None,
            "fractional_kelly_fraction": None,
            "no_chase": True,
            "no_martingale": True,
        }

    if not production_validated:
        return {
            "stake_units": 0.0,
            "status": "BLOCKED_NOT_PRODUCTION_VALIDATED",
            "blockers": ["MARKET_MODEL_NOT_PRODUCTION_VALIDATED"],
            "kelly_fraction": None,
            "fractional_kelly_fraction": None,
            "no_chase": True,
            "no_martingale": True,
        }

    bankroll = _num(bankroll_units)
    if bankroll is None or bankroll <= 0:
        return {
            "stake_units": 0.0,
            "status": "BLOCKED_INVALID_BANKROLL",
            "blockers": ["INVALID_BANKROLL"],
            "kelly_fraction": None,
            "fractional_kelly_fraction": None,
            "no_chase": True,
            "no_martingale": True,
        }

    full_kelly = kelly_fraction(probability, decimal_price)
    if full_kelly is None or full_kelly <= 0:
        return {
            "stake_units": 0.0,
            "status": "NO_POSITIVE_KELLY_EDGE",
            "blockers": ["NO_POSITIVE_KELLY_EDGE"],
            "kelly_fraction": full_kelly,
            "fractional_kelly_fraction": 0.0,
            "no_chase": True,
            "no_martingale": True,
        }

    drawdown = max(_num(current_drawdown_units) or 0.0, 0.0)
    max_drawdown = float(policy["max_drawdown_units"])
    if drawdown >= max_drawdown:
        return {
            "stake_units": 0.0,
            "status": "STOP_DRAWDOWN_LIMIT",
            "blockers": ["MAX_DRAWDOWN_REACHED"],
            "kelly_fraction": round(full_kelly, 8),
            "fractional_kelly_fraction": 0.0,
            "no_chase": True,
            "no_martingale": True,
        }

    fractional_multiplier = float(policy["fractional_kelly_multiplier"])
    fractional_kelly = full_kelly * fractional_multiplier
    raw_stake = bankroll * fractional_kelly

    daily = max(_num(daily_exposure_units) or 0.0, 0.0)
    correlated = max(_num(correlated_exposure_units) or 0.0, 0.0)
    league = max(_num(league_exposure_units) or 0.0, 0.0)
    market = max(_num(market_exposure_units) or 0.0, 0.0)

    caps = {
        "max_stake": float(policy["max_stake_units"]),
        "daily_remaining": max(float(policy["max_daily_exposure_units"]) - daily, 0.0),
        "correlated_remaining": max(float(policy["max_correlated_exposure_units"]) - correlated, 0.0),
        "league_remaining": max(float(policy["max_league_exposure_units"]) - league, 0.0),
        "market_remaining": max(float(policy["max_market_exposure_units"]) - market, 0.0),
    }
    stake = min([raw_stake, *caps.values()])
    stake = max(stake, 0.0)

    blockers: list[str] = []
    if stake <= 0:
        blockers.append("EXPOSURE_CAP_REACHED")

    # recent_results is accepted only for audit. It never increases stake.
    recent_results = recent_results or []
    recent_losing_streak = 0
    for result in reversed(recent_results):
        if str(result).upper() == "LOSS":
            recent_losing_streak += 1
        else:
            break

    return {
        "stake_units": round(stake, 6),
        "status": "STAKE_PROPOSAL_RESEARCH_ONLY" if stake > 0 else "BLOCKED_EXPOSURE_CAP",
        "blockers": blockers,
        "kelly_fraction": round(full_kelly, 8),
        "fractional_kelly_fraction": round(fractional_kelly, 8),
        "raw_fractional_kelly_stake_units": round(raw_stake, 6),
        "caps": {key: round(value, 6) for key, value in caps.items()},
        "recent_losing_streak_audit_only": recent_losing_streak,
        "recent_results_can_increase_stake": False,
        "no_chase": True,
        "no_martingale": True,
        "production_execution_enabled": False,
    }


def build_report(promotion_report: dict[str, Any], oos_report: dict[str, Any]) -> dict[str, Any]:
    reviews = promotion_report.get("market_family_reviews") if isinstance(promotion_report.get("market_family_reviews"), list) else []
    production_markets = [
        row.get("market_family")
        for row in reviews
        if isinstance(row, dict)
        and str(row.get("current_state") or "").upper() in PRODUCTION_STATES
    ]
    realized = oos_report.get("realized_settlement_metrics") if isinstance(oos_report.get("realized_settlement_metrics"), dict) else {}

    blockers: list[str] = []
    if not production_markets:
        blockers.append("NO_PRODUCTION_VALIDATED_MARKETS")
    if oos_report.get("status") != "OOS_FRAMEWORK_READY_FOR_MODEL_PREDICTIONS":
        blockers.append("OOS_FRAMEWORK_NOT_READY")

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "phase": "FASE_20_BANKROLL_RISK",
        "status": "RISK_ENGINE_LOCKED" if blockers else "RISK_ENGINE_REVIEW_READY",
        "production_execution_enabled": False,
        "production_promotion_allowed": False,
        "manual_policy_configuration_required": True,
        "eligible_production_markets": production_markets,
        "risk_controls": [
            "unit_sizing",
            "fractional_kelly",
            "max_stake",
            "max_daily_exposure",
            "max_correlated_exposure",
            "league_exposure",
            "market_exposure",
            "drawdown_control",
            "stop_conditions",
        ],
        "hard_principles": {
            "no_chase": True,
            "no_martingale": True,
            "no_short_streak_stake_increase": True,
        },
        "current_realized_risk_context": {
            "max_drawdown_units": realized.get("max_drawdown_units"),
            "max_losing_streak": realized.get("max_losing_streak"),
            "return_volatility_stddev": realized.get("return_volatility_stddev"),
            "settled_rows": realized.get("settled_rows"),
        },
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "blockers": blockers,
        "notes": [
            "No default production stake caps are invented. Fractional Kelly and exposure limits require explicit reviewed policy values.",
            "The engine cannot size live stakes until at least one market/model is production validated by Phase 19.",
            "Recent losing or winning streaks never increase stake; only validated probability edge and hard exposure controls may affect sizing.",
        ],
    }


def _load_json(path: str) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 20 bankroll/risk engine report.")
    parser.add_argument("--promotion-report", required=True)
    parser.add_argument("--oos-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_report(_load_json(args.promotion_report), _load_json(args.oos_report))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
