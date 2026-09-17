from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _nb_pmf(k: int, alpha: float, beta_minutes: float, future_exposure_minutes: float) -> float:
    if k < 0 or alpha <= 0 or beta_minutes <= 0 or future_exposure_minutes < 0:
        return 0.0
    if future_exposure_minutes == 0:
        return 1.0 if k == 0 else 0.0
    log_coeff = math.lgamma(k + alpha) - math.lgamma(alpha) - math.lgamma(k + 1)
    p_prior = beta_minutes / (beta_minutes + future_exposure_minutes)
    p_future = future_exposure_minutes / (beta_minutes + future_exposure_minutes)
    return math.exp(log_coeff + alpha * math.log(p_prior) + k * math.log(p_future))


def _prob_at_least(threshold_count: int, alpha: float, beta: float, exposure: float) -> float:
    cdf = sum(_nb_pmf(k, alpha, beta, exposure) for k in range(threshold_count))
    return max(0.0, min(1.0, 1.0 - cdf))


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural sanity validation for research-only player yellow-card model.")
    ap.add_argument("--player-registry", default="soccer_edge_state/analysis/player_trend_model_registry.json")
    ap.add_argument("--cards-registry", default="soccer_edge_state/analysis/cards_rate_registry.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/player_cards_model_sanity.json")
    args = ap.parse_args()

    registry = _load(args.player_registry)
    cards_registry = _load(args.cards_registry)
    profiles = registry.get("profiles") if isinstance(registry.get("profiles"), dict) else {}
    checked = valid = 0
    failures: list[dict[str, Any]] = []

    for pid, profile in profiles.items():
        if not isinstance(profile, dict):
            continue
        role = profile.get("role_model") if isinstance(profile.get("role_model"), dict) else {}
        windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
        l20 = windows.get("last_20") if isinstance(windows.get("last_20"), dict) else {}
        metrics = l20.get("metrics") if isinstance(l20.get("metrics"), dict) else {}
        cards = metrics.get("yellow_cards") if isinstance(metrics.get("yellow_cards"), dict) else {}
        alpha = _num(cards.get("posterior_gamma_shape"))
        beta = _num(cards.get("posterior_gamma_rate_minutes"))
        minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
        if alpha is None or beta is None or minutes is None or alpha <= 0 or beta <= 0 or minutes <= 0:
            continue

        checked += 1
        p1 = _prob_at_least(1, alpha, beta, minutes)
        p2 = _prob_at_least(2, alpha, beta, minutes)
        p0 = 1.0 - p1
        bounded = all(0.0 <= p <= 1.0 for p in (p0, p1, p2))
        monotone = p1 >= p2
        complement = abs(p0 + p1 - 1.0) <= 2e-9
        if bounded and monotone and complement:
            valid += 1
        else:
            failures.append({
                "player_id": pid,
                "bounded": bounded,
                "p_1plus_ge_p_2plus": monotone,
                "p0_plus_p1plus_equals_one": complement,
                "p_1plus_yellow": round(p1, 8),
                "p_2plus_yellow": round(p2, 8),
            })

    if checked == 0:
        status = "BLOCKED_INSUFFICIENT_FINALIZED_PLAYER_CARD_DATA"
    elif checked == valid:
        status = "PASS"
    else:
        status = "FAIL"

    report = {
        "schema_version": "1.0.0",
        "status": status,
        "target": "PLAYER_YELLOW_CARD_COUNT",
        "profiles_checked": checked,
        "profiles_valid": valid,
        "failures": failures[:50],
        "player_registry_schema": registry.get("schema_version"),
        "cards_registry_schema": cards_registry.get("schema_version"),
        "cards_registry_loaded": cards_registry.get("status") == "RESEARCH_YELLOW_CARD_RATE_REGISTRY",
        "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
        "data_blocked": checked == 0,
        "data_block_reason": "NO_FINALIZED_PLAYER_YELLOW_CARD_COUNTS_PERSISTED_YET" if checked == 0 else None,
        "checks": [
            "player booked and 2+ yellow probabilities are bounded [0,1]",
            "P(1+ yellow) >= P(2+ yellow)",
            "P(0 yellow)+P(1+ yellow)=1",
            "at least one player has finalized yellow-card exposure",
        ],
        "red_cards_modeled": False,
        "bookmaker_card_scoring_rule_assumed": False,
        "oos_validation_complete": False,
        "actionable": False,
        "decision_weight": 0.0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
