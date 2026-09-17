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


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural sanity validation for research-only assist model.")
    ap.add_argument("--player-registry", default="soccer_edge_state/analysis/player_trend_model_registry.json")
    ap.add_argument("--team-trends", default="soccer_edge_state/analysis/trend_intelligence.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/player_assists_model_sanity.json")
    args = ap.parse_args()

    registry = _load(args.player_registry)
    team_trends = _load(args.team_trends)
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
        assists = metrics.get("assists") if isinstance(metrics.get("assists"), dict) else {}
        alpha = _num(assists.get("posterior_gamma_shape"))
        beta = _num(assists.get("posterior_gamma_rate_minutes"))
        minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
        if alpha is None or beta is None or minutes is None or alpha <= 0 or beta <= 0 or minutes <= 0:
            continue

        checked += 1
        p1 = _prob_at_least(1, alpha, beta, minutes)
        p2 = _prob_at_least(2, alpha, beta, minutes)
        bounded = 0.0 <= p2 <= p1 <= 1.0
        fair = (1.0 / p1) if p1 > 0 else None
        finite = fair is None or math.isfinite(fair)
        if bounded and finite:
            valid += 1
        else:
            failures.append({
                "player_id": pid,
                "p_1plus_assist": p1,
                "p_2plus_assists": p2,
                "bounded_and_monotone": bounded,
                "fair_price_finite": finite,
            })

    global_context = team_trends.get("global_context") if isinstance(team_trends.get("global_context"), dict) else {}
    report = {
        "schema_version": "1.0.0",
        "status": "PASS" if checked > 0 and checked == valid else "FAIL",
        "profiles_checked": checked,
        "profiles_valid": valid,
        "failures": failures[:50],
        "team_trends_schema": team_trends.get("schema_version"),
        "team_scoring_context_available": _num(global_context.get("avg_team_goals")) is not None,
        "global_avg_team_goals": global_context.get("avg_team_goals"),
        "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
        "checks": [
            "1+ assist probability is bounded [0,1]",
            "P(2+ assists) <= P(1+ assist)",
            "fair no-vig 1+ assist decimal is finite when probability is nonzero",
            "at least one modelable player profile exists",
        ],
        "oos_validation_complete": False,
        "actionable": False,
        "decision_weight": 0.0,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
