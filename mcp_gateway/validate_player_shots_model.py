from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

LINES = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)


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
    if threshold_count <= 0:
        return 1.0
    cdf = sum(_nb_pmf(k, alpha, beta, exposure) for k in range(threshold_count))
    return max(0.0, min(1.0, 1.0 - cdf))


def _line_table(alpha: float, beta: float, exposure: float) -> list[dict[str, Any]]:
    rows = []
    for line in LINES:
        need = int(math.floor(line)) + 1
        over = _prob_at_least(need, alpha, beta, exposure)
        rows.append({"line": line, "p_over": round(over, 6), "p_under": round(1.0 - over, 6)})
    return rows


def _load(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural sanity validation for research-only player shots model.")
    ap.add_argument("--player-registry", default="soccer_edge_state/analysis/player_trend_model_registry.json")
    ap.add_argument("--team-trends", default="soccer_edge_state/analysis/trend_intelligence.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/player_shots_model_sanity.json")
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
        shots = metrics.get("shots") if isinstance(metrics.get("shots"), dict) else {}
        alpha = _num(shots.get("posterior_gamma_shape"))
        beta = _num(shots.get("posterior_gamma_rate_minutes"))
        minutes = _num(role.get("expected_minutes_if_confirmed_starter"))
        if alpha is None or beta is None or minutes is None or alpha <= 0 or beta <= 0 or minutes <= 0:
            continue
        checked += 1
        lines = _line_table(alpha, beta, minutes)
        probs = [float(x["p_over"]) for x in lines]
        bounded = all(0.0 <= p <= 1.0 for p in probs)
        monotone = all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
        sums = all(abs(float(x["p_over"]) + float(x["p_under"]) - 1.0) <= 2e-6 for x in lines)
        if bounded and monotone and sums:
            valid += 1
        else:
            failures.append({
                "player_id": pid,
                "bounded": bounded,
                "monotone_over_by_line": monotone,
                "over_under_sum_to_one": sums,
                "probabilities": probs,
            })

    global_context = team_trends.get("global_context") if isinstance(team_trends.get("global_context"), dict) else {}
    report = {
        "schema_version": "1.0.0",
        "status": "PASS" if checked > 0 and checked == valid else "FAIL",
        "profiles_checked": checked,
        "profiles_valid": valid,
        "failures": failures[:50],
        "team_trends_schema": team_trends.get("schema_version"),
        "opponent_shot_context_available": _num(global_context.get("avg_team_shots")) is not None,
        "global_avg_team_shots": global_context.get("avg_team_shots"),
        "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
        "checks": [
            "all half-line probabilities are bounded [0,1]",
            "P(over) is monotone non-increasing as the shot threshold rises",
            "P(over)+P(under)=1 for half-lines",
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
