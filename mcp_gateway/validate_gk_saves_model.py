from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any

LINES = (1.5, 2.5, 3.5, 4.5, 5.5)
SAVE_PROXY_PRIOR_SOT = 30.0


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


def _poisson_cdf(k: int, lam: float) -> float:
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return max(0.0, min(1.0, total))


def _global_prior(profiles: dict[str, Any]) -> float | None:
    saves = conceded = 0.0
    for profile in profiles.values():
        if not isinstance(profile, dict):
            continue
        l20 = ((profile.get("windows") or {}).get("last_20") or {})
        s = _num(l20.get("save_result_proxy_saves"))
        c = _num(l20.get("save_result_proxy_goals_conceded"))
        if s is None or c is None or s + c <= 0:
            continue
        saves += s
        conceded += c
    denom = saves + conceded
    return saves / denom if denom > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Structural sanity validation for research-only GK saves model.")
    ap.add_argument("--goalkeeper-profiles", default="soccer_edge_state/analysis/goalkeeper_profiles.json")
    ap.add_argument("--team-trends", default="soccer_edge_state/analysis/trend_intelligence.json")
    ap.add_argument("--output", default="soccer_edge_state/analysis/gk_saves_model_sanity.json")
    args = ap.parse_args()

    gk = _load(args.goalkeeper_profiles)
    trends = _load(args.team_trends)
    profiles = gk.get("goalkeepers") if isinstance(gk.get("goalkeepers"), dict) else {}
    global_p = _global_prior(profiles)
    global_context = trends.get("global_context") if isinstance(trends.get("global_context"), dict) else {}
    global_sot = _num(global_context.get("avg_team_sot"))

    checked = valid = 0
    failures: list[dict[str, Any]] = []
    if global_p is not None and global_sot is not None and global_sot > 0:
        for pid, profile in profiles.items():
            if not isinstance(profile, dict):
                continue
            l20 = ((profile.get("windows") or {}).get("last_20") or {})
            saves = _num(l20.get("save_result_proxy_saves"))
            conceded = _num(l20.get("save_result_proxy_goals_conceded"))
            if saves is None or conceded is None or saves + conceded <= 0:
                continue
            observed = saves + conceded
            p_save = (saves + SAVE_PROXY_PRIOR_SOT * global_p) / (observed + SAVE_PROXY_PRIOR_SOT)
            lam = global_sot * p_save
            probs = []
            sums = True
            for line in LINES:
                need = int(math.floor(line)) + 1
                over = 1.0 - _poisson_cdf(need - 1, lam)
                under = 1.0 - over
                probs.append(over)
                sums = sums and abs(over + under - 1.0) <= 2e-9
            checked += 1
            bounded = all(0.0 <= p <= 1.0 for p in probs)
            monotone = all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))
            if bounded and monotone and sums:
                valid += 1
            else:
                failures.append({
                    "player_id": pid,
                    "bounded": bounded,
                    "monotone_over_by_line": monotone,
                    "over_under_sum_to_one": sums,
                    "probabilities": [round(x, 8) for x in probs],
                })

    report = {
        "schema_version": "1.0.0",
        "status": "PASS" if checked > 0 and checked == valid else "FAIL",
        "profiles_checked": checked,
        "profiles_valid": valid,
        "failures": failures[:50],
        "goalkeeper_profiles_schema": gk.get("schema_version"),
        "team_trends_schema": trends.get("schema_version"),
        "global_save_result_proxy": round(global_p, 6) if global_p is not None else None,
        "global_avg_team_sot": global_sot,
        "validation_scope": "STRUCTURAL_ONLY_NOT_OOS_PERFORMANCE",
        "checks": [
            "save-line probabilities are bounded [0,1]",
            "P(over) is monotone non-increasing as save line rises",
            "P(over)+P(under)=1 for half-lines",
            "at least one GK has explicit saves+conceded counts",
        ],
        "psxg_available": False,
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
