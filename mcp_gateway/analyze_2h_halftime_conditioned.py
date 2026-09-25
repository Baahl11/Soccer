from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

from mcp_gateway import analyze_2h_goals_model as pregame


def parse_dt(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")) if value else None
    except ValueError:
        return None


def scores(event: dict[str, Any]) -> tuple[int, int, int, int] | None:
    result = event.get("result") if isinstance(event.get("result"), dict) else {}
    score = result.get("score") if isinstance(result.get("score"), dict) else {}
    ht = score.get("halftime") if isinstance(score.get("halftime"), dict) else {}
    ft = score.get("fulltime") if isinstance(score.get("fulltime"), dict) else {}
    goals = result.get("goals") if isinstance(result.get("goals"), dict) else {}
    try:
        hh = int(ht.get("home")); ha = int(ht.get("away"))
        fh = int(ft.get("home", goals.get("home"))); fa = int(ft.get("away", goals.get("away")))
    except (TypeError, ValueError):
        return None
    sh = fh - hh; sa = fa - ha
    if min(hh, ha, sh, sa) < 0: return None
    return hh, ha, sh, sa


def state_bucket(hh: int, ha: int) -> str:
    result = "DRAW" if hh == ha else "HOME_LEAD" if hh > ha else "AWAY_LEAD"
    total = hh + ha
    total_bucket = "HT0" if total == 0 else "HT1" if total == 1 else "HT2_PLUS"
    return f"{result}|{total_bucket}"


def load(history_dir: str) -> list[dict[str, Any]]:
    fixtures: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try: tick = json.loads(line)
                except (json.JSONDecodeError, TypeError): continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict) or event.get("stage") != "POSTGAME": continue
                    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
                    fid = fx.get("fixture_id"); ko = parse_dt(fx.get("kickoff")); sc = scores(event)
                    if fid is None or ko is None or sc is None: continue
                    hh, ha, sh, sa = sc
                    fixtures[int(fid)] = {
                        "fixture_id": int(fid), "kickoff": ko, "kickoff_local": fx.get("kickoff"),
                        "league_id": fx.get("league_id"), "league": fx.get("league"),
                        "home_team_id": fx.get("home_team_id"), "home_team": fx.get("home_team"),
                        "away_team_id": fx.get("away_team_id"), "away_team": fx.get("away_team"),
                        "ht_home": hh, "ht_away": ha, "second_home": sh, "second_away": sa,
                        "ft_home": hh + sh, "ft_away": ha + sa, "state_bucket": state_bucket(hh, ha),
                    }
    return sorted(fixtures.values(), key=lambda r: (r["kickoff"], r["fixture_id"]))


def shrunk_state_multiplier(prior: list[dict[str, Any]], bucket: str, prior_games: float) -> tuple[float, int, float, float]:
    global_mean = sum(r["second_home"] + r["second_away"] for r in prior) / len(prior)
    state_rows = [r for r in prior if r["state_bucket"] == bucket]
    state_total = sum(r["second_home"] + r["second_away"] for r in state_rows)
    state_mean = (state_total + prior_games * global_mean) / (len(state_rows) + prior_games)
    multiplier = state_mean / max(global_mean, 1e-9)
    return max(0.70, min(1.35, multiplier)), len(state_rows), global_mean, state_mean


def logloss(p: float, y: int) -> float:
    q = min(max(p, 1e-12), 1 - 1e-12)
    return -(y * math.log(q) + (1-y) * math.log(1-q))


def full_registry(fixtures: list[dict[str, Any]], prior_games: float) -> dict[str, Any]:
    if not fixtures: return {}
    global_mean = sum(r["second_home"] + r["second_away"] for r in fixtures) / len(fixtures)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fixtures: buckets[row["state_bucket"]].append(row)
    output = {}
    for bucket, rows in sorted(buckets.items()):
        state_total = sum(r["second_home"] + r["second_away"] for r in rows)
        state_mean = (state_total + prior_games * global_mean) / (len(rows) + prior_games)
        output[bucket] = {"n": len(rows), "shrunk_2h_total_rate": round(state_mean, 6), "multiplier_vs_global": round(max(0.70, min(1.35, state_mean/max(global_mean,1e-9))), 6)}
    return {"global_2h_total_rate": round(global_mean, 6), "buckets": output}


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward halftime-conditioned 2H goals challenger; research only.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/two_h_halftime_conditioned.json")
    ap.add_argument("--minimum-training-fixtures", type=int, default=50)
    ap.add_argument("--state-prior-games", type=float, default=20.0)
    args = ap.parse_args()

    fixtures = load(args.history_dir); rows = []
    for idx, target in enumerate(fixtures):
        prior = fixtures[:idx]
        if len(prior) < args.minimum_training_fixtures: continue
        base = pregame.model_for_target(target, prior, 20.0, 5.0)
        if base is None: continue
        multiplier, state_n, global_mean, state_mean = shrunk_state_multiplier(prior, target["state_bucket"], args.state_prior_games)
        base_lam = float(base["total_lambda_2h"]); cond_lam = max(0.10, min(7.0, base_lam * multiplier))
        pb = pregame.over_prob(base_lam, 1.5); pc = pregame.over_prob(cond_lam, 1.5)
        actual = target["second_home"] + target["second_away"]; y = int(actual >= 2)
        rows.append({
            "fixture_id": target["fixture_id"], "league": target.get("league"), "state_bucket": target["state_bucket"],
            "ht_score": f"{target['ht_home']}-{target['ht_away']}", "actual_2h_total": actual,
            "baseline_2h_lambda": round(base_lam,6), "conditioned_2h_lambda": round(cond_lam,6),
            "state_multiplier": round(multiplier,6), "prior_state_n": state_n, "prior_global_2h_rate": round(global_mean,6), "prior_state_2h_rate_shrunk": round(state_mean,6),
            "baseline_p_over_1_5": round(pb,6), "conditioned_p_over_1_5": round(pc,6), "actual_over_1_5": y,
            "baseline_brier": (pb-y)**2, "conditioned_brier": (pc-y)**2,
            "baseline_log_loss": logloss(pb,y), "conditioned_log_loss": logloss(pc,y),
            "baseline_abs_lambda_error": abs(base_lam-actual), "conditioned_abs_lambda_error": abs(cond_lam-actual),
        })

    n = len(rows)
    def mean(key: str) -> float | None: return round(sum(float(r[key]) for r in rows)/n,6) if n else None
    report = {
        "schema_version": "1.0.0", "status": "RESEARCH_ONLY_HALFTIME_CONDITIONED_2H_CHALLENGER",
        "model": "PREGAME_2H_BASELINE_X_HALFTIME_STATE_MULTIPLIER_v0.1",
        "walk_forward_evaluated": n,
        "conditioning_features": ["halftime home/away score", "halftime lead state", "halftime total-goal bucket"],
        "not_yet_conditioned_on": ["halftime red cards", "halftime shots/SOT", "halftime xG", "substitutions", "live weather"],
        "baseline": {"brier_o1_5": mean("baseline_brier"), "log_loss_o1_5": mean("baseline_log_loss"), "mae_2h_lambda": mean("baseline_abs_lambda_error")},
        "challenger": {"brier_o1_5": mean("conditioned_brier"), "log_loss_o1_5": mean("conditioned_log_loss"), "mae_2h_lambda": mean("conditioned_abs_lambda_error")},
        "improvement": {
            "brier_delta_baseline_minus_conditioned": round(mean("baseline_brier")-mean("conditioned_brier"),6) if n else None,
            "log_loss_delta_baseline_minus_conditioned": round(mean("baseline_log_loss")-mean("conditioned_log_loss"),6) if n else None,
            "mae_delta_baseline_minus_conditioned": round(mean("baseline_abs_lambda_error")-mean("conditioned_abs_lambda_error"),6) if n else None,
        },
        "full_history_state_registry": full_registry(fixtures, args.state_prior_games),
        "promotion_gate": {"enabled": False, "minimum_oos_for_live_research_review": 200, "sample_gate_met": n >= 200, "requires": ["beats pregame 2H baseline on Brier/log-loss/MAE", "dedicated HT scheduler path", "verified current halftime score", "red-card state before production review", "live market price/CLV calibration"]},
        "live_status": "OFFLINE_MODEL_BUILT; DEDICATED_HT_RESEARCH_STAGE_IMPLEMENTED_RESEARCH_ONLY; LIVE_2H_PRICE_RED_CARD_SHOTS_SOT_PENDING",
        "anti_leakage": "Every target multiplier uses only earlier finalized fixtures; target second-half outcome never enters its own state multiplier.",
        "rows": rows[-500:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh: json.dump(report,fh,ensure_ascii=False,indent=2,sort_keys=True); fh.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k != "rows"},indent=2,sort_keys=True))


if __name__ == "__main__": main()
