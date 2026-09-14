from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime
from typing import Any


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def period_scores(event: dict[str, Any]) -> tuple[int, int, int, int] | None:
    result = event.get("result")
    if not isinstance(result, dict):
        return None
    score = result.get("score") or {}
    ht = score.get("halftime") or {}
    ft = score.get("fulltime") or {}
    goals = result.get("goals") or {}
    try:
        hh = int(ht.get("home"))
        ha = int(ht.get("away"))
        fh = int(ft.get("home") if ft.get("home") is not None else goals.get("home"))
        fa = int(ft.get("away") if ft.get("away") is not None else goals.get("away"))
    except (TypeError, ValueError):
        return None
    sh = fh - hh
    sa = fa - ha
    if min(hh, ha, fh, fa, sh, sa) < 0:
        return None
    return sh, sa, fh, fa


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def over_prob(lam: float, line: float, max_goals: int = 10) -> float:
    threshold = math.floor(line) + 1
    return max(0.0, min(1.0, sum(poisson_pmf(k, lam) for k in range(threshold, max_goals + 1))))


def shrunk_rate(total: float, n: int, prior_rate: float, prior_games: float) -> float:
    return (total + prior_games * prior_rate) / (n + prior_games)


def clamp(value: float, lo: float = 0.05, hi: float = 4.0) -> float:
    return max(lo, min(hi, value))


def load_final_fixtures(history_dir: str) -> list[dict[str, Any]]:
    fixtures: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fx = event.get("fixture") or {}
                    fid = fx.get("fixture_id")
                    ko = parse_dt(fx.get("kickoff"))
                    scores = period_scores(event)
                    if not fid or ko is None or scores is None:
                        continue
                    sh, sa, fh, fa = scores
                    fixtures[int(fid)] = {
                        "fixture_id": int(fid),
                        "kickoff": ko,
                        "kickoff_local": fx.get("kickoff"),
                        "league_id": fx.get("league_id"),
                        "league": fx.get("league"),
                        "home_team_id": fx.get("home_team_id"),
                        "home_team": fx.get("home_team"),
                        "away_team_id": fx.get("away_team_id"),
                        "away_team": fx.get("away_team"),
                        "second_home": sh,
                        "second_away": sa,
                        "ft_home": fh,
                        "ft_away": fa,
                    }
    return sorted(fixtures.values(), key=lambda r: (r["kickoff"], r["fixture_id"]))


def model_for_target(target: dict[str, Any], prior: list[dict[str, Any]], league_prior_games: float, team_prior_games: float) -> dict[str, float] | None:
    if not prior:
        return None
    global_home = sum(r["second_home"] for r in prior) / len(prior)
    global_away = sum(r["second_away"] for r in prior) / len(prior)
    if global_home <= 0 or global_away <= 0:
        return None
    league = [r for r in prior if r.get("league_id") == target.get("league_id")]
    league_home = shrunk_rate(sum(r["second_home"] for r in league), len(league), global_home, league_prior_games)
    league_away = shrunk_rate(sum(r["second_away"] for r in league), len(league), global_away, league_prior_games)

    hid = target.get("home_team_id")
    aid = target.get("away_team_id")
    home_games = [r for r in prior if r.get("home_team_id") == hid]
    away_games = [r for r in prior if r.get("away_team_id") == aid]
    home_for = shrunk_rate(sum(r["second_home"] for r in home_games), len(home_games), league_home, team_prior_games)
    home_against = shrunk_rate(sum(r["second_away"] for r in home_games), len(home_games), league_away, team_prior_games)
    away_for = shrunk_rate(sum(r["second_away"] for r in away_games), len(away_games), league_away, team_prior_games)
    away_against = shrunk_rate(sum(r["second_home"] for r in away_games), len(away_games), league_home, team_prior_games)

    home_lam = clamp(league_home * (home_for / max(league_home, 1e-6)) * (away_against / max(league_home, 1e-6)))
    away_lam = clamp(league_away * (away_for / max(league_away, 1e-6)) * (home_against / max(league_away, 1e-6)))
    total_lam = home_lam + away_lam
    return {
        "global_home_2h_rate": global_home,
        "global_away_2h_rate": global_away,
        "league_home_2h_rate": league_home,
        "league_away_2h_rate": league_away,
        "home_lambda_2h": home_lam,
        "away_lambda_2h": away_lam,
        "total_lambda_2h": total_lam,
        "p_over_0_5_2h": over_prob(total_lam, 0.5),
        "p_over_1_5_2h": over_prob(total_lam, 1.5),
        "p_over_2_5_2h": over_prob(total_lam, 2.5),
        "prior_league_matches": float(len(league)),
        "prior_home_home_matches": float(len(home_games)),
        "prior_away_away_matches": float(len(away_games)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Explicit pregame walk-forward second-half goals model; research only.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/two_h_goals_model.json")
    ap.add_argument("--minimum-training-fixtures", type=int, default=30)
    ap.add_argument("--league-prior-games", type=float, default=20.0)
    ap.add_argument("--team-prior-games", type=float, default=5.0)
    args = ap.parse_args()

    fixtures = load_final_fixtures(args.history_dir)
    predictions: list[dict[str, Any]] = []
    brier_15 = logloss_15 = abs_error = 0.0
    hits = observed = 0
    for idx, target in enumerate(fixtures):
        prior = fixtures[:idx]
        if len(prior) < args.minimum_training_fixtures:
            continue
        model = model_for_target(target, prior, args.league_prior_games, args.team_prior_games)
        if model is None:
            continue
        actual_total = target["second_home"] + target["second_away"]
        y = int(actual_total >= 2)
        p = model["p_over_1_5_2h"]
        brier_15 += (p-y) ** 2
        logloss_15 += -(y*math.log(max(p,1e-12)) + (1-y)*math.log(max(1-p,1e-12)))
        abs_error += abs(model["total_lambda_2h"] - actual_total)
        hits += int((p >= 0.5) == bool(y))
        observed += y
        predictions.append({
            "fixture_id": target["fixture_id"],
            "kickoff_local": target["kickoff_local"],
            "league_id": target.get("league_id"),
            "league": target.get("league"),
            "home_team": target.get("home_team"),
            "away_team": target.get("away_team"),
            "actual_second_half": {"home": target["second_home"], "away": target["second_away"], "total": actual_total},
            **{k: round(v, 6) for k, v in model.items()},
        })

    n = len(predictions)
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "timezone_basis": "America/Mexico_City",
        "model": "WALK_FORWARD_HIERARCHICAL_PREGAME_2H_POISSON_STRENGTH_v0.1",
        "target": "SECOND_HALF_HOME_AND_AWAY_GOALS_FROM_FT_MINUS_HT",
        "model_timing": "PREGAME_SECOND_HALF_MARKET_MODEL_NOT_LIVE_HALFTIME_MODEL",
        "explicit_period_model": True,
        "reuses_ft_probability": False,
        "anti_leakage": "Every target uses only finalized fixtures with kickoff strictly earlier than the target. FT and HT scores are used only to construct historical second-half targets.",
        "formula": {
            "hierarchy": "global 2H home/away rates -> league 2H rates shrunk to global -> venue-specific team 2H attack/defense shrunk to league",
            "home_lambda_2h": "league_home_2h_rate * home_2h_attack_strength * away_2h_defensive_weakness",
            "away_lambda_2h": "league_away_2h_rate * away_2h_attack_strength * home_2h_defensive_weakness",
            "league_prior_games": args.league_prior_games,
            "team_prior_games": args.team_prior_games,
        },
        "finalized_fixtures_with_second_half_target": len(fixtures),
        "minimum_training_fixtures": args.minimum_training_fixtures,
        "walk_forward_evaluated": n,
        "metrics": {
            "mean_abs_2h_total_goals_error": round(abs_error/n,4) if n else None,
            "over_1_5_accuracy_at_0_5_threshold": round(hits/n,4) if n else None,
            "over_1_5_brier": round(brier_15/n,4) if n else None,
            "over_1_5_log_loss": round(logloss_15/n,4) if n else None,
            "observed_over_1_5_rate": round(observed/n,4) if n else None,
            "mean_predicted_over_1_5_probability": round(sum(r["p_over_1_5_2h"] for r in predictions)/n,4) if n else None,
        },
        "promotion_gate": {
            "enabled": False,
            "decision": "KEEP_RESEARCH_ONLY",
            "minimum_oos_n_for_market_comparison": 100,
            "minimum_oos_n_for_actionable_review": 200,
            "sample_gate_met_for_market_comparison": n >= 100,
            "reason": "Dedicated 2H market calibration and larger out-of-sample validation are required. This pregame model must never be confused with a live halftime-conditioned model.",
        },
        "market_scope": "NO_ACTIONABLE_2H_MARKET_PRICING_YET",
        "predictions": predictions[-250:],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k:v for k,v in report.items() if k != "predictions"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
