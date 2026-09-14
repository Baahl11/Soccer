from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from datetime import datetime
from typing import Any

LABELS = ["H", "D", "A"]


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def final_score(result: dict[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(result, dict) or not result:
        return None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ft = score.get("fulltime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None


def outcome(h: int, a: int) -> str:
    return "H" if h > a else "A" if a > h else "D"


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def one_x_two(home_lam: float, away_lam: float, max_goals: int = 10) -> list[float]:
    hp = [poisson_pmf(i, home_lam) for i in range(max_goals + 1)]
    ap = [poisson_pmf(i, away_lam) for i in range(max_goals + 1)]
    h = d = a = 0.0
    for i, ph in enumerate(hp):
        for j, pa in enumerate(ap):
            p = ph * pa
            if i > j:
                h += p
            elif i == j:
                d += p
            else:
                a += p
    s = h + d + a
    return [h / s, d / s, a / s]


def metrics(rows: list[tuple[list[float], str]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}
    correct = 0
    brier = 0.0
    logloss = 0.0
    pred = Counter()
    actual = Counter()
    draw_prob = 0.0
    for p, y in rows:
        yi = LABELS.index(y)
        pi = max(range(3), key=lambda i: p[i])
        pred[LABELS[pi]] += 1
        actual[y] += 1
        correct += int(pi == yi)
        brier += sum((p[i] - (1.0 if i == yi else 0.0)) ** 2 for i in range(3))
        logloss += -math.log(max(1e-12, p[yi]))
        draw_prob += p[1]
    return {
        "n": n,
        "accuracy": round(correct / n, 4),
        "brier": round(brier / n, 4),
        "log_loss": round(logloss / n, 4),
        "mean_draw_probability": round(draw_prob / n, 4),
        "observed_draw_rate": round(actual.get("D", 0) / n, 4),
        "predicted_counts": dict(pred),
        "actual_counts": dict(actual),
    }


def shrunk_rate(total: float, n: int, prior_rate: float, prior_games: float) -> float:
    return (total + prior_games * prior_rate) / (n + prior_games)


def clamp(x: float, lo: float = 0.15, hi: float = 4.5) -> float:
    return max(lo, min(hi, x))


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward relative-strength 1X2 challenger; research only.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/one_x_two_relative_strength.json")
    ap.add_argument("--min-league-matches", type=int, default=20)
    ap.add_argument("--team-prior-games", type=float, default=5.0)
    args = ap.parse_args()

    rows: list[dict[str, Any]] = []
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # Merge all snapshots by fixture_id. Pre-kickoff projections and final results
    # usually live on different ledger rows, so they must be joined explicitly.
    fixtures: dict[int, dict[str, Any]] = {}
    for row in rows:
        fid = row.get("fixture_id")
        if not fid:
            continue
        fid = int(fid)
        rec = fixtures.setdefault(fid, {
            "fixture_id": fid,
            "kickoff": None,
            "league_id": None,
            "league": None,
            "home_team_id": None,
            "away_team_id": None,
            "home_team": None,
            "away_team": None,
            "score": None,
            "actual": None,
            "baseline": None,
            "baseline_ts": None,
        })

        ko = parse_dt(row.get("kickoff_local"))
        if ko is not None:
            rec["kickoff"] = ko
        for key in ("league_id", "league", "home_team_id", "away_team_id", "home_team", "away_team"):
            value = row.get(key)
            if value is not None:
                rec[key] = value

        score = final_score(row.get("result"))
        if score is not None:
            rec["score"] = score
            rec["actual"] = outcome(*score)

        raw = row.get("raw_projection") or {}
        p = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
        ts = parse_dt(row.get("generated_at_local"))
        effective_ko = rec.get("kickoff") or ko
        if ts is not None and effective_ko is not None and ts < effective_ko and all(v is not None and 0 <= v <= 1 for v in p):
            s = sum(p)
            if s > 0 and (rec["baseline_ts"] is None or ts > rec["baseline_ts"]):
                rec["baseline"] = [v / s for v in p]
                rec["baseline_ts"] = ts

    all_final = sorted(
        [r for r in fixtures.values() if r.get("score") is not None and r.get("kickoff") is not None],
        key=lambda r: (r["kickoff"], r["fixture_id"]),
    )
    eval_candidates = [
        r for r in all_final
        if r.get("baseline") is not None
        and r.get("league_id") is not None
        and r.get("home_team_id") is not None
        and r.get("away_team_id") is not None
    ]

    baseline_eval: list[tuple[list[float], str]] = []
    challenger_eval: list[tuple[list[float], str]] = []
    diagnostics: list[dict[str, Any]] = []

    for target in eval_candidates:
        prior = [
            r for r in all_final
            if r["kickoff"] < target["kickoff"]
            and r.get("league_id") == target.get("league_id")
            and r.get("home_team_id") is not None
            and r.get("away_team_id") is not None
        ]
        if len(prior) < args.min_league_matches:
            continue

        league_home_goals = sum(r["score"][0] for r in prior)
        league_away_goals = sum(r["score"][1] for r in prior)
        league_n = len(prior)
        home_avg = league_home_goals / league_n
        away_avg = league_away_goals / league_n
        if home_avg <= 0 or away_avg <= 0:
            continue

        hid = target["home_team_id"]
        aid = target["away_team_id"]
        h_games = [r for r in prior if r.get("home_team_id") == hid]
        a_games = [r for r in prior if r.get("away_team_id") == aid]

        h_gf = sum(r["score"][0] for r in h_games)
        h_ga = sum(r["score"][1] for r in h_games)
        a_gf = sum(r["score"][1] for r in a_games)
        a_ga = sum(r["score"][0] for r in a_games)

        h_attack_rate = shrunk_rate(h_gf, len(h_games), home_avg, args.team_prior_games)
        h_concede_rate = shrunk_rate(h_ga, len(h_games), away_avg, args.team_prior_games)
        a_attack_rate = shrunk_rate(a_gf, len(a_games), away_avg, args.team_prior_games)
        a_concede_rate = shrunk_rate(a_ga, len(a_games), home_avg, args.team_prior_games)

        h_attack_strength = h_attack_rate / home_avg
        h_def_weakness = h_concede_rate / away_avg
        a_attack_strength = a_attack_rate / away_avg
        a_def_weakness = a_concede_rate / home_avg

        home_lam = clamp(home_avg * h_attack_strength * a_def_weakness)
        away_lam = clamp(away_avg * a_attack_strength * h_def_weakness)
        challenger = one_x_two(home_lam, away_lam)

        baseline_eval.append((target["baseline"], target["actual"]))
        challenger_eval.append((challenger, target["actual"]))
        diagnostics.append({
            "fixture_id": target["fixture_id"],
            "league": target.get("league"),
            "home_team": target.get("home_team"),
            "away_team": target.get("away_team"),
            "prior_league_matches": league_n,
            "home_team_prior_home_matches": len(h_games),
            "away_team_prior_away_matches": len(a_games),
            "league_home_goal_rate": round(home_avg, 4),
            "league_away_goal_rate": round(away_avg, 4),
            "challenger_home_lambda": round(home_lam, 4),
            "challenger_away_lambda": round(away_lam, 4),
            "baseline_probs": [round(v, 6) for v in target["baseline"]],
            "challenger_probs": [round(v, 6) for v in challenger],
            "actual": target["actual"],
        })

    b = metrics(baseline_eval)
    c = metrics(challenger_eval)
    n = c.get("n", 0)
    result = {
        "schema_version": "1.1.0",
        "status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "timezone_basis": "America/Mexico_City",
        "method": "WALK_FORWARD_LEAGUE_BASELINE_RELATIVE_ATTACK_DEFENSE_STRENGTH",
        "formula": {
            "home_lambda": "league_home_goal_rate * home_attack_strength * away_defensive_weakness",
            "away_lambda": "league_away_goal_rate * away_attack_strength * home_defensive_weakness",
            "home_advantage": "implicit in separate league home/away goal baselines",
            "team_shrinkage_prior_games": args.team_prior_games,
        },
        "anti_leakage": "For every target fixture, league and team strengths use only finalized fixtures with kickoff strictly earlier than the target kickoff.",
        "finalized_fixtures_joined": len(all_final),
        "eligible_baseline_fixtures": len(eval_candidates),
        "walk_forward_evaluated": n,
        "minimum_prior_league_matches": args.min_league_matches,
        "baseline": b,
        "challenger": c,
        "improvement": {
            "accuracy_delta_pp": round((c.get("accuracy", 0) - b.get("accuracy", 0)) * 100, 2) if n else None,
            "brier_delta": round(c.get("brier", 0) - b.get("brier", 0), 4) if n else None,
            "log_loss_delta": round(c.get("log_loss", 0) - b.get("log_loss", 0), 4) if n else None,
            "draw_probability_delta_pp": round((c.get("mean_draw_probability", 0) - b.get("mean_draw_probability", 0)) * 100, 2) if n else None,
        },
        "promotion_gate": {
            "enabled": False,
            "decision": "KEEP_RESEARCH_ONLY",
            "minimum_evaluation_n": 100,
            "sample_gate_met": n >= 100,
            "reason": "Never promotes 1X2 automatically. Requires larger out-of-sample sample and stable improvements in Brier/log-loss plus H/D/A calibration.",
        },
        "notes": [
            "No market odds are used to construct challenger probabilities.",
            "No xG is fabricated; this challenger uses only historical final goals already persisted by Soccer Edge.",
            "Venue-specific team rates are shrunk toward league home/away scoring baselines to reduce small-sample extremes.",
            "Pre-kickoff baseline projections are joined to final results by fixture_id; they need not occur on the same ledger row.",
            "This report cannot upgrade BET/LEAN classifications and 1X2 remains research-only.",
        ],
        "diagnostics": diagnostics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
