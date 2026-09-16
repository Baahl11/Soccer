from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any


def fnum(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def pois_over(lam: float, line: float) -> float:
    threshold = int(math.floor(line)) + 1
    cdf = sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(threshold))
    return max(0.0, min(1.0, 1.0 - cdf))


def shrink_rate(total: float, n: int, prior: float, pseudo_n: float) -> float:
    return (total + pseudo_n * prior) / (n + pseudo_n)


def ratio(total: float, expected_total: float, n: int, pseudo_n: float = 8.0) -> float:
    if n <= 0 or expected_total <= 0:
        return 1.0
    raw = total / expected_total
    weight = n / (n + pseudo_n)
    return 1.0 + weight * (raw - 1.0)


def load_rows(history_dir: str) -> list[dict[str, Any]]:
    by_fixture: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    tick = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                for event in tick.get("events") or []:
                    if not isinstance(event, dict) or event.get("stage") != "POSTGAME":
                        continue
                    fx = event.get("fixture") or {}
                    fid = fx.get("fixture_id")
                    ko = parse_dt(fx.get("kickoff"))
                    if not fid or ko is None:
                        continue
                    tactical = event.get("postgame_tactical_stats")
                    if not isinstance(tactical, dict):
                        result = event.get("result") or {}
                        tactical = result.get("tactical_stats") if isinstance(result, dict) else None
                    if not isinstance(tactical, dict):
                        continue
                    teams = tactical.get("teams") or []
                    by_id = {
                        int(t.get("team_id")): t
                        for t in teams
                        if isinstance(t, dict) and t.get("team_id") is not None
                    }
                    hid = fx.get("home_team_id")
                    aid = fx.get("away_team_id")
                    if hid is None or aid is None:
                        continue
                    home = by_id.get(int(hid), {})
                    away = by_id.get(int(aid), {})
                    hy = fnum(home.get("yellow_cards"))
                    ay = fnum(away.get("yellow_cards"))
                    if hy is None or ay is None:
                        continue
                    by_fixture[int(fid)] = {
                        "fixture_id": int(fid),
                        "kickoff": ko,
                        "kickoff_local": fx.get("kickoff"),
                        "league_id": fx.get("league_id"),
                        "league": fx.get("league"),
                        "home_team_id": int(hid),
                        "home_team": fx.get("home_team"),
                        "away_team_id": int(aid),
                        "away_team": fx.get("away_team"),
                        "referee": fx.get("referee"),
                        "home_yellow": hy,
                        "away_yellow": ay,
                        "home_red": fnum(home.get("red_cards")) or 0.0,
                        "away_red": fnum(away.get("red_cards")) or 0.0,
                        "home_fouls": fnum(home.get("fouls")),
                        "away_fouls": fnum(away.get("fouls")),
                    }
    return sorted(by_fixture.values(), key=lambda r: (r["kickoff"], r["fixture_id"]))


def build_model(target: dict[str, Any], prior: list[dict[str, Any]], league_pseudo: float = 25.0) -> dict[str, Any] | None:
    if len(prior) < 30:
        return None
    global_home = sum(r["home_yellow"] for r in prior) / len(prior)
    global_away = sum(r["away_yellow"] for r in prior) / len(prior)
    league = [r for r in prior if r.get("league_id") == target.get("league_id")]
    league_home = shrink_rate(sum(r["home_yellow"] for r in league), len(league), global_home, league_pseudo)
    league_away = shrink_rate(sum(r["away_yellow"] for r in league), len(league), global_away, league_pseudo)

    hid = target["home_team_id"]
    aid = target["away_team_id"]
    home_as_home = [r for r in prior if r["home_team_id"] == hid]
    away_as_away = [r for r in prior if r["away_team_id"] == aid]

    home_own = ratio(sum(r["home_yellow"] for r in home_as_home), league_home * len(home_as_home), len(home_as_home))
    away_own = ratio(sum(r["away_yellow"] for r in away_as_away), league_away * len(away_as_away), len(away_as_away))
    # Opponent card-drawing effect: how many cards opponents received against this team in the same venue role.
    away_draws_home = ratio(sum(r["home_yellow"] for r in away_as_away), league_home * len(away_as_away), len(away_as_away))
    home_draws_away = ratio(sum(r["away_yellow"] for r in home_as_home), league_away * len(home_as_home), len(home_as_home))

    home_lambda = max(0.25, min(5.5, league_home * math.sqrt(max(0.20, home_own * away_draws_home))))
    away_lambda = max(0.25, min(5.5, league_away * math.sqrt(max(0.20, away_own * home_draws_away))))
    base_total = home_lambda + away_lambda

    referee = str(target.get("referee") or "").strip()
    ref_prior = [r for r in prior if referee and str(r.get("referee") or "").strip() == referee]
    referee_scale = 1.0
    if len(ref_prior) >= 8 and base_total > 0:
        league_total = league_home + league_away
        ref_avg = sum(r["home_yellow"] + r["away_yellow"] for r in ref_prior) / len(ref_prior)
        raw_scale = ref_avg / max(league_total, 1e-6)
        referee_scale = (len(ref_prior) * raw_scale + 12.0) / (len(ref_prior) + 12.0)
        referee_scale = max(0.75, min(1.35, referee_scale))

    total_lambda = base_total * referee_scale
    return {
        "home_yellow_lambda": home_lambda,
        "away_yellow_lambda": away_lambda,
        "base_total_yellow_lambda": base_total,
        "referee": referee or None,
        "referee_prior_n": len(ref_prior),
        "referee_scale": referee_scale,
        "total_yellow_lambda": total_lambda,
        "prior_league_n": len(league),
        "prior_home_home_n": len(home_as_home),
        "prior_away_away_n": len(away_as_away),
        **{f"p_over_{str(line).replace('.', '_')}_yellow": pois_over(total_lambda, line) for line in (2.5, 3.5, 4.5, 5.5)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Walk-forward yellow-card baseline with optional referee adjustment; research only.")
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/cards_baseline.json")
    args = ap.parse_args()

    rows = load_rows(args.history_dir)
    evals = []
    for idx, target in enumerate(rows):
        prior = rows[:idx]
        model = build_model(target, prior)
        if model is None:
            continue
        actual = target["home_yellow"] + target["away_yellow"]
        item = {
            "fixture_id": target["fixture_id"],
            "kickoff_local": target["kickoff_local"],
            "league_id": target.get("league_id"),
            "league": target.get("league"),
            "home_team": target.get("home_team"),
            "away_team": target.get("away_team"),
            "actual_home_yellow": target["home_yellow"],
            "actual_away_yellow": target["away_yellow"],
            "actual_total_yellow": actual,
            "actual_total_red": target["home_red"] + target["away_red"],
            **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in model.items()},
        }
        evals.append(item)

    metrics: dict[str, Any] = {
        "n": len(evals),
        "mae_total_yellow": round(sum(abs(x["total_yellow_lambda"] - x["actual_total_yellow"]) for x in evals) / len(evals), 6) if evals else None,
        "referee_adjusted_n": sum(1 for x in evals if x.get("referee_prior_n", 0) >= 8),
        "lines": {},
    }
    for line in (2.5, 3.5, 4.5, 5.5):
        key = str(line).replace(".", "_")
        ps = [float(x[f"p_over_{key}_yellow"]) for x in evals]
        ys = [int(x["actual_total_yellow"] > line) for x in evals]
        if not ps:
            metrics["lines"][str(line)] = {"brier": None, "log_loss": None}
            continue
        brier = sum((p-y)**2 for p, y in zip(ps, ys)) / len(ps)
        logloss = sum(-(y*math.log(max(p,1e-9)) + (1-y)*math.log(max(1-p,1e-9))) for p, y in zip(ps, ys)) / len(ps)
        metrics["lines"][str(line)] = {"brier": round(brier, 6), "log_loss": round(logloss, 6)}

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_CARDS_BASELINE",
        "model": "WALK_FORWARD_TEAM_DISCIPLINE_PLUS_OPTIONAL_REFEREE_YELLOW_CARDS_v0.1",
        "target": "TOTAL_YELLOW_CARDS",
        "policy": "SPORT_FIRST; NO_MARKET_DERIVED_PROBABILITY; RED_CARDS_RECORDED_SEPARATELY; BOOKMAKER_CARD_POINT_RULES_NOT_ASSUMED",
        "rows_with_verified_postgame_yellow_cards": len(rows),
        "metrics": metrics,
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_evaluations": 200,
            "minimum_referee_adjusted_evaluations": 100,
            "requires": [
                "stable Brier/log-loss versus league baseline",
                "stable performance across competitions",
                "verified referee assignment",
                "explicit mapping to each sportsbook card-scoring rule",
                "verified market-price CLV evidence",
            ],
        },
        "market_scope": "NO_ACTIONABLE_CARD_MARKET_MAPPING_YET",
        "evaluations": evals[-300:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in ("status", "rows_with_verified_postgame_yellow_cards", "metrics", "promotion_gate")}, indent=2))


if __name__ == "__main__":
    main()
