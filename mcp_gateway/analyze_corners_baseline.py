from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_formation_intelligence_v2 as formation_v2

REQUIRED_LINES = (8.5, 9.5, 10.5)
MIN_LEAGUE_LIFT_EVALS = 20
MIN_LEAGUES_FOR_STABILITY = 2


def fnum(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def pois_over(lam: float, line: float) -> float:
    threshold = int(math.floor(line)) + 1
    cdf = 0.0
    for k in range(threshold):
        cdf += math.exp(-lam) * (lam ** k) / math.factorial(k)
    return max(0.0, min(1.0, 1.0 - cdf))


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def logloss(p: float, y: int) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(y * math.log(p) + (1-y) * math.log(1-p))


def shrink_ratio(numer_sum: float, denom_sum: float, n: int, pseudo_n: float = 8.0) -> float:
    if n <= 0 or denom_sum <= 0:
        return 1.0
    raw = numer_sum / denom_sum
    weight = n / (n + pseudo_n)
    return 1.0 + weight * (raw - 1.0)


def metrics_for_evals(evals: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not evals:
        return {"n": 0, "mae_total_corners": None, "lines": {}}
    lam_key = "baseline_total_lambda" if prefix == "base" else "challenger_total_lambda"
    out = {
        "n": len(evals),
        "mae_total_corners": round(
            sum(abs(float(x[lam_key]) - x["actual_total_corners"]) for x in evals) / len(evals),
            6,
        ),
        "mean_predicted_total": round(sum(float(x[lam_key]) for x in evals) / len(evals), 4),
        "mean_actual_total": round(sum(x["actual_total_corners"] for x in evals) / len(evals), 4),
        "lines": {},
    }
    for line in REQUIRED_LINES:
        key = str(line).replace(".", "_")
        ps = [float(x[f"{prefix}_over_{key}"]) for x in evals]
        ys = [int(x["actual_total_corners"] > line) for x in evals]
        out["lines"][str(line)] = {
            "brier": round(sum(brier(p, y) for p, y in zip(ps, ys)) / len(ps), 6),
            "log_loss": round(sum(logloss(p, y) for p, y in zip(ps, ys)) / len(ps), 6),
            "accuracy_at_0_5": round(
                sum(int((p >= 0.5) == bool(y)) for p, y in zip(ps, ys)) / len(ps),
                4,
            ),
        }
    return out


def formation_lift_by_league(evals: list[dict[str, Any]]) -> dict[str, Any]:
    adjusted = [row for row in evals if int(row.get("prior_matchup_n") or 0) >= 8]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in adjusted:
        grouped[str(row.get("league_id"))].append(row)

    league_rows: dict[str, Any] = {}
    review_eligible: list[str] = []
    stable: list[str] = []
    negative: list[str] = []

    for league_id, rows in sorted(grouped.items()):
        baseline = metrics_for_evals(rows, "base")
        challenger = metrics_for_evals(rows, "challenger")
        mae_improves = (
            baseline.get("mae_total_corners") is not None
            and challenger.get("mae_total_corners") is not None
            and challenger["mae_total_corners"] < baseline["mae_total_corners"]
        )
        line_improvements: dict[str, Any] = {}
        all_lines_improve = True
        for line in REQUIRED_LINES:
            key = str(line)
            base_line = (baseline.get("lines") or {}).get(key) or {}
            challenger_line = (challenger.get("lines") or {}).get(key) or {}
            improves_brier = (
                challenger_line.get("brier") is not None
                and base_line.get("brier") is not None
                and challenger_line["brier"] < base_line["brier"]
            )
            improves_log_loss = (
                challenger_line.get("log_loss") is not None
                and base_line.get("log_loss") is not None
                and challenger_line["log_loss"] < base_line["log_loss"]
            )
            line_improvements[key] = {
                "baseline_brier": base_line.get("brier"),
                "challenger_brier": challenger_line.get("brier"),
                "baseline_log_loss": base_line.get("log_loss"),
                "challenger_log_loss": challenger_line.get("log_loss"),
                "improves_brier": improves_brier,
                "improves_log_loss": improves_log_loss,
            }
            all_lines_improve = all_lines_improve and improves_brier and improves_log_loss

        eligible = len(rows) >= MIN_LEAGUE_LIFT_EVALS
        stable_lift = eligible and mae_improves and all_lines_improve
        if eligible:
            review_eligible.append(league_id)
            if stable_lift:
                stable.append(league_id)
            else:
                negative.append(league_id)

        league_rows[league_id] = {
            "formation_adjusted_evaluations": len(rows),
            "minimum_evaluations": MIN_LEAGUE_LIFT_EVALS,
            "review_eligible": eligible,
            "mae_improves": mae_improves,
            "all_required_lines_improve_brier_and_log_loss": all_lines_improve,
            "stable_lift": stable_lift,
            "baseline": baseline,
            "formation_challenger": challenger,
            "line_improvements": line_improvements,
        }

    return {
        "formation_adjusted_evaluations": len(adjusted),
        "minimum_evaluations_per_league": MIN_LEAGUE_LIFT_EVALS,
        "minimum_review_leagues": MIN_LEAGUES_FOR_STABILITY,
        "league_count_with_adjusted_rows": len(league_rows),
        "review_eligible_leagues": review_eligible,
        "stable_lift_leagues": stable,
        "negative_lift_leagues": negative,
        "review_ready": (
            len(review_eligible) >= MIN_LEAGUES_FOR_STABILITY
            and len(stable) == len(review_eligible)
        ),
        "leagues": league_rows,
    }


def build_rows(history_dir: str) -> list[dict[str, Any]]:
    fixtures = formation_v2._enhanced_load_history(history_dir)
    rows: list[dict[str, Any]] = []
    for rec in fixtures.values():
        hf, af = formation_v2.base.chosen_formations(rec)
        hc, ac, tc = formation_v2.base.tactical_value(rec, "corners")
        if hc is None or ac is None or tc is None:
            continue
        rows.append({
            "fixture_id": rec.get("fixture_id"),
            "kickoff_local": rec.get("kickoff_local") or "",
            "league_id": rec.get("league_id"),
            "home_team_id": rec.get("home_team_id"),
            "away_team_id": rec.get("away_team_id"),
            "home_formation": hf,
            "away_formation": af,
            "matchup": f"{hf} vs {af}" if hf and af else None,
            "home_corners": float(hc),
            "away_corners": float(ac),
            "total_corners": float(tc),
        })
    rows.sort(key=lambda x: (x["kickoff_local"], int(x["fixture_id"] or 0)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = build_rows(args.history_dir)
    league_hist: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    global_hist: list[dict[str, Any]] = []
    matchup_residuals: dict[str, list[float]] = defaultdict(list)
    evals: list[dict[str, Any]] = []

    for r in rows:
        league_prior = league_hist[r["league_id"]]
        base_prior = league_prior if len(league_prior) >= 20 else global_hist
        if len(base_prior) >= 30:
            league_home = sum(x["home_corners"] for x in base_prior) / len(base_prior)
            league_away = sum(x["away_corners"] for x in base_prior) / len(base_prior)

            ht = [x for x in base_prior if x["home_team_id"] == r["home_team_id"]]
            at = [x for x in base_prior if x["away_team_id"] == r["away_team_id"]]
            # Attack = own corners relative to expected side baseline.
            h_att = shrink_ratio(sum(x["home_corners"] for x in ht), league_home * len(ht), len(ht))
            a_att = shrink_ratio(sum(x["away_corners"] for x in at), league_away * len(at), len(at))
            # Opponent weakness = corners conceded in the relevant venue role.
            away_as_visitor = [x for x in base_prior if x["away_team_id"] == r["away_team_id"]]
            home_as_host = [x for x in base_prior if x["home_team_id"] == r["home_team_id"]]
            a_def_weak = shrink_ratio(sum(x["home_corners"] for x in away_as_visitor), league_home * len(away_as_visitor), len(away_as_visitor))
            h_def_weak = shrink_ratio(sum(x["away_corners"] for x in home_as_host), league_away * len(home_as_host), len(home_as_host))

            home_lam = max(1.0, min(10.0, league_home * math.sqrt(max(0.25, h_att * a_def_weak))))
            away_lam = max(1.0, min(10.0, league_away * math.sqrt(max(0.25, a_att * h_def_weak))))
            base_lam = home_lam + away_lam
            matchup = r.get("matchup")
            hist = matchup_residuals.get(matchup or "", []) if matchup else []
            formation_scale = 1.0
            if matchup and len(hist) >= 8:
                raw = sum(hist) / len(hist)
                formation_scale = (len(hist) * raw + 20.0) / (len(hist) + 20.0)
                formation_scale = max(0.85, min(1.15, formation_scale))
            challenger_lam = base_lam * formation_scale
            item = {
                "fixture_id": r["fixture_id"], "kickoff_local": r["kickoff_local"],
                "league_id": r["league_id"], "matchup": matchup,
                "prior_pool_n": len(base_prior), "prior_matchup_n": len(hist),
                "baseline_home_lambda": round(home_lam, 5), "baseline_away_lambda": round(away_lam, 5),
                "baseline_total_lambda": round(base_lam, 5), "formation_scale": round(formation_scale, 5),
                "challenger_total_lambda": round(challenger_lam, 5),
                "actual_home_corners": r["home_corners"], "actual_away_corners": r["away_corners"],
                "actual_total_corners": r["total_corners"],
            }
            for line in (8.5, 9.5, 10.5):
                key = str(line).replace(".", "_")
                item[f"base_over_{key}"] = round(pois_over(base_lam, line), 6)
                item[f"challenger_over_{key}"] = round(pois_over(challenger_lam, line), 6)
            evals.append(item)
            if matchup and base_lam > 0:
                matchup_residuals[matchup].append(r["total_corners"] / base_lam)
        # Residual history must only be generated after a baseline prediction exists.
        league_hist[r["league_id"]].append(r)
        global_hist.append(r)

    bm = metrics_for_evals(evals, "base"); cm = metrics_for_evals(evals, "challenger")
    formation_evals = sum(1 for x in evals if x["prior_matchup_n"] >= 8)
    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_CORNERS_BASELINE",
        "policy": "NO_ACTIONABLE_CORNERS_PICK; WALK_FORWARD_ONLY; FORMATION_MUST_ADD_OOS_LIFT_OVER_TEAM_LEAGUE_BASELINE",
        "rows_with_verified_postgame_corners": len(rows),
        "walk_forward_evaluations": len(evals),
        "formation_adjusted_evaluations": formation_evals,
        "baseline_method": "LEAGUE_HOME_AWAY_CORNERS_X_SHRUNK_TEAM_ATTACK_X_OPPONENT_DEFENSE_WEAKNESS",
        "baseline": bm,
        "formation_challenger": cm,
        "formation_lift_by_league": formation_lift_by_league(evals),
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_evaluations": 150,
            "minimum_formation_adjusted_evaluations": 100,
            "requires": ["lower total-corners MAE", "lower Brier and log-loss across relevant market lines", "stable lift across leagues", "verified market-price CLV evidence"],
        },
        "evaluations": evals[-250:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ("status","rows_with_verified_postgame_corners","walk_forward_evaluations","formation_adjusted_evaluations")}, indent=2))


if __name__ == "__main__":
    main()
