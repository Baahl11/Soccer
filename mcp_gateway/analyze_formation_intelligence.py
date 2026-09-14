from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
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


def norm_formation(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "")
    if not text:
        return None
    text = text.replace("–", "-").replace("—", "-")
    nums = re.findall(r"\d+", text)
    if len(nums) < 3:
        return None
    # Keep provider structure but normalize punctuation, e.g. 4-2-3-1.
    return "-".join(nums)


def pois_over(lam: float, line: float) -> float:
    # Integer-goal totals at x.5 lines.
    threshold = int(math.floor(line)) + 1
    cdf = 0.0
    for k in range(threshold):
        cdf += math.exp(-lam) * (lam ** k) / math.factorial(k)
    return max(0.0, min(1.0, 1.0 - cdf))


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def logloss(p: float, y: int) -> float:
    p = max(1e-9, min(1 - 1e-9, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def final_scores(result: Any) -> tuple[int | None, int | None, int | None, int | None]:
    if not isinstance(result, dict):
        return None, None, None, None
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ht = score.get("halftime") or {}
    try:
        fh, fa = int(goals.get("home")), int(goals.get("away"))
    except (TypeError, ValueError):
        fh = fa = None
    try:
        hh, ha = int(ht.get("home")), int(ht.get("away"))
    except (TypeError, ValueError):
        hh = ha = None
    return fh, fa, hh, ha


def load_history(history_dir: str) -> dict[int, dict[str, Any]]:
    fixtures: dict[int, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    continue
                generated = parse_dt(tick.get("generated_at_local"))
                for event in tick.get("events") or []:
                    if not isinstance(event, dict):
                        continue
                    fx = event.get("fixture") or {}
                    fid = fx.get("fixture_id")
                    if not fid:
                        continue
                    fid = int(fid)
                    rec = fixtures.setdefault(fid, {
                        "fixture_id": fid,
                        "kickoff_local": fx.get("kickoff"),
                        "league_id": fx.get("league_id"),
                        "league": fx.get("league"),
                        "home_team_id": fx.get("home_team_id"),
                        "home_team": fx.get("home_team"),
                        "away_team_id": fx.get("away_team_id"),
                        "away_team": fx.get("away_team"),
                        "lineup_obs": [],
                        "projection_obs": [],
                        "result": None,
                        "tactical_stats": None,
                    })
                    if fx.get("kickoff"):
                        rec["kickoff_local"] = fx.get("kickoff")
                    lineup = event.get("lineups")
                    if isinstance(lineup, dict) and lineup.get("both_xi_confirmed"):
                        teams = lineup.get("teams") or []
                        forms: dict[int, str] = {}
                        for t in teams:
                            if not isinstance(t, dict):
                                continue
                            tid = t.get("team_id")
                            form = norm_formation(t.get("formation"))
                            if tid is not None and form:
                                forms[int(tid)] = form
                        hf = forms.get(int(rec.get("home_team_id") or -1))
                        af = forms.get(int(rec.get("away_team_id") or -1))
                        if hf and af:
                            rec["lineup_obs"].append((generated, hf, af, event.get("stage")))
                    raw = event.get("raw_projection")
                    if isinstance(raw, dict):
                        lam = fnum(raw.get("raw_total_goals"))
                        if lam is not None and lam > 0:
                            rec["projection_obs"].append((generated, lam, event.get("stage")))
                    if event.get("result"):
                        rec["result"] = event.get("result")
                    if isinstance(event.get("postgame_tactical_stats"), dict):
                        rec["tactical_stats"] = event.get("postgame_tactical_stats")
    return fixtures


def chosen_formations(rec: dict[str, Any]) -> tuple[str | None, str | None]:
    obs = [x for x in rec.get("lineup_obs") or [] if x[0] is not None]
    if not obs:
        return None, None
    obs.sort(key=lambda x: x[0])
    _, hf, af, _ = obs[-1]
    return hf, af


def chosen_lambda(rec: dict[str, Any]) -> float | None:
    obs = [x for x in rec.get("projection_obs") or [] if x[0] is not None]
    if not obs:
        return None
    obs.sort(key=lambda x: x[0])
    return fnum(obs[-1][1])


def tactical_value(rec: dict[str, Any], metric: str) -> tuple[float | None, float | None, float | None]:
    stats = rec.get("tactical_stats") or {}
    teams = stats.get("teams") or []
    by_id = {int(x.get("team_id")): x for x in teams if isinstance(x, dict) and x.get("team_id") is not None}
    h = by_id.get(int(rec.get("home_team_id") or -1), {}).get(metric)
    a = by_id.get(int(rec.get("away_team_id") or -1), {}).get(metric)
    h = fnum(h); a = fnum(a)
    total = h + a if h is not None and a is not None else fnum((stats.get("totals") or {}).get(metric))
    return h, a, total


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    def avg(key: str) -> float | None:
        vals = [fnum(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    def rate(key: str) -> float | None:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        return round(sum(int(v) for v in vals) / len(vals), 4) if vals else None
    return {
        "n": n,
        "avg_ft_goals": avg("ft_goals"),
        "avg_1h_goals": avg("h1_goals"),
        "avg_2h_goals": avg("h2_goals"),
        "over_2_5_rate": rate("over_2_5"),
        "btts_rate": rate("btts"),
        "avg_total_corners": avg("total_corners"),
        "avg_home_corners": avg("home_corners"),
        "avg_away_corners": avg("away_corners"),
        "avg_total_shots": avg("total_shots"),
        "avg_shots_on_goal": avg("shots_on_goal"),
        "avg_total_fouls": avg("fouls"),
        "avg_total_yellow_cards": avg("yellow_cards"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    fixtures = load_history(args.history_dir)
    rows: list[dict[str, Any]] = []
    for rec in fixtures.values():
        hf, af = chosen_formations(rec)
        fh, fa, hh, ha = final_scores(rec.get("result"))
        if not hf or not af or fh is None or fa is None:
            continue
        ft = fh + fa
        h1 = hh + ha if hh is not None and ha is not None else None
        h2 = ft - h1 if h1 is not None else None
        hc, ac, tc = tactical_value(rec, "corners")
        _, _, tsh = tactical_value(rec, "total_shots")
        _, _, sog = tactical_value(rec, "shots_on_goal")
        _, _, fouls = tactical_value(rec, "fouls")
        _, _, yc = tactical_value(rec, "yellow_cards")
        rows.append({
            "fixture_id": rec["fixture_id"],
            "kickoff_local": rec.get("kickoff_local"),
            "league_id": rec.get("league_id"),
            "league": rec.get("league"),
            "home_team": rec.get("home_team"),
            "away_team": rec.get("away_team"),
            "home_formation": hf,
            "away_formation": af,
            "matchup": f"{hf} vs {af}",
            "baseline_total_lambda": chosen_lambda(rec),
            "ft_goals": ft,
            "h1_goals": h1,
            "h2_goals": h2,
            "over_2_5": int(ft >= 3),
            "btts": int(fh > 0 and fa > 0),
            "home_corners": hc,
            "away_corners": ac,
            "total_corners": tc,
            "total_shots": tsh,
            "shots_on_goal": sog,
            "fouls": fouls,
            "yellow_cards": yc,
        })

    rows.sort(key=lambda r: (r.get("kickoff_local") or "", r["fixture_id"]))

    formation_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matchup_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        formation_groups[f"HOME:{r['home_formation']}"].append(r)
        formation_groups[f"AWAY:{r['away_formation']}"].append(r)
        matchup_groups[r["matchup"]].append(r)

    formation_summary = [
        {"formation_role": k, **summarize_rows(v)}
        for k, v in formation_groups.items()
    ]
    formation_summary.sort(key=lambda x: (-x["n"], x["formation_role"]))
    matchup_summary = [
        {"matchup": k, **summarize_rows(v)}
        for k, v in matchup_groups.items()
    ]
    matchup_summary.sort(key=lambda x: (-x["n"], x["matchup"]))

    # Walk-forward residual test. Formation matchup may only modify an already
    # available baseline total-goals lambda, and only from prior same-matchup
    # residuals. This controls much of the team-strength/locality signal already
    # present in the baseline instead of crediting formations for raw team quality.
    prior: dict[str, list[float]] = defaultdict(list)
    evals: list[dict[str, Any]] = []
    for r in rows:
        lam = fnum(r.get("baseline_total_lambda"))
        actual = fnum(r.get("ft_goals"))
        matchup = r["matchup"]
        if lam is None or lam <= 0 or actual is None:
            continue
        hist = prior[matchup]
        if len(hist) >= 5:
            raw_ratio = sum(hist) / len(hist)
            # Strong shrinkage toward no formation effect; 12 pseudo-observations.
            scale = (len(hist) * raw_ratio + 12.0) / (len(hist) + 12.0)
            scale = max(0.75, min(1.25, scale))
            ch_lam = lam * scale
            y = int(actual >= 3)
            bp = pois_over(lam, 2.5)
            cp = pois_over(ch_lam, 2.5)
            evals.append({
                "fixture_id": r["fixture_id"],
                "kickoff_local": r.get("kickoff_local"),
                "matchup": matchup,
                "prior_matchup_n": len(hist),
                "baseline_lambda": round(lam, 6),
                "formation_scale": round(scale, 6),
                "challenger_lambda": round(ch_lam, 6),
                "actual_ft_goals": int(actual),
                "baseline_p_over_2_5": round(bp, 6),
                "challenger_p_over_2_5": round(cp, 6),
                "actual_over_2_5": y,
            })
        prior[matchup].append(actual / lam)

    def metrics(which: str) -> dict[str, Any]:
        if not evals:
            return {"n": 0, "brier": None, "log_loss": None, "accuracy_at_0_5": None}
        ps = [float(x[which]) for x in evals]
        ys = [int(x["actual_over_2_5"]) for x in evals]
        return {
            "n": len(evals),
            "brier": round(sum(brier(p, y) for p, y in zip(ps, ys)) / len(evals), 6),
            "log_loss": round(sum(logloss(p, y) for p, y in zip(ps, ys)) / len(evals), 6),
            "accuracy_at_0_5": round(sum(int((p >= 0.5) == bool(y)) for p, y in zip(ps, ys)) / len(evals), 4),
            "mean_predicted_over_2_5": round(sum(ps) / len(ps), 4),
            "observed_over_2_5": round(sum(ys) / len(ys), 4),
        }

    base_m = metrics("baseline_p_over_2_5")
    chal_m = metrics("challenger_p_over_2_5")
    corner_rows = sum(1 for r in rows if r.get("total_corners") is not None)
    stable_matchups = sum(1 for x in matchup_summary if x["n"] >= 8)

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_FORMATION_INTELLIGENCE",
        "policy": "DESCRIPTIVE_FORMATION_CORRELATION_NEVER_UPGRADES_BET; ONLY_WALK_FORWARD_RESIDUAL_LIFT_MAY_JUSTIFY_A_FEATURE",
        "timezone_basis": "America/Mexico_City",
        "fixtures_with_confirmed_both_formations_and_final": len(rows),
        "fixtures_with_postgame_corners_and_formations": corner_rows,
        "unique_matchups": len(matchup_groups),
        "matchups_with_n_ge_8": stable_matchups,
        "descriptive": {
            "formations": formation_summary[:100],
            "matchups": matchup_summary[:150],
        },
        "walk_forward_residual_test_over_2_5": {
            "method": "BASE_TOTAL_LAMBDA_X_SHRUNK_PRIOR_MATCHUP_ACTUAL_TO_EXPECTED_RATIO",
            "minimum_prior_same_matchup": 5,
            "shrinkage_pseudo_n": 12,
            "scale_clip": [0.75, 1.25],
            "baseline": base_m,
            "challenger": chal_m,
            "improvement": {
                "brier_delta": round((chal_m["brier"] - base_m["brier"]), 6) if base_m.get("brier") is not None else None,
                "log_loss_delta": round((chal_m["log_loss"] - base_m["log_loss"]), 6) if base_m.get("log_loss") is not None else None,
                "accuracy_delta_pp": round((chal_m["accuracy_at_0_5"] - base_m["accuracy_at_0_5"]) * 100, 2) if base_m.get("accuracy_at_0_5") is not None else None,
            },
            "evaluations": evals[:300],
        },
        "corner_feature_gate": {
            "status": "COLLECTING_POSTGAME_TACTICAL_STATS" if corner_rows < 100 else "READY_FOR_RESEARCH_MODEL",
            "minimum_formation_corner_rows": 100,
            "current_rows": corner_rows,
            "note": "Corner effects will be tested against a dedicated corner baseline, not raw averages, once enough compact postgame statistics accumulate.",
        },
        "promotion_gate": {
            "enabled": False,
            "minimum_oos_evaluations": 100,
            "requires": [
                "formation challenger improves both Brier and log-loss OOS",
                "effect remains after baseline team-strength/locality expectation",
                "no single-matchup tiny-sample tuning",
                "dedicated corners model for corner activation",
            ],
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: report[k] for k in (
        "status", "fixtures_with_confirmed_both_formations_and_final",
        "fixtures_with_postgame_corners_and_formations", "unique_matchups", "matchups_with_n_ge_8"
    )}, indent=2))


if __name__ == "__main__":
    main()
