from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

MODEL_VERSION = "FORMATION_MATCHUP_FM3_OOS_V1.0.0"
SCHEMA_VERSION = "1.0.0"
MIN_PRIOR_MATCHUP_N = 8
MIN_FORMATION_ADJUSTED_REVIEW_N = 100
TEAM_ROLE_SHRINK_PSEUDO_N = 5
LEAGUE_SHRINK_PSEUDO_N = 20
FORMATION_SHRINK_PSEUDO_N = 20
FORMATION_RATIO_CLIP = (0.85, 1.15)

TARGETS = {
    "SHOTS": ("home_shots", "away_shots"),
    "SOT": ("home_sot", "away_sot"),
    "GOALS": ("home_goals", "away_goals"),
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _dt(value: Any) -> datetime:
    text = str(value or "").strip().replace("Z", "+00:00")
    try:
        out = datetime.fromisoformat(text)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if out.tzinfo is None:
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _shrunk_mean(values: list[float], prior: float | None, pseudo_n: int) -> float | None:
    if prior is None:
        return _mean(values)
    if not values:
        return prior
    return (sum(values) + float(pseudo_n) * prior) / (len(values) + float(pseudo_n))


def _safe_ratio(actual: float | None, baseline: float | None) -> float | None:
    if actual is None or baseline is None or baseline <= 0.10:
        return None
    return actual / baseline


def _formation_multiplier(values: list[float]) -> float | None:
    if len(values) < MIN_PRIOR_MATCHUP_N:
        return None
    raw = (sum(values) + FORMATION_SHRINK_PSEUDO_N * 1.0) / (
        len(values) + FORMATION_SHRINK_PSEUDO_N
    )
    return max(FORMATION_RATIO_CLIP[0], min(FORMATION_RATIO_CLIP[1], raw))


def _metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    errors = []
    signed = []
    for row in rows:
        actual = _num(row.get("actual"))
        pred = _num(row.get(prefix))
        if actual is None or pred is None:
            continue
        delta = pred - actual
        errors.append(abs(delta))
        signed.append(delta)
    return {
        "n": len(errors),
        "mae": round(sum(errors) / len(errors), 6) if errors else None,
        "rmse": (
            round(math.sqrt(sum(value * value for value in signed) / len(signed)), 6)
            if signed
            else None
        ),
        "mean_error": round(sum(signed) / len(signed), 6) if signed else None,
    }


def _delta(base: float | None, challenger: float | None) -> float | None:
    if base is None or challenger is None:
        return None
    return round(challenger - base, 6)


def _improves(base: float | None, challenger: float | None) -> bool:
    return base is not None and challenger is not None and challenger < base


def _load_report(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def _role_prior(
    *,
    league_values: list[float],
    global_values: list[float],
) -> float | None:
    global_mean = _mean(global_values)
    return _shrunk_mean(league_values, global_mean, LEAGUE_SHRINK_PSEUDO_N)


def _baseline_side(
    *,
    league_values: list[float],
    global_values: list[float],
    team_attack_values: list[float],
    opponent_concession_values: list[float],
) -> float | None:
    role_prior = _role_prior(league_values=league_values, global_values=global_values)
    if role_prior is None:
        return None
    attack = _shrunk_mean(team_attack_values, role_prior, TEAM_ROLE_SHRINK_PSEUDO_N)
    concession = _shrunk_mean(
        opponent_concession_values, role_prior, TEAM_ROLE_SHRINK_PSEUDO_N
    )
    components = [value for value in (attack, concession) if value is not None]
    return sum(components) / len(components) if components else role_prior


def _family_state(
    formation_n: int,
    baseline: dict[str, Any],
    challenger: dict[str, Any],
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if formation_n < MIN_FORMATION_ADJUSTED_REVIEW_N:
        blockers.append(
            f"FORMATION_ADJUSTED_{formation_n}_LT_{MIN_FORMATION_ADJUSTED_REVIEW_N}"
        )
    for side in ("home", "away", "total"):
        base_mae = _num((baseline.get(side) or {}).get("mae"))
        challenger_mae = _num((challenger.get(side) or {}).get("mae"))
        if not _improves(base_mae, challenger_mae):
            blockers.append(f"{side.upper()}_MAE_NOT_BETTER_THAN_BASELINE")
    return ("OOS_REVIEW_ELIGIBLE" if not blockers else "RESEARCH_HOLD", blockers)


def build_report(source: dict[str, Any]) -> dict[str, Any]:
    raw_rows = source.get("rows")
    raw_rows = raw_rows if isinstance(raw_rows, list) else []
    rows = [dict(row) for row in raw_rows if isinstance(row, dict)]
    rows.sort(key=lambda row: (_dt(row.get("kickoff_local")), int(row.get("fixture_id") or 0)))

    reports: dict[str, dict[str, Any]] = {}
    evaluation_rows: list[dict[str, Any]] = []

    for target, (home_key, away_key) in TARGETS.items():
        global_home: list[float] = []
        global_away: list[float] = []
        league_home: dict[str, list[float]] = defaultdict(list)
        league_away: dict[str, list[float]] = defaultdict(list)

        home_attack: dict[int, list[float]] = defaultdict(list)
        home_concede: dict[int, list[float]] = defaultdict(list)
        away_attack: dict[int, list[float]] = defaultdict(list)
        away_concede: dict[int, list[float]] = defaultdict(list)

        matchup_home_ratio: dict[str, list[float]] = defaultdict(list)
        matchup_away_ratio: dict[str, list[float]] = defaultdict(list)
        matchup_total_ratio: dict[str, list[float]] = defaultdict(list)

        baseline_all_home: list[dict[str, Any]] = []
        baseline_all_away: list[dict[str, Any]] = []
        baseline_all_total: list[dict[str, Any]] = []
        eligible_home: list[dict[str, Any]] = []
        eligible_away: list[dict[str, Any]] = []
        eligible_total: list[dict[str, Any]] = []
        direct_total: list[dict[str, Any]] = []
        league_eval: dict[str, list[dict[str, Any]]] = defaultdict(list)
        matchup_eval_counts: dict[str, int] = defaultdict(int)

        observed_rows = 0
        baseline_evaluations = 0
        formation_evaluations = 0

        for row in rows:
            actual_home = _num(row.get(home_key))
            actual_away = _num(row.get(away_key))
            if actual_home is None or actual_away is None:
                continue
            observed_rows += 1

            league = str(row.get("league_id") or row.get("league") or "UNKNOWN")
            home_team_id = int(row.get("home_team_id") or 0)
            away_team_id = int(row.get("away_team_id") or 0)
            matchup = str(row.get("matchup_key") or "")
            if not matchup:
                continue

            base_home = _baseline_side(
                league_values=league_home[league],
                global_values=global_home,
                team_attack_values=home_attack[home_team_id],
                opponent_concession_values=away_concede[away_team_id],
            )
            base_away = _baseline_side(
                league_values=league_away[league],
                global_values=global_away,
                team_attack_values=away_attack[away_team_id],
                opponent_concession_values=home_concede[home_team_id],
            )
            base_total = (
                base_home + base_away
                if base_home is not None and base_away is not None
                else None
            )

            if base_home is not None and base_away is not None and base_total is not None:
                baseline_evaluations += 1
                baseline_all_home.append({"actual": actual_home, "baseline": base_home})
                baseline_all_away.append({"actual": actual_away, "baseline": base_away})
                baseline_all_total.append(
                    {"actual": actual_home + actual_away, "baseline": base_total}
                )

                mh = _formation_multiplier(matchup_home_ratio[matchup])
                ma = _formation_multiplier(matchup_away_ratio[matchup])
                mt = _formation_multiplier(matchup_total_ratio[matchup])
                if mh is not None and ma is not None and mt is not None:
                    formation_evaluations += 1
                    matchup_eval_counts[matchup] += 1
                    challenger_home = max(0.0, base_home * mh)
                    challenger_away = max(0.0, base_away * ma)
                    challenger_total = challenger_home + challenger_away
                    challenger_direct_total = max(0.0, base_total * mt)
                    actual_total = actual_home + actual_away

                    eligible_home.append(
                        {
                            "actual": actual_home,
                            "baseline": base_home,
                            "challenger": challenger_home,
                        }
                    )
                    eligible_away.append(
                        {
                            "actual": actual_away,
                            "baseline": base_away,
                            "challenger": challenger_away,
                        }
                    )
                    eligible_total.append(
                        {
                            "actual": actual_total,
                            "baseline": base_total,
                            "challenger": challenger_total,
                        }
                    )
                    direct_total.append(
                        {
                            "actual": actual_total,
                            "baseline": base_total,
                            "challenger": challenger_direct_total,
                        }
                    )
                    compact = {
                        "fixture_id": row.get("fixture_id"),
                        "kickoff_local": row.get("kickoff_local"),
                        "league_id": row.get("league_id"),
                        "league": row.get("league"),
                        "matchup_key": matchup,
                        "target": target,
                        "prior_matchup_n": min(
                            len(matchup_home_ratio[matchup]),
                            len(matchup_away_ratio[matchup]),
                            len(matchup_total_ratio[matchup]),
                        ),
                        "baseline_home": round(base_home, 6),
                        "baseline_away": round(base_away, 6),
                        "baseline_total": round(base_total, 6),
                        "home_multiplier": round(mh, 6),
                        "away_multiplier": round(ma, 6),
                        "total_multiplier": round(mt, 6),
                        "challenger_home": round(challenger_home, 6),
                        "challenger_away": round(challenger_away, 6),
                        "challenger_total": round(challenger_total, 6),
                        "challenger_direct_total": round(challenger_direct_total, 6),
                        "actual_home": actual_home,
                        "actual_away": actual_away,
                        "actual_total": actual_total,
                    }
                    evaluation_rows.append(compact)
                    league_eval[league].append(compact)

                rh = _safe_ratio(actual_home, base_home)
                ra = _safe_ratio(actual_away, base_away)
                rt = _safe_ratio(actual_home + actual_away, base_total)
                if rh is not None:
                    matchup_home_ratio[matchup].append(rh)
                if ra is not None:
                    matchup_away_ratio[matchup].append(ra)
                if rt is not None:
                    matchup_total_ratio[matchup].append(rt)

            global_home.append(actual_home)
            global_away.append(actual_away)
            league_home[league].append(actual_home)
            league_away[league].append(actual_away)

            home_attack[home_team_id].append(actual_home)
            home_concede[home_team_id].append(actual_away)
            away_attack[away_team_id].append(actual_away)
            away_concede[away_team_id].append(actual_home)

        full_baseline = {
            "home": _metrics(baseline_all_home, "baseline"),
            "away": _metrics(baseline_all_away, "baseline"),
            "total": _metrics(baseline_all_total, "baseline"),
        }
        eligible_baseline = {
            "home": _metrics(eligible_home, "baseline"),
            "away": _metrics(eligible_away, "baseline"),
            "total": _metrics(eligible_total, "baseline"),
        }
        challenger = {
            "home": _metrics(eligible_home, "challenger"),
            "away": _metrics(eligible_away, "challenger"),
            "total": _metrics(eligible_total, "challenger"),
        }
        challenger_direct = {
            "total": _metrics(direct_total, "challenger"),
        }

        improvement = {
            "home_mae_delta": _delta(
                (eligible_baseline["home"] or {}).get("mae"),
                (challenger["home"] or {}).get("mae"),
            ),
            "away_mae_delta": _delta(
                (eligible_baseline["away"] or {}).get("mae"),
                (challenger["away"] or {}).get("mae"),
            ),
            "total_mae_delta": _delta(
                (eligible_baseline["total"] or {}).get("mae"),
                (challenger["total"] or {}).get("mae"),
            ),
            "direct_total_mae_delta": _delta(
                (eligible_baseline["total"] or {}).get("mae"),
                (challenger_direct["total"] or {}).get("mae"),
            ),
            "home_mae_improves": _improves(
                (eligible_baseline["home"] or {}).get("mae"),
                (challenger["home"] or {}).get("mae"),
            ),
            "away_mae_improves": _improves(
                (eligible_baseline["away"] or {}).get("mae"),
                (challenger["away"] or {}).get("mae"),
            ),
            "total_mae_improves": _improves(
                (eligible_baseline["total"] or {}).get("mae"),
                (challenger["total"] or {}).get("mae"),
            ),
            "direct_total_mae_improves": _improves(
                (eligible_baseline["total"] or {}).get("mae"),
                (challenger_direct["total"] or {}).get("mae"),
            ),
        }

        status, blockers = _family_state(
            formation_evaluations, eligible_baseline, challenger
        )
        league_diagnostics: dict[str, Any] = {}
        for league, group in sorted(league_eval.items()):
            baseline_total_rows = [
                {"actual": row["actual_total"], "baseline": row["baseline_total"]}
                for row in group
            ]
            challenger_total_rows = [
                {"actual": row["actual_total"], "challenger": row["challenger_total"]}
                for row in group
            ]
            league_diagnostics[league] = {
                "n": len(group),
                "baseline_total_mae": _metrics(
                    baseline_total_rows, "baseline"
                ).get("mae"),
                "challenger_total_mae": _metrics(
                    challenger_total_rows, "challenger"
                ).get("mae"),
            }

        max_matchup_n = max(matchup_eval_counts.values(), default=0)
        reports[target] = {
            "status": status,
            "observed_rows": observed_rows,
            "baseline_evaluations": baseline_evaluations,
            "formation_adjusted_evaluations": formation_evaluations,
            "minimum_formation_adjusted_review": MIN_FORMATION_ADJUSTED_REVIEW_N,
            "minimum_prior_same_matchup": MIN_PRIOR_MATCHUP_N,
            "baseline_all_rows": full_baseline,
            "eligible_sample_baseline": eligible_baseline,
            "formation_challenger": challenger,
            "formation_challenger_direct_total": challenger_direct,
            "improvement": improvement,
            "matchup_concentration": {
                "eligible_matchups": len(matchup_eval_counts),
                "largest_matchup_evaluations": max_matchup_n,
                "largest_matchup_share": (
                    round(max_matchup_n / formation_evaluations, 6)
                    if formation_evaluations
                    else None
                ),
            },
            "by_league": league_diagnostics,
            "blockers": blockers,
            "production_enabled": False,
            "decision_weight": 0.0,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "RESEARCH_ONLY_FM3_SHOTS_SOT_GOALS",
        "source_model_version": source.get("model_version"),
        "source_status": source.get("status"),
        "source_fixtures": int(
            source.get("fixtures_with_verified_formation_pair_and_final") or len(rows)
        ),
        "policy": (
            "SPORT_FIRST; MARKET_NEVER_CONSUMED; PRIOR_ONLY_WALK_FORWARD; "
            "BASELINE_EXCLUDES_FORMATION; FORMATION_CHALLENGER_REQUIRES_PRIOR_MATCHUP_N_GE_8; "
            "SHRINK_TO_1_AND_CLIP; MISSING_IS_NOT_ZERO"
        ),
        "minimum_prior_same_matchup": MIN_PRIOR_MATCHUP_N,
        "minimum_formation_adjusted_review": MIN_FORMATION_ADJUSTED_REVIEW_N,
        "team_role_shrink_pseudo_n": TEAM_ROLE_SHRINK_PSEUDO_N,
        "league_shrink_pseudo_n": LEAGUE_SHRINK_PSEUDO_N,
        "formation_shrink_pseudo_n": FORMATION_SHRINK_PSEUDO_N,
        "formation_ratio_clip": list(FORMATION_RATIO_CLIP),
        "targets": reports,
        "health": {
            "leakage_policy": "STRICT_PRIOR_ONLY_CHRONOLOGICAL_WALK_FORWARD",
            "odds_consumed": False,
            "market_prices_consumed": False,
            "production_enabled": False,
            "decision_weight": 0.0,
            "provider_requests_added": 0,
            "model_weights_changed": False,
            "canonical_bet_logic_changed": False,
            "blockers": sorted(
                {
                    f"{target}:{blocker}"
                    for target, report in reports.items()
                    for blocker in report.get("blockers", [])
                }
            ),
        },
        "evaluation_rows": evaluation_rows[-1000:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FM-3 walk-forward formation residual research for shots, SOT and goals."
    )
    parser.add_argument("--formation-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    report = build_report(_load_report(args.formation_report))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        json.dumps(
            {
                "model_version": report["model_version"],
                "status": report["status"],
                "targets": {
                    key: {
                        "status": value["status"],
                        "observed_rows": value["observed_rows"],
                        "baseline_evaluations": value["baseline_evaluations"],
                        "formation_adjusted_evaluations": value[
                            "formation_adjusted_evaluations"
                        ],
                        "improvement": value["improvement"],
                        "blockers": value["blockers"],
                    }
                    for key, value in report["targets"].items()
                },
                "health": report["health"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
