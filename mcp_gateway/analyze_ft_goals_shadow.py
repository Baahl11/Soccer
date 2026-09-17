from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

LINES = (1.5, 2.5, 3.5)


def fnum(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def final_total(result: Any) -> int | None:
    if not isinstance(result, dict) or not result:
        return None
    goals = result.get("goals") or {}
    ft = (result.get("score") or {}).get("fulltime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    try:
        return int(h) + int(a)
    except (TypeError, ValueError):
        return None


def pkey(line: float) -> str:
    return f"raw_over_{str(line).replace('.', '_')}_prob"


def brier(p: float, y: int) -> float:
    return (p - y) ** 2


def logloss(p: float, y: int) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(y * math.log(p) + (1-y) * math.log(1-p))


def metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "lambda_mae": None, "lines": {}}
    lam_err = [abs(float(r[f"{prefix}_lambda"]) - r["final_total"]) for r in rows if r.get(f"{prefix}_lambda") is not None]
    out: dict[str, Any] = {
        "n": len(rows),
        "lambda_mae": round(sum(lam_err) / len(lam_err), 6) if lam_err else None,
        "lines": {},
    }
    for line in LINES:
        key = str(line)
        pairs = []
        for r in rows:
            p = fnum((r.get(f"{prefix}_probs") or {}).get(key))
            if p is None or not 0 <= p <= 1:
                continue
            y = int(r["final_total"] > line)
            pairs.append((p, y))
        out["lines"][key] = {
            "n": len(pairs),
            "brier": round(sum(brier(p, y) for p, y in pairs) / len(pairs), 6) if pairs else None,
            "log_loss": round(sum(logloss(p, y) for p, y in pairs) / len(pairs), 6) if pairs else None,
            "mean_probability": round(sum(p for p, _ in pairs) / len(pairs), 6) if pairs else None,
            "observed_rate": round(sum(y for _, y in pairs) / len(pairs), 6) if pairs else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Direct OOS validation of persisted canonical FT-goals projection versus relative-strength shadow.")
    ap.add_argument("--ledger", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    ap.add_argument("--output", default="soccer_edge_state/analysis/ft_goals_shadow_validation.json")
    args = ap.parse_args()

    fixtures: dict[int, dict[str, Any]] = {}
    with open(args.ledger, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fid = row.get("fixture_id")
            if not fid:
                continue
            fid = int(fid)
            rec = fixtures.setdefault(fid, {
                "fixture_id": fid,
                "kickoff": None,
                "league_id": None,
                "league": None,
                "home_team": None,
                "away_team": None,
                "final_total": None,
                "prediction": None,
                "prediction_ts": None,
            })
            ko = parse_dt(row.get("kickoff_local"))
            if ko is not None:
                rec["kickoff"] = ko
            for key in ("league_id", "league", "home_team", "away_team"):
                if row.get(key) is not None:
                    rec[key] = row.get(key)
            total = final_total(row.get("result"))
            if total is not None:
                rec["final_total"] = total

            raw = row.get("raw_projection")
            if not isinstance(raw, dict):
                continue
            shadow = raw.get("relative_strength_shadow")
            if not isinstance(shadow, dict) or shadow.get("status") != "RESEARCH_ONLY_SHADOW":
                continue
            baseline = shadow.get("baseline") if isinstance(shadow.get("baseline"), dict) else {}
            challenger = shadow.get("challenger") if isinstance(shadow.get("challenger"), dict) else {}
            b_lam = fnum(baseline.get("raw_total_goals"))
            c_lam = fnum(challenger.get("raw_total_goals"))
            b_probs = {str(line): fnum(baseline.get(pkey(line))) for line in LINES}
            c_probs = {str(line): fnum(challenger.get(pkey(line))) for line in LINES}
            if b_lam is None or c_lam is None or b_probs["2.5"] is None or c_probs["2.5"] is None:
                continue
            ts = parse_dt(row.get("generated_at_local"))
            effective_ko = rec.get("kickoff") or ko
            if ts is None or effective_ko is None or ts >= effective_ko:
                continue
            if rec["prediction_ts"] is None or ts > rec["prediction_ts"]:
                rec["prediction_ts"] = ts
                rec["prediction"] = {
                    "canonical_lambda": b_lam,
                    "challenger_lambda": c_lam,
                    "canonical_probs": b_probs,
                    "challenger_probs": c_probs,
                    "stage": row.get("stage"),
                    "generated_at_local": row.get("generated_at_local"),
                }

    evals: list[dict[str, Any]] = []
    for rec in fixtures.values():
        pred = rec.get("prediction")
        if rec.get("final_total") is None or not isinstance(pred, dict):
            continue
        evals.append({
            "fixture_id": rec["fixture_id"],
            "league_id": rec.get("league_id"),
            "league": rec.get("league"),
            "home_team": rec.get("home_team"),
            "away_team": rec.get("away_team"),
            "final_total": rec["final_total"],
            **pred,
        })
    evals.sort(key=lambda r: (str(r.get("generated_at_local") or ""), r["fixture_id"]))

    canonical = metrics(evals, "canonical")
    challenger = metrics(evals, "challenger")
    by_league: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in evals:
        by_league[str(r.get("league") or r.get("league_id") or "UNKNOWN")].append(r)

    line_deltas: dict[str, Any] = {}
    for line in LINES:
        key = str(line)
        b = (canonical.get("lines") or {}).get(key) or {}
        c = (challenger.get("lines") or {}).get(key) or {}
        line_deltas[key] = {
            "n": c.get("n"),
            "brier_delta_challenger_minus_canonical": round(float(c["brier"]) - float(b["brier"]), 6) if c.get("brier") is not None and b.get("brier") is not None else None,
            "log_loss_delta_challenger_minus_canonical": round(float(c["log_loss"]) - float(b["log_loss"]), 6) if c.get("log_loss") is not None and b.get("log_loss") is not None else None,
        }

    n = len(evals)
    o25 = line_deltas.get("2.5") or {}
    lambda_delta = (
        round(float(challenger["lambda_mae"]) - float(canonical["lambda_mae"]), 6)
        if challenger.get("lambda_mae") is not None and canonical.get("lambda_mae") is not None else None
    )
    stable_leagues = 0
    league_rows = {}
    for league, rows in sorted(by_league.items()):
        if len(rows) < 10:
            continue
        bm = metrics(rows, "canonical")
        cm = metrics(rows, "challenger")
        bd = ((bm.get("lines") or {}).get("2.5") or {}).get("brier")
        cd = ((cm.get("lines") or {}).get("2.5") or {}).get("brier")
        delta = round(float(cd) - float(bd), 6) if bd is not None and cd is not None else None
        if delta is not None and delta < 0:
            stable_leagues += 1
        league_rows[league] = {"n": len(rows), "o2_5_brier_delta": delta}

    gate_met = (
        n >= 100
        and o25.get("brier_delta_challenger_minus_canonical") is not None
        and o25["brier_delta_challenger_minus_canonical"] < 0
        and o25.get("log_loss_delta_challenger_minus_canonical") is not None
        and o25["log_loss_delta_challenger_minus_canonical"] < 0
        and lambda_delta is not None and lambda_delta < 0
        and stable_leagues >= 3
    )

    report = {
        "schema_version": "1.0.0",
        "status": "RESEARCH_ONLY_FT_GOALS_SHADOW_VALIDATION",
        "method": "DIRECT_PERSISTED_PREKICKOFF_CANONICAL_VS_RELATIVE_STRENGTH_SHADOW",
        "anti_leakage": "Uses the latest persisted shadow snapshot strictly before kickoff and joins the final result later by fixture_id; no final result or market odds enter either stored sport projection.",
        "evaluated_fixtures": n,
        "canonical": canonical,
        "challenger": challenger,
        "improvement": {
            "lambda_mae_delta_challenger_minus_canonical": lambda_delta,
            "lines": line_deltas,
            "leagues_with_10plus_evaluations": len(league_rows),
            "leagues_where_challenger_improves_o2_5_brier": stable_leagues,
        },
        "league_stability": league_rows,
        "promotion_gate": {
            "enabled": False,
            "decision": "KEEP_RESEARCH_ONLY",
            "minimum_evaluation_n": 100,
            "sample_gate_met": n >= 100,
            "requires": [
                "lower O2.5 Brier",
                "lower O2.5 log-loss",
                "lower total-goals lambda MAE",
                "O2.5 Brier improvement in at least 3 leagues with >=10 evaluations each",
                "separate CLV review before production promotion",
            ],
            "all_statistical_gates_currently_met": bool(gate_met),
            "automatic_promotion_allowed": False,
        },
        "policy": "NO_WEIGHT_OR_PRODUCTION_CHANGE_FROM_THIS_REPORT; OBSERVE_AND_ACCUMULATE_OOS_ONLY",
        "evaluations": evals[-300:],
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"evaluations", "league_stability"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
