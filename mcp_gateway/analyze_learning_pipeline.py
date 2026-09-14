from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any

STAGE_ORDER = {"T-90": 0, "T-60": 1, "T-40": 2, "T-30": 3, "T-20": 4, "T-10": 5, "CLOSE": 6, "POSTGAME": 7}
CANONICAL_FT = {"match winner", "winner", "goals over/under", "over/under", "both teams to score", "btts"}
DERIVATIVE_KEYWORDS = ("first half", "1st half", "1h", "second half", "2nd half", "2h", "team total", "corners", "cards", "player", "both halves", "halftime", "half time")


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def norm(x: Any) -> str:
    return " ".join(str(x or "").strip().lower().split())


def load_jsonl(path: str) -> list[dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def final_score(result: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not isinstance(result, dict):
        return None, None
    goals = result.get("goals") or {}
    ft = (result.get("score") or {}).get("fulltime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    try:
        return int(h), int(a)
    except (TypeError, ValueError):
        return None, None


def implied_prob(price: float | None) -> float | None:
    return (1.0 / price) if price and price > 1.0 else None


def canonical_market(m: dict[str, Any] | None) -> bool:
    if not isinstance(m, dict):
        return False
    market = norm(m.get("market"))
    if any(k in market for k in DERIVATIVE_KEYWORDS):
        return False
    return market in CANONICAL_FT


def sport_strength(row: dict[str, Any]) -> float | None:
    sl = row.get("sporting_shortlist") or row.get("sporting_screen_refined") or row.get("sporting_screen_initial") or {}
    vals = [fnum(sl.get(k)) for k in ("rank", "side_edge_score", "goal_environment_score", "two_way_scoring_score")]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def model_confidence(row: dict[str, Any]) -> float | None:
    raw = row.get("raw_projection") or {}
    probs = [fnum(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob", "raw_btts_yes_prob", "raw_over_2_5_prob")]
    probs = [p for p in probs if p is not None]
    if not probs:
        return None
    # distance from coin-flip / diffuse 1X2, descriptive only
    return round(max(abs(p - 0.5) for p in probs), 4)


def data_quality(row: dict[str, Any]) -> float:
    tier = str(row.get("data_tier") or "").upper()
    tier_score = {"A": 1.0, "B": 0.8, "C": 0.55}.get(tier, 0.4)
    raw = 1.0 if row.get("raw_projection") else 0.0
    lineup = row.get("lineups") or {}
    xi = 1.0 if lineup.get("both_xi_confirmed") else 0.0
    gk = 1.0 if lineup.get("both_goalkeepers_confirmed") else 0.0
    return round(0.55 * tier_score + 0.25 * raw + 0.10 * xi + 0.10 * gk, 3)


def market_edge(row: dict[str, Any]) -> float | None:
    m = row.get("best_market") or {}
    return fnum(m.get("prob_edge_pp"))


def build_decision_decomposition(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out = []
    cls = Counter()
    for r in rows:
        rec = {
            "fixture_id": r.get("fixture_id"),
            "timestamp": r.get("generated_at_local"),
            "stage": r.get("stage"),
            "classification": r.get("classification"),
            "sport_signal_strength": sport_strength(r),
            "model_confidence": model_confidence(r),
            "market_edge_pp": market_edge(r),
            "availability_confidence": fnum(r.get("availability_confidence")),
            "data_quality": data_quality(r),
            "market_verified": bool(r.get("best_market")),
            "canonical_ft_market": canonical_market(r.get("best_market")),
            "final_classification": r.get("classification"),
            "separation_rule": "SPORT_SIGNAL_IS_NOT_MARKET_EDGE",
        }
        out.append(rec)
        cls[str(r.get("classification") or "UNKNOWN")] += 1
    summary = {
        "schema_version": "1.0.0",
        "status": "ACTIVE",
        "rows": len(out),
        "classification_counts": dict(cls),
        "fields": ["sport_signal_strength", "model_confidence", "market_edge_pp", "availability_confidence", "data_quality", "final_classification"],
        "rule": "A strong sporting signal does not become a wager without a verified supported market, sufficient availability confidence, and backend BET/LEAN classification.",
    }
    return out, summary


def discrepancy_recheck(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flagged = []
    results = {}
    for r in rows:
        if r.get("result"):
            results[int(r["fixture_id"])] = r["result"]
    for r in rows:
        m = r.get("best_market") or {}
        raw = r.get("raw_projection") or {}
        if norm(m.get("market")) not in {"match winner", "winner"}:
            continue
        p_model = fnum(m.get("p_shrunk")) or fnum(m.get("p_raw"))
        p_mkt = fnum(m.get("p_market_fair")) or implied_prob(fnum(m.get("decimal_price")))
        if p_model is None or p_mkt is None:
            continue
        gap = (p_model - p_mkt) * 100
        if abs(gap) < 12:
            continue
        fid = int(r["fixture_id"])
        h, a = final_score(results.get(fid))
        lineup = r.get("lineups") or {}
        reasons = []
        if (fnum(r.get("availability_confidence")) or 0) < 0.85:
            reasons.append("AVAILABILITY_LOW")
        if not lineup.get("both_xi_confirmed"):
            reasons.append("XI_UNCONFIRMED")
        if not lineup.get("both_goalkeepers_confirmed"):
            reasons.append("GK_UNCONFIRMED")
        if data_quality(r) < 0.75:
            reasons.append("DATA_QUALITY_LOW")
        if not reasons:
            reasons.append("MODEL_MARKET_DISAGREEMENT_REQUIRES_REVIEW")
        flagged.append({
            "fixture_id": fid,
            "fixture": f"{r.get('home_team')} vs {r.get('away_team')}",
            "stage": r.get("stage"),
            "selection": m.get("selection"),
            "price": fnum(m.get("decimal_price")),
            "model_prob": p_model,
            "market_prob": p_mkt,
            "gap_pp": round(gap, 3),
            "classification": r.get("classification"),
            "recheck_status": "WATCH_RECHECK_NOT_A_BET",
            "reasons": reasons,
            "final_score": [h, a] if h is not None else None,
        })
    flagged.sort(key=lambda x: abs(x["gap_pp"]), reverse=True)
    return {
        "schema_version": "1.0.0",
        "status": "ACTIVE_RESEARCH_GATE",
        "threshold_pp": 12.0,
        "flagged_observations": len(flagged),
        "top_discrepancies": flagged[:100],
        "rule": "A >=12pp raw/model-vs-market 1X2 discrepancy triggers RECHECK and can never be upgraded automatically to BET.",
    }


def lineup_impact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results: dict[int, tuple[int, int]] = {}
    by_fixture: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        fid = int(r.get("fixture_id") or 0)
        if not fid:
            continue
        h, a = final_score(r.get("result"))
        if h is not None:
            results[fid] = (h, a)
        if r.get("raw_projection") and r.get("stage") in {"T-40", "T-20", "T-10"}:
            by_fixture[fid].append(r)
    comparisons = []
    stage_accuracy: dict[str, list[int]] = defaultdict(list)
    stage_total_abs_err: dict[str, list[float]] = defaultdict(list)
    for fid, recs in by_fixture.items():
        if fid not in results:
            continue
        h, a = results[fid]
        actual = "H" if h > a else "A" if a > h else "D"
        for r in recs:
            raw = r.get("raw_projection") or {}
            probs = [fnum(raw.get("raw_home_win_prob")), fnum(raw.get("raw_draw_prob")), fnum(raw.get("raw_away_win_prob"))]
            if all(p is not None for p in probs):
                pred = ["H", "D", "A"][max(range(3), key=lambda i: probs[i])]
                stage_accuracy[str(r.get("stage"))].append(int(pred == actual))
            total = fnum(raw.get("raw_total_goals"))
            if total is not None:
                stage_total_abs_err[str(r.get("stage"))].append(abs(total - (h + a)))
        recs.sort(key=lambda r: STAGE_ORDER.get(str(r.get("stage")), -1))
        if len(recs) >= 2:
            first, last = recs[0], recs[-1]
            comparisons.append({
                "fixture_id": fid,
                "first_stage": first.get("stage"),
                "last_stage": last.get("stage"),
                "availability_first": fnum(first.get("availability_confidence")),
                "availability_last": fnum(last.get("availability_confidence")),
                "xi_first": bool((first.get("lineups") or {}).get("both_xi_confirmed")),
                "xi_last": bool((last.get("lineups") or {}).get("both_xi_confirmed")),
                "gk_first": bool((first.get("lineups") or {}).get("both_goalkeepers_confirmed")),
                "gk_last": bool((last.get("lineups") or {}).get("both_goalkeepers_confirmed")),
            })
    stage = {}
    for s in ("T-40", "T-20", "T-10"):
        acc = stage_accuracy.get(s, [])
        err = stage_total_abs_err.get(s, [])
        stage[s] = {
            "n_1x2": len(acc),
            "top1_accuracy": round(sum(acc)/len(acc), 4) if acc else None,
            "n_totals": len(err),
            "mean_abs_total_goals_error": round(sum(err)/len(err), 4) if err else None,
        }
    return {
        "schema_version": "1.0.0",
        "status": "MEASUREMENT_ACTIVE_NO_CAUSAL_CLAIM",
        "multi_stage_fixture_comparisons": len(comparisons),
        "by_stage": stage,
        "comparisons": comparisons[:200],
        "note": "Stage differences are observational. Larger samples are required before deciding whether XI/GK refresh cost improves predictive accuracy.",
    }


def api_efficiency(history_dir: str) -> dict[str, Any]:
    ticks = []
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        ticks.extend(load_jsonl(path))
    sums = Counter()
    modes = Counter()
    for t in ticks:
        for key in ("api_calls_this_tick", "due_fixture_count", "actionable_refresh_count", "deep_dive_processed_count", "shortlist_event_count", "screened_out_low_data_count", "market_requests_avoided_by_sport_screen", "deferred_due_to_priority", "deferred_due_to_budget", "urgent_late_shortlist_due", "urgent_late_shortlist_processed"):
            v = fnum(t.get(key))
            if v is not None:
                sums[key] += v
        modes[str(t.get("daily_budget_mode") or "UNKNOWN")] += 1
        gm = t.get("galaxy_first_metrics") or {}
        for key in ("galaxy_fixture_reads", "galaxy_fixture_hits", "galaxy_fixture_misses", "galaxy_raw_used", "galaxy_raw_fallbacks", "api_team_stats_calls_avoided", "api_recent_calls_avoided", "api_odds_calls_avoided", "galaxy_odds_used", "galaxy_slate_reads", "galaxy_slate_hits", "galaxy_slate_fallbacks", "galaxy_slate_stale_rejections", "duplicate_requests_detected", "provider_requests_avoided"):
            v = fnum(gm.get(key))
            if v is not None:
                sums[key] += v
    calls = float(sums["api_calls_this_tick"])
    processed = float(sums["deep_dive_processed_count"])
    shortlist_events = float(sums["shortlist_event_count"])
    return {
        "schema_version": "1.0.0",
        "ticks": len(ticks),
        "totals": dict(sums),
        "budget_modes": dict(modes),
        "api_calls_per_deep_dive": round(calls/processed, 3) if processed else None,
        "api_calls_per_shortlist_event": round(calls/shortlist_events, 3) if shortlist_events else None,
        "urgent_window_completion_rate": round(sums["urgent_late_shortlist_processed"] / sums["urgent_late_shortlist_due"], 4) if sums["urgent_late_shortlist_due"] else None,
        "status": "ACTIVE",
    }


def derivative_registry(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for r in rows:
        m = norm((r.get("best_market") or {}).get("market"))
        if not m:
            continue
        if "first half" in m or "1st half" in m or "1h" in m:
            counts["1H_GOALS"] += 1
        if "second half" in m or "2nd half" in m or "2h" in m:
            counts["2H_GOALS"] += 1
        if "team total" in m:
            counts["TEAM_TOTALS"] += 1
        if "corner" in m:
            counts["CORNERS"] += 1
        if "card" in m:
            counts["CARDS"] += 1
        if "player" in m and ("shot" in m or "shots" in m):
            counts["PLAYER_SHOTS"] += 1
    models = {}
    requirements = {
        "1H_GOALS": ["period-specific target", "1H feature set", "out-of-sample calibration"],
        "2H_GOALS": ["period-specific target", "2H state/context features", "out-of-sample calibration"],
        "TEAM_TOTALS": ["team-specific scoring distribution", "opponent defensive model", "market-line calibration"],
        "CORNERS": ["corner event history", "cross/shot pressure features", "team/league calibration"],
        "CARDS": ["referee/card history", "foul/tackle context", "league calibration"],
        "PLAYER_SHOTS": ["verified starters/minutes", "player shot volume", "opponent matchup", "lineup gate"],
    }
    for name, req in requirements.items():
        models[name] = {
            "status": "BLOCKED_PENDING_EXPLICIT_MODEL",
            "historical_market_observations": counts[name],
            "requirements": req,
            "actionable": False,
        }
    return {
        "schema_version": "1.0.0",
        "policy": "NEVER_REUSE_FT_PROBABILITIES_FOR_DERIVATIVES",
        "models": models,
        "note": "Framework/readiness gates are implemented. No derivative is enabled until a dedicated model and validation dataset exist.",
    }


def learning_dashboard(rows: list[dict[str, Any]], ft_path: str, one_x_two_path: str) -> dict[str, Any]:
    results = {}
    for r in rows:
        if r.get("result"):
            results[int(r["fixture_id"])] = r["result"]
    # unique fixture+track to avoid repeated stages
    sig: dict[tuple[int, str], dict[str, Any]] = {}
    for r in rows:
        sl = r.get("sporting_shortlist") or {}
        fid = int(r.get("fixture_id") or 0)
        if not fid:
            continue
        for track in sl.get("tracks") or []:
            key = (fid, str(track))
            old = sig.get(key)
            if old is None or STAGE_ORDER.get(str(r.get("stage")), -1) > STAGE_ORDER.get(str(old.get("stage")), -1):
                sig[key] = r
    groups: dict[str, list[int]] = defaultdict(list)
    rank_groups: dict[str, list[int]] = defaultdict(list)
    for (fid, track), r in sig.items():
        h, a = final_score(results.get(fid))
        if h is None:
            continue
        total = h + a
        if track == "GOALS_OVER": hit = int(total >= 3)
        elif track == "GOALS_UNDER": hit = int(total <= 2)
        elif track == "TWO_WAY": hit = int(h > 0 and a > 0)
        elif track == "SIDE":
            raw = r.get("raw_projection") or {}
            probs = [fnum(raw.get("raw_home_win_prob")), fnum(raw.get("raw_draw_prob")), fnum(raw.get("raw_away_win_prob"))]
            if not all(p is not None for p in probs):
                continue
            pred = ["H", "D", "A"][max(range(3), key=lambda i: probs[i])]
            actual = "H" if h > a else "A" if a > h else "D"
            hit = int(pred == actual)
        else:
            continue
        groups[track].append(hit)
        rank = fnum((r.get("sporting_shortlist") or {}).get("rank"))
        if rank is not None:
            bucket = ">=80" if rank >= 80 else "70-79" if rank >= 70 else "<70"
            rank_groups[f"{track}:{bucket}"].append(hit)
    def summarize(v: list[int]) -> dict[str, Any]:
        return {"n": len(v), "hit_rate": round(sum(v)/len(v), 4) if v else None, "sample_gate_met": len(v) >= 50}
    ft = json.load(open(ft_path, encoding="utf-8")) if os.path.exists(ft_path) else None
    ox = json.load(open(one_x_two_path, encoding="utf-8")) if os.path.exists(one_x_two_path) else None
    return {
        "schema_version": "1.0.0",
        "status": "ACTIVE_LEARNING_DASHBOARD",
        "signal_performance": {k: summarize(v) for k, v in sorted(groups.items())},
        "signal_rank_buckets": {k: summarize(v) for k, v in sorted(rank_groups.items())},
        "ft_totals": ft,
        "one_x_two": ox,
        "minimum_sample_policy": {"directional_decision": 50, "strong_model_change": 200, "note": "Do not retune thresholds from tiny cohorts."},
    }


def write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--history-dir", required=True)
    ap.add_argument("--analysis-dir", required=True)
    args = ap.parse_args()
    rows = load_jsonl(args.ledger)

    decisions, decision_summary = build_decision_decomposition(rows)
    os.makedirs(args.analysis_dir, exist_ok=True)
    with open(os.path.join(args.analysis_dir, "decision_decomposition.jsonl"), "w", encoding="utf-8") as fh:
        for r in decisions:
            fh.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
    write_json(os.path.join(args.analysis_dir, "decision_decomposition_summary.json"), decision_summary)
    write_json(os.path.join(args.analysis_dir, "discrepancy_recheck.json"), discrepancy_recheck(rows))
    write_json(os.path.join(args.analysis_dir, "lineup_impact.json"), lineup_impact(rows))
    write_json(os.path.join(args.analysis_dir, "api_efficiency.json"), api_efficiency(args.history_dir))
    write_json(os.path.join(args.analysis_dir, "derivative_model_registry.json"), derivative_registry(rows))
    write_json(os.path.join(args.analysis_dir, "learning_dashboard.json"), learning_dashboard(rows, os.path.join(args.analysis_dir, "ft_totals_validation.json"), os.path.join(args.analysis_dir, "one_x_two_calibration.json")))

    print(json.dumps({
        "decision_rows": len(decisions),
        "discrepancy_report": "discrepancy_recheck.json",
        "lineup_report": "lineup_impact.json",
        "api_efficiency_report": "api_efficiency.json",
        "derivative_registry": "derivative_model_registry.json",
        "dashboard": "learning_dashboard.json",
    }, indent=2))


if __name__ == "__main__":
    main()
