from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

CLASS_RANK = {"PASS": 0, "WATCH": 1, "LEAN": 2, "BET": 3}
CANONICAL_MARKETS = {"match winner", "winner", "goals over/under", "over/under", "both teams to score", "btts"}


def fnum(x: Any) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def norm(x: Any) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip()).lower()


def parse_line(best: dict[str, Any]) -> float | None:
    line = fnum(best.get("line"))
    if line is not None:
        return line
    for field in (best.get("selection"), best.get("market")):
        m = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(field or ""), re.I)
        if m:
            return float(m.group(1))
    return None


def period_type(market: str) -> str:
    n = norm(market)
    if any(s in n for s in ("first half", "1st half", "1h", "half time", "halftime")):
        return "1H"
    if any(s in n for s in ("second half", "2nd half", "2h")):
        return "2H"
    return "FT"


def canonical_full_match(best: dict[str, Any]) -> bool:
    market = norm(best.get("market"))
    if period_type(market) != "FT":
        return False
    banned = ("team total", "corners", "cards", "player", "double chance", "draw no bet", "asian", "handicap", "correct score", "odd/even", "both halves", "result/total", "winner &", "win and")
    if any(x in market for x in banned):
        return False
    return market in CANONICAL_MARKETS


def final_score(result: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    goals = result.get("goals") or {}
    score = result.get("score") or {}
    ft = score.get("fulltime") or {}
    ht = score.get("halftime") or {}
    h = goals.get("home", ft.get("home"))
    a = goals.get("away", ft.get("away"))
    hh = ht.get("home")
    ha = ht.get("away")
    try: h = int(h) if h is not None else None
    except Exception: h = None
    try: a = int(a) if a is not None else None
    except Exception: a = None
    try: hh = int(hh) if hh is not None else None
    except Exception: hh = None
    try: ha = int(ha) if ha is not None else None
    except Exception: ha = None
    return h, a, hh, ha


def grade(best: dict[str, Any], result: dict[str, Any], fixture: dict[str, Any]) -> str:
    if not result:
        return "NO_FINAL"
    market = norm(best.get("market"))
    selection = norm(best.get("selection"))
    h, a, hh, ha = final_score(result)
    if h is None or a is None:
        return "NO_FINAL"
    p = period_type(market)
    if p == "2H" and hh is not None and ha is not None:
        ph, pa = h - hh, a - ha
    elif p == "1H" and hh is not None and ha is not None:
        ph, pa = hh, ha
    elif p == "FT":
        ph, pa = h, a
    else:
        return "UNGRADABLE"

    if "both teams to score" in market or market == "btts":
        yes = ph > 0 and pa > 0
        if selection in {"yes", "y", "btts yes"}:
            return "WIN" if yes else "LOSS"
        if selection in {"no", "n", "btts no"}:
            return "WIN" if not yes else "LOSS"
        return "UNGRADABLE"

    if "over/under" in market or market in {"goals over/under", "over/under"} or " over " in f" {selection} " or " under " in f" {selection} ":
        line = parse_line(best)
        if line is None:
            return "UNGRADABLE"
        total = ph + pa
        is_over = selection.startswith("over") or " over " in f" {selection} "
        is_under = selection.startswith("under") or " under " in f" {selection} "
        if not (is_over or is_under):
            return "UNGRADABLE"
        if math.isclose(total, line):
            return "PUSH"
        if is_over:
            return "WIN" if total > line else "LOSS"
        return "WIN" if total < line else "LOSS"

    if market in {"match winner", "winner"}:
        home_name = norm(fixture.get("home_team"))
        away_name = norm(fixture.get("away_team"))
        if h > a:
            actual = "home"
        elif a > h:
            actual = "away"
        else:
            actual = "draw"
        if selection in {"home", "1", home_name}:
            pick = "home"
        elif selection in {"away", "2", away_name}:
            pick = "away"
        elif selection in {"draw", "x"}:
            pick = "draw"
        else:
            return "UNGRADABLE"
        return "WIN" if pick == actual else "LOSS"

    return "UNGRADABLE"


def watch_reason(event: dict[str, Any]) -> str:
    md = event.get("market_decision") or {}
    reason = str(md.get("reason") or "")
    notes = " ".join(str(x) for x in (event.get("notes") or []))
    text = (reason + " " + notes).lower()
    if "first_half" in text or "first half" in text or "period" in text or "not modeled" in text:
        return "UNSUPPORTED_MARKET_MODEL"
    if "lineup" in text or "goalkeeper" in text or "availability" in text or "xi" in text:
        return "AVAILABILITY_NOT_VERIFIED"
    if "market" in text and ("not verified" in text or "missing" in text or "no " in text):
        return "MARKET_NOT_VERIFIED"
    if "raw_projection" in text or "raw projection" in text:
        return "RAW_PROJECTION_NOT_AVAILABLE"
    if "edge" in text or "threshold" in text:
        return "EDGE_BELOW_THRESHOLD"
    if reason:
        return reason[:100]
    return "GENERIC_WATCH"


def summarize_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    c = Counter(r["outcome"] for r in rows)
    decided = c["WIN"] + c["LOSS"]
    return {
        "n": len(rows), "win": c["WIN"], "loss": c["LOSS"], "push": c["PUSH"],
        "ungradable": c["UNGRADABLE"], "no_final": c["NO_FINAL"],
        "hit_rate_ex_push": round(c["WIN"] / decided, 4) if decided else None,
    }


def bucket(value: float | None, cuts: list[tuple[float, float, str]]) -> str | None:
    if value is None:
        return None
    for lo, hi, name in cuts:
        if lo <= value < hi:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history-dir", default="soccer_edge_state/history")
    ap.add_argument("--output", default="soccer_edge_state/analysis/latest.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.history_dir, "*.jsonl")))
    ticks = 0
    bad_lines = 0
    versions = Counter()
    first_ts = None
    last_ts = None
    results: dict[int, dict[str, Any]] = {}
    fixtures: dict[int, dict[str, Any]] = {}
    calls: dict[tuple, dict[str, Any]] = {}
    generic_watches: dict[tuple, dict[str, Any]] = {}
    sporting: dict[int, dict[str, Any]] = {}

    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                ticks += 1
                ver = str(tick.get("version") or "unknown")
                versions[ver] += 1
                ts = tick.get("generated_at_local") or tick.get("generated_at_utc")
                if ts:
                    first_ts = min(first_ts, ts) if first_ts else ts
                    last_ts = max(last_ts, ts) if last_ts else ts
                for ev in tick.get("events") or []:
                    fx = ev.get("fixture") or {}
                    fid = fx.get("fixture_id")
                    if not fid:
                        continue
                    fid = int(fid)
                    fixtures[fid] = fx
                    res = ev.get("result") or {}
                    if res and (res.get("status") in {"FT", "AET", "PEN"} or (res.get("goals") or {}).get("home") is not None):
                        results[fid] = res
                    elif fx.get("status") in {"FT", "AET", "PEN"} and (fx.get("goals") or {}).get("home") is not None:
                        results[fid] = {"goals": fx.get("goals"), "score": fx.get("score"), "status": fx.get("status")}

                    sl = ev.get("sporting_shortlist") or {}
                    raw = ev.get("raw_projection") or {}
                    if sl.get("shortlisted"):
                        rec = sporting.setdefault(fid, {"fixture_id": fid, "fixture": fx, "tracks": set(), "scores": {}, "raw": {}, "versions": set()})
                        rec["tracks"].update(sl.get("tracks") or [])
                        for k in ("side_edge_score", "goal_environment_score", "two_way_scoring_score", "rank"):
                            v = fnum(sl.get(k))
                            if v is not None:
                                rec["scores"][k] = v
                        if raw:
                            rec["raw"] = raw
                        rec["versions"].add(ver)

                    best = ev.get("best_market")
                    cls = str(ev.get("classification") or "").upper()
                    stage = str(ev.get("stage") or "")
                    if isinstance(best, dict) and best:
                        key = (
                            fid, norm(best.get("family")), norm(best.get("market")),
                            norm(best.get("selection")), fnum(best.get("line")),
                        )
                        rec = calls.get(key)
                        current = {
                            "fixture_id": fid, "fixture": fx, "family": best.get("family"), "market": best.get("market"),
                            "selection": best.get("selection"), "line": fnum(best.get("line")), "price": fnum(best.get("decimal_price")),
                            "bookmaker": best.get("bookmaker"), "classification": cls, "tier": ev.get("tier") or best.get("tier"),
                            "stage": stage, "version": ver, "timestamp": ts, "p_raw": fnum(best.get("p_raw")),
                            "p_shrunk": fnum(best.get("p_shrunk")), "p_breakeven": fnum(best.get("p_breakeven")),
                            "p_market_fair": fnum(best.get("p_market_fair")), "edge_pp": fnum(best.get("prob_edge_pp")),
                            "availability_confidence": fnum(ev.get("availability_confidence")),
                            "canonical_ft": canonical_full_match(best),
                        }
                        if rec is None:
                            current["first_seen"] = ts
                            current["last_seen"] = ts
                            current["stages"] = [stage]
                            current["classifications"] = [cls]
                            calls[key] = current
                        else:
                            rec["last_seen"] = ts
                            if stage and stage not in rec["stages"]: rec["stages"].append(stage)
                            if cls and cls not in rec["classifications"]: rec["classifications"].append(cls)
                            if CLASS_RANK.get(cls, -1) >= CLASS_RANK.get(rec.get("classification"), -1):
                                for k2, v2 in current.items():
                                    if k2 not in {"first_seen", "stages", "classifications"}:
                                        rec[k2] = v2
                    elif cls in {"WATCH", "LEAN", "BET"}:
                        reason = watch_reason(ev)
                        key = (fid, stage, reason)
                        generic_watches[key] = {"fixture_id": fid, "fixture": fx, "stage": stage, "classification": cls, "reason": reason, "version": ver, "timestamp": ts}

    rows = []
    for rec in calls.values():
        rec["outcome"] = grade(rec, results.get(rec["fixture_id"], {}), rec["fixture"])
        rec["legacy_invalid_model"] = not rec["canonical_ft"]
        rows.append(rec)
    rows.sort(key=lambda r: (r.get("first_seen") or "", r["fixture_id"], str(r.get("market"))))

    valid = [r for r in rows if r["canonical_ft"]]
    legacy = [r for r in rows if not r["canonical_ft"]]
    watch_valid = [r for r in valid if r.get("classification") == "WATCH"]
    bet_valid = [r for r in valid if r.get("classification") == "BET"]
    lean_valid = [r for r in valid if r.get("classification") == "LEAN"]

    by_market = {}
    for k, group in defaultdict(list).items():
        pass
    market_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in valid:
        market_groups[norm(r.get("market")) or norm(r.get("family"))].append(r)
    by_market = {k: summarize_outcomes(v) for k, v in sorted(market_groups.items())}

    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in valid:
        by_tier[str(r.get("tier") or "NONE")].append(r)

    # Sporting-signal diagnostics are descriptive, not wager grading.
    signal = {
        "GOALS_OVER": {"n": 0, "with_final": 0, "avg_goals": None, "pct_3plus": None},
        "GOALS_UNDER": {"n": 0, "with_final": 0, "avg_goals": None, "pct_0_2": None},
        "TWO_WAY": {"n": 0, "with_final": 0, "btts_rate": None},
        "SIDE": {"n": 0, "with_final": 0, "predicted_side_win_rate": None},
    }
    totals_by_track: dict[str, list[int]] = defaultdict(list)
    btts_vals: list[int] = []
    side_vals: list[int] = []
    score_bins: dict[str, dict[str, list[int]]] = {
        "two_way": defaultdict(list), "goal_environment": defaultdict(list), "side_edge": defaultdict(list)
    }
    for fid, rec in sporting.items():
        tracks = rec["tracks"]
        for t in tracks:
            if t in signal: signal[t]["n"] += 1
        res = results.get(fid)
        if not res:
            continue
        h, a, _, _ = final_score(res)
        if h is None or a is None:
            continue
        total = h + a
        if "GOALS_OVER" in tracks:
            signal["GOALS_OVER"]["with_final"] += 1; totals_by_track["GOALS_OVER"].append(total)
        if "GOALS_UNDER" in tracks:
            signal["GOALS_UNDER"]["with_final"] += 1; totals_by_track["GOALS_UNDER"].append(total)
        if "TWO_WAY" in tracks:
            signal["TWO_WAY"]["with_final"] += 1; btts_vals.append(1 if h > 0 and a > 0 else 0)
        if "SIDE" in tracks:
            raw = rec.get("raw") or {}
            hp = fnum(raw.get("raw_home_win_prob")); ap = fnum(raw.get("raw_away_win_prob"))
            if hp is not None and ap is not None and not math.isclose(hp, ap):
                pred_home = hp > ap
                won = (h > a) if pred_home else (a > h)
                side_vals.append(1 if won else 0); signal["SIDE"]["with_final"] += 1
        tw = fnum(rec["scores"].get("two_way_scoring_score"))
        ge = fnum(rec["scores"].get("goal_environment_score"))
        se = fnum(rec["scores"].get("side_edge_score"))
        if tw is not None:
            b = bucket(tw, [(65,75,"65-74"),(75,85,"75-84"),(85,101,"85+")]);
            if b: score_bins["two_way"][b].append(1 if h>0 and a>0 else 0)
        if ge is not None:
            b = bucket(ge, [(0,30,"<30"),(30,50,"30-49"),(50,70,"50-69"),(70,80,"70-79"),(80,90,"80-89"),(90,101,"90+")]);
            if b: score_bins["goal_environment"][b].append(total)
        if se is not None and "SIDE" in tracks:
            b = bucket(se, [(60,70,"60-69"),(70,80,"70-79"),(80,90,"80-89"),(90,101,"90+")]);
            raw = rec.get("raw") or {}; hp=fnum(raw.get("raw_home_win_prob")); ap=fnum(raw.get("raw_away_win_prob"))
            if b and hp is not None and ap is not None and not math.isclose(hp,ap):
                pred_home=hp>ap; score_bins["side_edge"][b].append(1 if ((h>a) if pred_home else (a>h)) else 0)

    if totals_by_track["GOALS_OVER"]:
        xs=totals_by_track["GOALS_OVER"]; signal["GOALS_OVER"]["avg_goals"]=round(statistics.mean(xs),2); signal["GOALS_OVER"]["pct_3plus"]=round(sum(x>=3 for x in xs)/len(xs),4)
    if totals_by_track["GOALS_UNDER"]:
        xs=totals_by_track["GOALS_UNDER"]; signal["GOALS_UNDER"]["avg_goals"]=round(statistics.mean(xs),2); signal["GOALS_UNDER"]["pct_0_2"]=round(sum(x<=2 for x in xs)/len(xs),4)
    if btts_vals: signal["TWO_WAY"]["btts_rate"]=round(sum(btts_vals)/len(btts_vals),4)
    if side_vals: signal["SIDE"]["predicted_side_win_rate"]=round(sum(side_vals)/len(side_vals),4)

    bins_out = {
        "two_way": {k:{"n":len(v),"btts_rate":round(sum(v)/len(v),4) if v else None} for k,v in score_bins["two_way"].items()},
        "goal_environment": {k:{"n":len(v),"avg_goals":round(statistics.mean(v),2) if v else None,"pct_3plus":round(sum(x>=3 for x in v)/len(v),4) if v else None} for k,v in score_bins["goal_environment"].items()},
        "side_edge": {k:{"n":len(v),"win_rate":round(sum(v)/len(v),4) if v else None} for k,v in score_bins["side_edge"].items()},
    }

    reason_counts = Counter(x["reason"] for x in generic_watches.values())
    out = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "history_files": [os.path.basename(x) for x in files],
        "period": {"first_tick": first_ts, "last_tick": last_ts},
        "ticks_parsed": ticks, "bad_json_lines": bad_lines, "versions": dict(versions),
        "final_results_found": len(results), "unique_fixtures_with_shortlist": len(sporting),
        "market_calls": {
            "unique_total": len(rows), "canonical_ft": len(valid), "legacy_invalid_model": len(legacy),
            "canonical_watch": len(watch_valid), "canonical_lean": len(lean_valid), "canonical_bet": len(bet_valid),
            "canonical_outcomes": summarize_outcomes(valid), "watch_outcomes": summarize_outcomes(watch_valid),
            "lean_outcomes": summarize_outcomes(lean_valid), "bet_outcomes": summarize_outcomes(bet_valid),
            "legacy_outcomes": summarize_outcomes(legacy), "by_market": by_market,
            "by_tier": {k:summarize_outcomes(v) for k,v in sorted(by_tier.items())},
        },
        "generic_watch_count": len(generic_watches), "generic_watch_reasons": dict(reason_counts.most_common()),
        "sporting_signal_diagnostics": signal,
        "score_bin_diagnostics": bins_out,
        "market_call_rows": rows,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
