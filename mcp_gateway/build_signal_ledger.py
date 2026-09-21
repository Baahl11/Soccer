from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter
from typing import Any


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_snapshot(best: Any) -> dict[str, Any] | None:
    if not isinstance(best, dict) or not best:
        return None
    numeric = {"line", "decimal_price", "p_raw", "p_shrunk", "p_breakeven", "p_market_fair", "prob_edge_pp", "ev_pct"}
    return {k: (_f(best.get(k)) if k in numeric else best.get(k)) for k in ("family", "market", "selection", "line", "decimal_price", "bookmaker", "p_raw", "p_shrunk", "p_breakeven", "p_market_fair", "prob_edge_pp", "ev_pct", "tier", "team", "team_name", "participant")}


def _lineup_snapshot(lineups: Any) -> dict[str, Any] | None:
    if not isinstance(lineups, dict) or not lineups:
        return None
    teams = [{"team_id": t.get("team_id"), "team": t.get("team"), "formation": t.get("formation"), "goalkeepers": t.get("goalkeepers")} for t in (lineups.get("teams") or []) if isinstance(t, dict)]
    return {"lineup_state": lineups.get("lineup_state"), "both_xi_confirmed": lineups.get("both_xi_confirmed"), "both_goalkeepers_confirmed": lineups.get("both_goalkeepers_confirmed"), "teams": teams}


def _raw_snapshot(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    out = {
        "status": raw.get("status"), "projection_model": raw.get("projection_model"), "sport_source": raw.get("sport_source"), "market_independent": raw.get("market_independent"),
        "galaxy_model_versions": raw.get("galaxy_model_versions"), "galaxy_data_cutoff": raw.get("galaxy_data_cutoff"),
        "raw_home_goal_rate": _f(raw.get("raw_home_goal_rate")), "raw_away_goal_rate": _f(raw.get("raw_away_goal_rate")), "raw_total_goals": _f(raw.get("raw_total_goals")),
        "raw_home_win_prob": _f(raw.get("raw_home_win_prob")), "raw_draw_prob": _f(raw.get("raw_draw_prob")), "raw_away_win_prob": _f(raw.get("raw_away_win_prob")),
        "raw_btts_yes_prob": _f(raw.get("raw_btts_yes_prob")), "raw_over_2_5_prob": _f(raw.get("raw_over_2_5_prob")), "scoring_path": raw.get("scoring_path"), "screen_scores": raw.get("screen_scores")
    }
    shadow = raw.get("relative_strength_shadow")
    if isinstance(shadow, dict):
        out["relative_strength_shadow"] = shadow
    return out


def _shortlist_snapshot(s: Any) -> dict[str, Any] | None:
    if not isinstance(s, dict) or not s:
        return None
    return {"shortlisted": s.get("shortlisted"), "rank": _f(s.get("rank")), "tracks": s.get("tracks") or [], "side_edge_score": _f(s.get("side_edge_score")), "goal_environment_score": _f(s.get("goal_environment_score")), "two_way_scoring_score": _f(s.get("two_way_scoring_score")), "reason": s.get("reason")}


def _result_snapshot(event: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any] | None:
    result = event.get("result")
    out: dict[str, Any] | None = dict(result) if isinstance(result, dict) and result else None
    if out is None and fixture.get("status") in {"FT", "AET", "PEN"} and (fixture.get("goals") or {}).get("home") is not None:
        out = {"goals": fixture.get("goals"), "score": fixture.get("score"), "status": fixture.get("status")}
    tactical = event.get("postgame_tactical_stats")
    if not isinstance(tactical, dict) and isinstance(out, dict):
        tactical = out.get("tactical_stats")
    if isinstance(tactical, dict):
        if out is None:
            out = {}
        out.setdefault("tactical_stats", tactical)
    return out


def _event_key(row: dict[str, Any]) -> str:
    identity = {k: row.get(k) for k in ("fixture_id", "generated_at_local", "stage", "classification", "best_market", "availability_confidence", "lineups", "result")}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()[:20]


def build_rows(history_dir: str):
    rows = []
    seen = set()
    bad_lines = 0
    ticks = 0
    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    tick = json.loads(line)
                except Exception:
                    bad_lines += 1
                    continue
                ticks += 1
                for event in tick.get("events") or []:
                    fixture = event.get("fixture") or {}
                    fid = fixture.get("fixture_id")
                    if not fid:
                        continue
                    md = event.get("market_decision") or {}
                    coverage = event.get("coverage") or {}
                    row = {
                        "generated_at_local": tick.get("generated_at_local"), "generated_at_utc": tick.get("generated_at_utc"), "timezone": tick.get("timezone") or "America/Mexico_City", "service_version": tick.get("version"), "model_version": tick.get("model_version"),
                        "fixture_id": int(fid), "kickoff_local": fixture.get("kickoff"), "league_id": fixture.get("league_id"), "league": fixture.get("league"), "country": fixture.get("country"), "season": fixture.get("season"), "home_team_id": fixture.get("home_team_id"), "home_team": fixture.get("home_team"), "away_team_id": fixture.get("away_team_id"), "away_team": fixture.get("away_team"), "fixture_status": fixture.get("status"),
                        "event_type": event.get("event_type"), "stage": event.get("stage"), "classification": event.get("classification"), "tier": event.get("tier"), "stake_units": _f(event.get("stake_units")), "bet_eligible": event.get("bet_eligible"), "availability_confidence": _f(event.get("availability_confidence")), "data_tier": coverage.get("data_tier") if isinstance(coverage, dict) else None,
                        "sporting_shortlist": _shortlist_snapshot(event.get("sporting_shortlist")), "sporting_screen_initial": _shortlist_snapshot(event.get("sporting_screen_initial")), "sporting_screen_refined": _shortlist_snapshot(event.get("sporting_screen_refined")), "raw_projection": _raw_snapshot(event.get("raw_projection")), "lineups": _lineup_snapshot(event.get("lineups")), "best_market": _market_snapshot(event.get("best_market")),
                        "market_decision": {"status": md.get("status"), "reason": md.get("reason"), "first_half_markets_ignored": md.get("first_half_markets_ignored"), "period_markets_ignored": md.get("period_markets_ignored"), "btts_market_mode": md.get("btts_market_mode")} if isinstance(md, dict) and md else None,
                        "notes": event.get("notes") or [], "result": _result_snapshot(event, fixture), "error": event.get("error")
                    }
                    row["event_key"] = _event_key(row)
                    if row["event_key"] in seen:
                        continue
                    seen.add(row["event_key"])
                    rows.append(row)
    rows.sort(key=lambda r: (r.get("generated_at_local") or "", r["fixture_id"], r.get("stage") or ""))
    stage_counts = Counter(str(r.get("stage") or "UNKNOWN") for r in rows)
    class_counts = Counter(str(r.get("classification") or "UNKNOWN") for r in rows)
    signal_counts = Counter()
    fixtures = set()
    finals = set()
    with_market = with_raw = with_lineups = with_shadow = with_tactical = 0
    for r in rows:
        fixtures.add(r["fixture_id"])
        if r.get("result"):
            finals.add(r["fixture_id"])
        with_market += bool(r.get("best_market"))
        with_raw += bool(r.get("raw_projection"))
        with_lineups += bool(r.get("lineups"))
        with_shadow += bool((r.get("raw_projection") or {}).get("relative_strength_shadow"))
        with_tactical += bool((r.get("result") or {}).get("tactical_stats"))
        for track in (r.get("sporting_shortlist") or {}).get("tracks") or []:
            signal_counts[str(track)] += 1
    summary = {"schema_version": "1.2.0", "source": "soccer_edge_state/history/*.jsonl", "timezone_basis": "America/Mexico_City", "ticks_read": ticks, "bad_lines": bad_lines, "rows": len(rows), "unique_fixtures": len(fixtures), "fixtures_with_final_result": len(finals), "rows_with_raw_projection": with_raw, "rows_with_relative_strength_shadow": with_shadow, "rows_with_market": with_market, "rows_with_lineups": with_lineups, "rows_with_tactical_stats": with_tactical, "stage_counts": dict(sorted(stage_counts.items())), "classification_counts": dict(sorted(class_counts.items())), "sport_signal_counts": dict(sorted(signal_counts.items())), "first_generated_at_local": rows[0].get("generated_at_local") if rows else None, "last_generated_at_local": rows[-1].get("generated_at_local") if rows else None}
    return rows, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history-dir", default="soccer_edge_state/history")
    p.add_argument("--output", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    p.add_argument("--summary-output", default="soccer_edge_state/analysis/signal_ledger_summary.json")
    a = p.parse_args()
    rows, summary = build_rows(a.history_dir)
    os.makedirs(os.path.dirname(a.output), exist_ok=True)
    with open(a.output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(a.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
