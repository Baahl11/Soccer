from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
from collections import Counter, defaultdict
from typing import Any


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _market_snapshot(best: Any) -> dict[str, Any] | None:
    if not isinstance(best, dict) or not best:
        return None
    return {
        "family": best.get("family"),
        "market": best.get("market"),
        "selection": best.get("selection"),
        "line": _f(best.get("line")),
        "decimal_price": _f(best.get("decimal_price")),
        "bookmaker": best.get("bookmaker"),
        "p_raw": _f(best.get("p_raw")),
        "p_shrunk": _f(best.get("p_shrunk")),
        "p_breakeven": _f(best.get("p_breakeven")),
        "p_market_fair": _f(best.get("p_market_fair")),
        "prob_edge_pp": _f(best.get("prob_edge_pp")),
        "ev_pct": _f(best.get("ev_pct")),
        "tier": best.get("tier"),
    }


def _lineup_snapshot(lineups: Any) -> dict[str, Any] | None:
    if not isinstance(lineups, dict) or not lineups:
        return None
    teams = []
    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        teams.append({
            "team_id": team.get("team_id"),
            "team": team.get("team"),
            "formation": team.get("formation"),
            "goalkeepers": team.get("goalkeepers"),
        })
    return {
        "lineup_state": lineups.get("lineup_state"),
        "both_xi_confirmed": lineups.get("both_xi_confirmed"),
        "both_goalkeepers_confirmed": lineups.get("both_goalkeepers_confirmed"),
        "teams": teams,
    }


def _raw_snapshot(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict) or not raw:
        return None
    return {
        "status": raw.get("status"),
        "projection_model": raw.get("projection_model"),
        "sport_source": raw.get("sport_source"),
        "market_independent": raw.get("market_independent"),
        "galaxy_model_versions": raw.get("galaxy_model_versions"),
        "galaxy_data_cutoff": raw.get("galaxy_data_cutoff"),
        "raw_home_goal_rate": _f(raw.get("raw_home_goal_rate")),
        "raw_away_goal_rate": _f(raw.get("raw_away_goal_rate")),
        "raw_total_goals": _f(raw.get("raw_total_goals")),
        "raw_home_win_prob": _f(raw.get("raw_home_win_prob")),
        "raw_draw_prob": _f(raw.get("raw_draw_prob")),
        "raw_away_win_prob": _f(raw.get("raw_away_win_prob")),
        "raw_btts_yes_prob": _f(raw.get("raw_btts_yes_prob")),
        "raw_over_2_5_prob": _f(raw.get("raw_over_2_5_prob")),
        "scoring_path": raw.get("scoring_path"),
        "screen_scores": raw.get("screen_scores"),
    }


def _shortlist_snapshot(shortlist: Any) -> dict[str, Any] | None:
    if not isinstance(shortlist, dict) or not shortlist:
        return None
    return {
        "shortlisted": shortlist.get("shortlisted"),
        "rank": _f(shortlist.get("rank")),
        "tracks": shortlist.get("tracks") or [],
        "side_edge_score": _f(shortlist.get("side_edge_score")),
        "goal_environment_score": _f(shortlist.get("goal_environment_score")),
        "two_way_scoring_score": _f(shortlist.get("two_way_scoring_score")),
        "reason": shortlist.get("reason"),
    }


def _result_snapshot(event: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any] | None:
    result = event.get("result")
    if isinstance(result, dict) and result:
        return result
    if fixture.get("status") in {"FT", "AET", "PEN"} and (fixture.get("goals") or {}).get("home") is not None:
        return {
            "goals": fixture.get("goals"),
            "score": fixture.get("score"),
            "status": fixture.get("status"),
        }
    return None


def _event_key(row: dict[str, Any]) -> str:
    identity = {
        "fixture_id": row.get("fixture_id"),
        "generated_at_local": row.get("generated_at_local"),
        "stage": row.get("stage"),
        "classification": row.get("classification"),
        "market": row.get("best_market"),
        "availability_confidence": row.get("availability_confidence"),
        "lineups": row.get("lineups"),
        "result": row.get("result"),
    }
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def build_rows(history_dir: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    bad_lines = 0
    ticks = 0

    for path in sorted(glob.glob(os.path.join(history_dir, "*.jsonl"))):
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
                generated_local = tick.get("generated_at_local")
                generated_utc = tick.get("generated_at_utc")
                for event in tick.get("events") or []:
                    fixture = event.get("fixture") or {}
                    fixture_id = fixture.get("fixture_id")
                    if not fixture_id:
                        continue
                    md = event.get("market_decision") or {}
                    coverage = event.get("coverage") or {}
                    row = {
                        "generated_at_local": generated_local,
                        "generated_at_utc": generated_utc,
                        "timezone": tick.get("timezone") or "America/Mexico_City",
                        "service_version": tick.get("version"),
                        "model_version": tick.get("model_version"),
                        "fixture_id": int(fixture_id),
                        "kickoff_local": fixture.get("kickoff"),
                        "league_id": fixture.get("league_id"),
                        "league": fixture.get("league"),
                        "country": fixture.get("country"),
                        "season": fixture.get("season"),
                        "home_team_id": fixture.get("home_team_id"),
                        "home_team": fixture.get("home_team"),
                        "away_team_id": fixture.get("away_team_id"),
                        "away_team": fixture.get("away_team"),
                        "fixture_status": fixture.get("status"),
                        "event_type": event.get("event_type"),
                        "stage": event.get("stage"),
                        "classification": event.get("classification"),
                        "tier": event.get("tier"),
                        "stake_units": _f(event.get("stake_units")),
                        "bet_eligible": event.get("bet_eligible"),
                        "availability_confidence": _f(event.get("availability_confidence")),
                        "data_tier": coverage.get("data_tier") if isinstance(coverage, dict) else None,
                        "sporting_shortlist": _shortlist_snapshot(event.get("sporting_shortlist")),
                        "sporting_screen_initial": _shortlist_snapshot(event.get("sporting_screen_initial")),
                        "sporting_screen_refined": _shortlist_snapshot(event.get("sporting_screen_refined")),
                        "raw_projection": _raw_snapshot(event.get("raw_projection")),
                        "lineups": _lineup_snapshot(event.get("lineups")),
                        "best_market": _market_snapshot(event.get("best_market")),
                        "market_decision": {
                            "status": md.get("status"),
                            "reason": md.get("reason"),
                            "first_half_markets_ignored": md.get("first_half_markets_ignored"),
                            "period_markets_ignored": md.get("period_markets_ignored"),
                            "btts_market_mode": md.get("btts_market_mode"),
                        } if isinstance(md, dict) and md else None,
                        "notes": event.get("notes") or [],
                        "result": _result_snapshot(event, fixture),
                        "error": event.get("error"),
                    }
                    row["event_key"] = _event_key(row)
                    if row["event_key"] in seen:
                        continue
                    seen.add(row["event_key"])
                    rows.append(row)

    rows.sort(key=lambda r: (r.get("generated_at_local") or "", r["fixture_id"], r.get("stage") or ""))

    stage_counts = Counter(str(r.get("stage") or "UNKNOWN") for r in rows)
    class_counts = Counter(str(r.get("classification") or "UNKNOWN") for r in rows)
    signal_counts: Counter[str] = Counter()
    fixtures = set()
    final_fixtures = set()
    with_market = 0
    with_raw = 0
    with_lineups = 0
    for r in rows:
        fixtures.add(r["fixture_id"])
        if r.get("result"):
            final_fixtures.add(r["fixture_id"])
        if r.get("best_market"):
            with_market += 1
        if r.get("raw_projection"):
            with_raw += 1
        if r.get("lineups"):
            with_lineups += 1
        sl = r.get("sporting_shortlist") or {}
        for track in sl.get("tracks") or []:
            signal_counts[str(track)] += 1

    summary = {
        "schema_version": "1.0.0",
        "source": "soccer_edge_state/history/*.jsonl",
        "timezone_basis": "America/Mexico_City",
        "ticks_read": ticks,
        "bad_lines": bad_lines,
        "rows": len(rows),
        "unique_fixtures": len(fixtures),
        "fixtures_with_final_result": len(final_fixtures),
        "rows_with_raw_projection": with_raw,
        "rows_with_market": with_market,
        "rows_with_lineups": with_lineups,
        "stage_counts": dict(sorted(stage_counts.items())),
        "classification_counts": dict(sorted(class_counts.items())),
        "sport_signal_counts": dict(sorted(signal_counts.items())),
        "first_generated_at_local": rows[0].get("generated_at_local") if rows else None,
        "last_generated_at_local": rows[-1].get("generated_at_local") if rows else None,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized Soccer Edge historical signal ledger without new provider/API calls.")
    parser.add_argument("--history-dir", default="soccer_edge_state/history")
    parser.add_argument("--output", default="soccer_edge_state/analysis/signal_ledger.jsonl")
    parser.add_argument("--summary-output", default="soccer_edge_state/analysis/signal_ledger_summary.json")
    args = parser.parse_args()

    rows, summary = build_rows(args.history_dir)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(args.summary_output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
