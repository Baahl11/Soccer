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


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().upper().split())


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


def _same_direction(left: Any, right: Any) -> bool:
    a = _norm(left)
    b = _norm(right)
    if not a or not b:
        return True
    for token in ("OVER", "UNDER", "YES", "NO", "HOME", "DRAW", "AWAY"):
        a_has = token in a
        b_has = token in b
        if a_has or b_has:
            return a_has == b_has
    return a == b


def _phase16_calibration_snapshot(tick: dict[str, Any], fixture_id: int, best: Any) -> dict[str, Any] | None:
    """Preserve same-tick Phase16 calibration lineage for future audits.

    This is metadata-only. It never changes the selected market, classification,
    probability, price, tier or stake. The match is fixture + line + selection
    direction, with nearest observed price used only to disambiguate duplicate rows.
    """
    if not isinstance(best, dict) or not best:
        return None
    rows = tick.get("match_table_rows")
    if not isinstance(rows, list):
        return None

    best_line = _f(best.get("line"))
    best_price = _f(best.get("decimal_price"))
    best_selection = best.get("selection")
    best_family = _norm(best.get("family"))
    best_market = _norm(best.get("market"))

    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            row_fixture = int(row.get("fixture_id"))
        except (TypeError, ValueError):
            continue
        if row_fixture != int(fixture_id):
            continue

        row_line = _f(row.get("line"))
        if best_line is not None or row_line is not None:
            if best_line is None or row_line is None or abs(best_line - row_line) > 1e-9:
                continue
        if not _same_direction(best_selection, row.get("selection")):
            continue

        row_family = _norm(row.get("market_family"))
        row_market = _norm(row.get("market"))
        if best_family and row_family and best_family != row_family:
            family_aliases = {
                "FT_TOTALS": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
                "TOTAL": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
                "FT_TOTALS_RESEARCH": {"FT_TOTALS", "TOTAL", "FT_TOTALS_RESEARCH"},
                "BTTS": {"BTTS", "FT_BTTS", "FT_BTTS_RESEARCH"},
                "FT_BTTS": {"BTTS", "FT_BTTS", "FT_BTTS_RESEARCH"},
                "1X2": {"1X2", "FT_1X2", "FT_1X2_RESEARCH", "MATCH_WINNER"},
                "FT_1X2": {"1X2", "FT_1X2", "FT_1X2_RESEARCH", "MATCH_WINNER"},
            }
            allowed = family_aliases.get(best_family, {best_family})
            if row_family not in allowed:
                continue
        elif best_market and row_market and best_market != row_market:
            continue
        candidates.append(row)

    if not candidates:
        return None

    def distance(row: dict[str, Any]) -> float:
        row_price = _f(row.get("price"))
        if best_price is None or row_price is None:
            return 999999.0
        return abs(best_price - row_price)

    row = min(candidates, key=distance)
    diag = row.get("phase16_binary_calibration_diagnostics")
    diag = diag if isinstance(diag, dict) else {}
    calibrated_fields: dict[str, float] = {}
    for key in ("p_model_calibrated", "p_calibrated", "calibrated_probability", "model_probability_calibrated"):
        value = _f(row.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            calibrated_fields[key] = value

    diag_snapshot = None
    if diag:
        discrimination = diag.get("discrimination") if isinstance(diag.get("discrimination"), dict) else {}
        diag_snapshot = {
            "target": diag.get("target"),
            "source_model_version": diag.get("source_model_version"),
            "requested_model_version": diag.get("requested_model_version"),
            "source_model_version_matches": diag.get("source_model_version_matches"),
            "rows": diag.get("rows"),
            "positive_count": diag.get("positive_count"),
            "negative_count": diag.get("negative_count"),
            "auc": diag.get("auc") if diag.get("auc") is not None else discrimination.get("auc"),
            "auc_lower_95": diag.get("auc_lower_95") if diag.get("auc_lower_95") is not None else discrimination.get("auc_lower_95"),
            "brier_delta": diag.get("brier_delta"),
            "log_loss_delta": diag.get("log_loss_delta"),
            "eligible_for_phase16_research": diag.get("eligible_for_phase16_research"),
            "calibrator_status": diag.get("calibrator_status"),
        }

    return {
        "source": "SAME_TICK_MATCH_TABLE_ROW",
        "market_family": row.get("market_family"),
        "market": row.get("market"),
        "selection": row.get("selection"),
        "line": _f(row.get("line")),
        "price": _f(row.get("price")),
        "bookmaker": row.get("bookmaker"),
        "calibration_status": row.get("phase16_calibration_status"),
        "calibration_target": diag.get("target") or row.get("phase16_calibration_target"),
        "promotion_shadow_eligible": row.get("phase16_calibration_promotion_shadow_eligible") is True,
        "calibrated_probability_fields": calibrated_fields,
        "binary_calibration_diagnostics": diag_snapshot,
    }


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
                    best_market = _market_snapshot(event.get("best_market"))
                    row = {
                        "generated_at_local": tick.get("generated_at_local"), "generated_at_utc": tick.get("generated_at_utc"), "timezone": tick.get("timezone") or "America/Mexico_City", "service_version": tick.get("version"), "model_version": tick.get("model_version"),
                        "fixture_id": int(fid), "kickoff_local": fixture.get("kickoff"), "league_id": fixture.get("league_id"), "league": fixture.get("league"), "country": fixture.get("country"), "season": fixture.get("season"), "home_team_id": fixture.get("home_team_id"), "home_team": fixture.get("home_team"), "away_team_id": fixture.get("away_team_id"), "away_team": fixture.get("away_team"), "fixture_status": fixture.get("status"),
                        "event_type": event.get("event_type"), "stage": event.get("stage"), "classification": event.get("classification"), "tier": event.get("tier"), "stake_units": _f(event.get("stake_units")), "bet_eligible": event.get("bet_eligible"), "availability_confidence": _f(event.get("availability_confidence")), "data_tier": coverage.get("data_tier") if isinstance(coverage, dict) else None,
                        "sporting_shortlist": _shortlist_snapshot(event.get("sporting_shortlist")), "sporting_screen_initial": _shortlist_snapshot(event.get("sporting_screen_initial")), "sporting_screen_refined": _shortlist_snapshot(event.get("sporting_screen_refined")), "raw_projection": _raw_snapshot(event.get("raw_projection")), "lineups": _lineup_snapshot(event.get("lineups")), "best_market": best_market,
                        "phase16_calibration_provenance": _phase16_calibration_snapshot(tick, int(fid), best_market),
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
    with_market = with_raw = with_lineups = with_shadow = with_tactical = with_phase16_provenance = phase16_promotion_shadow_eligible = 0
    for r in rows:
        fixtures.add(r["fixture_id"])
        if r.get("result"):
            finals.add(r["fixture_id"])
        with_market += bool(r.get("best_market"))
        with_raw += bool(r.get("raw_projection"))
        with_lineups += bool(r.get("lineups"))
        with_shadow += bool((r.get("raw_projection") or {}).get("relative_strength_shadow"))
        with_tactical += bool((r.get("result") or {}).get("tactical_stats"))
        provenance = r.get("phase16_calibration_provenance")
        with_phase16_provenance += isinstance(provenance, dict) and bool(provenance)
        phase16_promotion_shadow_eligible += isinstance(provenance, dict) and provenance.get("promotion_shadow_eligible") is True
        for track in (r.get("sporting_shortlist") or {}).get("tracks") or []:
            signal_counts[str(track)] += 1
    summary = {"schema_version": "1.3.0", "source": "soccer_edge_state/history/*.jsonl", "timezone_basis": "America/Mexico_City", "ticks_read": ticks, "bad_lines": bad_lines, "rows": len(rows), "unique_fixtures": len(fixtures), "fixtures_with_final_result": len(finals), "rows_with_raw_projection": with_raw, "rows_with_relative_strength_shadow": with_shadow, "rows_with_market": with_market, "rows_with_lineups": with_lineups, "rows_with_tactical_stats": with_tactical, "rows_with_phase16_calibration_provenance": with_phase16_provenance, "rows_phase16_promotion_shadow_eligible": phase16_promotion_shadow_eligible, "stage_counts": dict(sorted(stage_counts.items())), "classification_counts": dict(sorted(class_counts.items())), "sport_signal_counts": dict(sorted(signal_counts.items())), "first_generated_at_local": rows[0].get("generated_at_local") if rows else None, "last_generated_at_local": rows[-1].get("generated_at_local") if rows else None, "provider_requests_added": 0, "canonical_bet_logic_changed": False, "model_weights_changed": False}
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
