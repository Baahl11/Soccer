from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
REPORT_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/formation_intelligence.json"
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_formation(value: Any) -> str | None:
    text = str(value or "").strip().upper().replace(" ", "").replace("–", "-").replace("—", "-")
    nums = re.findall(r"\d+", text)
    return "-".join(nums) if len(nums) >= 3 else None


def load_report() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("formation_live_report", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REPORT_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "RESEARCH_ONLY_FORMATION_INTELLIGENCE":
        return None
    base._cache_set("formation_live_report", "latest", payload, now)
    return payload


def _current_pair(event: dict[str, Any]) -> tuple[str, str] | None:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    if not lineups.get("both_xi_confirmed"):
        return None
    teams = lineups.get("teams") or []
    by_id: dict[int, str] = {}
    for row in teams:
        if not isinstance(row, dict) or row.get("team_id") is None:
            continue
        formation = _norm_formation(row.get("formation"))
        if formation:
            by_id[int(row["team_id"])] = formation
    hid = fixture.get("home_team_id")
    aid = fixture.get("away_team_id")
    if hid is None or aid is None:
        return None
    home = by_id.get(int(hid)); away = by_id.get(int(aid))
    return (home, away) if home and away else None


def _lookup_matchup(report: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    descriptive = report.get("descriptive") if isinstance(report, dict) and isinstance(report.get("descriptive"), dict) else {}
    for row in descriptive.get("matchups") or []:
        if isinstance(row, dict) and str(row.get("matchup") or "") == key:
            return row
    return None


def build(event: dict[str, Any], report: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    pair = _current_pair(event)
    if pair is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "BOTH_CONFIRMED_XI_FORMATIONS_NOT_AVAILABLE",
            "actionable": False,
            "decision_weight": 0.0,
        }
    home_form, away_form = pair
    key = f"{home_form} vs {away_form}"
    matchup = _lookup_matchup(report, key)
    wf = report.get("walk_forward_residual_test_over_2_5") if isinstance(report, dict) and isinstance(report.get("walk_forward_residual_test_over_2_5"), dict) else {}
    improvement = wf.get("improvement") if isinstance(wf.get("improvement"), dict) else {}
    baseline = wf.get("baseline") if isinstance(wf.get("baseline"), dict) else {}
    brier_delta = _num(improvement.get("brier_delta"))
    log_delta = _num(improvement.get("log_loss_delta"))
    eval_n = int(baseline.get("n") or 0)
    aggregate_lift = bool(eval_n >= 100 and brier_delta is not None and log_delta is not None and brier_delta < 0 and log_delta < 0)
    matchup_n = int((matchup or {}).get("n") or 0)

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_PROFILE",
        "home_formation": home_form,
        "away_formation": away_form,
        "matchup": key,
        "formation_source": "CONFIRMED_API_STARTING_XI",
        "historical_matchup": matchup if matchup is not None else {"n": 0, "status": "NO_MATCHUP_HISTORY"},
        "sample_band": "HIGH" if matchup_n >= 20 else "MEDIUM" if matchup_n >= 8 else "LOW",
        "aggregate_oos_residual_test": {
            "evaluations": eval_n,
            "brier_delta_challenger_minus_baseline": brier_delta,
            "log_loss_delta_challenger_minus_baseline": log_delta,
            "aggregate_lift_gate_passed": aggregate_lift,
        },
        "feature_candidate": bool(aggregate_lift and matchup_n >= 8),
        "actionable": False,
        "decision_weight": 0.0,
        "promotion_block": "FORMATION_EFFECT_NOT_VERSIONED_OR_PRODUCTION_APPROVED",
        "policy": "DESCRIPTIVE MATCHUP HISTORY NEVER UPGRADES BET; ONLY OOS RESIDUAL LIFT MAY JUSTIFY FUTURE FEATURE WEIGHT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    report = load_report()
    confirmed = profiled = candidates = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, report)
        event["formation_live_intelligence"] = intel
        if intel.get("home_formation") and intel.get("away_formation"):
            confirmed += 1
        if intel.get("status") == "LIVE_RESEARCH_PROFILE":
            profiled += 1
        if intel.get("feature_candidate"):
            candidates += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["formations"] = intel
    return {
        "formation_report_loaded": bool(report),
        "confirmed_formation_pairs": confirmed,
        "profiled_events": profiled,
        "oos_feature_candidate_events": candidates,
        "provider_requests_added": 0,
    }
