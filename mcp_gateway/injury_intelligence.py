from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base

SCHEMA_VERSION = "1.0.0"
PLAYER_TRENDS_URL = "https://raw.githubusercontent.com/Baahl11/Soccer/soccer-edge-state/soccer_edge_state/analysis/player_trends.json"
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def load_player_trends() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("injury_player_trends", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(PLAYER_TRENDS_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "RESEARCH_ONLY_PLAYER_TRENDS":
        return None
    base._cache_set("injury_player_trends", "latest", payload, now)
    return payload


def _player_index(report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not isinstance(report, dict):
        return out
    for row in report.get("players") or []:
        if isinstance(row, dict) and row.get("player_id") is not None:
            out[str(row["player_id"])] = row
    return out


def _starter_ids(event: dict[str, Any]) -> set[str]:
    lineups = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    out: set[str] = set()
    for team in lineups.get("teams") or []:
        if not isinstance(team, dict):
            continue
        for player in team.get("starters") or []:
            if isinstance(player, dict) and player.get("id") is not None:
                out.add(str(player["id"]))
    return out


def _reported_category(row: dict[str, Any]) -> str:
    text = f"{_norm(row.get('type'))} {_norm(row.get('reason'))}"
    if any(token in text for token in ("suspension", "suspended", "ban", "banned")):
        return "SUSPENSION_REPORTED"
    if any(token in text for token in ("injury", "injured", "muscle", "knee", "ankle", "hamstring", "illness", "sick")):
        return "INJURY_REPORTED"
    return "REPORTED_AVAILABILITY_ISSUE_UNCLASSIFIED"


def _role_exposure(profile: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(profile, dict):
        return {
            "history_available": False,
            "role_exposure_score": None,
            "materiality_review_required": False,
        }
    windows = profile.get("windows") if isinstance(profile.get("windows"), dict) else {}
    block = windows.get("last_10") if isinstance(windows.get("last_10"), dict) else {}
    avg_minutes = _num(block.get("avg_minutes"))
    played_n = int(block.get("played_n") or 0)
    if avg_minutes is None:
        score = None
    else:
        sample_factor = min(1.0, played_n / 8.0)
        score = max(0.0, min(1.0, (avg_minutes / 90.0) * sample_factor))
    position = profile.get("last_known_position")
    material = bool(
        (score is not None and score >= 0.55 and played_n >= 5)
        or str(position or "").upper() in {"G", "GK", "GOALKEEPER"} and played_n >= 5
    )
    return {
        "history_available": True,
        "last_known_position": position,
        "last10_played_n": played_n,
        "last10_avg_minutes": avg_minutes,
        "role_exposure_score": round(score, 4) if score is not None else None,
        "role_exposure_definition": "avg_minutes/90 shrunk by captured last-10 sample; descriptive usage only, NOT player impact",
        "materiality_review_required": material,
    }


def build(event: dict[str, Any], trends: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    injuries = event.get("injuries")
    if injuries == "NOT VERIFIED" or injuries is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "CURRENT_INJURY_REPORT_NOT_AVAILABLE",
            "actionable": False,
            "decision_weight": 0.0,
        }
    if not isinstance(injuries, list):
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "status": "NOT_VERIFIED",
            "reason": "UNSUPPORTED_INJURY_PAYLOAD_SHAPE",
            "actionable": False,
            "decision_weight": 0.0,
        }

    profiles = _player_index(trends)
    starters = _starter_ids(event)
    lineup = event.get("lineups") if isinstance(event.get("lineups"), dict) else {}
    both_xi = bool(lineup.get("both_xi_confirmed"))
    rows = []
    conflicts = 0
    material = 0
    suspensions = 0

    for report in injuries:
        if not isinstance(report, dict):
            continue
        pid = report.get("player_id")
        exposure = _role_exposure(profiles.get(str(pid)) if pid is not None else None)
        category = _reported_category(report)
        if category == "SUSPENSION_REPORTED":
            suspensions += 1
        starter_conflict = bool(pid is not None and str(pid) in starters)
        if starter_conflict:
            conflicts += 1
        if exposure.get("materiality_review_required"):
            material += 1
        rows.append({
            "player_id": pid,
            "player": report.get("player"),
            "team_id": report.get("team_id"),
            "team": report.get("team"),
            "provider_type": report.get("type"),
            "provider_reason": report.get("reason"),
            "reported_category": category,
            "role_exposure": exposure,
            "appears_in_confirmed_starting_xi": starter_conflict,
            "data_conflict": starter_conflict,
            "official_independent_verification": False,
            "quantified_team_goal_impact": None,
            "replacement_quality": "NOT_MODELED",
        })

    status = "LIVE_RESEARCH_CONTEXT"
    if conflicts:
        status = "DATA_CONFLICT_REQUIRES_RECHECK"
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": status,
        "provider_report_count": len(rows),
        "suspension_report_count": suspensions,
        "materiality_review_count": material,
        "starter_conflict_count": conflicts,
        "both_xi_confirmed": both_xi,
        "reports": rows,
        "material_availability_recheck_required": bool(material > 0 and not both_xi) or bool(conflicts),
        "actionable": False,
        "decision_weight": 0.0,
        "canonical_availability_confidence_changed": False,
        "missing": [
            "INDEPENDENT_OFFICIAL_TEAM_SOURCE_VERIFICATION_NOT_AUTOMATED",
            "PLAYER_REPLACEMENT_QUALITY_MODEL_NOT_AVAILABLE",
            "PLAYER_TO_TEAM_GOAL_IMPACT_NOT_OOS_VALIDATED",
            "BENCH_PLAYER_IDENTITIES_NOT_RETAINED_IN_CURRENT_LINEUP_COMPACT",
        ],
        "policy": "PROVIDER AVAILABILITY REPORT + HISTORICAL ROLE EXPOSURE ONLY; ROLE EXPOSURE IS NOT IMPACT; STARTER CONFLICT TRIGGERS RECHECK; NO CANONICAL PROBABILITY CHANGE",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    events = [
        event for event in (payload.get("events") or [])
        if isinstance(event, dict)
        and event.get("event_type") == "SOCCER_REFRESH"
        and event.get("stage") not in {"POSTGAME", "HT", "CLOSE"}
    ]
    needs_trends = any(
        isinstance(event.get("injuries"), list) and bool(event.get("injuries"))
        for event in events
    )
    trends = load_player_trends() if needs_trends else None
    context_events = material_events = conflict_events = reports = 0
    for event in events:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event, trends)
        event["injury_intelligence"] = intel
        if intel.get("status") in {"LIVE_RESEARCH_CONTEXT", "DATA_CONFLICT_REQUIRES_RECHECK"}:
            context_events += 1
        if intel.get("material_availability_recheck_required"):
            material_events += 1
        if int(intel.get("starter_conflict_count") or 0) > 0:
            conflict_events += 1
        reports += int(intel.get("provider_report_count") or 0)
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["injuries"] = intel
    return {
        "player_trends_loaded": bool(trends),
        "events_with_availability_context": context_events,
        "provider_reports_profiled": reports,
        "events_requiring_material_availability_recheck": material_events,
        "events_with_report_vs_starting_xi_conflict": conflict_events,
        "provider_requests_added": 0,
    }
