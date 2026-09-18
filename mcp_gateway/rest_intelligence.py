from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "1.0.0"
_FINISHED = {"FT", "AET", "PEN"}


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if out.tzinfo is None:
            out = out.replace(tzinfo=timezone.utc)
        return out.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _recent(event: dict[str, Any], role: str) -> list[dict[str, Any]]:
    sporting = event.get("sporting") if isinstance(event.get("sporting"), dict) else {}
    rows = sporting.get(f"{role}_recent")
    return rows if isinstance(rows, list) else []


def _profile(rows: list[dict[str, Any]], kickoff: datetime) -> dict[str, Any]:
    completed: list[datetime] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") not in _FINISHED:
            continue
        dt = _dt(row.get("kickoff"))
        if dt is not None and dt < kickoff:
            completed.append(dt)
    completed.sort(reverse=True)
    if not completed:
        return {
            "status": "NO_VERIFIED_COMPLETED_RECENT_FIXTURE",
            "last_match_utc": None,
            "rest_hours": None,
            "rest_days": None,
            "rest_bucket": "NOT_VERIFIED",
        }
    delta_hours = (kickoff - completed[0]).total_seconds() / 3600.0
    days = delta_hours / 24.0
    if days < 3.0:
        bucket = "UNDER_3_DAYS"
    elif days < 4.0:
        bucket = "3_TO_4_DAYS"
    elif days < 7.0:
        bucket = "4_TO_7_DAYS"
    else:
        bucket = "7_PLUS_DAYS"
    return {
        "status": "VERIFIED_FROM_RECENT_FIXTURES",
        "last_match_utc": completed[0].isoformat(),
        "rest_hours": round(delta_hours, 2),
        "rest_days": round(days, 3),
        "rest_bucket": bucket,
        "materiality": "NOT_OOS_VALIDATED",
    }


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    kickoff = _dt(fixture.get("kickoff"))
    if kickoff is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "CURRENT_KICKOFF_NOT_VERIFIED",
            "actionable": False,
            "decision_weight": 0.0,
        }
    home = _profile(_recent(event, "home"), kickoff)
    away = _profile(_recent(event, "away"), kickoff)
    available = home.get("rest_days") is not None and away.get("rest_days") is not None
    rest_gap = None
    if available:
        rest_gap = round(float(home["rest_days"]) - float(away["rest_days"]), 3)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_REST_CONTEXT" if available else "REST_CONTEXT_INCOMPLETE",
        "home": {"team": fixture.get("home_team"), **home},
        "away": {"team": fixture.get("away_team"), **away},
        "home_minus_away_rest_days": rest_gap,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "promotion_gate": {
            "minimum_oos_feature_lift_sample": 500,
            "requires_league_schedule_calibration": True,
            "requires_no_brier_logloss_degradation": True,
        },
        "policy": (
            "REST IS CALCULATED ONLY FROM VERIFIED COMPLETED RECENT FIXTURE DATES. "
            "NO LAMBDA OR BET ADJUSTMENT UNTIL OOS MATERIALITY IS VALIDATED."
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    attached = incomplete = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        if event.get("stage") in {"POSTGAME", "HT", "CLOSE"}:
            continue
        intel = build(event)
        event["rest_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_REST_CONTEXT":
            attached += 1
        else:
            incomplete += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["rest"] = intel
    return {"context_events": attached, "incomplete_events": incomplete, "provider_requests_added": 0}
