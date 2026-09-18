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
    ages: list[float] = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("status") or "") not in _FINISHED:
            continue
        dt = _dt(row.get("kickoff"))
        if dt is None or dt >= kickoff:
            continue
        ages.append((kickoff - dt).total_seconds() / 86400.0)
    return {
        "verified_completed_recent_fixtures": len(ages),
        "matches_last_7d": sum(1 for x in ages if x <= 7.0),
        "matches_last_14d": sum(1 for x in ages if x <= 14.0),
        "matches_last_21d": sum(1 for x in ages if x <= 21.0),
        "minutes_load": "NOT_VERIFIED",
        "rotation_pressure": "NOT_MODELED",
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
    home_rows = _recent(event, "home")
    away_rows = _recent(event, "away")
    home = _profile(home_rows, kickoff)
    away = _profile(away_rows, kickoff)
    available = bool(home_rows) and bool(away_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "status": "LIVE_RESEARCH_CONGESTION_CONTEXT" if available else "CONGESTION_CONTEXT_INCOMPLETE",
        "home": {"team": fixture.get("home_team"), **home},
        "away": {"team": fixture.get("away_team"), **away},
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "CONTEXT_ONLY_NOT_ACTIONABLE",
        "promotion_gate": {
            "minimum_oos_feature_lift_sample": 500,
            "requires_minutes_or_rotation_enrichment_for_full_model": True,
            "requires_competition_specific_calibration": True,
        },
        "policy": (
            "FIXTURE DENSITY ONLY FROM VERIFIED COMPLETED RECENT FIXTURES. "
            "MATCH COUNTS ARE NOT PLAYER MINUTES OR ROTATION PRESSURE. ZERO MODEL WEIGHT UNTIL OOS LIFT."
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
        event["congestion_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_CONGESTION_CONTEXT":
            attached += 1
        else:
            incomplete += 1
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["congestion"] = intel
    return {"context_events": attached, "incomplete_events": incomplete, "provider_requests_added": 0}
