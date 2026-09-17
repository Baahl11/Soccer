from __future__ import annotations

from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

from mcp_gateway import automation as base
from mcp_gateway import automation_v6 as v6
from mcp_gateway import automation_v35 as v35
from mcp_gateway import match_intelligence_v1 as intelligence

MODEL_VERSION = v35.MODEL_VERSION
AUTOMATION_VERSION = "3.12.0"
PROJECTION_KEY_PREFIX = "_sport_projection_v1:"
PROJECTION_TTL = timedelta(hours=12)
LATE_STAGES = {"T-60", "T-40", "T-30", "T-20", "T-10", "CLOSE"}


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        out = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if out.tzinfo is None:
        out = out.replace(tzinfo=dt_timezone.utc)
    return out.astimezone(dt_timezone.utc)


def _usable_projection(raw: Any) -> bool:
    return isinstance(raw, dict) and any(
        raw.get(key) is not None
        for key in (
            "raw_total_goals",
            "raw_over_2_5_prob",
            "raw_btts_yes_prob",
            "raw_home_win_prob",
            "raw_draw_prob",
            "raw_away_win_prob",
            "raw_home_goal_rate",
            "raw_away_goal_rate",
        )
    )


def _fixture_signature(event: dict[str, Any]) -> dict[str, Any]:
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    return {
        "fixture_id": fx.get("fixture_id"),
        "kickoff": fx.get("kickoff"),
        "home_team_id": fx.get("home_team_id"),
        "away_team_id": fx.get("away_team_id"),
    }


def _same_fixture(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if str(a.get("fixture_id")) != str(b.get("fixture_id")):
        return False
    if a.get("kickoff") and b.get("kickoff") and str(a.get("kickoff")) != str(b.get("kickoff")):
        return False
    for key in ("home_team_id", "away_team_id"):
        if a.get(key) is not None and b.get(key) is not None and str(a.get(key)) != str(b.get(key)):
            return False
    return True


def _projection_key(fixture_id: Any) -> str:
    return f"{PROJECTION_KEY_PREFIX}{fixture_id}"


def _cache_current_projection(event: dict[str, Any], now: datetime) -> bool:
    raw = event.get("raw_projection")
    if not _usable_projection(raw):
        return False
    sig = _fixture_signature(event)
    fixture_id = sig.get("fixture_id")
    if fixture_id is None:
        return False
    record = {
        "kind": "SPORT_PROJECTION_CONTINUITY",
        "fixture": sig,
        "captured_at_utc": now.isoformat(),
        "source_stage": event.get("stage"),
        "raw_projection": dict(raw),
    }
    # Use sport_shortlist so v6 export/import makes continuity durable through
    # soccer-edge-state, not dependent on one Render process lifetime.
    base._cache_set("sport_shortlist", _projection_key(fixture_id), record, now)
    return True


def _restore_missing_projection(event: dict[str, Any], now: datetime) -> bool:
    if event.get("stage") not in LATE_STAGES or _usable_projection(event.get("raw_projection")):
        return False
    sig = _fixture_signature(event)
    fixture_id = sig.get("fixture_id")
    if fixture_id is None:
        return False
    record = base._cache_get("sport_shortlist", _projection_key(fixture_id), PROJECTION_TTL, now)
    if not isinstance(record, dict) or record.get("kind") != "SPORT_PROJECTION_CONTINUITY":
        return False
    if not _same_fixture(sig, record.get("fixture") or {}):
        return False
    raw = record.get("raw_projection")
    if not _usable_projection(raw):
        return False
    captured = _dt(record.get("captured_at_utc"))
    age_minutes = round((now - captured).total_seconds() / 60.0, 1) if captured else None
    carried = dict(raw)
    carried["continuity_status"] = "CARRIED_FORWARD_RESEARCH_ONLY"
    carried["continuity_source_stage"] = record.get("source_stage")
    carried["continuity_captured_at_utc"] = record.get("captured_at_utc")
    carried["continuity_age_minutes"] = age_minutes
    carried["continuity_actionable"] = False
    carried["continuity_policy"] = (
        "SAME_FIXTURE_LAST_VALID_SPORT_PROJECTION; RESEARCH_ONLY; "
        "CANNOT ALONE SUPPORT BET_OR_LEAN; CURRENT AVAILABILITY AND MARKET GATES STILL REQUIRED"
    )
    event["raw_projection"] = carried
    event["projection_continuity"] = {
        "status": "RESTORED_LAST_VALID_SPORT_PROJECTION",
        "research_only": True,
        "source_stage": record.get("source_stage"),
        "captured_at_utc": record.get("captured_at_utc"),
        "age_minutes": age_minutes,
        "bet_promotion_allowed": False,
    }
    notes = event.get("notes")
    if not isinstance(notes, list):
        notes = []
        event["notes"] = notes
    notes.append("Last valid same-fixture sport projection restored for continuity; research-only.")
    return True


def _apply_continuity(payload: dict[str, Any]) -> dict[str, int]:
    now = _dt(payload.get("generated_at_utc")) or datetime.now(dt_timezone.utc)
    events = [
        event for event in payload.get("events") or []
        if isinstance(event, dict) and event.get("event_type") == "SOCCER_REFRESH"
    ]
    cached = 0
    restored = 0

    # Cache every naturally generated projection first.
    for event in events:
        if _cache_current_projection(event, now):
            cached += 1

    # Restore only a missing late-stage projection for the exact same fixture.
    for event in events:
        if _restore_missing_projection(event, now):
            restored += 1

    if restored:
        intelligence.attach(payload)

    # Refresh durable handoff after continuity records are written.
    payload["shortlist_state"] = v6.export_shortlist_state()
    payload["shortlist_state_count"] = len(payload["shortlist_state"])
    return {"cached": cached, "restored": restored}


async def run_tick() -> dict[str, Any]:
    payload = await v35.run_tick()
    metrics = _apply_continuity(payload)
    payload["sport_projection_continuity"] = {
        "schema_version": "1.0.0",
        "cached_current_tick": metrics["cached"],
        "restored_missing_late_stage": metrics["restored"],
        "ttl_hours": 12,
        "durable_via_shortlist_state": True,
        "research_only": True,
        "galaxy_rebuild_from_carried_projection": False,
        "bet_promotion_allowed": False,
        "provider_requests_added": 0,
    }
    payload["v312_provider_requests_added"] = 0
    payload["v312_model_weights_changed"] = False
    payload["v312_canonical_bet_logic_changed"] = False
    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    return payload
