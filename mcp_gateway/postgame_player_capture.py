from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v2 as v2
from mcp_gateway import automation_v4 as v4
from mcp_gateway import player_trends

MAX_POSTGAME_PLAYER_FIXTURES_PER_TICK = 2
MIN_CALLS_TO_KEEP_FREE = 3


async def attach(payload: dict[str, Any]) -> dict[str, Any]:
    mode = str(payload.get("daily_budget_mode") or "UNKNOWN")
    cap = int(payload.get("effective_max_api_calls_per_tick") or v2.MAX_API_CALLS_PER_TICK)
    start_calls = int(v2._API_CALLS_THIS_TICK)
    captured = attempted = deferred = 0

    if mode != "NORMAL":
        return {
            "capture_enabled_this_tick": False,
            "reason": "POSTGAME_PLAYER_CAPTURE_REQUIRES_NORMAL_BUDGET_MODE",
            "attempted": 0,
            "captured": 0,
            "deferred": sum(
                1 for e in payload.get("events") or []
                if isinstance(e, dict) and e.get("stage") == "POSTGAME"
            ),
            "provider_requests_added": 0,
        }

    for event in payload.get("events") or []:
        if attempted >= MAX_POSTGAME_PLAYER_FIXTURES_PER_TICK:
            break
        if not isinstance(event, dict) or event.get("stage") != "POSTGAME":
            continue
        coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
        fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fid = fixture.get("fixture_id")
        if not coverage.get("statistics_players") or not fid:
            continue
        if int(v2._API_CALLS_THIS_TICK) > cap - MIN_CALLS_TO_KEEP_FREE - 1:
            deferred += 1
            continue

        attempted += 1
        try:
            raw = await v4._paced_api_get("fixtures/players", {"fixture": int(fid)})
            compact = player_trends._compact(raw)
            compact["fixture_id"] = int(fid)
            compact["capture_phase"] = "POSTGAME"
            compact["finalized_fixture_required"] = True
            compact["goalkeeper_fields_retained"] = ["saves", "goals_conceded"]
            event["postgame_player_stats"] = compact
            if any((team.get("players") or []) for team in compact.get("teams") or [] if isinstance(team, dict)):
                captured += 1
        except Exception as exc:
            event["postgame_player_stats"] = {
                "status": "UNAVAILABLE",
                "fixture_id": fid,
                "capture_phase": "POSTGAME",
                "error": str(exc)[:180],
                "actionable": False,
                "decision_weight": 0.0,
            }

    added = max(0, int(v2._API_CALLS_THIS_TICK) - start_calls)
    payload["api_calls_this_tick"] = int(v2._API_CALLS_THIS_TICK)
    payload["last_daily_remaining"] = v2._LAST_DAILY_REMAINING
    return {
        "capture_enabled_this_tick": True,
        "attempted": attempted,
        "captured": captured,
        "deferred": deferred,
        "max_postgame_player_fixtures_per_tick": MAX_POSTGAME_PLAYER_FIXTURES_PER_TICK,
        "minimum_calls_kept_free": MIN_CALLS_TO_KEEP_FREE,
        "provider_requests_added": added,
        "policy": "LOW_PRIORITY_POSTGAME_ONLY; NORMAL_BUDGET_ONLY; NEVER DISPLACES PREGAME CALLS; MAX_2_FIXTURES_PER_TICK",
    }
