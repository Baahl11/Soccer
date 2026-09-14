from __future__ import annotations

from typing import Any

from mcp_gateway import automation_v16 as v16

MODEL_VERSION = "SOCCER EDGE ENGINE v1.6"
AUTOMATION_VERSION = "2.6.0"

_STAT_ALIASES = {
    "corner kicks": "corners",
    "total shots": "total_shots",
    "shots on goal": "shots_on_goal",
    "ball possession": "possession",
    "fouls": "fouls",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "blocked shots": "blocked_shots",
    "shots insidebox": "shots_inside_box",
    "shots inside box": "shots_inside_box",
    "shots outsidebox": "shots_outside_box",
    "shots outside box": "shots_outside_box",
}


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _num(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace("%", "")
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _compact_team_stats(entry: dict[str, Any]) -> dict[str, Any]:
    team = entry.get("team") or {}
    out: dict[str, Any] = {
        "team_id": team.get("id"),
        "team": team.get("name"),
    }
    for stat in entry.get("statistics") or []:
        if not isinstance(stat, dict):
            continue
        key = _STAT_ALIASES.get(_norm(stat.get("type")))
        if not key:
            continue
        value = _num(stat.get("value"))
        if value is not None:
            out[key] = value
    return out


def _compact_postgame_stats(event: dict[str, Any]) -> dict[str, Any] | None:
    if event.get("stage") != "POSTGAME":
        return None
    raw = event.get("match_stats")
    if not isinstance(raw, list) or not raw:
        return None
    teams = [_compact_team_stats(x) for x in raw if isinstance(x, dict)]
    teams = [x for x in teams if x.get("team_id") is not None or x.get("team")]
    if not teams:
        return None

    totals: dict[str, float] = {}
    for metric in set(_STAT_ALIASES.values()):
        vals = [x.get(metric) for x in teams if isinstance(x.get(metric), (int, float))]
        if vals:
            totals[metric] = round(sum(float(v) for v in vals), 4)

    return {
        "schema_version": "1.0.0",
        "source": "API_FOOTBALL_FIXTURE_STATISTICS",
        "teams": teams,
        "totals": totals,
    }


async def run_tick() -> dict[str, Any]:
    payload = await v16.run_tick()
    captured = 0
    with_corners = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        compact = _compact_postgame_stats(event)
        if compact is None:
            continue
        event["postgame_tactical_stats"] = compact
        result = event.get("result")
        if not isinstance(result, dict):
            result = {}
            event["result"] = result
        result["tactical_stats"] = compact
        captured += 1
        if (compact.get("totals") or {}).get("corners") is not None:
            with_corners += 1

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["postgame_tactical_stats_captured_this_tick"] = captured
    payload["postgame_corner_results_captured_this_tick"] = with_corners
    payload["formation_intelligence_policy"] = (
        "FORMATION_AND_FORMATION_MATCHUP_ARE_RESEARCH_FEATURES; EVALUATE_RESIDUAL_LIFT_OOS; "
        "DO_NOT_PROMOTE_FROM_DESCRIPTIVE_CORRELATION"
    )
    return payload
