from __future__ import annotations

import re
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


def _line(selection: Any) -> float | None:
    m = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(selection or ""), re.I)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
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


def _corner_groups(event: dict[str, Any]) -> list[dict[str, Any]]:
    if event.get("stage") not in {"T-40", "T-20", "T-10"}:
        return []
    market = event.get("market")
    if not isinstance(market, dict):
        return []
    groups: list[dict[str, Any]] = []
    for row in market.get("markets") or []:
        if not isinstance(row, dict):
            continue
        name = _norm(row.get("market"))
        if "corner" not in name:
            continue
        values = []
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            selection = value.get("selection")
            try:
                price = float(value.get("price"))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            values.append({
                "selection": selection,
                "line": _line(selection),
                "decimal_price": round(price, 6),
            })
        if not values:
            continue
        family = "TEAM_CORNERS" if any(token in name for token in ("team", "home", "away")) else "CORNERS"
        groups.append({
            "family": family,
            "bookmaker": row.get("bookmaker"),
            "market": row.get("market"),
            "provider_update": row.get("provider_update"),
            "values": values,
        })
        if len(groups) >= 20:
            break
    return groups


async def run_tick() -> dict[str, Any]:
    payload = await v16.run_tick()
    captured = 0
    with_corners = 0
    corner_market_groups = 0
    corner_market_quotes = 0

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue

        compact = _compact_postgame_stats(event)
        if compact is not None:
            event["postgame_tactical_stats"] = compact
            result = event.get("result")
            if not isinstance(result, dict):
                result = {}
                event["result"] = result
            # Scheduler already persists `result`, so tactical outcomes survive
            # without widening the compact-state contract.
            result["tactical_stats"] = compact
            captured += 1
            if (compact.get("totals") or {}).get("corners") is not None:
                with_corners += 1

        corner_groups = _corner_groups(event)
        if corner_groups:
            snap = event.get("derivative_research_market_snapshot")
            if not isinstance(snap, dict):
                snap = {
                    "schema_version": "1.1.0",
                    "captured_at_local": payload.get("generated_at_local"),
                    "research_only": True,
                    "actionable": False,
                    "policy": "EXPLICIT_DERIVATIVE_MODEL_REQUIRED; NEVER_REUSE_FT_PROBABILITIES",
                    "group_count": 0,
                    "quote_count": 0,
                    "groups": [],
                }
                event["derivative_research_market_snapshot"] = snap
            snap.setdefault("groups", []).extend(corner_groups)
            add_quotes = sum(len(g.get("values") or []) for g in corner_groups)
            snap["group_count"] = len(snap.get("groups") or [])
            snap["quote_count"] = int(snap.get("quote_count") or 0) + add_quotes
            snap["contains_corner_markets"] = True
            corner_market_groups += len(corner_groups)
            corner_market_quotes += add_quotes

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["postgame_tactical_stats_captured_this_tick"] = captured
    payload["postgame_corner_results_captured_this_tick"] = with_corners
    payload["corner_market_groups_this_tick"] = corner_market_groups
    payload["corner_market_quotes_this_tick"] = corner_market_quotes
    payload["formation_intelligence_policy"] = (
        "FORMATION_AND_FORMATION_MATCHUP_ARE_RESEARCH_FEATURES; EVALUATE_RESIDUAL_LIFT_OOS; "
        "DO_NOT_PROMOTE_FROM_DESCRIPTIVE_CORRELATION"
    )
    payload["corner_research_policy"] = (
        "CAPTURE_MARKET_AND_POSTGAME_OUTCOME_WITHOUT_EXTRA_ODDS_CALLS; RESEARCH_ONLY_UNTIL_EXPLICIT_CORNER_MODEL_OOS_GATE"
    )
    return payload
