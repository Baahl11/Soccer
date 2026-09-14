from __future__ import annotations

import re
from typing import Any

from mcp_gateway import automation_v15 as v15

MODEL_VERSION = "SOCCER EDGE ENGINE v1.5"
AUTOMATION_VERSION = "2.5.0"


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _period_family(name: Any) -> str | None:
    n = _norm(name)
    goal_market = "over/under" in n or "over under" in n or "goals" in n or "total" in n
    if not goal_market:
        return None
    if any(token in n for token in ("first half", "1st half", "first-half", "1st-half", "half time", "half-time", "halftime")):
        return "1H_GOALS"
    if any(token in n for token in ("second half", "2nd half", "second-half", "2nd-half")):
        return "2H_GOALS"
    return None


def _line(selection: Any) -> float | None:
    match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", str(selection or ""), re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _period_research_snapshot(event: dict[str, Any], captured_at_local: Any) -> dict[str, Any] | None:
    if event.get("stage") not in {"T-40", "T-20", "T-10"}:
        return None
    market = event.get("market")
    if not isinstance(market, dict):
        return None
    groups: list[dict[str, Any]] = []
    quotes = 0
    for row in market.get("markets") or []:
        if not isinstance(row, dict):
            continue
        family = _period_family(row.get("market"))
        if family not in {"1H_GOALS", "2H_GOALS"}:
            continue
        values = []
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            selection = value.get("selection")
            if not any(token in _norm(selection) for token in ("over", "under")):
                continue
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
            quotes += 1
        if values:
            groups.append({
                "family": family,
                "bookmaker": row.get("bookmaker"),
                "market": row.get("market"),
                "provider_update": row.get("provider_update"),
                "values": values,
            })
        if len(groups) >= 30:
            break
    if not groups:
        return None
    return {
        "schema_version": "1.0.0",
        "captured_at_local": captured_at_local,
        "research_only": True,
        "actionable": False,
        "policy": "EXPLICIT_PERIOD_MODEL_REQUIRED; NEVER_REUSE_FT_PROBABILITIES",
        "group_count": len(groups),
        "quote_count": quotes,
        "groups": groups,
    }


async def run_tick() -> dict[str, Any]:
    payload = await v15.run_tick()
    captured = payload.get("generated_at_local")
    snapshots = 0
    quotes = 0
    one_h_groups = 0
    two_h_groups = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        snap = _period_research_snapshot(event, captured)
        if snap is None:
            continue
        event["derivative_research_market_snapshot"] = snap
        snapshots += 1
        quotes += int(snap.get("quote_count") or 0)
        for group in snap.get("groups") or []:
            if group.get("family") == "1H_GOALS":
                one_h_groups += 1
            elif group.get("family") == "2H_GOALS":
                two_h_groups += 1

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["derivative_research_snapshots_this_tick"] = snapshots
    payload["derivative_research_quotes_this_tick"] = quotes
    payload["one_h_market_groups_this_tick"] = one_h_groups
    payload["two_h_market_groups_this_tick"] = two_h_groups
    payload["derivative_market_capture_policy"] = (
        "PERIOD_GOALS_RESEARCH_ONLY; NO_ACTIONABLE_CLASSIFICATION; "
        "EXPLICIT_PERIOD_MODEL_AND_OOS_CALIBRATION_REQUIRED"
    )
    return payload
