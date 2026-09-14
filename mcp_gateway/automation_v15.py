from __future__ import annotations

import re
from typing import Any

from mcp_gateway import automation_v14 as v14

MODEL_VERSION = "SOCCER EDGE ENGINE v1.4"
AUTOMATION_VERSION = "2.4.0"

_CANONICAL = {
    "match winner",
    "winner",
    "goals over/under",
    "over/under",
    "goals over under",
    "both teams to score",
    "both teams score",
}
_PERIOD_TOKENS = (
    "first half", "second half", "1st half", "2nd half",
    "first-half", "second-half", "half time", "half-time",
    "halftime", "both halves", "1h ", " 1h ", "2h ", " 2h ",
)
_DERIVATIVE_TOKENS = (
    "team total", "corner", "card", "player", "double chance",
    "draw no bet", "asian", "handicap", "correct score", "odd/even",
    "result/total", "winner &", "win and",
)


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _canonical_ft_market(name: Any) -> bool:
    market = _norm(name)
    if market not in _CANONICAL:
        return False
    padded = f" {market} "
    if any(token in padded for token in _PERIOD_TOKENS):
        return False
    if any(token in market for token in _DERIVATIVE_TOKENS):
        return False
    return True


def _line_from_selection(selection: Any) -> float | None:
    text = str(selection or "")
    match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _compact_close_snapshot(event: dict[str, Any], captured_at_local: Any) -> dict[str, Any] | None:
    if event.get("stage") != "CLOSE":
        return None
    market = event.get("market")
    if not isinstance(market, dict):
        return None
    rows = market.get("markets") or []
    if not isinstance(rows, list):
        return None

    groups: list[dict[str, Any]] = []
    quote_count = 0
    for row in rows:
        if not isinstance(row, dict) or not _canonical_ft_market(row.get("market")):
            continue
        values = []
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            try:
                price = float(value.get("price"))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:
                continue
            selection = value.get("selection")
            values.append({
                "selection": selection,
                "line": _line_from_selection(selection),
                "decimal_price": round(price, 6),
            })
            quote_count += 1
        if not values:
            continue
        groups.append({
            "bookmaker": row.get("bookmaker"),
            "market": row.get("market"),
            "provider_update": row.get("provider_update"),
            "values": values,
        })
        if len(groups) >= 40:
            break

    if not groups:
        return None
    return {
        "schema_version": "1.0.0",
        "captured_at_local": captured_at_local,
        "source": event.get("market_source") or "API_FOOTBALL_OR_GALAXY",
        "canonical_ft_only": True,
        "group_count": len(groups),
        "quote_count": quote_count,
        "groups": groups,
    }


async def run_tick() -> dict[str, Any]:
    payload = await v14.run_tick()
    captured_at_local = payload.get("generated_at_local")
    close_snapshots = 0
    close_quotes = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            continue
        snap = _compact_close_snapshot(event, captured_at_local)
        if snap is None:
            continue
        event["closing_market_snapshot"] = snap
        close_snapshots += 1
        close_quotes += int(snap.get("quote_count") or 0)

    payload["version"] = AUTOMATION_VERSION
    payload["model_version"] = MODEL_VERSION
    payload["dedicated_close_snapshots_this_tick"] = close_snapshots
    payload["dedicated_close_quotes_this_tick"] = close_quotes
    payload["clv_close_policy"] = (
        "PERSIST_CANONICAL_FT_CLOSE_QUOTES; TRUE_CLV_ONLY_WHEN_CAPTURED_BEFORE_KICKOFF; "
        "LATE_CLOSE_SNAPSHOTS_RETAINED_BUT_NOT_LABELED_TRUE_PREKICKOFF_CLOSE"
    )
    return payload
