from __future__ import annotations

import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
WATCH_CLASSIFICATION = "WATCH"
TARGET_BET_FAMILIES = {"FT_TOTALS", "2H_TOTALS", "2H_BTTS", "1H_OTHER"}
DEMOTION_REASON = "SETTLEMENT_GUARD_RUNTIME_DEMOTION_NEGATIVE_OR_INSUFFICIENT_SAMPLE"
DEMOTION_POLICY = (
    "BET rows in FT_TOTALS, 2H_TOTALS, 2H_BTTS and 1H_OTHER are downgraded to WATCH/research "
    "until settlement shows 20+ settled decisions, positive ROI, CLV review and stability. LEAN rows are not touched."
)


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def period_type(market: Any) -> str:
    m = norm(market)
    if any(token in m for token in ("first half", "1st half", "1h", "half time", "halftime")):
        return "1H"
    if any(token in m for token in ("second half", "2nd half", "2h")):
        return "2H"
    return "FT"


def market_family_from_values(market: Any = None, selection: Any = None, explicit: Any = None) -> str:
    if explicit:
        return str(explicit).upper()
    market_n = norm(market)
    selection_n = norm(selection)
    period = period_type(market_n)
    if "both teams to score" in market_n or market_n == "btts" or "btts" in market_n:
        return f"{period}_BTTS"
    if (
        "over/under" in market_n
        or market_n in {"goals over/under", "over/under"}
        or selection_n.startswith("over")
        or selection_n.startswith("under")
        or " over " in f" {selection_n} "
        or " under " in f" {selection_n} "
    ):
        return f"{period}_TOTALS"
    if market_n in {"match winner", "winner"}:
        return f"{period}_1X2"
    return f"{period}_OTHER"


def market_family(row: dict[str, Any]) -> str:
    best = row.get("best_market") if isinstance(row.get("best_market"), dict) else {}
    explicit = row.get("market_family") or (best or {}).get("market_family")
    market = row.get("market") or (best or {}).get("market")
    selection = row.get("selection") or (best or {}).get("selection")
    return market_family_from_values(market, selection, explicit)


def is_target_bet(row: dict[str, Any]) -> tuple[bool, str]:
    classification = str(row.get("classification") or "").upper()
    family = market_family(row)
    return classification == "BET" and family in TARGET_BET_FAMILIES, family


def _append_unique(row: dict[str, Any], key: str, value: str) -> None:
    items = row.get(key)
    if not isinstance(items, list):
        items = []
    if value not in items:
        items.append(value)
    row[key] = items


def demote_row(row: dict[str, Any], family: str) -> bool:
    row["original_classification"] = row.get("classification")
    row["classification"] = WATCH_CLASSIFICATION
    row["bet_eligible"] = False
    row["tier"] = None
    if "stake_units" in row:
        row["stake_units"] = 0.0
    row["runtime_demotion_guard"] = {
        "schema_version": SCHEMA_VERSION,
        "reason": DEMOTION_REASON,
        "market_family": family,
        "from_classification": "BET",
        "to_classification": WATCH_CLASSIFICATION,
        "policy": DEMOTION_POLICY,
        "settlement_inputs": {
            "FT_TOTALS_BET": "7 settled, 3-4, -0.5652u",
            "2H_TOTALS_BET": "2 settled, 0-2, -0.72u",
            "2H_BTTS_BET": "1 settled, 0-1, -0.36u",
            "1H_OTHER_BET": "0 settled, 1 ungraded",
        },
        "lean_preserved": True,
    }
    _append_unique(row, "notes", DEMOTION_REASON)
    _append_unique(row, "block_reasons", DEMOTION_REASON)
    if row.get("reason") and DEMOTION_REASON not in str(row.get("reason")):
        row["reason"] = f"{row.get('reason')} | {DEMOTION_REASON}"
    elif not row.get("reason"):
        row["reason"] = DEMOTION_REASON
    return True


def apply(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "status": "ACTIVE_RUNTIME_DEMOTION_GUARD",
        "target_bet_market_families": sorted(TARGET_BET_FAMILIES),
        "policy": DEMOTION_POLICY,
        "events_checked": 0,
        "events_demoted": 0,
        "match_table_rows_checked": 0,
        "match_table_rows_demoted": 0,
        "lean_rows_preserved": True,
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "runtime_promotion_added": False,
    }

    for row in payload.get("events") or []:
        if not isinstance(row, dict):
            continue
        metrics["events_checked"] += 1
        should_demote, family = is_target_bet(row)
        if should_demote and demote_row(row, family):
            metrics["events_demoted"] += 1

    for row in payload.get("match_table_rows") or []:
        if not isinstance(row, dict):
            continue
        metrics["match_table_rows_checked"] += 1
        should_demote, family = is_target_bet(row)
        if should_demote and demote_row(row, family):
            metrics["match_table_rows_demoted"] += 1

    payload["runtime_bet_demotion_guard"] = metrics
    return metrics
