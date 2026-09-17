from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "1.0.0"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _decimal(value: Any) -> float | None:
    out = _num(value)
    return out if out is not None and 1.0 < out <= 1000.0 else None


def _is_btts_market(name: Any) -> bool:
    n = _norm(name)
    return "both teams" in n and ("score" in n or "to score" in n)


def _side(value: Any) -> str | None:
    n = _norm(value)
    if n in {"yes", "btts yes", "both teams to score - yes", "both teams score - yes"} or n.endswith(" yes"):
        return "YES"
    if n in {"no", "btts no", "both teams to score - no", "both teams score - no"} or n.endswith(" no"):
        return "NO"
    return None


def _fair_pair(yes_price: float, no_price: float) -> tuple[float, float]:
    yi = 1.0 / yes_price
    ni = 1.0 / no_price
    total = yi + ni
    return yi / total, ni / total


def _observed_rows(event: dict[str, Any], p_yes: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    source = provenance.get("source") or "NOT_VERIFIED"
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_btts_market(market_row.get("market")):
            continue
        prices: dict[str, float] = {}
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _side(value.get("selection"))
            price = _decimal(value.get("price"))
            if side is None or price is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "reason": "UNPARSED_BTTS_SELECTION_OR_PRICE",
                })
                continue
            prices[side] = price
        market_fair_yes = market_fair_no = None
        if prices.get("YES") is not None and prices.get("NO") is not None:
            market_fair_yes, market_fair_no = _fair_pair(prices["YES"], prices["NO"])

        for side, p_model, p_market_fair in (
            ("YES", p_yes, market_fair_yes),
            ("NO", 1.0 - p_yes, market_fair_no),
        ):
            price = prices.get(side)
            if price is None:
                continue
            rows.append({
                "selection": side,
                "probability_model_raw": round(p_model, 6),
                "fair_decimal_model_raw": round(1.0 / p_model, 4) if 0 < p_model < 1 else None,
                "bookmaker": market_row.get("bookmaker"),
                "bookmaker_id": market_row.get("bookmaker_id"),
                "market": market_row.get("market"),
                "market_id": market_row.get("market_id"),
                "decimal_price": round(price, 4),
                "p_breakeven": round(1.0 / price, 6),
                "p_market_fair": round(p_market_fair, 6) if p_market_fair is not None else None,
                "raw_edge_vs_market_fair_pp": round((p_model - p_market_fair) * 100.0, 3) if p_market_fair is not None else None,
                "raw_ev_at_observed_price": round(p_model * price - 1.0, 6),
                "provider_update": market_row.get("provider_update"),
                "market_source": source,
                "market_fresh": fresh,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "market_shrinkage": "NOT_CALIBRATED_FOR_BTTS_PRODUCTION",
                "promotion_block": "BTTS_NOT_OOS_CALIBRATED_OR_PRODUCTION_APPROVED",
            })

    rows.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") or -999.0),
            float(row.get("probability_model_raw") or 0.0),
        ),
        reverse=True,
    )
    return rows[:16], unsupported[:16]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    p_yes = _num(raw.get("raw_btts_yes_prob"))
    if p_yes is None or not 0 < p_yes < 1:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "CANONICAL_BTTS_PROBABILITY_MISSING",
        }

    observed, unsupported = _observed_rows(event, p_yes)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": "CANONICAL_SCORE_MATRIX_BTTS_v0.1",
        "model_source": "EXISTING_CANONICAL_RAW_BTTS_YES_PROBABILITY",
        "p_yes": round(p_yes, 6),
        "p_no": round(1.0 - p_yes, 6),
        "fair_yes_decimal": round(1.0 / p_yes, 4),
        "fair_no_decimal": round(1.0 / (1.0 - p_yes), 4),
        "observed_market_rows": observed,
        "observed_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_fixtures_for_market_comparison": 150,
            "minimum_oos_fixtures_for_actionable_review": 300,
            "requires": [
                "stable Brier/log-loss calibration overall and by competition",
                "verified historical BTTS prices and true CLV evidence",
                "validated market shrinkage policy",
                "stable performance by probability bucket and data tier",
                "no material degradation versus canonical FT-goals parent model",
            ],
        },
        "policy": (
            "SPORT_FIRST; USE EXISTING CANONICAL SCORE-MATRIX BTTS PROBABILITY; "
            "COMPARE ONLY TO OBSERVED YES/NO BTTS MARKETS; NO-VIG ONLY WHEN BOTH SIDES EXIST; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION BEFORE CALIBRATION"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = observed_events = observed_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["btts_intelligence"] = intelligence
        if intelligence.get("status") == "LIVE_RESEARCH_MODELED":
            modeled += 1
        count = int(intelligence.get("observed_market_count") or 0)
        if count:
            observed_events += 1
            observed_rows += count

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict):
                areas["btts"] = {
                    "status": intelligence.get("status"),
                    "p_yes": intelligence.get("p_yes"),
                    "p_no": intelligence.get("p_no"),
                    "observed_market_rows": intelligence.get("observed_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }
    return {
        "modeled_events": modeled,
        "events_with_observed_btts_markets": observed_events,
        "observed_btts_market_rows": observed_rows,
    }
