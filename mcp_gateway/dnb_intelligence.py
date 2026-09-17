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


def _canonical_probs(raw: dict[str, Any]) -> dict[str, float] | None:
    values = [_num(raw.get(k)) for k in ("raw_home_win_prob", "raw_draw_prob", "raw_away_win_prob")]
    if any(v is None or v < 0 for v in values):
        return None
    total = sum(float(v) for v in values if v is not None)
    if total <= 0:
        return None
    h, d, a = [float(v) / total for v in values if v is not None]
    return {"HOME": h, "DRAW": d, "AWAY": a}


def _is_dnb_market(name: Any) -> bool:
    n = _norm(name)
    return "draw no bet" in n or n == "dnb" or " dnb" in n or "dnb " in n


def _side(value: Any, fixture: dict[str, Any]) -> str | None:
    text = _norm(value)
    home_name = _norm(fixture.get("home_team") or fixture.get("home_name"))
    away_name = _norm(fixture.get("away_team") or fixture.get("away_name"))
    if text in {"home", "1", "home dnb", "1 dnb"} or (home_name and text == home_name):
        return "HOME"
    if text in {"away", "2", "away dnb", "2 dnb"} or (away_name and text == away_name):
        return "AWAY"
    return None


def _conditional_probs(parent: dict[str, float]) -> dict[str, float] | None:
    non_draw = parent["HOME"] + parent["AWAY"]
    if non_draw <= 0:
        return None
    return {"HOME": parent["HOME"] / non_draw, "AWAY": parent["AWAY"] / non_draw}


def _fair_pair(home_price: float, away_price: float) -> dict[str, float]:
    hi = 1.0 / home_price
    ai = 1.0 / away_price
    total = hi + ai
    return {"HOME": hi / total, "AWAY": ai / total}


def _observed_rows(event: dict[str, Any], parent: dict[str, float], conditional: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    source = provenance.get("source") or "NOT_VERIFIED"
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_dnb_market(market_row.get("market")):
            continue
        prices: dict[str, float] = {}
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _side(value.get("selection"), fixture)
            price = _decimal(value.get("price"))
            if side is None or price is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "reason": "UNPARSED_DNB_SELECTION_OR_PRICE",
                })
                continue
            prices[side] = price
        fair_conditional = _fair_pair(prices["HOME"], prices["AWAY"]) if set(prices) == {"HOME", "AWAY"} else None

        for side in ("HOME", "AWAY"):
            price = prices.get(side)
            if price is None:
                continue
            p_cond = conditional[side]
            p_win = parent[side]
            p_draw = parent["DRAW"]
            p_loss = parent["AWAY" if side == "HOME" else "HOME"]
            p_market_fair = fair_conditional.get(side) if fair_conditional else None
            push_aware_ev = p_win * price + p_draw - 1.0
            rows.append({
                "selection": side,
                "probability_win_unconditional": round(p_win, 6),
                "probability_push_draw": round(p_draw, 6),
                "probability_loss_unconditional": round(p_loss, 6),
                "probability_win_given_no_draw": round(p_cond, 6),
                "fair_decimal_dnb": round(1.0 / p_cond, 4) if 0 < p_cond < 1 else None,
                "bookmaker": market_row.get("bookmaker"),
                "bookmaker_id": market_row.get("bookmaker_id"),
                "market": market_row.get("market"),
                "market_id": market_row.get("market_id"),
                "decimal_price": round(price, 4),
                "p_breakeven_conditional": round(1.0 / price, 6),
                "p_market_fair_conditional": round(p_market_fair, 6) if p_market_fair is not None else None,
                "raw_edge_vs_market_fair_pp": round((p_cond - p_market_fair) * 100.0, 3) if p_market_fair is not None else None,
                "raw_edge_vs_breakeven_pp": round((p_cond - 1.0 / price) * 100.0, 3),
                "raw_ev_push_aware": round(push_aware_ev, 6),
                "ev_formula": "P(win)*decimal_price + P(draw)*1 + P(loss)*0 - 1",
                "provider_update": market_row.get("provider_update"),
                "market_source": source,
                "market_fresh": fresh,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "promotion_block": "PARENT_1X2_NOT_PRODUCTION_APPROVED_AND_DNB_NOT_OOS_CALIBRATED",
            })

    rows.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") if row.get("raw_edge_vs_market_fair_pp") is not None else row.get("raw_edge_vs_breakeven_pp") or -999.0),
            float(row.get("probability_win_given_no_draw") or 0.0),
        ),
        reverse=True,
    )
    return rows[:16], unsupported[:16]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    parent = _canonical_probs(raw)
    conditional = _conditional_probs(parent) if parent else None
    if parent is None or conditional is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "PARENT_CANONICAL_1X2_PROBABILITIES_MISSING",
        }

    observed, unsupported = _observed_rows(event, parent, conditional)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": "CANONICAL_1X2_CONDITIONAL_DNB_v0.1",
        "parent_1x2": {key: round(value, 6) for key, value in parent.items()},
        "push_probability": round(parent["DRAW"], 6),
        "probability_win_given_no_draw": {key: round(value, 6) for key, value in conditional.items()},
        "fair_decimal_dnb": {key: round(1.0 / value, 4) if value > 0 else None for key, value in conditional.items()},
        "observed_market_rows": observed,
        "observed_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_PARENT_1X2_NOT_APPROVED",
        "calibration_gate": {
            "minimum_non_draw_oos_for_market_comparison": 200,
            "minimum_non_draw_oos_for_actionable_review": 400,
            "requires": [
                "parent 1X2 production model selected and approved",
                "stable conditional HOME/AWAY calibration on non-draw outcomes",
                "verified historical DNB prices and true CLV",
                "push-aware EV and settlement verified across books",
                "validated shrinkage after parent-model selection",
            ],
        },
        "policy": (
            "SPORT_FIRST; DNB FAIR PROBABILITY IS P(WIN|NO_DRAW); DRAW IS A PUSH; "
            "PUSH-AWARE EV=P(WIN)*PRICE+P(DRAW)-1; TWO-SIDED DNB NO-VIG IS CONDITIONAL ON NO DRAW; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = observed_events = observed_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["dnb_intelligence"] = intelligence
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
                areas["draw_no_bet"] = {
                    "status": intelligence.get("status"),
                    "push_probability": intelligence.get("push_probability"),
                    "probability_win_given_no_draw": intelligence.get("probability_win_given_no_draw") or {},
                    "observed_market_rows": intelligence.get("observed_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }
    return {
        "modeled_events": modeled,
        "events_with_observed_dnb_markets": observed_events,
        "observed_dnb_market_rows": observed_rows,
    }
