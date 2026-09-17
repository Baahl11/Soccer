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


def _is_1x2_market(name: Any) -> bool:
    n = _norm(name)
    if any(token in n for token in ("double chance", "draw no bet", "dnb", "handicap")):
        return False
    return n in {"winner", "match winner", "1x2", "fulltime result", "full time result"} or "match winner" in n


def _is_dc_market(name: Any) -> bool:
    n = _norm(name)
    return "double chance" in n


def _one_x_two_side(value: Any, fixture: dict[str, Any]) -> str | None:
    text = _norm(value)
    home_name = _norm(fixture.get("home_team") or fixture.get("home_name"))
    away_name = _norm(fixture.get("away_team") or fixture.get("away_name"))
    if text in {"home", "1"} or (home_name and text == home_name):
        return "HOME"
    if text in {"draw", "x", "tie"}:
        return "DRAW"
    if text in {"away", "2"} or (away_name and text == away_name):
        return "AWAY"
    return None


def _dc_side(value: Any) -> str | None:
    text = _norm(value)
    compact = text.replace(" ", "").replace("/", "").replace("-", "").replace("_", "")
    if compact in {"1x", "x1", "homedraw", "drawhome", "homeordraw", "draworhome"}:
        return "1X"
    if compact in {"x2", "2x", "drawaway", "awaydraw", "draworaway", "awayordraw"}:
        return "X2"
    if compact in {"12", "21", "homeaway", "awayhome", "homeoraway", "awayorhome"}:
        return "12"
    has_home = "home" in text
    has_draw = "draw" in text or "tie" in text
    has_away = "away" in text
    if has_home and has_draw and not has_away:
        return "1X"
    if has_away and has_draw and not has_home:
        return "X2"
    if has_home and has_away and not has_draw:
        return "12"
    return None


def _fair_three_way(prices: dict[str, float]) -> dict[str, float] | None:
    if set(prices) != {"HOME", "DRAW", "AWAY"}:
        return None
    implied = {key: 1.0 / value for key, value in prices.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in implied.items()}


def _parent_fair_by_book(event: dict[str, Any]) -> dict[str, dict[str, float]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    output: dict[str, dict[str, float]] = {}
    for row in market.get("markets") or []:
        if not isinstance(row, dict) or not _is_1x2_market(row.get("market")):
            continue
        book = str(row.get("bookmaker") or "").strip()
        if not book:
            continue
        prices: dict[str, float] = {}
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _one_x_two_side(value.get("selection"), fixture)
            price = _decimal(value.get("price"))
            if side and price:
                prices[side] = price
        fair = _fair_three_way(prices)
        if fair:
            output[book] = fair
    return output


def _dc_probs(parent: dict[str, float]) -> dict[str, float]:
    return {
        "1X": parent["HOME"] + parent["DRAW"],
        "X2": parent["DRAW"] + parent["AWAY"],
        "12": parent["HOME"] + parent["AWAY"],
    }


def _observed_rows(event: dict[str, Any], model: dict[str, float]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    source = provenance.get("source") or "NOT_VERIFIED"
    parent_fair = _parent_fair_by_book(event)
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_dc_market(market_row.get("market")):
            continue
        book = str(market_row.get("bookmaker") or "").strip()
        fair_parent = parent_fair.get(book)
        fair_dc = _dc_probs(fair_parent) if fair_parent else None
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _dc_side(value.get("selection"))
            price = _decimal(value.get("price"))
            if side is None or price is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "reason": "UNPARSED_DOUBLE_CHANCE_SELECTION_OR_PRICE",
                })
                continue
            p_model = model[side]
            p_market_fair = fair_dc.get(side) if fair_dc else None
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
                "p_market_fair_from_same_book_1x2": round(p_market_fair, 6) if p_market_fair is not None else None,
                "market_fair_method": "SAME_BOOK_COMPLETE_1X2_NO_VIG_DERIVATION" if p_market_fair is not None else "NOT_AVAILABLE",
                "raw_edge_vs_market_fair_pp": round((p_model - p_market_fair) * 100.0, 3) if p_market_fair is not None else None,
                "raw_edge_vs_breakeven_pp": round((p_model - 1.0 / price) * 100.0, 3),
                "raw_ev_at_observed_price": round(p_model * price - 1.0, 6),
                "provider_update": market_row.get("provider_update"),
                "market_source": source,
                "market_fresh": fresh,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "promotion_block": "PARENT_1X2_NOT_PRODUCTION_APPROVED_AND_DC_NOT_OOS_CALIBRATED",
            })

    rows.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") if row.get("raw_edge_vs_market_fair_pp") is not None else row.get("raw_edge_vs_breakeven_pp") or -999.0),
            float(row.get("probability_model_raw") or 0.0),
        ),
        reverse=True,
    )
    return rows[:24], unsupported[:24]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    parent = _canonical_probs(raw)
    if parent is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "PARENT_CANONICAL_1X2_PROBABILITIES_MISSING",
        }
    model = _dc_probs(parent)
    observed, unsupported = _observed_rows(event, model)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": "CANONICAL_1X2_DERIVED_DOUBLE_CHANCE_v0.1",
        "parent_1x2": {key: round(value, 6) for key, value in parent.items()},
        "probabilities": {key: round(value, 6) for key, value in model.items()},
        "fair_decimal": {key: round(1.0 / value, 4) if value > 0 else None for key, value in model.items()},
        "observed_market_rows": observed,
        "observed_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_PARENT_1X2_NOT_APPROVED",
        "calibration_gate": {
            "minimum_oos_fixtures_for_market_comparison": 200,
            "minimum_oos_fixtures_for_actionable_review": 400,
            "requires": [
                "parent 1X2 production model selected and approved",
                "stable binary calibration for 1X/X2/12",
                "verified historical Double Chance prices and true CLV",
                "market fair comparison derived from same-book complete 1X2 where available",
                "validated shrinkage after parent-model promotion",
            ],
        },
        "policy": (
            "SPORT_FIRST; P(1X)=P(H)+P(D), P(X2)=P(D)+P(A), P(12)=P(H)+P(A); "
            "DOUBLE-CHANCE SELECTIONS OVERLAP SO NEVER NORMALIZE THEIR THREE PRICES AS A NO-VIG MARKET; "
            "WHEN AVAILABLE DERIVE MARKET-FAIR DC FROM SAME-BOOK COMPLETE 1X2 NO-VIG PROBABILITIES; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY PROMOTION"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = observed_events = observed_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["double_chance_intelligence"] = intelligence
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
                areas["double_chance"] = {
                    "status": intelligence.get("status"),
                    "probabilities": intelligence.get("probabilities") or {},
                    "observed_market_rows": intelligence.get("observed_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }
    return {
        "modeled_events": modeled,
        "events_with_observed_double_chance_markets": observed_events,
        "observed_double_chance_market_rows": observed_rows,
    }
