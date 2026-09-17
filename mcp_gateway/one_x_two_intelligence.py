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


def _normalized_probs(values: list[Any]) -> list[float] | None:
    probs = [_num(v) for v in values]
    if any(v is None or v < 0 for v in probs):
        return None
    total = sum(float(v) for v in probs if v is not None)
    if total <= 0:
        return None
    return [float(v) / total for v in probs if v is not None]


def _is_1x2_market(name: Any) -> bool:
    n = _norm(name)
    if any(token in n for token in ("double chance", "draw no bet", "dnb", "handicap")):
        return False
    return n in {"winner", "match winner", "1x2", "fulltime result", "full time result"} or "match winner" in n


def _selection(value: Any, fixture: dict[str, Any]) -> str | None:
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


def _fair_three_way(prices: dict[str, float]) -> dict[str, float] | None:
    if set(prices) != {"HOME", "DRAW", "AWAY"}:
        return None
    implied = {key: 1.0 / value for key, value in prices.items()}
    total = sum(implied.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in implied.items()}


def _canonical(raw: dict[str, Any]) -> dict[str, Any]:
    probs = _normalized_probs([
        raw.get("raw_home_win_prob"),
        raw.get("raw_draw_prob"),
        raw.get("raw_away_win_prob"),
    ])
    if probs is None:
        return {"status": "NOT_MODELED_THIS_TICK"}
    home, draw, away = probs
    return {
        "status": "LIVE_RESEARCH_MODELED",
        "model": raw.get("projection_model") or "CANONICAL_SCORE_MATRIX_1X2",
        "p_home": round(home, 6),
        "p_draw": round(draw, 6),
        "p_away": round(away, 6),
        "fair_home_decimal": round(1.0 / home, 4) if home > 0 else None,
        "fair_draw_decimal": round(1.0 / draw, 4) if draw > 0 else None,
        "fair_away_decimal": round(1.0 / away, 4) if away > 0 else None,
    }


def _relative_strength_shadow(raw: dict[str, Any], canonical: dict[str, Any]) -> dict[str, Any]:
    shadow = raw.get("relative_strength_shadow")
    if not isinstance(shadow, dict) or shadow.get("status") != "RESEARCH_ONLY_SHADOW":
        return {
            "status": "NOT_AVAILABLE_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
        }
    challenger = shadow.get("challenger") if isinstance(shadow.get("challenger"), dict) else {}
    probs = _normalized_probs([
        challenger.get("raw_home_win_prob"),
        challenger.get("raw_draw_prob"),
        challenger.get("raw_away_win_prob"),
    ])
    if probs is None:
        return {
            "status": "SHADOW_PRESENT_BUT_1X2_PROBS_MISSING",
            "actionable": False,
            "decision_weight": 0.0,
        }
    c = [canonical.get("p_home"), canonical.get("p_draw"), canonical.get("p_away")]
    deltas = [round((probs[i] - float(c[i])) * 100.0, 3) if c[i] is not None else None for i in range(3)]
    return {
        "status": "LIVE_RESEARCH_SHADOW",
        "model": shadow.get("projection_model") or "RELATIVE_STRENGTH_1X2_SHADOW",
        "p_home": round(probs[0], 6),
        "p_draw": round(probs[1], 6),
        "p_away": round(probs[2], 6),
        "delta_vs_canonical_pp": {"HOME": deltas[0], "DRAW": deltas[1], "AWAY": deltas[2]},
        "max_abs_disagreement_pp": max(abs(v) for v in deltas if v is not None) if any(v is not None for v in deltas) else None,
        "actionable": False,
        "decision_weight": 0.0,
        "policy": "SHADOW_ONLY; OOS MODEL-SELECTION EVIDENCE REQUIRED BEFORE ANY PROMOTION",
    }


def _observed_rows(event: dict[str, Any], canonical: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    source = provenance.get("source") or "NOT_VERIFIED"
    model_probs = {
        "HOME": _num(canonical.get("p_home")),
        "DRAW": _num(canonical.get("p_draw")),
        "AWAY": _num(canonical.get("p_away")),
    }
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_1x2_market(market_row.get("market")):
            continue
        prices: dict[str, float] = {}
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _selection(value.get("selection"), fixture)
            price = _decimal(value.get("price"))
            if side is None or price is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "reason": "UNPARSED_1X2_SELECTION_OR_PRICE",
                })
                continue
            prices[side] = price
        fair = _fair_three_way(prices)
        for side in ("HOME", "DRAW", "AWAY"):
            price = prices.get(side)
            p_model = model_probs.get(side)
            if price is None or p_model is None or not 0 < p_model < 1:
                continue
            p_market_fair = fair.get(side) if fair else None
            rows.append({
                "selection": side,
                "probability_model_raw": round(p_model, 6),
                "fair_decimal_model_raw": round(1.0 / p_model, 4),
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
                "market_shrinkage": "NOT_PRODUCTION_CALIBRATED_FOR_1X2",
                "promotion_block": "1X2_MODEL_SELECTION_AND_OOS_CLV_GATES_NOT_PASSED",
            })

    rows.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") or -999.0),
            float(row.get("probability_model_raw") or 0.0),
        ),
        reverse=True,
    )
    return rows[:24], unsupported[:24]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    canonical = _canonical(raw)
    if canonical.get("status") != "LIVE_RESEARCH_MODELED":
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "CANONICAL_1X2_PROBABILITIES_MISSING",
        }

    shadow = _relative_strength_shadow(raw, canonical)
    observed, unsupported = _observed_rows(event, canonical)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "canonical": canonical,
        "relative_strength_shadow": shadow,
        "offline_challengers": {
            "dixon_coles": "OOS_VALIDATOR_AVAILABLE; NOT_APPLIED_LIVE",
            "class_prior_calibration": "OOS_VALIDATOR_AVAILABLE; NOT_APPLIED_LIVE",
        },
        "observed_market_rows": observed,
        "observed_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_MODEL_SELECTION_PENDING",
        "calibration_gate": {
            "minimum_same_cohort_oos_for_model_selection": 200,
            "minimum_oos_for_actionable_review": 400,
            "requires": [
                "challenger beats matched canonical baseline on Brier and log-loss",
                "H/D/A calibration stable by probability bucket and competition",
                "verified historical 1X2 prices and true CLV evidence",
                "validated market shrinkage after model selection",
                "no automatic promotion or weight changes",
            ],
        },
        "policy": (
            "SPORT_FIRST; CANONICAL 1X2 AND CHALLENGERS REMAIN RESEARCH-ONLY; "
            "COMPARE ONLY TO EXACT OBSERVED THREE-WAY MARKET; NO-VIG ONLY WITH HOME/DRAW/AWAY COMPLETE; "
            "ZERO DECISION WEIGHT; MODEL-SELECTION REPORT REQUIRED BEFORE PROMOTION REVIEW"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = observed_events = observed_rows = shadow_events = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["one_x_two_intelligence"] = intelligence
        if intelligence.get("status") == "LIVE_RESEARCH_MODELED":
            modeled += 1
        if ((intelligence.get("relative_strength_shadow") or {}).get("status")) == "LIVE_RESEARCH_SHADOW":
            shadow_events += 1
        count = int(intelligence.get("observed_market_count") or 0)
        if count:
            observed_events += 1
            observed_rows += count

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict):
                areas["one_x_two"] = {
                    "status": intelligence.get("status"),
                    "canonical": intelligence.get("canonical"),
                    "relative_strength_shadow": intelligence.get("relative_strength_shadow"),
                    "observed_market_rows": intelligence.get("observed_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }

    return {
        "modeled_events": modeled,
        "events_with_relative_strength_shadow": shadow_events,
        "events_with_observed_1x2_markets": observed_events,
        "observed_1x2_market_rows": observed_rows,
    }
