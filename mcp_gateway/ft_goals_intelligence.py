from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0.0"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 6) -> float | None:
    x = _num(value)
    return round(x, digits) if x is not None else None


def _canonical_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    total = _num(raw.get("raw_total_goals"))
    status = "LIVE_MODELED" if total is not None else "NOT_MODELED_THIS_TICK"
    return {
        "status": status,
        "model_version": raw.get("model_version"),
        "projection_model": raw.get("projection_model"),
        "home_lambda": _round(raw.get("raw_home_goal_rate"), 4),
        "away_lambda": _round(raw.get("raw_away_goal_rate"), 4),
        "total_lambda": _round(raw.get("raw_total_goals"), 4),
        "p_over_1_5": _round(raw.get("raw_over_1_5_prob")),
        "p_over_2_5": _round(raw.get("raw_over_2_5_prob")),
        "p_over_3_5": _round(raw.get("raw_over_3_5_prob")),
        "scoring_path": raw.get("scoring_path"),
        "sample": raw.get("sample") if isinstance(raw.get("sample"), dict) else {},
        "advanced_metrics": raw.get("advanced_metrics") or "NOT_VERIFIED",
    }


def _shadow_snapshot(raw: dict[str, Any]) -> dict[str, Any]:
    shadow = raw.get("relative_strength_shadow")
    if not isinstance(shadow, dict) or shadow.get("status") != "RESEARCH_ONLY_SHADOW":
        return {
            "status": "NOT_AVAILABLE_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
        }
    baseline = shadow.get("baseline") if isinstance(shadow.get("baseline"), dict) else {}
    challenger = shadow.get("challenger") if isinstance(shadow.get("challenger"), dict) else {}
    deltas: dict[str, float | None] = {}
    for key in (
        "raw_home_goal_rate",
        "raw_away_goal_rate",
        "raw_total_goals",
        "raw_over_1_5_prob",
        "raw_over_2_5_prob",
        "raw_over_3_5_prob",
        "raw_btts_yes_prob",
    ):
        b = _num(baseline.get(key))
        c = _num(challenger.get(key))
        deltas[key] = round(c - b, 6) if b is not None and c is not None else None
    return {
        "status": "LIVE_RESEARCH_SHADOW",
        "actionable": False,
        "decision_weight": 0.0,
        "projection_model": shadow.get("projection_model"),
        "baseline_source": shadow.get("baseline_source"),
        "sample": shadow.get("sample") if isinstance(shadow.get("sample"), dict) else {},
        "strengths": shadow.get("strengths") if isinstance(shadow.get("strengths"), dict) else {},
        "baseline": {k: baseline.get(k) for k in baseline},
        "challenger": {k: challenger.get(k) for k in challenger},
        "delta_challenger_minus_canonical": deltas,
        "policy": "SHADOW_ONLY; MUST_OUTPERFORM_CANONICAL_OOS_BEFORE_ANY_PROMOTION",
    }


def _market_ladder(event: dict[str, Any]) -> list[dict[str, Any]]:
    decision = event.get("market_decision") if isinstance(event.get("market_decision"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    rows: list[dict[str, Any]] = []
    for item in decision.get("decisions") or []:
        if not isinstance(item, dict) or str(item.get("family") or "").upper() != "TOTAL":
            continue
        line = _num(item.get("line"))
        price = _num(item.get("decimal_price"))
        p_shrunk = _num(item.get("p_shrunk"))
        fair_decimal = (1.0 / p_shrunk) if p_shrunk is not None and 0 < p_shrunk < 1 else None
        rows.append(
            {
                "selection": str(item.get("selection") or "").upper(),
                "line": round(line, 2) if line is not None else None,
                "bookmaker": item.get("bookmaker"),
                "decimal_price": round(price, 4) if price is not None else None,
                "provider_update": item.get("provider_update"),
                "market_fresh": fresh,
                "p_breakeven": _round(item.get("p_breakeven")),
                "p_market_fair": _round(item.get("p_market_fair")),
                "p_raw": _round(item.get("p_raw")),
                "shrink_weight": _round(item.get("shrink_weight"), 3),
                "p_shrunk": _round(item.get("p_shrunk")),
                "fair_decimal_from_p_shrunk": round(fair_decimal, 4) if fair_decimal is not None else None,
                "prob_edge_pp": _round(item.get("prob_edge_pp"), 3),
                "estimated_ev": _round(item.get("estimated_ev")),
                "tier": item.get("tier"),
                "classification": item.get("classification"),
                "discrepancy_recheck": bool(item.get("discrepancy_recheck")),
                "reasons": list(item.get("reasons") or []),
                "exact_observed_market": bool(line is not None and price is not None and item.get("bookmaker")),
            }
        )
    rank = {"BET": 4, "LEAN": 3, "WATCH": 2, "PASS": 1}
    rows.sort(
        key=lambda r: (
            rank.get(str(r.get("classification") or ""), 0),
            float(r.get("prob_edge_pp") or -999),
            float(r.get("estimated_ev") or -999),
        ),
        reverse=True,
    )
    return rows[:16]


def build(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    ladder = _market_ladder(event)
    canonical = _canonical_snapshot(raw)
    shadow = _shadow_snapshot(raw)
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    availability = _round(event.get("availability_confidence"), 2)
    best = ladder[0] if ladder else None
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": ((event.get("fixture") or {}).get("fixture_id") if isinstance(event.get("fixture"), dict) else None),
        "stage": event.get("stage"),
        "canonical": canonical,
        "relative_strength_shadow": shadow,
        "observed_total_market_ladder": ladder,
        "observed_total_market_count": len(ladder),
        "best_exact_total_market": best,
        "checkpoint": {
            "data": "LIVE_BASE_GOAL_RATES" if canonical.get("status") == "LIVE_MODELED" else "INSUFFICIENT_BASE_DATA",
            "model": "LIVE_CANONICAL" if canonical.get("status") == "LIVE_MODELED" else "NOT_MODELED",
            "shadow_model": shadow.get("status"),
            "market_pricing": "LIVE_EXACT_OBSERVED_LINE_PRICING" if ladder else "NO_VERIFIED_TOTAL_LINE_PRICED_THIS_TICK",
            "production_family": "FT_TOTALS_CANONICAL_ACTIONABLE_SCOPE",
            "data_tier": coverage.get("data_tier"),
            "availability_confidence": availability,
        },
        "remaining_model_gaps": [
            "ADVANCED_XG_NPXG_NOT_LIVE",
            "SHOT_QUALITY_NOT_LIVE",
            "QUANTIFIED_GOALKEEPER_IMPACT_PENDING",
            "SET_PIECE_GOAL_MODEL_PENDING",
            "REST_CONGESTION_TRAVEL_FEATURES_PENDING",
            "WEATHER_MATERIALITY_SOURCE_PENDING",
        ],
        "policy": "OBSERVABILITY_AND_SHADOW_VALIDATION_ONLY; NO_CANONICAL_WEIGHT_THRESHOLD_CLASSIFICATION_OR_STAKE_CHANGE",
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = 0
    shadow = 0
    ladder_events = 0
    ladder_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["ft_goals_intelligence"] = intelligence
        if ((intelligence.get("canonical") or {}).get("status")) == "LIVE_MODELED":
            modeled += 1
        if ((intelligence.get("relative_strength_shadow") or {}).get("status")) == "LIVE_RESEARCH_SHADOW":
            shadow += 1
        count = int(intelligence.get("observed_total_market_count") or 0)
        if count:
            ladder_events += 1
            ladder_rows += count

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict) and isinstance(areas.get("goals_full_match"), dict):
                goals = areas["goals_full_match"]
                goals["canonical_projection_model"] = (intelligence.get("canonical") or {}).get("projection_model")
                goals["observed_total_market_ladder"] = intelligence.get("observed_total_market_ladder") or []
                goals["relative_strength_shadow"] = intelligence.get("relative_strength_shadow")
                goals["checkpoint"] = intelligence.get("checkpoint")
                goals["remaining_model_gaps"] = intelligence.get("remaining_model_gaps")

    return {
        "modeled_events": modeled,
        "shadow_events": shadow,
        "events_with_exact_total_market_ladder": ladder_events,
        "exact_total_market_rows": ladder_rows,
    }
