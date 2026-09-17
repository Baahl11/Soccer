from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
MAX_EXACT_GOALS_PER_TEAM = 10


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


def _pmf(goals: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** goals) / math.factorial(goals)


def _score_probability(home_goals: int, away_goals: int, home_lambda: float, away_lambda: float) -> float:
    return max(0.0, min(1.0, _pmf(home_goals, home_lambda) * _pmf(away_goals, away_lambda)))


def _parse_score(value: Any) -> tuple[int, int] | None:
    text = _norm(value)
    # Accept common exact-score sportsbook formats: 1-0, 1:0, 1 - 0.
    match = re.search(r"(?<!\d)(\d{1,2})\s*[-:]\s*(\d{1,2})(?!\d)", text)
    if not match:
        return None
    home = int(match.group(1))
    away = int(match.group(2))
    if home > MAX_EXACT_GOALS_PER_TEAM or away > MAX_EXACT_GOALS_PER_TEAM:
        return None
    return home, away


def _is_correct_score_market(value: Any) -> bool:
    name = _norm(value)
    return "correct score" in name or "exact score" in name


def _top_scores(home_lambda: float, away_lambda: float, limit: int = 10) -> list[dict[str, Any]]:
    rows: list[tuple[float, int, int]] = []
    for home in range(0, 8):
        for away in range(0, 8):
            rows.append((_score_probability(home, away, home_lambda, away_lambda), home, away))
    rows.sort(reverse=True)
    return [
        {
            "home": home,
            "away": away,
            "score": f"{home}-{away}",
            "probability_model": round(probability, 6),
            "fair_decimal_model": round(1.0 / probability, 4) if probability > 0 else None,
        }
        for probability, home, away in rows[:limit]
    ]


def _observed_rows(event: dict[str, Any], home_lambda: float, away_lambda: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    market_fresh = provenance.get("fresh") is True
    market_source = provenance.get("source") or "NOT_VERIFIED"
    supported: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_correct_score_market(market_row.get("market")):
            continue

        priced_values: list[tuple[dict[str, Any], float]] = []
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            price = _decimal(value.get("price"))
            if price is not None:
                priced_values.append((value, price))
        total_implied = sum(1.0 / price for _, price in priced_values)

        for value, price in priced_values:
            score = _parse_score(value.get("selection"))
            if score is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "decimal_price": round(price, 4),
                    "reason": "NON_EXACT_SCORE_BUCKET_OR_UNSUPPORTED_SCORE",
                })
                continue
            home_goals, away_goals = score
            probability = _score_probability(home_goals, away_goals, home_lambda, away_lambda)
            market_fair = (1.0 / price) / total_implied if total_implied > 0 else None
            supported.append({
                "score": f"{home_goals}-{away_goals}",
                "home_goals": home_goals,
                "away_goals": away_goals,
                "probability_model_raw": round(probability, 6),
                "fair_decimal_model_raw": round(1.0 / probability, 4) if probability > 0 else None,
                "bookmaker": market_row.get("bookmaker"),
                "bookmaker_id": market_row.get("bookmaker_id"),
                "market": market_row.get("market"),
                "market_id": market_row.get("market_id"),
                "decimal_price": round(price, 4),
                "p_breakeven": round(1.0 / price, 6),
                "p_market_fair_full_market": round(market_fair, 6) if market_fair is not None else None,
                "raw_edge_vs_market_fair_pp": round((probability - market_fair) * 100.0, 3) if market_fair is not None else None,
                "raw_ev_at_observed_price": round(probability * price - 1.0, 6),
                "provider_update": market_row.get("provider_update"),
                "market_source": market_source,
                "market_fresh": market_fresh,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "market_shrinkage": "NOT_CALIBRATED_FOR_CORRECT_SCORE",
                "promotion_block": "CORRECT_SCORE_NOT_OOS_CALIBRATED_OR_PRODUCTION_APPROVED",
            })

    supported.sort(
        key=lambda row: (
            1.0 if row.get("market_fresh") else 0.0,
            float(row.get("raw_edge_vs_market_fair_pp") or -999.0),
            float(row.get("probability_model_raw") or 0.0),
        ),
        reverse=True,
    )
    return supported[:40], unsupported[:40]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    home_lambda = _num(raw.get("raw_home_goal_rate"))
    away_lambda = _num(raw.get("raw_away_goal_rate"))
    modeled = home_lambda is not None and away_lambda is not None and home_lambda > 0 and away_lambda > 0

    if modeled:
        top_scores = _top_scores(home_lambda, away_lambda)
        observed, unsupported = _observed_rows(event, home_lambda, away_lambda)
    else:
        top_scores, observed, unsupported = [], [], []

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED" if modeled else "NOT_MODELED_THIS_TICK",
        "model": "CANONICAL_HOME_AWAY_LAMBDA_INDEPENDENT_POISSON_EXACT_SCORE_v0.1",
        "model_source": "EXISTING_CANONICAL_RAW_HOME_AWAY_GOAL_RATES",
        "home_lambda": round(home_lambda, 4) if home_lambda is not None else None,
        "away_lambda": round(away_lambda, 4) if away_lambda is not None else None,
        "top_scorelines": top_scores,
        "observed_exact_score_market_rows": observed,
        "observed_exact_score_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_fixtures_for_market_comparison": 300,
            "minimum_oos_fixtures_for_actionable_review": 500,
            "requires": [
                "multiclass Brier and negative log likelihood stability",
                "top-1/top-3/top-5 hit-rate calibration by competition",
                "verified correct-score market history and CLV evidence",
                "validated market shrinkage for sparse exact-score outcomes",
                "no material degradation versus canonical FT-goals calibration",
            ],
        },
        "policy": (
            "SPORT_FIRST; EXACT SCORE PROBABILITY DERIVED ONLY FROM CANONICAL HOME/AWAY LAMBDAS; "
            "ONLY OBSERVED CORRECT_SCORE/EXACT_SCORE PRICES ARE COMPARED; NO SYNTHETIC SCORE MARKET; "
            "NO BET_LEAN_GALAXY PROMOTION"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled_events = 0
    observed_events = 0
    observed_rows = 0
    unsupported_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["correct_score_intelligence"] = intelligence
        if intelligence.get("status") == "LIVE_RESEARCH_MODELED":
            modeled_events += 1
        count = int(intelligence.get("observed_exact_score_market_count") or 0)
        if count:
            observed_events += 1
            observed_rows += count
        unsupported_rows += len(intelligence.get("unsupported_market_rows") or [])

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict):
                areas["correct_score"] = {
                    "status": intelligence.get("status"),
                    "home_lambda": intelligence.get("home_lambda"),
                    "away_lambda": intelligence.get("away_lambda"),
                    "top_scorelines": intelligence.get("top_scorelines") or [],
                    "observed_exact_score_market_rows": intelligence.get("observed_exact_score_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }

    return {
        "modeled_events": modeled_events,
        "events_with_observed_correct_score_markets": observed_events,
        "observed_exact_score_market_rows": observed_rows,
        "unsupported_market_rows": unsupported_rows,
    }
