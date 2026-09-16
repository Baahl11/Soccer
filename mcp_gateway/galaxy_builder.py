from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = "0.1.0"
MIN_RESEARCH_JOINT_PROB = 0.58
MIN_RESEARCH_AVAILABILITY = 0.70
BET_AVAILABILITY_GATE = 0.85
TARGET_EDGE_PP = 3.5
MAX_CANDIDATES = 5

# v0.1 deliberately stays inside the best-supported Soccer Edge family: full-match goals.
# Each two-leg corridor is a single total-goals interval, so the joint probability is
# calculated directly from the score-count distribution rather than multiplying legs.
CORRIDORS: tuple[dict[str, Any], ...] = (
    {
        "key": "O0.5_U3.5",
        "min_goals": 1,
        "max_goals": 3,
        "legs": (
            {"family": "FT_GOALS", "selection": "OVER", "line": 0.5},
            {"family": "FT_GOALS", "selection": "UNDER", "line": 3.5},
        ),
    },
    {
        "key": "O1.5_U4.5",
        "min_goals": 2,
        "max_goals": 4,
        "legs": (
            {"family": "FT_GOALS", "selection": "OVER", "line": 1.5},
            {"family": "FT_GOALS", "selection": "UNDER", "line": 4.5},
        ),
    },
    {
        "key": "O1.5_U5.5",
        "min_goals": 2,
        "max_goals": 5,
        "legs": (
            {"family": "FT_GOALS", "selection": "OVER", "line": 1.5},
            {"family": "FT_GOALS", "selection": "UNDER", "line": 5.5},
        ),
    },
    {
        "key": "O0.5_U4.5",
        "min_goals": 1,
        "max_goals": 4,
        "legs": (
            {"family": "FT_GOALS", "selection": "OVER", "line": 0.5},
            {"family": "FT_GOALS", "selection": "UNDER", "line": 4.5},
        ),
    },
)


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _poisson_pmf(goals: int, lam: float) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def _interval_probability(lam: float, minimum: int, maximum: int) -> float:
    if lam <= 0 or minimum < 0 or maximum < minimum:
        return 0.0
    return max(0.0, min(1.0, sum(_poisson_pmf(k, lam) for k in range(minimum, maximum + 1))))


def _decimal_to_american(decimal_price: float | None) -> int | None:
    if decimal_price is None or decimal_price <= 1.0:
        return None
    if decimal_price >= 2.0:
        return int(round((decimal_price - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_price - 1.0)))


def _prices(probability: float) -> dict[str, Any]:
    fair_decimal = 1.0 / probability if probability > 0 else None
    edge_prob = probability - TARGET_EDGE_PP / 100.0
    min_decimal = 1.0 / edge_prob if edge_prob > 0 else None
    return {
        "fair_decimal": round(fair_decimal, 3) if fair_decimal else None,
        "fair_american": _decimal_to_american(fair_decimal),
        "minimum_sgp_decimal_for_target_edge": round(min_decimal, 3) if min_decimal else None,
        "minimum_sgp_american_for_target_edge": _decimal_to_american(min_decimal),
        "target_probability_edge_pp": TARGET_EDGE_PP,
    }


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("½", ".5").split())


def _line_from_text(value: Any) -> float | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)", str(value or ""))
    return _num(match.group(1)) if match else None


def _is_full_match_goal_market(name: Any) -> bool:
    text = _normalize(name)
    if not text:
        return False
    if any(token in text for token in ("first half", "1st half", "second half", "2nd half", "team total", "corner", "card")):
        return False
    return any(token in text for token in ("goals over/under", "over/under", "total goals", "goals total"))


def _observed_leg(event: dict[str, Any], leg: dict[str, Any]) -> dict[str, Any]:
    market = event.get("market")
    if not isinstance(market, dict):
        return {"observed": False, "best_price": None, "bookmaker": None, "provider_update": None}

    desired_side = str(leg.get("selection") or "").lower()
    desired_line = _num(leg.get("line"))
    matches: list[dict[str, Any]] = []
    for row in market.get("markets") or []:
        if not isinstance(row, dict) or not _is_full_match_goal_market(row.get("market")):
            continue
        for value in row.get("values") or []:
            if not isinstance(value, dict):
                continue
            selection_text = _normalize(value.get("selection"))
            if desired_side not in selection_text:
                continue
            parsed_line = _line_from_text(value.get("selection"))
            if desired_line is not None and (parsed_line is None or abs(parsed_line - desired_line) > 1e-6):
                continue
            price = _num(value.get("price"))
            if price is None or price <= 1.0:
                continue
            matches.append(
                {
                    "price": price,
                    "bookmaker": row.get("bookmaker"),
                    "provider_update": row.get("provider_update"),
                }
            )
    if not matches:
        return {"observed": False, "best_price": None, "bookmaker": None, "provider_update": None}
    best = max(matches, key=lambda item: item["price"])
    return {
        "observed": True,
        "best_price": round(float(best["price"]), 4),
        "bookmaker": best.get("bookmaker"),
        "provider_update": best.get("provider_update"),
    }


def _fixture_label(fx: dict[str, Any]) -> str:
    home = fx.get("home_team") or fx.get("home_name") or "NOT VERIFIED"
    away = fx.get("away_team") or fx.get("away_name") or "NOT VERIFIED"
    return f"{home} vs {away}"


def _candidate(event: dict[str, Any], corridor: dict[str, Any], probability: float) -> dict[str, Any]:
    fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    availability = _num(event.get("availability_confidence"))
    legs = []
    all_legs_observed = True
    for leg in corridor["legs"]:
        observed = _observed_leg(event, leg)
        all_legs_observed = all_legs_observed and bool(observed["observed"])
        legs.append({**leg, **observed})

    # API-Football exposes component markets, not the sportsbook's correlation-adjusted
    # SGP quote. Therefore v0.1 can never synthesize an actionable combined price.
    exact_sgp_quote = None
    availability_ok_for_bet = availability is not None and availability >= BET_AVAILABILITY_GATE
    status = "GALAXY WATCH — QUOTE/VERIFICATION NEEDED"
    block_reasons: list[str] = []
    if not all_legs_observed:
        block_reasons.append("ONE_OR_MORE_EXACT_LEGS_NOT_OBSERVED")
    if not availability_ok_for_bet:
        block_reasons.append("AVAILABILITY_BELOW_0.85")
    block_reasons.append("EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED")

    prices = _prices(probability)
    return {
        "fixture_id": fx.get("fixture_id"),
        "kickoff": fx.get("kickoff"),
        "country": fx.get("country"),
        "league": fx.get("league"),
        "match": _fixture_label(fx),
        "data_tier": event.get("tier") or coverage.get("data_tier"),
        "stage": event.get("stage"),
        "availability_confidence": round(availability, 3) if availability is not None else None,
        "projection_source": raw.get("sport_source") or raw.get("projection_model"),
        "raw_total_goals": round(float(raw.get("raw_total_goals")), 4),
        "corridor_key": corridor["key"],
        "goal_interval": {"minimum": corridor["min_goals"], "maximum": corridor["max_goals"]},
        "legs": legs,
        "joint_model_probability": round(probability, 6),
        **prices,
        "target_sgp_price_zone_american": {"minimum": 100, "maximum": 160},
        "exact_sgp_quote": exact_sgp_quote,
        "all_exact_legs_observed": all_legs_observed,
        "status": status,
        "bet_eligible": False,
        "block_reasons": block_reasons,
        "joint_probability_method": "DIRECT_POISSON_GOAL_INTERVAL; NEVER_MULTIPLY_CORRELATED_LEGS",
    }


def build(payload: dict[str, Any]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    seen_fixtures: set[Any] = set()

    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH":
            continue
        if event.get("stage") in {"POSTGAME", "CLOSE"}:
            continue
        fx = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
        fixture_id = fx.get("fixture_id")
        if fixture_id in seen_fixtures:
            continue
        coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
        tier = event.get("tier") or coverage.get("data_tier")
        if tier not in {"A", "B"}:
            continue
        availability = _num(event.get("availability_confidence"))
        if availability is None or availability < MIN_RESEARCH_AVAILABILITY:
            continue
        raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
        lam = _num(raw.get("raw_total_goals"))
        if lam is None or not (0.15 <= lam <= 7.0):
            continue

        options: list[tuple[float, dict[str, Any]]] = []
        for corridor in CORRIDORS:
            probability = _interval_probability(lam, corridor["min_goals"], corridor["max_goals"])
            if probability >= MIN_RESEARCH_JOINT_PROB:
                options.append((probability, corridor))
        if not options:
            continue

        # Prefer a useful research corridor over an almost-certain redundant one:
        # rank near 65% first, then by absolute probability. This keeps the builder
        # oriented toward eventual +100/+160 combinations without inventing a quote.
        probability, corridor = min(options, key=lambda item: (abs(item[0] - 0.65), -item[0]))
        candidates.append(_candidate(event, corridor, probability))
        seen_fixtures.add(fixture_id)

    candidates.sort(
        key=lambda row: (
            bool(row.get("all_exact_legs_observed")),
            float(row.get("availability_confidence") or 0.0),
            float(row.get("joint_model_probability") or 0.0),
        ),
        reverse=True,
    )
    candidates = candidates[:MAX_CANDIDATES]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "RESEARCH_ONLY_ACTIVE",
        "generated_at_local": payload.get("generated_at_local"),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "research_gates": {
            "data_tier": ["A", "B"],
            "minimum_availability_confidence": MIN_RESEARCH_AVAILABILITY,
            "minimum_joint_probability": MIN_RESEARCH_JOINT_PROB,
            "target_probability_edge_pp": TARGET_EDGE_PP,
        },
        "actionable_gate": (
            "BLOCKED_UNTIL_EXACT_SPORTSBOOK_SGP_QUOTE_AND_ALL_LEGS_VERIFIED; "
            "AVAILABILITY_CONFIDENCE_MUST_BE_AT_LEAST_0.85"
        ),
        "allowed_v0_1_families": ["FT_GOALS_CORRIDORS_ONLY"],
        "future_shadow_families": ["SIDE_PROTECTION", "BTTS", "2H_GOALS", "CORNERS", "CARDS", "GK_SAVES", "PLAYER_SHOTS"],
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "policy": (
            "SPORT_FIRST; COMBINATION_THIRD; DIRECT_JOINT_PROBABILITY; NEVER_MULTIPLY_CORRELATED_LEGS; "
            "NEVER_SYNTHESIZE_A_SPORTSBOOK_SGP_PRICE"
        ),
    }
