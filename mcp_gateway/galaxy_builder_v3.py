from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any

from mcp_gateway import galaxy_builder_v2 as v2

SCHEMA_VERSION = "0.3.0"
EXPLICIT_MODEL_FIELD = "explicit_derivative_models"
TARGET_DECIMAL = v2.TARGET_DECIMAL
TARGET_EDGE_PP = v2.TARGET_EDGE_PP
MIN_RESEARCH_AVAILABILITY = v2.MIN_AVAILABILITY
MIN_ACTIONABLE_AVAILABILITY = 0.85
MAX_DERIVATIVE_MULTI = 6

_CANONICAL_FAMILIES = {"FT_GOALS", "BTTS", "DOUBLE_CHANCE"}
_ALLOWED_MARKET_SOURCES = {"GALAXY_ODDS", "API_FALLBACK_ODDS"}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _american(decimal_price: float | None) -> int | None:
    if decimal_price is None or decimal_price <= 1:
        return None
    if decimal_price >= 2:
        return int(round((decimal_price - 1) * 100))
    return int(round(-100 / (decimal_price - 1)))


def _validated_quotes(row: dict[str, Any]) -> list[dict[str, Any]]:
    quotes: list[dict[str, Any]] = []
    for quote in row.get("quotes") or []:
        if not isinstance(quote, dict):
            continue
        bookmaker = _clean(quote.get("bookmaker"))
        market = _clean(quote.get("market"))
        selection_text = _clean(quote.get("selection_text") or quote.get("selection"))
        provider_update = _clean(quote.get("provider_update") or quote.get("timestamp"))
        price = _num(quote.get("price"))
        if not bookmaker or not market or not selection_text or not provider_update or price is None or price <= 1:
            continue
        quotes.append(
            {
                "bookmaker": bookmaker,
                "market": market,
                "selection_text": selection_text,
                "price": round(price, 4),
                "provider_update": provider_update,
            }
        )
    return quotes


def _validate_explicit_leg(row: Any) -> tuple[dict[str, Any] | None, str | None]:
    if not isinstance(row, dict):
        return None, "ROW_NOT_OBJECT"

    family = (_clean(row.get("family")) or "").upper()
    selection = (_clean(row.get("selection")) or "").upper()
    probability = _num(row.get("probability"))
    probability_source = _clean(row.get("probability_source"))
    model_version = _clean(row.get("model_version"))
    line = _num(row.get("line"))

    if not family or not selection:
        return None, "FAMILY_OR_SELECTION_MISSING"
    if family in _CANONICAL_FAMILIES:
        return None, "CANONICAL_FAMILY_ALREADY_MODELED_BY_GALAXY_BUILDER"
    if probability is None or probability <= 0 or probability >= 1:
        return None, "INVALID_MODEL_PROBABILITY"
    if row.get("sport_model_verified") is not True:
        return None, "SPORT_MODEL_NOT_VERIFIED"
    if row.get("sport_first_projection") is not True:
        return None, "SPORT_FIRST_PROJECTION_NOT_ATTESTED"
    if not probability_source:
        return None, "PROBABILITY_SOURCE_MISSING"
    if not model_version:
        return None, "MODEL_VERSION_MISSING"

    production_approved = row.get("production_approved") is True
    requested_actionable = row.get("actionable_model") is True
    actionable = bool(requested_actionable and production_approved)
    research_reason = None
    if requested_actionable and not production_approved:
        research_reason = "ACTIONABLE_REQUEST_BLOCKED_PRODUCTION_APPROVAL_MISSING"
    elif not requested_actionable:
        research_reason = _clean(row.get("research_only_reason")) or "EXPLICIT_MODEL_RESEARCH_ONLY"

    return (
        {
            "leg_id": _clean(row.get("leg_id")),
            "family": family,
            "selection": selection,
            "line": line,
            "probability": round(probability, 6),
            "probability_source": probability_source,
            "model_version": model_version,
            "model_timestamp": _clean(row.get("model_timestamp")),
            "sport_model_verified": True,
            "sport_first_projection": True,
            "production_approved": production_approved,
            "actionable_model": actionable,
            "research_only_reason": research_reason,
            "quotes": _validated_quotes(row),
            "source_contract": EXPLICIT_MODEL_FIELD,
        },
        None,
    )


def _explicit_models(event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[str] = []
    for row in event.get(EXPLICIT_MODEL_FIELD) or []:
        leg, reason = _validate_explicit_leg(row)
        if leg is not None:
            accepted.append(leg)
        elif reason:
            rejected.append(reason)
    return accepted, rejected


def _event_context(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    availability = _num(event.get("availability_confidence"))
    return {
        "fixture_id": fixture.get("fixture_id"),
        "match": f"{fixture.get('home_team') or fixture.get('home_name')} vs {fixture.get('away_team') or fixture.get('away_name')}",
        "kickoff": fixture.get("kickoff"),
        "league": fixture.get("league"),
        "country": fixture.get("country"),
        "data_tier": event.get("tier") or coverage.get("data_tier"),
        "availability_confidence": availability,
        "market_source": provenance.get("source"),
        "market_fresh": provenance.get("fresh") is True,
    }


def _best_prices_by_book(leg: dict[str, Any]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for quote in leg.get("quotes") or []:
        bookmaker = _clean(quote.get("bookmaker"))
        price = _num(quote.get("price"))
        if bookmaker and price is not None and price > 1:
            prices[bookmaker] = max(prices.get(bookmaker, 0.0), price)
    return prices


def _public_leg(leg: dict[str, Any]) -> dict[str, Any]:
    return dict(leg)


def _derivative_cross_match_candidates(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []

    for event in events:
        if event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "CLOSE"}:
            continue

        context = _event_context(event)
        availability = context.get("availability_confidence")
        if context.get("data_tier") not in {"A", "B"}:
            continue
        if availability is None or availability < MIN_RESEARCH_AVAILABILITY:
            continue
        if context.get("market_source") not in _ALLOWED_MARKET_SOURCES or not context.get("market_fresh"):
            continue

        explicit, _ = _explicit_models(event)
        canonical, _ = v2._legs(event)
        legs = [*canonical, *explicit]

        for leg in legs:
            probability = _num(leg.get("probability"))
            if probability is None or probability < 0.70:
                continue
            prices = _best_prices_by_book(leg)
            if not prices:
                continue
            pool.append(
                {
                    **context,
                    "leg": leg,
                    "by_book": prices,
                    "is_explicit_derivative": leg.get("source_contract") == EXPLICIT_MODEL_FIELD,
                }
            )

    candidates: list[dict[str, Any]] = []
    for left, right in itertools.combinations(pool, 2):
        if left.get("fixture_id") == right.get("fixture_id"):
            continue
        if not (left.get("is_explicit_derivative") or right.get("is_explicit_derivative")):
            continue

        common_books = set(left["by_book"]) & set(right["by_book"])
        for book in common_books:
            decimal_reference = left["by_book"][book] * right["by_book"][book]
            if not (TARGET_DECIMAL <= decimal_reference <= 3.50):
                continue

            raw_joint = float(left["leg"]["probability"]) * float(right["leg"]["probability"])
            conservative_joint = raw_joint * 0.98
            edge_pp = (conservative_joint - 1 / decimal_reference) * 100
            if edge_pp < TARGET_EDGE_PP:
                continue

            all_actionable = bool(left["leg"].get("actionable_model")) and bool(
                right["leg"].get("actionable_model")
            )
            min_availability = min(
                float(left["availability_confidence"]), float(right["availability_confidence"])
            )
            block_reasons = ["FINAL_PARLAY_QUOTE_NOT_VERIFIED"]
            if not all_actionable:
                block_reasons.append("RESEARCH_ONLY_LEG_PRESENT")
            if min_availability < MIN_ACTIONABLE_AVAILABILITY:
                block_reasons.append("AVAILABILITY_BELOW_0.85")

            candidates.append(
                {
                    "type": "MULTI_MATCH_PARLAY",
                    "candidate_source": "GALAXY_BUILDER_V0.3_EXPLICIT_DERIVATIVE_CONTRACT",
                    "bookmaker": book,
                    "legs": [
                        {
                            "fixture_id": left["fixture_id"],
                            "match": left["match"],
                            "kickoff": left["kickoff"],
                            "league": left["league"],
                            "availability_confidence": round(float(left["availability_confidence"]), 3),
                            **_public_leg(left["leg"]),
                            "decimal_price": round(left["by_book"][book], 4),
                        },
                        {
                            "fixture_id": right["fixture_id"],
                            "match": right["match"],
                            "kickoff": right["kickoff"],
                            "league": right["league"],
                            "availability_confidence": round(float(right["availability_confidence"]), 3),
                            **_public_leg(right["leg"]),
                            "decimal_price": round(right["by_book"][book], 4),
                        },
                    ],
                    "component_product_decimal_reference": round(decimal_reference, 4),
                    "component_product_american_reference": _american(decimal_reference),
                    "conservative_joint_probability": round(conservative_joint, 6),
                    "probability_edge_vs_component_product_pp": round(edge_pp, 2),
                    "exact_parlay_quote": None,
                    "status": "GALAXY MULTI WATCH — FINAL PARLAY QUOTE NEEDED",
                    "bet_eligible": False,
                    "all_leg_models_actionable": all_actionable,
                    "block_reasons": block_reasons,
                    "joint_probability_method": (
                        "DISTINCT_FIXTURES_INDEPENDENCE_WITH_2_PERCENT_UNCERTAINTY_HAIRCUT"
                    ),
                }
            )

    candidates.sort(
        key=lambda row: (
            row.get("all_leg_models_actionable") is True,
            row.get("probability_edge_vs_component_product_pp") or 0,
            row.get("conservative_joint_probability") or 0,
        ),
        reverse=True,
    )
    return candidates[:MAX_DERIVATIVE_MULTI]


def _contract_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    accepted_count = 0
    rejected = Counter()
    families = Counter()
    actionable_count = 0
    quoted_count = 0

    for event in events:
        if not isinstance(event, dict):
            continue
        accepted, rejected_reasons = _explicit_models(event)
        accepted_count += len(accepted)
        rejected.update(rejected_reasons)
        for leg in accepted:
            families[str(leg.get("family") or "UNKNOWN")] += 1
            actionable_count += int(bool(leg.get("actionable_model")))
            quoted_count += int(bool(leg.get("quotes")))

    return {
        "schema_version": "1.0.0",
        "input_field": EXPLICIT_MODEL_FIELD,
        "accepted_model_leg_count": accepted_count,
        "actionable_model_leg_count": actionable_count,
        "accepted_with_verified_quote_shape_count": quoted_count,
        "accepted_family_counts": dict(families),
        "rejected_count": sum(rejected.values()),
        "rejected_reason_counts": dict(rejected),
        "required_fields": [
            "family",
            "selection",
            "probability",
            "probability_source",
            "model_version",
            "sport_model_verified=true",
            "sport_first_projection=true",
        ],
        "actionable_extra_gate": "actionable_model=true AND production_approved=true",
        "quote_shape_required_for_parlay_pool": (
            "bookmaker + market + selection_text + decimal price + provider_update"
        ),
        "same_game_rule": (
            "MARGINAL DERIVATIVE PROBABILITIES ARE NEVER MULTIPLIED INSIDE ONE MATCH. "
            "A DERIVATIVE SGP REQUIRES AN EXPLICIT CORRELATION-AWARE JOINT MODEL CONTRACT."
        ),
        "cross_match_rule": (
            "EXPLICIT DERIVATIVE LEGS MAY ENTER RESEARCH MULTI-MATCH CANDIDATES ONLY WITH "
            "VERIFIED FRESH MARKET PROVENANCE; DISTINCT-FIXTURE JOINT P USES THE EXISTING "
            "2_PERCENT_UNCERTAINTY_HAIRCUT AND STILL REQUIRES A FINAL PARLAY QUOTE."
        ),
    }


def build(payload: dict[str, Any]) -> dict[str, Any]:
    base = v2.build(payload)
    events = [event for event in payload.get("events") or [] if isinstance(event, dict)]

    derivative_multis = _derivative_cross_match_candidates(events)
    existing_multis = [
        row for row in base.get("multi_match_candidates") or [] if isinstance(row, dict)
    ]

    combined_multis = existing_multis + derivative_multis
    combined_multis.sort(
        key=lambda row: (
            row.get("all_leg_models_actionable") is True,
            row.get("probability_edge_vs_component_product_pp") or 0,
            row.get("conservative_joint_probability") or 0,
        ),
        reverse=True,
    )
    combined_multis = combined_multis[: v2.MAX_MULTI]

    same_game = [
        row for row in base.get("same_game_candidates") or [] if isinstance(row, dict)
    ]
    contract = _contract_summary(events)

    family_policy = dict(base.get("family_policy") or {})
    family_policy["OTHER_MARKETS"] = (
        "INGEST_ONLY_FROM_EXPLICIT_DERIVATIVE_MODELS CONTRACT; "
        "NO MARKET-DERIVED OR FABRICATED PROBABILITIES"
    )
    family_policy["DERIVATIVE_SGP"] = (
        "BLOCKED_UNLESS_EXPLICIT_CORRELATION_AWARE_JOINT_MODEL_EXISTS"
    )

    return {
        **base,
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(same_game) + len(combined_multis),
        "same_game_candidate_count": len(same_game),
        "multi_match_candidate_count": len(combined_multis),
        "same_game_candidates": same_game,
        "multi_match_candidates": combined_multis,
        "explicit_derivative_multi_candidate_count": len(derivative_multis),
        "derivative_model_contract": contract,
        "family_policy": family_policy,
        "actionable_gate": (
            "FINAL_SPORTSBOOK_QUOTE + ALL_LEGS_MODELED + PRODUCTION_APPROVED_DERIVATIVE_MODELS "
            "+ AVAILABILITY>=0.85 + EDGE_GATE"
        ),
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
        "policy": (
            "SPORT FIRST; MARKET SECOND; COMBINATION THIRD; TARGET +110 OR BETTER; "
            "EXPLICIT DERIVATIVE MODEL CONTRACT; NEVER INVENT PRICES, PROBABILITIES, OR CORRELATION"
        ),
    }
