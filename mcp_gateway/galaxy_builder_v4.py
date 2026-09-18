from __future__ import annotations

import itertools
from typing import Any

from mcp_gateway import galaxy_builder_v2 as v2
from mcp_gateway import galaxy_builder_v3 as v3
from mcp_gateway import quote_freshness as qf

SCHEMA_VERSION = "0.4.2"
TARGET_DECIMAL = v2.TARGET_DECIMAL
TARGET_EDGE_PP = v2.TARGET_EDGE_PP
MIN_AVAILABILITY = v2.MIN_AVAILABILITY
MIN_ACTIONABLE_AVAILABILITY = 0.85
MAX_SGP = 6
MAX_MULTI = v2.MAX_MULTI
MIN_INCREMENTAL_PROBABILITY_DROP = 0.01
_ALLOWED_MARKET_SOURCES = {"GALAXY_ODDS", "API_FALLBACK_ODDS"}


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _verified_quotes(leg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for quote in leg.get("quotes") or []:
        if not isinstance(quote, dict):
            continue
        bookmaker = str(quote.get("bookmaker") or "").strip()
        market = str(quote.get("market") or "").strip()
        selection_text = str(quote.get("selection_text") or "").strip()
        price = _num(quote.get("price"))
        provider_update = quote.get("provider_update")
        if not bookmaker or not market or not selection_text or price is None or price <= 1:
            continue
        if not qf.is_fresh(provider_update):
            continue
        age = qf.age_minutes(provider_update)
        out.append(
            {
                "bookmaker": bookmaker,
                "market": market,
                "selection_text": selection_text,
                "price": round(price, 4),
                "provider_update": provider_update,
                "quote_age_minutes": round(age, 2) if age is not None else None,
                "quote_fresh": True,
                "quote_freshness_anchor": "PROVIDER_UPDATE",
            }
        )
    return out


def _canonical_ft_goals_legs(event: dict[str, Any]) -> list[dict[str, Any]]:
    decision = event.get("market_decision") if isinstance(event.get("market_decision"), dict) else {}
    rows = decision.get("decisions") if isinstance(decision.get("decisions"), list) else []
    rank = {"BET": 4, "LEAN": 3, "WATCH": 2, "PASS": 1}
    best: dict[tuple[str, float, str], dict[str, Any]] = {}

    for item in rows:
        if not isinstance(item, dict) or str(item.get("family") or "").upper() != "TOTAL":
            continue

        selection = str(item.get("selection") or "").strip().upper()
        line = _num(item.get("line"))
        price = _num(item.get("decimal_price"))
        probability = _num(item.get("p_shrunk"))
        raw_probability = _num(item.get("p_raw"))
        bookmaker = str(item.get("bookmaker") or "").strip()
        market = str(item.get("market") or "").strip()
        classification = str(item.get("classification") or "WATCH").upper()

        if selection not in {"OVER", "UNDER"}:
            continue
        if line is None or price is None or price <= 1 or probability is None or not (0 < probability < 1):
            continue
        if not bookmaker or not market or classification == "PASS":
            continue

        discrepancy = bool(item.get("discrepancy_recheck"))
        actionable = classification in {"BET", "LEAN"} and not discrepancy
        row = {
            "family": "FT_GOALS",
            "selection": selection,
            "line": line,
            "probability": round(probability, 6),
            "probability_source": "FT_GOALS_CANONICAL_MARKET_LADDER_P_SHRUNK",
            "score_matrix_marginal_probability": round(raw_probability, 6) if raw_probability is not None else None,
            "actionable_model": actionable,
            "research_only_reason": (
                None
                if actionable
                else f"FT_GOALS_CANONICAL_LADDER_{classification}"
            ),
            "canonical_market_ladder": True,
            "canonical_classification": classification,
            "canonical_tier": item.get("tier"),
            "canonical_prob_edge_pp": item.get("prob_edge_pp"),
            "canonical_estimated_ev": item.get("estimated_ev"),
            "canonical_discrepancy_recheck": discrepancy,
            "canonical_reasons": list(item.get("reasons") or []),
            "market_line_verified": True,
            "market_backed": True,
            "quotes": [
                {
                    "bookmaker": bookmaker,
                    "market": market,
                    "selection_text": f"{selection.title()} {line:g}",
                    "price": round(price, 4),
                    "provider_update": item.get("provider_update"),
                    "canonical_p_shrunk": round(probability, 6),
                    "canonical_p_raw": round(raw_probability, 6) if raw_probability is not None else None,
                    "canonical_classification": classification,
                    "canonical_prob_edge_pp": item.get("prob_edge_pp"),
                }
            ],
        }

        key = (selection, round(float(line), 4), bookmaker)
        score = (
            rank.get(classification, 0),
            float(item.get("prob_edge_pp") or -999.0),
            float(item.get("estimated_ev") or -999.0),
        )
        current = best.get(key)
        if current is None or score > tuple(current.get("_canonical_rank") or (0, -999.0, -999.0)):
            row["_canonical_rank"] = score
            best[key] = row

    out = list(best.values())
    out.sort(key=lambda row: tuple(row.get("_canonical_rank") or (0, -999.0, -999.0)), reverse=True)
    for row in out:
        row.pop("_canonical_rank", None)
    return out


def _market_backed_legs(event: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[int, int, float]]]:
    base_legs, matrix = v2._legs(event)
    backed: list[dict[str, Any]] = []

    # FT Goals must come from the canonical evaluated market ladder, not from a
    # second synthetic fixed-line ladder. This keeps Galaxy on the exact observed
    # bookmaker/line and the same p_shrunk used by the canonical FT Goals layer.
    backed.extend(_canonical_ft_goals_legs(event))

    # BTTS / Double Chance still use the existing score-matrix marginal model;
    # their production status remains research-only as before.
    for leg in base_legs:
        if str(leg.get("family") or "").upper() == "FT_GOALS":
            continue
        quotes = _verified_quotes(leg)
        if not quotes:
            continue
        row = dict(leg)
        row["quotes"] = quotes
        row["market_line_verified"] = True
        row["market_backed"] = True
        backed.append(row)
    return backed, matrix

def _common_book_reference(legs: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_leg: list[dict[str, dict[str, Any]]] = []
    common: set[str] | None = None
    for leg in legs:
        prices: dict[str, dict[str, Any]] = {}
        for quote in _verified_quotes(leg):
            bookmaker = quote["bookmaker"]
            current = prices.get(bookmaker)
            if current is None or float(quote["price"]) > float(current["price"]):
                prices[bookmaker] = quote
        if not prices:
            return None
        by_leg.append(prices)
        common = set(prices) if common is None else common & set(prices)
    if not common:
        return None

    def product(book: str) -> float:
        result = 1.0
        for prices in by_leg:
            result *= float(prices[book]["price"])
        return result

    book = max(common, key=product)
    decimal_reference = product(book)
    components = [prices[book] for prices in by_leg]
    return {
        "bookmaker": book,
        "component_product_decimal_reference": round(decimal_reference, 4),
        "component_product_american_reference": v2._american(decimal_reference),
        "components": components,
        "target_decimal": TARGET_DECIMAL,
        "target_american": 110,
        "target_component_reference_gate_passed": decimal_reference >= TARGET_DECIMAL,
        "warning": "REFERENCE_ONLY_NOT_FINAL_SGP_QUOTE; EXACT_CORRELATION_ADJUSTED_SGP_PRICE_REQUIRED",
    }


def _redundancy_diagnostics(matrix: list[tuple[int, int, float]], legs: list[dict[str, Any]], joint: float) -> tuple[bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    redundant = False
    for idx, leg in enumerate(legs):
        without = [x for j, x in enumerate(legs) if j != idx]
        p_without = v2._prob(matrix, without)
        incremental_drop = max(0.0, p_without - joint)
        is_redundant = incremental_drop < MIN_INCREMENTAL_PROBABILITY_DROP
        redundant = redundant or is_redundant
        diagnostics.append(
            {
                "family": leg.get("family"),
                "selection": leg.get("selection"),
                "line": leg.get("line"),
                "probability_without_leg": round(p_without, 6),
                "joint_probability": round(joint, 6),
                "incremental_probability_drop": round(incremental_drop, 6),
                "redundant": is_redundant,
            }
        )
    return redundant, diagnostics


def _minimum_decimal_for_edge(probability: float) -> float | None:
    max_market_probability = probability - TARGET_EDGE_PP / 100.0
    if max_market_probability <= 0:
        return None
    return 1.0 / max_market_probability


def _candidate_rank(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        1.0 if row.get("all_leg_models_actionable") else 0.0,
        float(row.get("reference_probability_edge_pp") or 0.0),
        float(row.get("joint_model_probability") or 0.0),
        float(((row.get("component_price_reference") or {}).get("component_product_decimal_reference")) or 0.0),
    )


def _sgp_market_backed(event: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    tier = event.get("tier") or coverage.get("data_tier")
    availability = _num(event.get("availability_confidence"))
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    if tier not in {"A", "B"} or availability is None or availability < MIN_AVAILABILITY:
        return []
    if provenance.get("source") not in _ALLOWED_MARKET_SOURCES or provenance.get("fresh") is not True:
        return []

    legs, matrix = _market_backed_legs(event)
    if len(legs) < 2 or not matrix:
        return []

    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    candidates: list[dict[str, Any]] = []
    for n in (2, 3):
        for combo_tuple in itertools.combinations(legs, n):
            combo = list(combo_tuple)
            if len({str(x.get("family")) for x in combo}) != n:
                continue

            reference = _common_book_reference(combo)
            if reference is None:
                continue
            reference_decimal = float(reference["component_product_decimal_reference"])
            # The builder's stated objective is +110 or better. If even the product
            # of verified component quotes from one bookmaker is below +110, the
            # combination is not a credible +110 candidate and must not be surfaced.
            if reference_decimal < TARGET_DECIMAL:
                continue

            joint = v2._prob(matrix, combo)
            if joint <= 0 or joint >= 1:
                continue
            redundant, redundancy = _redundancy_diagnostics(matrix, combo, joint)
            if redundant:
                continue

            reference_edge_pp = (joint - 1.0 / reference_decimal) * 100.0
            if reference_edge_pp < TARGET_EDGE_PP:
                continue

            all_actionable = all(bool(x.get("actionable_model")) for x in combo)
            blocks = ["EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED"]
            if availability < MIN_ACTIONABLE_AVAILABILITY:
                blocks.append("AVAILABILITY_BELOW_0.85")
            if not all_actionable:
                blocks.append("RESEARCH_ONLY_LEG_PRESENT")

            minimum_decimal = _minimum_decimal_for_edge(joint)
            candidates.append(
                {
                    "type": "SAME_GAME_PARLAY",
                    "candidate_source": "GALAXY_BUILDER_V0.4_MARKET_BACKED",
                    "fixture_id": fixture.get("fixture_id"),
                    "kickoff": fixture.get("kickoff"),
                    "country": fixture.get("country"),
                    "league": fixture.get("league"),
                    "match": f"{fixture.get('home_team') or fixture.get('home_name')} vs {fixture.get('away_team') or fixture.get('away_name')}",
                    "data_tier": tier,
                    "availability_confidence": round(float(availability), 3),
                    "market_source": provenance.get("source"),
                    "market_fresh": True,
                    "market_backed": True,
                    "all_lines_verified_in_observed_market": True,
                    "legs": [dict(x) for x in combo],
                    "joint_model_probability": round(joint, 6),
                    "fair_decimal": round(1.0 / joint, 4),
                    "fair_american": v2._american(1.0 / joint),
                    "minimum_sgp_decimal_for_target_edge": round(minimum_decimal, 4) if minimum_decimal else None,
                    "minimum_sgp_american_for_target_edge": v2._american(minimum_decimal) if minimum_decimal else None,
                    "reference_probability_edge_pp": round(reference_edge_pp, 2),
                    "reference_edge_warning": "COMPONENT_PRODUCT_IS_A_SCREENING_REFERENCE_ONLY; NOT_THE_SGP_PRICE",
                    "component_price_reference": reference,
                    "exact_parlay_quote": None,
                    "target": {"minimum_decimal": TARGET_DECIMAL, "minimum_american": 110},
                    "redundancy_check": {"passed": True, "legs": redundancy},
                    "status": "GALAXY MARKET-BACKED SGP WATCH — EXACT SGP QUOTE NEEDED",
                    "bet_eligible": False,
                    "all_leg_models_actionable": all_actionable,
                    "block_reasons": blocks,
                    "joint_probability_method": "DIRECT_SCORE_MATRIX_CORRELATION_AWARE",
                }
            )

    candidates.sort(key=_candidate_rank, reverse=True)
    # One primary thesis per fixture. Alternative DC/BTTS variants from the same
    # match are deliberately suppressed so the output is not a wall of near-dupes.
    return candidates[:1]


def _dedupe_multis(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        fixture_ids: list[int] = []
        valid = True
        for leg in row.get("legs") or []:
            if not isinstance(leg, dict):
                valid = False
                break
            try:
                fixture_ids.append(int(leg.get("fixture_id")))
            except (TypeError, ValueError):
                valid = False
                break
            verified_quotes = _verified_quotes(leg)
            direct_price = _num(leg.get("decimal_price"))
            direct_price_fresh = (
                direct_price is not None
                and direct_price > 1
                and qf.is_fresh(leg.get("provider_update"))
            )
            if not verified_quotes and not direct_price_fresh:
                valid = False
                break
        if not valid or len(set(fixture_ids)) != len(fixture_ids):
            continue
        key = tuple(sorted(fixture_ids))
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_rank = (
            1.0 if current.get("all_leg_models_actionable") else 0.0,
            float(current.get("probability_edge_vs_component_product_pp") or 0.0),
            float(current.get("conservative_joint_probability") or 0.0),
        )
        new_rank = (
            1.0 if row.get("all_leg_models_actionable") else 0.0,
            float(row.get("probability_edge_vs_component_product_pp") or 0.0),
            float(row.get("conservative_joint_probability") or 0.0),
        )
        if new_rank > current_rank:
            best[key] = row
    rows_out = list(best.values())
    rows_out.sort(
        key=lambda row: (
            1.0 if row.get("all_leg_models_actionable") else 0.0,
            float(row.get("probability_edge_vs_component_product_pp") or 0.0),
            float(row.get("conservative_joint_probability") or 0.0),
        ),
        reverse=True,
    )
    return rows_out[:MAX_MULTI]


def build(payload: dict[str, Any]) -> dict[str, Any]:
    base = v3.build(payload)
    events = [event for event in payload.get("events") or [] if isinstance(event, dict)]

    same_game: list[dict[str, Any]] = []
    for event in events:
        if event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "CLOSE"}:
            continue
        same_game.extend(_sgp_market_backed(event))
    same_game.sort(key=_candidate_rank, reverse=True)

    # _sgp_market_backed already returns at most one candidate per event; guard
    # again by fixture in case multiple event rows for a fixture occur in one tick.
    one_per_fixture: dict[Any, dict[str, Any]] = {}
    for row in same_game:
        fid = row.get("fixture_id")
        current = one_per_fixture.get(fid)
        if current is None or _candidate_rank(row) > _candidate_rank(current):
            one_per_fixture[fid] = row
    same_game = sorted(one_per_fixture.values(), key=_candidate_rank, reverse=True)[:MAX_SGP]

    multi_match = _dedupe_multis(
        [row for row in base.get("multi_match_candidates") or [] if isinstance(row, dict)]
    )

    policy = dict(base.get("family_policy") or {})
    policy["SGP_CONSTRUCTION"] = (
        "MARKET_BACKED_ONLY; FT_GOALS LEGS COME FROM THE CANONICAL MARKET_DECISION LADDER; "
        "EVERY LEG MUST EXIST IN VERIFIED CURRENT MARKET; SAME BOOK COMPONENT REFERENCE REQUIRED; "
        "COMPONENT PRODUCT MUST BE >= TARGET +110; LOGICALLY REDUNDANT LEGS BLOCKED; ONE PRIMARY SGP PER FIXTURE"
    )

    return {
        **base,
        "schema_version": SCHEMA_VERSION,
        "candidate_count": len(same_game) + len(multi_match),
        "same_game_candidate_count": len(same_game),
        "multi_match_candidate_count": len(multi_match),
        "same_game_candidates": same_game,
        "multi_match_candidates": multi_match,
        "family_policy": policy,
        "market_backed_sgp_policy": {
            "every_leg_requires_verified_observed_quote": True,
            "same_book_component_reference_required": True,
            "minimum_component_product_decimal_reference": TARGET_DECIMAL,
            "minimum_component_product_american_reference": 110,
            "component_product_is_exact_sgp_quote": False,
            "exact_sgp_quote_required_for_bet": True,
            "logical_redundancy_block": True,
            "minimum_incremental_probability_drop_per_leg": MIN_INCREMENTAL_PROBABILITY_DROP,
            "maximum_primary_sgp_candidates_per_fixture": 1,
            "synthetic_unquoted_lines_allowed": False,
            "quote_freshness_anchor": "PROVIDER_UPDATE",
            "quote_freshness_limit_minutes": qf.DEFAULT_MAX_AGE_MINUTES,
            "missing_or_stale_quote_allowed": False,
            "ft_goals_leg_source": "CANONICAL_MARKET_DECISION_LADDER",
            "ft_goals_probability_source": "P_SHRUNK_MATCHED_TO_THE_SAME_OBSERVED_BOOKMAKER_QUOTE",
            "ft_goals_score_matrix_role": "JOINT_SGP_CORRELATION_ONLY",
        },
        "policy": (
            "SPORT FIRST; MARKET SECOND; COMBINATION THIRD; MARKET-BACKED LEGS ONLY; FT_GOALS USE THE CANONICAL "
            "OBSERVED MARKET LADDER; TARGET +110 OR BETTER; NO SYNTHETIC UNQUOTED LINES; "
            "NO LOGICALLY REDUNDANT LEGS; ONE PRIMARY SGP PER FIXTURE; "
            "EXACT SPORTSBOOK SGP QUOTE REQUIRED FOR FINAL BET"
        ),
        "provider_requests_added": 0,
        "model_weights_changed": False,
        "canonical_bet_logic_changed": False,
    }
