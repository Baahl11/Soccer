from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any

from mcp_gateway import galaxy_builder_v2 as v2
from mcp_gateway import galaxy_builder_v4 as v4
from mcp_gateway import quote_freshness as qf

SCHEMA_VERSION = "0.6.0"
ACTIONABLE_STAGES = {"T-40", "T-20", "T-10"}
OBSERVATION_STAGES = {"T-90", "T-60", "T-30"}
MAX_EVENT_RESEARCH_CANDIDATES = 3
MAX_GLOBAL_RESEARCH_CANDIDATES = 12
MAX_SETTLEMENT_DIAGNOSTICS_PER_EVENT = 8
MIN_INCREMENTAL_PROBABILITY_DROP = v4.MIN_INCREMENTAL_PROBABILITY_DROP
TARGET_DECIMAL = v4.TARGET_DECIMAL
TARGET_EDGE_PP = v4.TARGET_EDGE_PP


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _matrix(event: dict[str, Any]) -> list[tuple[int, int, float]]:
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    home = _num(raw.get("raw_home_goal_rate"))
    away = _num(raw.get("raw_away_goal_rate"))
    if home is None or away is None or home <= 0 or away <= 0:
        return []
    return v2._matrix(home, away)


def _pred(leg: dict[str, Any], home: int, away: int) -> bool:
    family = str(leg.get("family") or "")
    selection = str(leg.get("selection") or "").upper()
    line = _num(leg.get("line"))

    if family in {"FT_GOALS", "BTTS", "DOUBLE_CHANCE"}:
        return v2._pred(leg, home, away)

    if family == "ONE_X_TWO":
        if selection == "HOME":
            return home > away
        if selection == "DRAW":
            return home == away
        if selection == "AWAY":
            return away > home
        return False

    if family == "TEAM_TOTAL":
        role = str(leg.get("team_role") or "").upper()
        goals = home if role == "HOME" else away if role == "AWAY" else None
        if goals is None or line is None:
            return False
        if selection == "OVER":
            return goals > line
        if selection == "UNDER":
            return goals < line
        return False

    if family == "CORRECT_SCORE":
        target_home = leg.get("home_goals")
        target_away = leg.get("away_goals")
        try:
            return home == int(target_home) and away == int(target_away)
        except (TypeError, ValueError):
            return False

    if family == "ASIAN_HANDICAP_HALF":
        if line is None:
            return False
        margin = (home - away) if selection == "HOME" else (away - home)
        return margin + line > 1e-9

    return False


def _joint_probability(
    matrix: list[tuple[int, int, float]],
    legs: list[dict[str, Any]],
) -> float:
    return max(
        0.0,
        min(
            1.0,
            sum(
                probability
                for home, away, probability in matrix
                if all(_pred(leg, home, away) for leg in legs)
            ),
        ),
    )


def _quote_from_row(row: dict[str, Any], selection_text: str) -> dict[str, Any] | None:
    price = _num(row.get("decimal_price"))
    bookmaker = str(row.get("bookmaker") or "").strip()
    market = str(row.get("market") or "").strip()
    provider_update = row.get("provider_update")
    if (
        price is None
        or price <= 1
        or not bookmaker
        or not market
        or not qf.is_fresh(provider_update)
    ):
        return None
    age = qf.age_minutes(provider_update)
    return {
        "bookmaker": bookmaker,
        "market": market,
        "selection_text": selection_text,
        "price": round(price, 4),
        "provider_update": provider_update,
        "quote_age_minutes": round(age, 2) if age is not None else None,
        "quote_fresh": True,
        "quote_freshness_anchor": "PROVIDER_UPDATE",
    }


def _research_leg(
    family: str,
    selection: str,
    probability: float,
    quote: dict[str, Any],
    *,
    line: float | None = None,
    extra: dict[str, Any] | None = None,
    reason: str,
) -> dict[str, Any]:
    row = {
        "family": family,
        "selection": selection,
        "line": line,
        "probability": round(probability, 6),
        "probability_source": "DIRECT_SCORE_MATRIX_SCORE_DERIVED_RESEARCH",
        "actionable_model": False,
        "research_only_reason": reason,
        "score_matrix_expansion": True,
        "market_line_verified": True,
        "market_backed": True,
        "quotes": [quote],
    }
    if extra:
        row.update(extra)
    return row


def _binary_extra_legs(
    event: dict[str, Any],
    matrix: list[tuple[int, int, float]],
) -> list[dict[str, Any]]:
    legs: list[dict[str, Any]] = []

    one_x_two = (
        event.get("one_x_two_intelligence")
        if isinstance(event.get("one_x_two_intelligence"), dict)
        else {}
    )
    for row in one_x_two.get("observed_market_rows") or []:
        if not isinstance(row, dict):
            continue
        selection = str(row.get("selection") or "").upper()
        if selection not in {"HOME", "DRAW", "AWAY"}:
            continue
        quote = _quote_from_row(row, selection)
        if quote is None:
            continue
        leg = {"family": "ONE_X_TWO", "selection": selection, "line": None}
        probability = _joint_probability(matrix, [leg])
        if probability <= 0:
            continue
        legs.append(
            _research_leg(
                "ONE_X_TWO",
                selection,
                probability,
                quote,
                reason="1X2_RESEARCH_ONLY_MODEL_SELECTION_PENDING",
            )
        )

    team_totals = (
        event.get("team_totals_intelligence")
        if isinstance(event.get("team_totals_intelligence"), dict)
        else {}
    )
    for row in team_totals.get("observed_exact_market_rows") or []:
        if not isinstance(row, dict):
            continue
        selection = str(row.get("selection") or "").upper()
        role = str(row.get("team_role") or "").upper()
        line = _num(row.get("line"))
        if selection not in {"OVER", "UNDER"} or role not in {"HOME", "AWAY"} or line is None:
            continue
        quote = _quote_from_row(row, f"{role} {selection.title()} {line:g}")
        if quote is None:
            continue
        leg = {
            "family": "TEAM_TOTAL",
            "selection": selection,
            "line": line,
            "team_role": role,
        }
        probability = _joint_probability(matrix, [leg])
        if probability <= 0:
            continue
        legs.append(
            _research_leg(
                "TEAM_TOTAL",
                selection,
                probability,
                quote,
                line=line,
                extra={
                    "team_role": role,
                    "team_id": row.get("team_id"),
                    "team": row.get("team"),
                },
                reason="TEAM_TOTALS_RESEARCH_ONLY_NOT_OOS_CALIBRATED",
            )
        )

    correct_score = (
        event.get("correct_score_intelligence")
        if isinstance(event.get("correct_score_intelligence"), dict)
        else {}
    )
    for row in correct_score.get("observed_exact_market_rows") or []:
        if not isinstance(row, dict):
            continue
        try:
            home_goals = int(row.get("home_goals"))
            away_goals = int(row.get("away_goals"))
        except (TypeError, ValueError):
            continue
        score = f"{home_goals}-{away_goals}"
        quote = _quote_from_row(row, score)
        if quote is None:
            continue
        leg = {
            "family": "CORRECT_SCORE",
            "selection": score,
            "line": None,
            "home_goals": home_goals,
            "away_goals": away_goals,
        }
        probability = _joint_probability(matrix, [leg])
        if probability <= 0:
            continue
        legs.append(
            _research_leg(
                "CORRECT_SCORE",
                score,
                probability,
                quote,
                extra={"home_goals": home_goals, "away_goals": away_goals},
                reason="CORRECT_SCORE_RESEARCH_ONLY_NOT_OOS_CALIBRATED",
            )
        )

    asian = (
        event.get("asian_handicap_intelligence")
        if isinstance(event.get("asian_handicap_intelligence"), dict)
        else {}
    )
    for row in asian.get("observed_market_rows") or []:
        if not isinstance(row, dict):
            continue
        selection = str(row.get("selection") or "").upper()
        line = _num(row.get("handicap"))
        if selection not in {"HOME", "AWAY"} or line is None:
            continue
        # Only true half-goal AH lines are binary. Integer and quarter lines
        # remain in the settlement-aware diagnostic path below.
        is_half_goal = (
            abs(line * 2.0 - round(line * 2.0)) < 1e-8
            and abs(line - round(line)) > 1e-8
        )
        if not is_half_goal:
            continue
        quote = _quote_from_row(row, f"{selection} {line:+g}")
        if quote is None:
            continue
        leg = {
            "family": "ASIAN_HANDICAP_HALF",
            "selection": selection,
            "line": line,
        }
        probability = _joint_probability(matrix, [leg])
        if probability <= 0:
            continue
        legs.append(
            _research_leg(
                "ASIAN_HANDICAP_HALF",
                selection,
                probability,
                quote,
                line=line,
                reason="ASIAN_HANDICAP_RESEARCH_ONLY; HALF_GOAL_BINARY_SCORE_MATRIX_PATH",
            )
        )

    # Bound the combinatorial pool while preserving the best observed/modelled
    # representatives from each new score-derived family.
    grouped: dict[str, list[dict[str, Any]]] = {}
    for leg in legs:
        key = str(leg.get("family") or "")
        if key == "TEAM_TOTAL":
            key = f"TEAM_TOTAL_{leg.get('team_role')}"
        grouped.setdefault(key, []).append(leg)

    bounded: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda leg: float(leg.get("probability") or 0.0), reverse=True)
        bounded.extend(rows[:4])
    return bounded


def _correlation_key(leg: dict[str, Any]) -> str:
    family = str(leg.get("family") or "")
    if family == "TEAM_TOTAL":
        return f"TEAM_TOTAL_{leg.get('team_role')}"
    return family


def _redundancy_diagnostics(
    matrix: list[tuple[int, int, float]],
    legs: list[dict[str, Any]],
    joint: float,
) -> tuple[bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    redundant = False
    for index, leg in enumerate(legs):
        without = [item for j, item in enumerate(legs) if j != index]
        p_without = _joint_probability(matrix, without)
        incremental_drop = max(0.0, p_without - joint)
        is_redundant = incremental_drop < MIN_INCREMENTAL_PROBABILITY_DROP
        redundant = redundant or is_redundant
        diagnostics.append(
            {
                "family": leg.get("family"),
                "selection": leg.get("selection"),
                "line": leg.get("line"),
                "team_role": leg.get("team_role"),
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


def _expanded_candidates(
    event: dict[str, Any],
    matrix: list[tuple[int, int, float]],
    extra_legs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_legs, _ = v4._market_backed_legs(event)
    pool = [*base_legs, *extra_legs]
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    coverage = event.get("coverage") if isinstance(event.get("coverage"), dict) else {}
    tier = event.get("tier") or coverage.get("data_tier")
    availability = _num(event.get("availability_confidence"))

    if tier not in {"A", "B"} or availability is None or availability < v4.MIN_AVAILABILITY:
        return []

    candidates: list[dict[str, Any]] = []
    for count in (2, 3):
        for combo_tuple in itertools.combinations(pool, count):
            combo = list(combo_tuple)
            if not any(bool(leg.get("score_matrix_expansion")) for leg in combo):
                continue
            keys = [_correlation_key(leg) for leg in combo]
            if len(set(keys)) != len(keys):
                continue

            reference = v4._common_book_reference(combo)
            if reference is None:
                continue
            reference_decimal = float(reference.get("component_product_decimal_reference") or 0.0)
            if reference_decimal < TARGET_DECIMAL:
                continue

            joint = _joint_probability(matrix, combo)
            if joint <= 0 or joint >= 1:
                continue

            redundant, redundancy = _redundancy_diagnostics(matrix, combo, joint)
            if redundant:
                continue

            reference_edge_pp = (joint - 1.0 / reference_decimal) * 100.0
            if reference_edge_pp < TARGET_EDGE_PP:
                continue

            minimum_decimal = _minimum_decimal_for_edge(joint)
            candidates.append(
                {
                    "type": "SAME_GAME_PARLAY_RESEARCH",
                    "candidate_source": "GALAXY_SCORE_MATRIX_EXPANSION_V0.6",
                    "fixture_id": fixture.get("fixture_id"),
                    "kickoff": fixture.get("kickoff"),
                    "country": fixture.get("country"),
                    "league": fixture.get("league"),
                    "match": f"{fixture.get('home_team') or fixture.get('home_name')} vs {fixture.get('away_team') or fixture.get('away_name')}",
                    "data_tier": tier,
                    "availability_confidence": round(float(availability), 3),
                    "market_source": (
                        (event.get("market_provenance") or {}).get("source")
                        if isinstance(event.get("market_provenance"), dict)
                        else None
                    ),
                    "market_fresh": True,
                    "legs": [dict(leg) for leg in combo],
                    "families": [_correlation_key(leg) for leg in combo],
                    "joint_model_probability": round(joint, 6),
                    "fair_decimal": round(1.0 / joint, 4),
                    "fair_american": v2._american(1.0 / joint),
                    "minimum_sgp_decimal_for_target_edge": (
                        round(minimum_decimal, 4) if minimum_decimal else None
                    ),
                    "minimum_sgp_american_for_target_edge": (
                        v2._american(minimum_decimal) if minimum_decimal else None
                    ),
                    "reference_probability_edge_pp": round(reference_edge_pp, 2),
                    "component_price_reference": reference,
                    "component_product_is_exact_sgp_quote": False,
                    "exact_parlay_quote": None,
                    "redundancy_check": {"passed": True, "legs": redundancy},
                    "status": "GALAXY SCORE-MATRIX EXPANSION — RESEARCH ONLY / EXACT SGP QUOTE NEEDED",
                    "bet_eligible": False,
                    "all_leg_models_actionable": False,
                    "block_reasons": [
                        "SCORE_MATRIX_EXPANSION_RESEARCH_ONLY",
                        "AT_LEAST_ONE_UNPROMOTED_FAMILY_PRESENT",
                        "EXACT_CORRELATION_ADJUSTED_SGP_QUOTE_NOT_EXPOSED",
                    ],
                    "joint_probability_method": "DIRECT_SCORE_MATRIX_INTERSECTION_NO_MARGINAL_MULTIPLICATION",
                }
            )

    candidates.sort(
        key=lambda row: (
            float(row.get("reference_probability_edge_pp") or 0.0),
            float(row.get("joint_model_probability") or 0.0),
        ),
        reverse=True,
    )
    return candidates[:MAX_EVENT_RESEARCH_CANDIDATES]


def _split_ah(line: float) -> list[float]:
    if abs(line * 4.0 - round(line * 4.0)) > 1e-8:
        return []
    if abs(line * 2.0 - round(line * 2.0)) < 1e-8:
        return [round(line, 2)]
    low = math.floor(line * 2.0) / 2.0
    high = math.ceil(line * 2.0) / 2.0
    return [round(low, 2), round(high, 2)]


def _settlement_fraction(
    family: str,
    selection: str,
    line: float | None,
    home: int,
    away: int,
) -> tuple[float, float, float]:
    if family == "DNB":
        if home == away:
            return 0.0, 1.0, 0.0
        win = home > away if selection == "HOME" else away > home
        return (1.0, 0.0, 0.0) if win else (0.0, 0.0, 1.0)

    if family == "ASIAN_HANDICAP_SETTLEMENT" and line is not None:
        components = _split_ah(line)
        if not components:
            return 0.0, 0.0, 1.0
        margin = (home - away) if selection == "HOME" else (away - home)
        win = push = loss = 0.0
        for component in components:
            adjusted = margin + component
            if adjusted > 1e-9:
                win += 1.0 / len(components)
            elif adjusted < -1e-9:
                loss += 1.0 / len(components)
            else:
                push += 1.0 / len(components)
        return win, push, loss

    return 0.0, 0.0, 1.0


def _settlement_legs(event: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    dnb = event.get("dnb_intelligence") if isinstance(event.get("dnb_intelligence"), dict) else {}
    for row in dnb.get("observed_market_rows") or []:
        if not isinstance(row, dict):
            continue
        selection = str(row.get("selection") or "").upper()
        if selection not in {"HOME", "AWAY"}:
            continue
        quote = _quote_from_row(row, f"{selection} DNB")
        if quote is None:
            continue
        out.append(
            {
                "family": "DNB",
                "selection": selection,
                "line": 0.0,
                "quotes": [quote],
                "score_matrix_expansion": True,
                "settlement_aware": True,
                "binary_sgp_candidate_eligible": False,
                "block_reason": "DRAW_PUSH_REQUIRES_SETTLEMENT_AWARE_EXACT_SGP_QUOTE_RULES",
            }
        )

    asian = event.get("asian_handicap_intelligence") if isinstance(event.get("asian_handicap_intelligence"), dict) else {}
    for row in asian.get("observed_market_rows") or []:
        if not isinstance(row, dict):
            continue
        selection = str(row.get("selection") or "").upper()
        line = _num(row.get("handicap"))
        if selection not in {"HOME", "AWAY"} or line is None:
            continue
        is_binary_half = (
            abs(line * 2.0 - round(line * 2.0)) < 1e-8
            and abs(line - round(line)) > 1e-8
        )
        if is_binary_half:
            continue
        quote = _quote_from_row(row, f"{selection} {line:+g}")
        if quote is None:
            continue
        out.append(
            {
                "family": "ASIAN_HANDICAP_SETTLEMENT",
                "selection": selection,
                "line": line,
                "quotes": [quote],
                "score_matrix_expansion": True,
                "settlement_aware": True,
                "binary_sgp_candidate_eligible": False,
                "split_components": _split_ah(line),
                "block_reason": "INTEGER_OR_QUARTER_SETTLEMENT_REQUIRES_EXACT_SGP_SETTLEMENT_RULES",
            }
        )
    return out


def _settlement_diagnostics(
    matrix: list[tuple[int, int, float]],
    settlement_legs: list[dict[str, Any]],
    binary_legs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    binary_ranked = sorted(
        binary_legs,
        key=lambda leg: float(leg.get("probability") or 0.0),
        reverse=True,
    )[:6]

    for settlement_leg in settlement_legs:
        for binary_leg in binary_ranked:
            if _correlation_key(binary_leg) in {"DNB", "ASIAN_HANDICAP_SETTLEMENT"}:
                continue
            reference = v4._common_book_reference([settlement_leg, binary_leg])
            if reference is None:
                continue

            full_win = push_fraction = settlement_loss = binary_failure = 0.0
            for home, away, probability in matrix:
                if not _pred(binary_leg, home, away):
                    binary_failure += probability
                    continue
                win, push, loss = _settlement_fraction(
                    str(settlement_leg.get("family") or ""),
                    str(settlement_leg.get("selection") or "").upper(),
                    _num(settlement_leg.get("line")),
                    home,
                    away,
                )
                full_win += probability * win
                push_fraction += probability * push
                settlement_loss += probability * loss

            diagnostics.append(
                {
                    "families": [
                        _correlation_key(binary_leg),
                        settlement_leg.get("family"),
                    ],
                    "binary_leg": {
                        "family": binary_leg.get("family"),
                        "selection": binary_leg.get("selection"),
                        "line": binary_leg.get("line"),
                        "team_role": binary_leg.get("team_role"),
                    },
                    "settlement_leg": {
                        "family": settlement_leg.get("family"),
                        "selection": settlement_leg.get("selection"),
                        "line": settlement_leg.get("line"),
                        "split_components": settlement_leg.get("split_components"),
                    },
                    "joint_binary_success_settlement_win_fraction": round(full_win, 6),
                    "joint_binary_success_settlement_push_fraction": round(push_fraction, 6),
                    "joint_binary_success_settlement_loss_fraction": round(settlement_loss, 6),
                    "binary_leg_failure_probability": round(binary_failure, 6),
                    "same_book_component_reference": reference,
                    "method": "DIRECT_SCORE_MATRIX_SETTLEMENT_INTEGRATION",
                    "bet_eligible": False,
                    "status": "RESEARCH_ONLY_SETTLEMENT_DIAGNOSTIC",
                    "block_reason": settlement_leg.get("block_reason"),
                }
            )

    diagnostics.sort(
        key=lambda row: float(row.get("joint_binary_success_settlement_win_fraction") or 0.0),
        reverse=True,
    )
    return diagnostics[:MAX_SETTLEMENT_DIAGNOSTICS_PER_EVENT]


def attach(payload: dict[str, Any]) -> dict[str, Any]:
    all_candidates: list[dict[str, Any]] = []
    all_settlement: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    events_modeled = 0
    gate_counts: Counter[str] = Counter()

    for event in payload.get("events") or []:
        if not isinstance(event, dict):
            gate_counts["NOT_DICT"] += 1
            continue
        if event.get("event_type") != "SOCCER_REFRESH":
            gate_counts["NOT_SOCCER_REFRESH"] += 1
            continue
        stage = str(event.get("stage") or "")
        gate_counts[f"STAGE_{stage or 'MISSING'}"] += 1
        if stage in OBSERVATION_STAGES:
            gate_counts["PRE_ACTIONABLE_OBSERVED"] += 1
            provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
            if provenance.get("fresh") is True:
                gate_counts["PRE_ACTIONABLE_MARKET_FRESH"] += 1
            else:
                gate_counts["PRE_ACTIONABLE_MARKET_NOT_FRESH"] += 1
            raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
            if _num(raw.get("raw_home_goal_rate")) is not None and _num(raw.get("raw_away_goal_rate")) is not None:
                gate_counts["PRE_ACTIONABLE_GOAL_RATES_PRESENT"] += 1
            else:
                gate_counts["PRE_ACTIONABLE_GOAL_RATES_MISSING"] += 1
            continue
        if stage not in ACTIONABLE_STAGES:
            continue
        gate_counts["ACTIONABLE_STAGE"] += 1
        provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
        if provenance.get("fresh") is not True:
            gate_counts["MARKET_NOT_FRESH"] += 1
            source = str(provenance.get("source") or "UNKNOWN")
            gate_counts[f"MARKET_NOT_FRESH_SOURCE_{source}"] += 1
            if provenance.get("latest_market_timestamp") is None:
                gate_counts["MARKET_TIMESTAMP_MISSING"] += 1
            else:
                age = _num(provenance.get("age_minutes"))
                if age is not None and age > qf.DEFAULT_MAX_AGE_MINUTES:
                    gate_counts["MARKET_TIMESTAMP_TOO_OLD"] += 1
                elif age is not None and age < -qf.MAX_FUTURE_SKEW_MINUTES:
                    gate_counts["MARKET_TIMESTAMP_FUTURE_SKEW"] += 1
                else:
                    gate_counts["MARKET_TIMESTAMP_OTHER_FRESHNESS_FAILURE"] += 1
            continue
        gate_counts["MARKET_FRESH"] += 1

        matrix = _matrix(event)
        if not matrix:
            raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
            if _num(raw.get("raw_home_goal_rate")) is None or _num(raw.get("raw_away_goal_rate")) is None:
                gate_counts["MISSING_RAW_GOAL_RATES"] += 1
            else:
                gate_counts["INVALID_RAW_GOAL_RATES"] += 1
            continue
        gate_counts["MATRIX_READY"] += 1

        base_legs, _ = v4._market_backed_legs(event)
        extra_legs = _binary_extra_legs(event, matrix)
        settlement_legs = _settlement_legs(event)
        for leg in extra_legs:
            family_counts[_correlation_key(leg)] += 1
        for leg in settlement_legs:
            family_counts[str(leg.get("family") or "")] += 1

        candidates = _expanded_candidates(event, matrix, extra_legs)
        settlement = _settlement_diagnostics(
            matrix,
            settlement_legs,
            [*base_legs, *extra_legs],
        )
        if extra_legs or settlement_legs:
            events_modeled += 1
            gate_counts["EXPANDED_LEGS_PRESENT"] += 1
        else:
            gate_counts["NO_FRESH_EXPANSION_MARKETS"] += 1

        event["galaxy_score_matrix_expansion"] = {
            "schema_version": SCHEMA_VERSION,
            "status": "LIVE_RESEARCH_SCORE_MATRIX_EXPANDED",
            "binary_joint_families": [
                "ONE_X_TWO",
                "TEAM_TOTAL_HOME",
                "TEAM_TOTAL_AWAY",
                "CORRECT_SCORE",
                "ASIAN_HANDICAP_HALF",
            ],
            "settlement_joint_families": [
                "DNB",
                "ASIAN_HANDICAP_INTEGER_OR_QUARTER",
            ],
            "new_binary_leg_count": len(extra_legs),
            "settlement_leg_count": len(settlement_legs),
            "research_candidate_count": len(candidates),
            "settlement_diagnostic_count": len(settlement),
            "research_candidates": candidates,
            "settlement_diagnostics": settlement,
            "same_game_marginal_multiplication_allowed": False,
            "joint_method": "DIRECT_SCORE_MATRIX_INTERSECTION",
            "settlement_method": "DIRECT_SCORE_MATRIX_SETTLEMENT_INTEGRATION",
            "production_promotion_allowed": False,
        }
        all_candidates.extend(candidates)
        all_settlement.extend(settlement)

    all_candidates.sort(
        key=lambda row: (
            float(row.get("reference_probability_edge_pp") or 0.0),
            float(row.get("joint_model_probability") or 0.0),
        ),
        reverse=True,
    )
    all_candidates = all_candidates[:MAX_GLOBAL_RESEARCH_CANDIDATES]

    builder = payload.get("galaxy_builder") if isinstance(payload.get("galaxy_builder"), dict) else {}
    builder = dict(builder)
    builder["score_matrix_expansion"] = {
        "schema_version": SCHEMA_VERSION,
        "status": "LIVE_RESEARCH",
        "events_modeled": events_modeled,
        "gate_counts": dict(gate_counts),
        "family_leg_counts": dict(family_counts),
        "research_candidate_count": len(all_candidates),
        "research_candidates": all_candidates,
        "settlement_diagnostic_count": len(all_settlement),
        "same_game_marginal_multiplication_allowed": False,
        "binary_joint_method": "DIRECT_SCORE_MATRIX_INTERSECTION",
        "settlement_joint_method": "DIRECT_SCORE_MATRIX_SETTLEMENT_INTEGRATION",
        "production_promotion_allowed": False,
        "candidate_merge_into_primary_galaxy_feed": False,
        "policy": (
            "SCORE-DERIVED FAMILIES EXPAND IN RESEARCH LANE ONLY; EXACT OBSERVED FRESH QUOTES REQUIRED; "
            "1X2/TEAM_TOTAL/CORRECT_SCORE/AH_HALF USE DIRECT SCORE-MATRIX INTERSECTION; "
            "DNB AND INTEGER/QUARTER AH USE SETTLEMENT-AWARE MATRIX DIAGNOSTICS AND DO NOT PRETEND TO BE BINARY; "
            "NO MARGINAL MULTIPLICATION; NO PRODUCTION PROMOTION"
        ),
    }
    payload["galaxy_builder"] = builder

    return {
        "events_modeled": events_modeled,
        "gate_counts": dict(gate_counts),
        "family_leg_counts": dict(family_counts),
        "research_candidate_count": len(all_candidates),
        "settlement_diagnostic_count": len(all_settlement),
    }
