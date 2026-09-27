from __future__ import annotations

import math
import re
import statistics
from typing import Any

SCHEMA_VERSION = "1.0.0"
MAX_TOTAL_GOALS = 20
MAX_CAPTURE_ROWS = 64


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def split_total_line(line: float) -> list[float]:
    value = float(line)
    if not math.isfinite(value) or value < 0.0 or value > 20.0:
        return []
    if abs(value * 4.0 - round(value * 4.0)) > 1e-8:
        return []
    if abs(value * 2.0 - round(value * 2.0)) < 1e-8:
        return [round(value, 2)]
    lower = math.floor(value * 2.0) / 2.0
    upper = math.ceil(value * 2.0) / 2.0
    return [round(lower, 2), round(upper, 2)]


def _component_fractions(total_goals: int, side: str, line: float) -> tuple[float, float, float]:
    side_u = str(side or "").upper()
    if side_u not in {"OVER", "UNDER"}:
        return 0.0, 0.0, 0.0
    total = float(total_goals)
    if math.isclose(total, float(line), abs_tol=1e-9):
        return 0.0, 1.0, 0.0
    won = total > float(line) if side_u == "OVER" else total < float(line)
    return (1.0, 0.0, 0.0) if won else (0.0, 0.0, 1.0)


def settlement_from_total_dist(
    total_dist: dict[int, float], side: str, line: float
) -> dict[str, float] | None:
    components = split_total_line(line)
    if not components or str(side or "").upper() not in {"OVER", "UNDER"}:
        return None
    win = push = loss = mass = 0.0
    for goals_raw, probability_raw in total_dist.items():
        try:
            goals = int(goals_raw)
            probability = float(probability_raw)
        except (TypeError, ValueError):
            continue
        if probability < 0.0 or not math.isfinite(probability):
            continue
        mass += probability
        for component in components:
            cw, cp, cl = _component_fractions(goals, side, component)
            weight = 1.0 / len(components)
            win += probability * cw * weight
            push += probability * cp * weight
            loss += probability * cl * weight
    if mass <= 0.0:
        return None
    return {
        "win_fraction": win / mass,
        "push_fraction": push / mass,
        "loss_fraction": loss / mass,
    }


def _poisson_total_dist(total_lambda: float) -> dict[int, float]:
    lam = float(total_lambda)
    if not math.isfinite(lam) or lam <= 0.0:
        return {}
    probs = {
        goals: math.exp(-lam) * (lam ** goals) / math.factorial(goals)
        for goals in range(MAX_TOTAL_GOALS + 1)
    }
    mass = sum(probs.values())
    if mass <= 0.0:
        return {}
    return {goals: probability / mass for goals, probability in probs.items()}


def settlement_distribution(total_lambda: float, side: str, line: float) -> dict[str, float] | None:
    return settlement_from_total_dist(_poisson_total_dist(total_lambda), side, line)


def fair_decimal_model(distribution: dict[str, float]) -> float | None:
    win = _num(distribution.get("win_fraction"))
    push = _num(distribution.get("push_fraction"))
    if win is None or push is None or win <= 0.0:
        return None
    price = (1.0 - push) / win
    return price if price >= 1.0 else 1.0


def _is_ft_total_market(name: Any) -> bool:
    return _norm(name) in {"goals over/under", "over/under", "goals over under"}


def _value_side_line(value: dict[str, Any]) -> tuple[str | None, float | None]:
    raw_selection = value.get("selection")
    if raw_selection is None:
        raw_selection = value.get("value")
    text = _norm(raw_selection)
    side = "OVER" if text.startswith("over") else "UNDER" if text.startswith("under") else None
    line = _num(value.get("line"))
    if line is None:
        line = _num(value.get("handicap"))
    if line is None:
        match = re.search(r"(?:over|under)\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
        if match:
            line = _num(match.group(1))
    if side is None or line is None or not split_total_line(line):
        return side, None
    return side, round(line, 2)


def _value_price(value: dict[str, Any]) -> float | None:
    for key in ("decimal_price", "odd", "price"):
        price = _num(value.get(key))
        if price is not None and 1.0 < price <= 1000.0:
            return price
    return None


def _raw_total_lambda(raw: dict[str, Any]) -> float | None:
    total = _num(raw.get("raw_total_goals"))
    if total is not None and total > 0.0:
        return total
    home = _num(raw.get("raw_home_goal_rate"))
    away = _num(raw.get("raw_away_goal_rate"))
    if home is None or away is None or home <= 0.0 or away <= 0.0:
        return None
    return home + away


def capture_event(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    market_payload = event.get("market") if isinstance(event.get("market"), dict) else {}
    total_lambda = _raw_total_lambda(raw)
    source = str(market_payload.get("source") or "NOT_VERIFIED")
    resolution_status = str(market_payload.get("resolution_status") or "")
    cache_replay = "CACHE" in source.upper() or "CACHE" in resolution_status.upper()
    fresh_provider = source.upper() == "API_FOOTBALL_ODDS_V3" and not cache_replay

    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for market in market_payload.get("markets") or []:
        if not isinstance(market, dict) or not _is_ft_total_market(market.get("market")):
            continue
        for value in market.get("values") or []:
            if not isinstance(value, dict):
                continue
            side, line = _value_side_line(value)
            price = _value_price(value)
            if side is None or line is None or price is None:
                continue
            grouped.setdefault((side, line), []).append(
                {
                    "side": side,
                    "line": line,
                    "price": price,
                    "bookmaker": market.get("bookmaker"),
                    "bookmaker_id": market.get("bookmaker_id"),
                    "market": market.get("market"),
                    "market_id": market.get("market_id"),
                    "provider_update": market.get("provider_update"),
                    "legacy_binary_fair_probability": _num(value.get("fair_probability")),
                }
            )

    rows: list[dict[str, Any]] = []
    if total_lambda is not None:
        for (side, line), offers in grouped.items():
            prices = [float(offer["price"]) for offer in offers]
            median_price = statistics.median(prices)
            chosen = min(
                offers,
                key=lambda offer: (
                    abs(float(offer["price"]) - median_price),
                    str(offer.get("bookmaker") or ""),
                ),
            )
            distribution = settlement_distribution(total_lambda, side, line)
            if distribution is None:
                continue
            fair = fair_decimal_model(distribution)
            expected_return = (
                distribution["win_fraction"] * float(chosen["price"])
                + distribution["push_fraction"]
            )
            rows.append(
                {
                    "selection": side,
                    "line": line,
                    "split_components": split_total_line(line),
                    "bookmaker": chosen.get("bookmaker"),
                    "bookmaker_id": chosen.get("bookmaker_id"),
                    "bookmaker_count": len(offers),
                    "market": chosen.get("market"),
                    "market_id": chosen.get("market_id"),
                    "decimal_price": round(float(chosen["price"]), 4),
                    "provider_update": chosen.get("provider_update"),
                    "market_source": source,
                    "market_resolution_status": resolution_status or None,
                    "market_fresh": fresh_provider,
                    "win_fraction_model": round(distribution["win_fraction"], 6),
                    "push_fraction_model": round(distribution["push_fraction"], 6),
                    "loss_fraction_model": round(distribution["loss_fraction"], 6),
                    "fair_decimal_model": round(fair, 4) if fair is not None else None,
                    "expected_return_model": round(expected_return, 6),
                    "raw_ev": round(expected_return - 1.0, 6),
                    "legacy_binary_fair_probability": chosen.get("legacy_binary_fair_probability"),
                    "market_no_vig_status": "NOT_USED_FOR_ASIAN_SETTLEMENT_RESEARCH",
                    "settlement_basis": "SPLIT_STAKE_INTEGER_HALF_QUARTER_TOTALS",
                    "research_only": True,
                    "actionable": False,
                    "decision_weight": 0.0,
                    "classification": "RESEARCH_ONLY",
                    "production_promotion_allowed": False,
                }
            )

    rows.sort(key=lambda row: (float(row["line"]), str(row["selection"])))
    rows = rows[:MAX_CAPTURE_ROWS]
    observed_lines = sorted({float(row["line"]) for row in rows})
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_SETTLEMENT_AWARE_CAPTURE" if rows else "NO_SETTLEMENT_AWARE_TOTALS_CAPTURED",
        "total_lambda": round(total_lambda, 4) if total_lambda is not None else None,
        "observed_rows": rows,
        "observed_row_count": len(rows),
        "observed_lines": observed_lines,
        "fresh_provider": fresh_provider,
        "cache_replay": cache_replay,
        "provider_requests_added": 0,
        "research_only": True,
        "actionable": False,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
        "policy": (
            "OBSERVE_EXACT_FT_TOTAL_LINES_FROM_ALREADY_RESOLVED_MARKET_PAYLOAD; "
            "INTEGER_HALF_QUARTER_SETTLEMENT_EXPLICIT; QUARTER_LINES_SPLIT_INTO_ADJACENT_HALF_LINES; "
            "NO_BINARY_PROBABILITY_SUBSTITUTION; NO_THRESHOLD_TIER_STAKE_OR_CANONICAL_DECISION_CHANGE"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, Any]:
    events_with_rows = 0
    rows_captured = 0
    fresh_provider_rows = 0
    cache_replay_rows = 0
    unique_lines: set[float] = set()
    fixture_ids: set[int] = set()

    for event in payload.get("events") or []:
        if (
            not isinstance(event, dict)
            or event.get("event_type") != "SOCCER_REFRESH"
            or str(event.get("stage") or "").upper() == "POSTGAME"
        ):
            continue
        capture = capture_event(event)
        event["ft_totals_settlement_capture"] = capture
        rows = capture.get("observed_rows") or []
        if rows:
            events_with_rows += 1
            rows_captured += len(rows)
            if capture.get("fresh_provider") is True:
                fresh_provider_rows += len(rows)
            if capture.get("cache_replay") is True:
                cache_replay_rows += len(rows)
            fixture_id = capture.get("fixture_id")
            if fixture_id is not None:
                try:
                    fixture_ids.add(int(fixture_id))
                except (TypeError, ValueError):
                    pass
            unique_lines.update(float(line) for line in capture.get("observed_lines") or [])

        ft_goals = event.get("ft_goals_intelligence")
        if isinstance(ft_goals, dict):
            ft_goals["observed_settlement_aware_total_markets"] = rows
            ft_goals["observed_settlement_aware_total_market_count"] = len(rows)
            ft_goals["observed_settlement_aware_lines"] = capture.get("observed_lines") or []
            ft_goals["settlement_aware_capture_policy"] = capture.get("policy")

        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict) and isinstance(areas.get("goals_full_match"), dict):
                goals = areas["goals_full_match"]
                goals["observed_settlement_aware_total_markets"] = rows
                goals["observed_settlement_aware_lines"] = capture.get("observed_lines") or []

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FT_TOTALS_SETTLEMENT_AWARE_CAPTURE_ACTIVE",
        "events_with_rows": events_with_rows,
        "unique_fixtures_with_rows": len(fixture_ids),
        "rows_captured": rows_captured,
        "fresh_provider_rows": fresh_provider_rows,
        "cache_replay_rows": cache_replay_rows,
        "observed_lines": sorted(unique_lines),
        "provider_requests_added": 0,
        "research_only": True,
        "decision_weight": 0.0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
    }
