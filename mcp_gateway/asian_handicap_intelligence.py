from __future__ import annotations

import math
import re
from typing import Any

SCHEMA_VERSION = "1.0.0"
MAX_GOALS = 12


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


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def _line_supported(line: float) -> bool:
    return abs(line * 4 - round(line * 4)) < 1e-8 and -5.0 <= line <= 5.0


def _split_line(line: float) -> list[float]:
    if not _line_supported(line):
        return []
    if abs(line * 2 - round(line * 2)) < 1e-8:
        return [round(line, 2)]
    low = math.floor(line * 2) / 2.0
    high = math.ceil(line * 2) / 2.0
    return [round(low, 2), round(high, 2)]


def _component_fractions(margin: int, line: float) -> tuple[float, float, float]:
    adjusted = margin + line
    if adjusted > 1e-9:
        return 1.0, 0.0, 0.0
    if adjusted < -1e-9:
        return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def _settlement_distribution(home_lam: float, away_lam: float, side: str, line: float) -> dict[str, float] | None:
    components = _split_line(line)
    if not components:
        return None
    win = push = loss = mass = 0.0
    for h in range(MAX_GOALS + 1):
        ph = _poisson(h, home_lam)
        for a in range(MAX_GOALS + 1):
            p = ph * _poisson(a, away_lam)
            mass += p
            margin = (h - a) if side == "HOME" else (a - h)
            cw = cp = cl = 0.0
            for component in components:
                w, pu, lo = _component_fractions(margin, component)
                cw += w / len(components)
                cp += pu / len(components)
                cl += lo / len(components)
            win += p * cw
            push += p * cp
            loss += p * cl
    if mass <= 0:
        return None
    return {
        "win_fraction": win / mass,
        "push_fraction": push / mass,
        "loss_fraction": loss / mass,
    }


def _fair_decimal(dist: dict[str, float]) -> float | None:
    win = dist["win_fraction"]
    push = dist["push_fraction"]
    if win <= 0:
        return None
    price = (1.0 - push) / win
    return price if price > 1.0 else 1.0


def _is_ah_market(name: Any) -> bool:
    n = _norm(name)
    return "asian handicap" in n or "asian hcap" in n


def _side_and_line(value: Any, fixture: dict[str, Any]) -> tuple[str | None, float | None]:
    text = _norm(value)
    home_name = _norm(fixture.get("home_team") or fixture.get("home_name"))
    away_name = _norm(fixture.get("away_team") or fixture.get("away_name"))
    side = None
    if text.startswith("home") or text.startswith("1 ") or (home_name and text.startswith(home_name)):
        side = "HOME"
    elif text.startswith("away") or text.startswith("2 ") or (away_name and text.startswith(away_name)):
        side = "AWAY"
    else:
        if home_name and home_name in text and not (away_name and away_name in text):
            side = "HOME"
        elif away_name and away_name in text and not (home_name and home_name in text):
            side = "AWAY"
    matches = re.findall(r"(?<!\d)([+-]?\d+(?:\.\d+)?)(?!\d)", text)
    line = None
    for token in reversed(matches):
        candidate = _num(token)
        if candidate is not None and -5.0 <= candidate <= 5.0:
            line = candidate
            break
    if side is None or line is None or not _line_supported(line):
        return side, None
    return side, round(line, 2)


def _observed_rows(event: dict[str, Any], home_lam: float, away_lam: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    fresh = provenance.get("fresh") is True
    source = provenance.get("source") or "NOT_VERIFIED"
    rows: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []

    for market_row in market.get("markets") or []:
        if not isinstance(market_row, dict) or not _is_ah_market(market_row.get("market")):
            continue
        for value in market_row.get("values") or []:
            if not isinstance(value, dict):
                continue
            side, line = _side_and_line(value.get("selection"), fixture)
            price = _decimal(value.get("price"))
            if side is None or line is None or price is None:
                unsupported.append({
                    "bookmaker": market_row.get("bookmaker"),
                    "market": market_row.get("market"),
                    "selection": value.get("selection"),
                    "price": value.get("price"),
                    "reason": "EXPLICIT_SIDE_LINE_AND_VALID_DECIMAL_PRICE_REQUIRED",
                })
                continue
            dist = _settlement_distribution(home_lam, away_lam, side, line)
            if dist is None:
                continue
            fair = _fair_decimal(dist)
            expected_return = dist["win_fraction"] * price + dist["push_fraction"]
            rows.append({
                "selection": side,
                "handicap": line,
                "split_components": _split_line(line),
                "win_fraction_model": round(dist["win_fraction"], 6),
                "push_fraction_model": round(dist["push_fraction"], 6),
                "loss_fraction_model": round(dist["loss_fraction"], 6),
                "fair_decimal_model": round(fair, 4) if fair is not None else None,
                "bookmaker": market_row.get("bookmaker"),
                "bookmaker_id": market_row.get("bookmaker_id"),
                "market": market_row.get("market"),
                "market_id": market_row.get("market_id"),
                "decimal_price": round(price, 4),
                "expected_return_model": round(expected_return, 6),
                "raw_ev": round(expected_return - 1.0, 6),
                "price_edge_decimal_vs_fair": round(price - fair, 4) if fair is not None else None,
                "market_no_vig_probability": None,
                "market_no_vig_status": "NOT_CALCULATED_FOR_PUSH_QUARTER_SETTLEMENT_WITHOUT_FULL_SETTLEMENT_AWARE_MARKET_TRANSFORM",
                "provider_update": market_row.get("provider_update"),
                "market_source": source,
                "market_fresh": fresh,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "promotion_block": "ASIAN_HANDICAP_NOT_OOS_CALIBRATED_AND_PARENT_GOAL_MODEL_LIMITED",
            })

    rows.sort(key=lambda r: (1.0 if r.get("market_fresh") else 0.0, float(r.get("raw_ev") or -999.0)), reverse=True)
    return rows[:32], unsupported[:32]


def build(event: dict[str, Any]) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    raw = event.get("raw_projection") if isinstance(event.get("raw_projection"), dict) else {}
    home_lam = _num(raw.get("raw_home_goal_rate"))
    away_lam = _num(raw.get("raw_away_goal_rate"))
    if home_lam is None or away_lam is None or home_lam <= 0 or away_lam <= 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "CANONICAL_HOME_AWAY_LAMBDAS_MISSING",
        }
    observed, unsupported = _observed_rows(event, home_lam, away_lam)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": "CANONICAL_SCORE_MARGIN_ASIAN_HANDICAP_v0.1",
        "home_lambda": round(home_lam, 4),
        "away_lambda": round(away_lam, 4),
        "settlement_support": "INTEGER_HALF_AND_QUARTER_LINES_AT_0.25_INCREMENT",
        "observed_market_rows": observed,
        "observed_market_count": len(observed),
        "unsupported_market_rows": unsupported,
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "RESEARCH_ONLY_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_bets_for_market_comparison": 250,
            "minimum_oos_bets_for_actionable_review": 500,
            "requires": [
                "settlement-unit return calibration by line bucket",
                "verified historical Asian Handicap prices and true CLV",
                "stable performance across competitions and favorite/underdog sides",
                "validated market shrinkage for push and quarter-line settlements",
                "parent score-margin distribution calibration materially adequate",
            ],
        },
        "policy": (
            "SPORT_FIRST SCORE-MARGIN DISTRIBUTION; INTEGER/HALF/QUARTER SETTLEMENT EXPLICIT; "
            "QUARTER LINES SPLIT INTO ADJACENT HALF-LINES; FAIR PRICE SOLVES EXPECTED RETURN=1; "
            "NO FAKE BINARY PROBABILITY OR NO-VIG NORMALIZATION; ZERO DECISION WEIGHT"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int]:
    modeled = observed_events = observed_rows = unsupported_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME":
            continue
        intelligence = build(event)
        event["asian_handicap_intelligence"] = intelligence
        if intelligence.get("status") == "LIVE_RESEARCH_MODELED":
            modeled += 1
        count = int(intelligence.get("observed_market_count") or 0)
        if count:
            observed_events += 1
            observed_rows += count
        unsupported_rows += len(intelligence.get("unsupported_market_rows") or [])
        match_intel = event.get("match_intelligence")
        if isinstance(match_intel, dict):
            areas = match_intel.get("areas")
            if isinstance(areas, dict):
                areas["asian_handicap"] = {
                    "status": intelligence.get("status"),
                    "home_lambda": intelligence.get("home_lambda"),
                    "away_lambda": intelligence.get("away_lambda"),
                    "observed_market_rows": intelligence.get("observed_market_rows") or [],
                    "actionable": False,
                    "decision_weight": 0.0,
                    "production_status": intelligence.get("production_status"),
                    "calibration_gate": intelligence.get("calibration_gate"),
                }
    return {
        "modeled_events": modeled,
        "events_with_observed_asian_handicap_markets": observed_events,
        "observed_asian_handicap_rows": observed_rows,
        "unsupported_asian_handicap_rows": unsupported_rows,
    }
