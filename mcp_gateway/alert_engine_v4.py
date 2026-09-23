from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_ALERT_ENGINE_V4_1.0.0"

SIGNAL_ORDER = {
    "INSUFFICIENT_DATA": 0,
    "NEUTRAL": 1,
    "WEAK": 2,
    "MODERATE": 3,
    "STRONG": 4,
    "VERY_STRONG": 5,
}

DECISION_ORDER = {
    "PASS": 0,
    "WATCH": 1,
    "LEAN": 2,
    "BET": 3,
}

CHANGE_TYPES = (
    "SIGNAL_UPGRADED",
    "PRICE_IMPROVED",
    "XI_CONFIRMED",
    "GK_CONFIRMED",
    "FRESH_QUOTE_APPEARED",
    "EDGE_CROSSED_THRESHOLD",
    "MARKET_BECAME_STALE",
    "MODEL_DISAGREEMENT_APPEARED",
    "PICK_DEMOTED",
)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _bool_nested(snapshot: dict[str, Any], key: str, nested: tuple[str, str] | None = None) -> bool:
    if bool(snapshot.get(key)):
        return True
    if nested:
        parent = snapshot.get(nested[0])
        if isinstance(parent, dict):
            return bool(parent.get(nested[1]))
    return False


def _is_stale(snapshot: dict[str, Any]) -> bool:
    status = str(snapshot.get("execution_status") or "").upper()
    blockers = {str(value or "").upper() for value in (snapshot.get("blockers") or [])}
    return status == "STALE_QUOTE" or any("STALE" in value for value in blockers)


def _has_disagreement(snapshot: dict[str, Any]) -> bool:
    status = str(snapshot.get("execution_status") or "").upper()
    signal = str(snapshot.get("model_signal") or "").upper()
    disagreement = str(snapshot.get("model_disagreement") or "").upper()
    return status == "MODEL_DISAGREEMENT" or signal == "MODEL_DISAGREEMENT" or disagreement == "HIGH"


def detect_changes(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    edge_threshold_pp: float | None = None,
) -> list[dict[str, Any]]:
    if not isinstance(previous, dict):
        return []

    changes: list[dict[str, Any]] = []

    previous_signal = str(previous.get("model_signal") or "INSUFFICIENT_DATA").upper()
    current_signal = str(current.get("model_signal") or "INSUFFICIENT_DATA").upper()
    if SIGNAL_ORDER.get(current_signal, 0) > SIGNAL_ORDER.get(previous_signal, 0):
        changes.append({
            "type": "SIGNAL_UPGRADED",
            "from": previous_signal,
            "to": current_signal,
        })

    previous_price = _num(previous.get("price"))
    current_price = _num(current.get("price"))
    if previous_price is not None and current_price is not None and current_price > previous_price:
        changes.append({
            "type": "PRICE_IMPROVED",
            "from": previous_price,
            "to": current_price,
        })

    if not _bool_nested(previous, "xi_confirmed", ("lineups", "both_xi_confirmed")) and _bool_nested(current, "xi_confirmed", ("lineups", "both_xi_confirmed")):
        changes.append({"type": "XI_CONFIRMED"})

    if not _bool_nested(previous, "gk_confirmed") and _bool_nested(current, "gk_confirmed"):
        changes.append({"type": "GK_CONFIRMED"})

    previous_quote_missing = _num(previous.get("price")) is None or _is_stale(previous)
    current_quote_fresh = _num(current.get("price")) is not None and not _is_stale(current)
    if previous_quote_missing and current_quote_fresh:
        changes.append({"type": "FRESH_QUOTE_APPEARED"})

    if edge_threshold_pp is not None:
        previous_edge = _num(previous.get("prob_edge_pp"))
        current_edge = _num(current.get("prob_edge_pp"))
        if current_edge is not None and current_edge >= edge_threshold_pp and (previous_edge is None or previous_edge < edge_threshold_pp):
            changes.append({
                "type": "EDGE_CROSSED_THRESHOLD",
                "threshold_pp": float(edge_threshold_pp),
                "from": previous_edge,
                "to": current_edge,
            })

    if not _is_stale(previous) and _is_stale(current):
        changes.append({"type": "MARKET_BECAME_STALE"})

    if not _has_disagreement(previous) and _has_disagreement(current):
        changes.append({"type": "MODEL_DISAGREEMENT_APPEARED"})

    previous_decision = str(previous.get("classification") or previous.get("final") or "PASS").upper()
    current_decision = str(current.get("classification") or current.get("final") or "PASS").upper()
    if DECISION_ORDER.get(current_decision, 0) < DECISION_ORDER.get(previous_decision, 0):
        changes.append({
            "type": "PICK_DEMOTED",
            "from": previous_decision,
            "to": current_decision,
        })

    return changes


def build_alert_payload(
    current: dict[str, Any],
    *,
    changes: list[dict[str, Any]],
    sport_probabilities: dict[str, Any] | None = None,
    model_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sport_probabilities = sport_probabilities or {}
    model_context = model_context or {}

    title = "🔥 STRONG SPORT SIGNAL"
    signal = str(current.get("model_signal") or "").upper()
    if signal not in {"STRONG", "VERY_STRONG"}:
        title = "⚽ SOCCER EDGE UPDATE"

    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "title": title,
        "fixture": {
            "fixture_id": current.get("fixture_id"),
            "home": current.get("home"),
            "away": current.get("away"),
            "league": current.get("league"),
            "kickoff": current.get("kickoff"),
        },
        "sport": {
            "p_home": sport_probabilities.get("p_home"),
            "p_draw": sport_probabilities.get("p_draw"),
            "p_away": sport_probabilities.get("p_away"),
            "p_over_2_5": sport_probabilities.get("p_over_2_5"),
            "p_btts": sport_probabilities.get("p_btts"),
            "home_tt_over_1_5": sport_probabilities.get("home_tt_over_1_5"),
            "corners_over_9_5": sport_probabilities.get("corners_over_9_5"),
        },
        "model": {
            "sport_confidence": current.get("model_signal_score") or model_context.get("sport_confidence"),
            "data_quality": current.get("data_tier") or model_context.get("data_quality"),
            "agreement": model_context.get("agreement"),
            "uncertainty": model_context.get("uncertainty"),
            "model_signal": current.get("model_signal"),
        },
        "best_market": {
            "market": current.get("market"),
            "selection": current.get("selection"),
            "line": current.get("line"),
            "book": current.get("bookmaker"),
            "price": current.get("price"),
            "fair": current.get("p_market_fair"),
            "edge_pp": current.get("prob_edge_pp"),
            "ev": current.get("estimated_ev"),
            "min_acceptable_price": current.get("min_acceptable_price"),
        },
        "execution": {
            "status": current.get("execution_status"),
            "blockers": current.get("blockers") or [],
        },
        "final": str(current.get("classification") or "WATCH").upper(),
        "changes": changes,
        "audit_only": True,
        "webhook_delivery_enabled": False,
        "production_action_enabled": False,
    }
    payload["text"] = format_alert_text(payload)
    return payload


def _fmt(value: Any) -> str:
    if value is None:
        return "N/V"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def format_alert_text(payload: dict[str, Any]) -> str:
    fixture = payload["fixture"]
    sport = payload["sport"]
    model = payload["model"]
    market = payload["best_market"]
    execution = payload["execution"]
    blockers = ", ".join(str(value) for value in execution.get("blockers") or []) or "None"

    return "\n".join([
        str(payload["title"]),
        "",
        f"Fixture: {_fmt(fixture.get('home'))} vs {_fmt(fixture.get('away'))}",
        f"League: {_fmt(fixture.get('league'))}",
        f"Kickoff: {_fmt(fixture.get('kickoff'))}",
        "",
        "SPORT",
        f"P(Home): {_fmt(sport.get('p_home'))}",
        f"P(Draw): {_fmt(sport.get('p_draw'))}",
        f"P(Away): {_fmt(sport.get('p_away'))}",
        f"P(O2.5): {_fmt(sport.get('p_over_2_5'))}",
        f"P(BTTS): {_fmt(sport.get('p_btts'))}",
        f"Home TT O1.5: {_fmt(sport.get('home_tt_over_1_5'))}",
        f"Corners O9.5: {_fmt(sport.get('corners_over_9_5'))}",
        "",
        "MODEL",
        f"Sport confidence: {_fmt(model.get('sport_confidence'))}",
        f"Data quality: {_fmt(model.get('data_quality'))}",
        f"Agreement: {_fmt(model.get('agreement'))}",
        f"Uncertainty: {_fmt(model.get('uncertainty'))}",
        "",
        "BEST MARKET",
        f"Market: {_fmt(market.get('market'))}",
        f"Selection: {_fmt(market.get('selection'))}",
        f"Line: {_fmt(market.get('line'))}",
        f"Book: {_fmt(market.get('book'))}",
        f"Price: {_fmt(market.get('price'))}",
        f"Fair: {_fmt(market.get('fair'))}",
        f"Edge: {_fmt(market.get('edge_pp'))}",
        f"EV: {_fmt(market.get('ev'))}",
        f"Min acceptable price: {_fmt(market.get('min_acceptable_price'))}",
        "",
        "EXECUTION",
        f"Status: {_fmt(execution.get('status'))}",
        f"Blockers: {blockers}",
        "",
        "FINAL",
        str(payload.get("final") or "WATCH"),
    ])
