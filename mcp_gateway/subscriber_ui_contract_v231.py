from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.1.0"
MODEL_VERSION = "SOCCER_SUBSCRIBER_UI_CONTRACT_V231"


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _probability(value: Any) -> float | None:
    number = _number(value)
    if number is None:
        return None
    if abs(number) > 1.0:
        number /= 100.0
    return number


def _team_name(row: dict[str, Any], side: str) -> str | None:
    direct = _first(row, f"{side}_team", f"{side}_team_name", f"{side}_name")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    value = row.get(side)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        name = value.get("name") or value.get("team_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    teams = row.get("teams")
    if isinstance(teams, dict):
        value = teams.get(side)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            name = value.get("name") or value.get("team_name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def adapt_market_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one persisted/runtime row into the stable subscriber UI contract.

    Presentation only: this never changes canonical model probabilities, thresholds,
    execution status, strict-close rules, budget, or BET logic.
    """
    model_probability = _probability(
        _first(
            row,
            "p_model_calibrated",
            "p_shrunk",
            "p_raw",
            "model_probability",
            "probability",
        )
    )
    market_probability = _probability(
        _first(
            row,
            "p_market_fair",
            "p_market_devig",
            "market_probability",
            "implied_probability",
        )
    )
    persisted_edge = _number(_first(row, "edge_pp", "prob_edge_pp", "edge", "edge_pct", "model_edge"))
    if persisted_edge is not None and abs(persisted_edge) <= 1.0:
        persisted_edge *= 100.0
    display_edge = persisted_edge
    edge_source = "persisted"
    if display_edge is None and model_probability is not None and market_probability is not None:
        display_edge = (model_probability - market_probability) * 100.0
        edge_source = "presentation_derived"

    home = _team_name(row, "home")
    away = _team_name(row, "away")
    fixture_id = _first(row, "fixture_id", "id")
    label = " vs ".join([name for name in (home, away) if name])
    if not label:
        label = _first(row, "fixture_name", "match_name") or (f"Fixture {fixture_id}" if fixture_id is not None else "Fixture")

    return {
        "schema_version": SCHEMA_VERSION,
        "match": {
            "fixture_id": fixture_id,
            "home": home,
            "away": away,
            "label": label,
            "league": _first(row, "league_name", "league"),
            "country": _first(row, "country"),
            "kickoff": _first(row, "kickoff", "fixture_date", "date"),
        },
        "market": {
            "family": _first(row, "market_family", "family", "market"),
            "selection": _first(row, "selection", "pick", "market_name", "market"),
            "period": _first(row, "period"),
            "line": _first(row, "line", "handicap", "total_line"),
        },
        "model": {
            "probability": model_probability,
            "confidence": _number(_first(row, "confidence", "confidence_score", "model_confidence")),
            "version": _first(row, "model_version"),
        },
        "pricing": {
            "market_probability": market_probability,
            "edge_pp": display_edge,
            "edge_source": edge_source if display_edge is not None else None,
            "price": _number(_first(row, "price", "odds", "decimal_odds", "entry_price")),
            "fair_price": _number(_first(row, "fair_price")),
            "bookmaker": _first(row, "bookmaker", "provider"),
        },
        "state": {
            "status": _first(row, "execution_status", "status", "stage", "classification") or "WATCH",
            "reason": _first(row, "reason", "blocker"),
            "stage": _first(row, "stage"),
            "data_quality": _first(row, "data_quality", "data_tier", "quality"),
            "lineup": _first(row, "lineup_status", "xi_status", "lineup"),
            "provider_update": _first(row, "provider_update", "provider_updated_at"),
            "generated_at": _first(row, "generated_at_utc", "generated_at", "signal_generated_at"),
        },
        "raw": row,
    }


def adapt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [adapt_market_row(row) for row in rows if isinstance(row, dict)]
