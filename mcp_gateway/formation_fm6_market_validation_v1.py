from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any

from mcp_gateway import soccer_model

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_FM6_EXACT_MARKET_VALIDATION_V1.0.0"
FM5_MODEL_VERSION = "FORMATION_FM5_RAW_SPORT_PROJECTION_V1.0.0"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _fair_probabilities(prices: dict[str, Any]) -> dict[str, float] | None:
    implied: dict[str, float] = {}
    for key, value in prices.items():
        price = _num(value)
        if price is None or price <= 1.0:
            return None
        implied[str(key).lower()] = 1.0 / price
    total = sum(implied.values())
    if total <= 0:
        return None
    return {key: value / total for key, value in implied.items()}


def _candidate_probability(
    candidate: dict[str, Any],
    *,
    family: str,
    selection: str,
    line: float | None,
) -> float | None:
    family = family.upper()
    selection = selection.upper()

    if family == "1X2":
        key = {
            "HOME": "raw_home_win_prob",
            "DRAW": "raw_draw_prob",
            "AWAY": "raw_away_win_prob",
        }.get(selection)
        return _num(candidate.get(key)) if key else None

    if family == "BTTS":
        yes = _num(candidate.get("raw_btts_yes_prob"))
        if yes is None:
            return None
        if selection == "YES":
            return yes
        if selection == "NO":
            return 1.0 - yes
        return None

    if family == "FT_TOTALS":
        if line is None or selection not in {"OVER", "UNDER"}:
            return None
        total_dist = candidate.get("_total_dist")
        if not isinstance(total_dist, dict):
            return None
        over = soccer_model._over_prob(total_dist, float(line))
        if over is None:
            return None
        return over if selection == "OVER" else 1.0 - over

    # Future formation components may expose exact sport-first probabilities
    # for derivative families. FM6 can consume them only if they were already
    # present in the FM5 sporting candidate before this market snapshot.
    rows = candidate.get("exact_sporting_probabilities")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_family = str(row.get("market_family") or "").upper()
            row_selection = str(row.get("selection") or "").upper()
            row_line = _num(row.get("line"))
            line_matches = (
                line is None and row_line is None
                or line is not None
                and row_line is not None
                and abs(float(line) - row_line) < 1e-9
            )
            if row_family == family and row_selection == selection and line_matches:
                return _num(row.get("probability"))
    return None


def build_exact_market_research_signal(
    fm5_report: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    blockers: list[str] = []
    report = fm5_report if isinstance(fm5_report, dict) else {}
    market = market_snapshot if isinstance(market_snapshot, dict) else {}

    if report.get("model_version") != FM5_MODEL_VERSION:
        blockers.append("FM5_RAW_PROJECTION_CONTRACT_REQUIRED")
    if report.get("status") != "FM5_RESEARCH_CANDIDATE_AVAILABLE":
        blockers.append("FM5_RESEARCH_CANDIDATE_NOT_AVAILABLE")
    if report.get("research_candidate_available") is not True:
        blockers.append("FM5_RESEARCH_CANDIDATE_FLAG_FALSE")
    if report.get("canonical_raw_projection_changed") is not False:
        blockers.append("FM5_CANONICAL_PROJECTION_MUTATION_DETECTED")
    if report.get("market_fields_consumed") is not False:
        blockers.append("FM5_MARKET_LEAKAGE_DETECTED")

    candidate = report.get("fm5_raw_projection_research")
    if not isinstance(candidate, dict):
        blockers.append("FM5_CANDIDATE_PAYLOAD_REQUIRED")
        candidate = {}

    captured_at = _dt(market.get("captured_at"))
    feature_as_of = _dt(candidate.get("feature_as_of"))
    kickoff = _dt(candidate.get("fixture_kickoff"))

    if captured_at is None:
        blockers.append("MARKET_CAPTURE_TIMESTAMP_REQUIRED")
    if not market.get("source"):
        blockers.append("MARKET_SOURCE_REQUIRED")
    if not market.get("bookmaker"):
        blockers.append("BOOKMAKER_REQUIRED")
    if feature_as_of is None:
        blockers.append("FM5_FEATURE_TIMESTAMP_REQUIRED")
    if kickoff is None:
        blockers.append("FIXTURE_KICKOFF_REQUIRED")
    if (
        captured_at is not None
        and feature_as_of is not None
        and captured_at < feature_as_of
    ):
        blockers.append("MARKET_SNAPSHOT_PRECEDES_RAW_SPORT_PROJECTION")
    if captured_at is not None and kickoff is not None and captured_at >= kickoff:
        blockers.append("MARKET_SNAPSHOT_NOT_STRICTLY_PREKICKOFF")

    family = str(market.get("market_family") or "").upper()
    selection = str(market.get("selection") or "").upper()
    line = _num(market.get("line"))
    decimal_price = _num(market.get("decimal_price"))
    if not family:
        blockers.append("MARKET_FAMILY_REQUIRED")
    if not selection:
        blockers.append("SELECTION_REQUIRED")
    if decimal_price is None or decimal_price <= 1.0:
        blockers.append("VALID_DECIMAL_PRICE_REQUIRED")

    raw_probability = _candidate_probability(
        candidate,
        family=family,
        selection=selection,
        line=line,
    )
    if raw_probability is None or raw_probability < 0.0 or raw_probability > 1.0:
        blockers.append("EXACT_RAW_SPORT_PROBABILITY_NOT_AVAILABLE")

    fair_prices = market.get("comparison_prices")
    fair = _fair_probabilities(fair_prices) if isinstance(fair_prices, dict) else None
    fair_key = selection.lower()
    market_fair_probability = fair.get(fair_key) if fair and fair_key in fair else None

    if blockers:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "status": "FM6_BLOCKED",
            "blockers": sorted(set(blockers)),
            "signal": None,
            "research_only": True,
            "decision_weight": 0.0,
            "bet_eligible": False,
            "production_enabled": False,
            "market_can_create_sporting_thesis": False,
            "production_promotion_allowed": False,
        }

    assert decimal_price is not None
    assert raw_probability is not None
    breakeven = 1.0 / decimal_price
    raw_edge_to_breakeven_pp = (raw_probability - breakeven) * 100.0
    fair_edge_pp = (
        (raw_probability - market_fair_probability) * 100.0
        if market_fair_probability is not None
        else None
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "status": "FM6_EXACT_MARKET_RESEARCH_SIGNAL",
        "blockers": [],
        "signal": {
            "source_fm5_model_version": report.get("model_version"),
            "source_component": candidate.get("source_component"),
            "source_artifact_id": candidate.get("source_artifact_id"),
            "raw_projection_feature_as_of": candidate.get("feature_as_of"),
            "fixture_kickoff": candidate.get("fixture_kickoff"),
            "market_captured_at": market.get("captured_at"),
            "market_source": market.get("source"),
            "bookmaker": market.get("bookmaker"),
            "market_family": family,
            "market": market.get("market"),
            "selection": selection,
            "line": line,
            "decimal_price": round(decimal_price, 6),
            "p_breakeven": round(breakeven, 8),
            "p_raw_sport": round(raw_probability, 8),
            "p_market_fair": (
                round(market_fair_probability, 8)
                if market_fair_probability is not None
                else None
            ),
            "raw_edge_to_breakeven_pp": round(raw_edge_to_breakeven_pp, 6),
            "raw_edge_to_market_fair_pp": (
                round(fair_edge_pp, 6) if fair_edge_pp is not None else None
            ),
            "comparison_prices": copy.deepcopy(fair_prices)
            if isinstance(fair_prices, dict)
            else None,
            "strict_exact_instrument": True,
            "classification": "WATCH",
            "bet_eligible": False,
            "decision_weight": 0.0,
        },
        "research_only": True,
        "decision_weight": 0.0,
        "bet_eligible": False,
        "production_enabled": False,
        "market_can_create_sporting_thesis": False,
        "market_shrinkage_applied": False,
        "estimated_ev_used_for_decision": False,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
    }
