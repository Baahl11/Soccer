from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "SOCCER_MARKET_MISMATCH_V4_1.3.0"
EVIDENCE_REGIME = "PHASE16_DISCRIMINATION_GATED_V2"

SPORT_WEIGHT = 0.30
EDGE_WEIGHT = 0.30
DATA_WEIGHT = 0.15
PRICE_WEIGHT = 0.15
UNCERTAINTY_WEIGHT = 0.10

DATA_TIER_SCORE = {
    "A": 1.00,
    "B": 0.80,
    "C": 0.60,
    "D": 0.40,
    "N/V": 0.20,
    "NV": 0.20,
}

CORRELATION_GROUPS = {
    "1X2": "MATCH_RESULT",
    "BTTS": "MATCH_GOALS",
    "FT_TOTALS": "MATCH_GOALS",
    "HOME_TT": "MATCH_GOALS",
    "AWAY_TT": "MATCH_GOALS",
    "1H": "PERIOD_GOALS",
    "2H": "PERIOD_GOALS",
    "FT_CORNERS": "CORNERS",
    "TEAM_CORNERS": "CORNERS",
    "CARDS": "DISCIPLINE",
    "PLAYER_CARDS": "DISCIPLINE",
    "SHOTS": "PLAYER_ATTACK",
    "SOT": "PLAYER_ATTACK",
    "GOALSCORER": "PLAYER_ATTACK",
    "ASSISTS": "PLAYER_ATTACK",
    "GK_SAVES": "PLAYER_GK",
}


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).upper()


def canonical_market_family(row: dict[str, Any]) -> str | None:
    family = _norm(row.get("market_family"))
    market = _norm(row.get("market"))
    selection = _norm(row.get("selection"))
    combined = f"{family} {market} {selection}"

    if "SPORTING_SCREEN" in family or "RESEARCH SCREEN" in combined:
        return None
    if family in {"1X2", "FT_1X2", "MATCH_WINNER"} or "MATCH WINNER" in market:
        return "1X2"
    if "BOTH TEAMS" in market or "BTTS" in combined:
        return "BTTS"
    if any(token in market for token in ("FIRST HALF", "1ST HALF", "1H")):
        return "1H"
    if any(token in market for token in ("SECOND HALF", "2ND HALF", "2H")):
        return "2H"
    if "CORNER" in combined:
        if "TEAM" in combined or "HOME" in market or "AWAY" in market:
            return "TEAM_CORNERS"
        return "FT_CORNERS"
    if "CARD" in combined or "BOOKING" in combined:
        if "PLAYER" in combined:
            return "PLAYER_CARDS"
        return "CARDS"
    if "GOALKEEPER SAVES" in combined or "GK SAVES" in combined:
        return "GK_SAVES"
    if "SHOTS ON TARGET" in combined or "SOT" in family:
        return "SOT"
    if "SHOT" in combined and "PLAYER" in combined:
        return "SHOTS"
    if "GOALSCORER" in combined or "ANYTIME SCORER" in combined:
        return "GOALSCORER"
    if "ASSIST" in combined:
        return "ASSISTS"
    if "TEAM TOTAL" in combined or "TEAM GOALS" in combined:
        if "AWAY" in combined:
            return "AWAY_TT"
        if "HOME" in combined:
            return "HOME_TT"
        return "HOME_TT"
    if family in {"TOTAL", "FT_TOTALS", "FT_TOTALS_RESEARCH"} or "GOALS OVER/UNDER" in market:
        return "FT_TOTALS"
    return None


def _calibrated_probability(row: dict[str, Any]) -> tuple[float | None, str | None]:
    for key in (
        "p_model_calibrated",
        "p_calibrated",
        "calibrated_probability",
        "model_probability_calibrated",
    ):
        value = _num(row.get(key))
        if value is not None and 0.0 <= value <= 1.0:
            return value, key
    return None, None


def _uncertainty(row: dict[str, Any]) -> tuple[float, str]:
    for key in ("uncertainty", "model_uncertainty", "uncertainty_score"):
        value = _num(row.get(key))
        if value is None:
            continue
        if value > 1.0 and value <= 100.0:
            value /= 100.0
        return min(max(value, 0.0), 1.0), key
    disagreement = _norm(row.get("model_disagreement"))
    mapped = {"LOW": 0.20, "MODERATE": 0.50, "HIGH": 0.80}
    if disagreement in mapped:
        return mapped[disagreement], "model_disagreement"
    return 0.50, "MISSING_NEUTRAL_RESEARCH_ONLY"


def _price_quality(row: dict[str, Any]) -> tuple[float, list[str]]:
    blockers = {_norm(value) for value in (row.get("blockers") or [])}
    price = _num(row.get("price"))
    p_market = _num(row.get("p_market_fair"))
    reasons: list[str] = []
    if price is None or p_market is None:
        return 0.0, ["PRICE_OR_FAIR_PROBABILITY_MISSING"]
    if any("STALE" in blocker for blocker in blockers):
        return 0.0, ["STALE_QUOTE"]
    if any("OUTLIER" in blocker for blocker in blockers):
        reasons.append("PRICE_OUTLIER_FLAGGED")
        return 0.35, reasons
    return 1.0, reasons


def analyze_row(row: dict[str, Any]) -> dict[str, Any] | None:
    family = canonical_market_family(row)
    fixture_id = row.get("fixture_id")
    if family is None or fixture_id is None:
        return None

    sport_score_raw = _num(row.get("model_signal_score"))
    sport_score = min(max((sport_score_raw or 0.0) / 100.0, 0.0), 1.0)
    data_score = DATA_TIER_SCORE.get(_norm(row.get("data_tier")), 0.20)
    price_score, price_reasons = _price_quality(row)
    uncertainty, uncertainty_source = _uncertainty(row)
    uncertainty_quality = 1.0 - uncertainty

    calibrated_probability, calibrated_source = _calibrated_probability(row)
    p_market = _num(row.get("p_market_fair"))
    calibrated_edge_pp = None
    if calibrated_probability is not None and p_market is not None and 0.0 <= p_market <= 1.0:
        calibrated_edge_pp = (calibrated_probability - p_market) * 100.0

    raw_edge_pp = _num(row.get("prob_edge_pp"))
    calibration_status = str(row.get("phase16_calibration_status") or "").strip().upper()
    calibration_missing_reason = (
        calibration_status
        if calibration_status and calibration_status != "RESEARCH_CALIBRATION_APPLIED"
        else "CALIBRATED_MODEL_PROBABILITY_MISSING"
    )

    blockers: list[str] = []
    if calibrated_probability is None:
        blockers.append(calibration_missing_reason)
    if p_market is None:
        blockers.append("MARKET_FAIR_PROBABILITY_MISSING")
    if price_score <= 0:
        blockers.extend(price_reasons)
    if sport_score_raw is None:
        blockers.append("SPORT_CONFIDENCE_MISSING")

    rankability_reasons: list[str] = []
    if calibrated_probability is None:
        rankability_reasons.append(calibration_missing_reason)
    elif p_market is None:
        rankability_reasons.append("MARKET_FAIR_PROBABILITY_MISSING")
    elif calibrated_edge_pp is None:
        rankability_reasons.append("CALIBRATED_EDGE_UNAVAILABLE")
    elif calibrated_edge_pp <= 0:
        rankability_reasons.append("CALIBRATED_EDGE_NOT_POSITIVE")
    if price_score <= 0:
        rankability_reasons.extend(price_reasons or ["PRICE_NOT_RANKABLE"])
    if sport_score_raw is None:
        rankability_reasons.append("SPORT_CONFIDENCE_MISSING")

    rankable = not rankability_reasons
    edge_component = min(max((calibrated_edge_pp or 0.0) / 15.0, 0.0), 1.0)
    research_score = 100.0 * (
        SPORT_WEIGHT * sport_score
        + EDGE_WEIGHT * edge_component
        + DATA_WEIGHT * data_score
        + PRICE_WEIGHT * price_score
        + UNCERTAINTY_WEIGHT * uncertainty_quality
    )

    raw_diagnostic_score = None
    if raw_edge_pp is not None:
        raw_component = min(max(raw_edge_pp / 15.0, 0.0), 1.0)
        raw_diagnostic_score = 100.0 * (
            SPORT_WEIGHT * sport_score
            + EDGE_WEIGHT * raw_component
            + DATA_WEIGHT * data_score
            + PRICE_WEIGHT * price_score
            + UNCERTAINTY_WEIGHT * uncertainty_quality
        )

    return {
        "fixture_id": fixture_id,
        "league": row.get("league"),
        "home": row.get("home"),
        "away": row.get("away"),
        "stage": row.get("stage"),
        "market_family": family,
        "correlation_group": CORRELATION_GROUPS.get(family, family),
        "market": row.get("market"),
        "selection": row.get("selection"),
        "line": row.get("line"),
        "price": row.get("price"),
        "bookmaker": row.get("bookmaker"),
        "sport_confidence_score": round(sport_score * 100.0, 3),
        "calibrated_probability": calibrated_probability,
        "calibrated_probability_source": calibrated_source,
        "phase16_calibration_source": row.get("phase16_calibration_source"),
        "phase16_calibration_policy": row.get("phase16_calibration_policy"),
        "phase16_calibration_status": row.get("phase16_calibration_status"),
        "phase16_1x2_class_discrimination_ready": row.get("phase16_1x2_class_discrimination_ready"),
        "phase16_1x2_family_discrimination_ready": row.get("phase16_1x2_family_discrimination_ready"),
        "phase16_1x2_not_ready_classes": list(row.get("phase16_1x2_not_ready_classes") or []),
        "promotion_shadow_eligible": row.get("phase16_calibration_promotion_shadow_eligible") is True,
        "market_fair_probability": p_market,
        "calibrated_edge_pp": round(calibrated_edge_pp, 4) if calibrated_edge_pp is not None else None,
        "raw_legacy_edge_pp": raw_edge_pp,
        "data_quality_score": round(data_score * 100.0, 3),
        "price_quality_score": round(price_score * 100.0, 3),
        "uncertainty": round(uncertainty, 6),
        "uncertainty_source": uncertainty_source,
        "rankable": rankable,
        "rankability_reasons": sorted(set(rankability_reasons)),
        "evidence_regime": EVIDENCE_REGIME,
        "mismatch_score": round(research_score, 4) if rankable else None,
        "raw_diagnostic_score": round(raw_diagnostic_score, 4) if raw_diagnostic_score is not None else None,
        "blockers": sorted(set(blockers)),
        "production_promotion_allowed": False,
        "bet_eligible": False,
    }


def find_mismatches(rows: Iterable[dict[str, Any]], *, top_n: int = 20) -> dict[str, Any]:
    analyzed = [candidate for row in rows if isinstance(row, dict) for candidate in [analyze_row(row)] if candidate is not None]
    by_fixture: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for candidate in analyzed:
        by_fixture[candidate["fixture_id"]].append(candidate)

    primary: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for fixture_id, fixture_rows in by_fixture.items():
        rankable = [row for row in fixture_rows if row["rankable"]]
        rankable.sort(key=lambda row: row["mismatch_score"] or -1.0, reverse=True)
        used_groups: set[str] = set()
        for row in rankable:
            group = row["correlation_group"]
            if group in used_groups:
                suppressed.append({
                    **row,
                    "suppression_reason": "CORRELATED_MARKET_GROUP_ALREADY_REPRESENTED",
                })
                continue
            used_groups.add(group)
            primary.append(row)

    primary.sort(key=lambda row: row["mismatch_score"] or -1.0, reverse=True)
    coverage_counts: dict[str, int] = defaultdict(int)
    rankable_counts: dict[str, int] = defaultdict(int)
    non_rankable_reason_counts: dict[str, int] = defaultdict(int)
    non_rankable_reason_counts_by_family: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in analyzed:
        coverage_counts[row["market_family"]] += 1
        if row["rankable"]:
            rankable_counts[row["market_family"]] += 1
            continue
        reasons = row.get("rankability_reasons") or ["UNSPECIFIED_NON_RANKABLE"]
        for reason in reasons:
            non_rankable_reason_counts[str(reason)] += 1
            non_rankable_reason_counts_by_family[row["market_family"]][str(reason)] += 1

    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "evidence_regime": EVIDENCE_REGIME,
        "status": "RESEARCH_MISMATCH_SCAN",
        "rows_analyzed": len(analyzed),
        "fixtures_analyzed": len(by_fixture),
        "rankable_rows": sum(1 for row in analyzed if row["rankable"]),
        "non_rankable_rows": sum(1 for row in analyzed if not row["rankable"]),
        "non_rankable_reason_counts": dict(sorted(non_rankable_reason_counts.items())),
        "non_rankable_reason_counts_by_family": {
            family: dict(sorted(counts.items()))
            for family, counts in sorted(non_rankable_reason_counts_by_family.items())
        },
        "primary_candidates": primary[: max(int(top_n), 0)],
        "correlated_candidates_suppressed": suppressed[: max(int(top_n), 0)],
        "market_family_coverage": dict(sorted(coverage_counts.items())),
        "rankable_market_family_coverage": dict(sorted(rankable_counts.items())),
        "ranking_weights": {
            "sport_confidence": SPORT_WEIGHT,
            "calibrated_edge": EDGE_WEIGHT,
            "data_quality": DATA_WEIGHT,
            "price_quality": PRICE_WEIGHT,
            "uncertainty_quality": UNCERTAINTY_WEIGHT,
        },
        "calibrated_probability_required_for_ranking": True,
        "raw_legacy_edge_used_for_diagnostic_only": True,
        "correlation_suppression_enabled": True,
        "provider_requests_added": 0,
        "production_promotion_allowed": False,
        "canonical_bet_logic_changed": False,
        "model_weights_changed": False,
    }
