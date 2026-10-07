from __future__ import annotations

import copy
import math
from datetime import datetime
from typing import Any

from mcp_gateway import soccer_model

SCHEMA_VERSION = "1.0.0"
MODEL_VERSION = "FORMATION_FM5_RAW_SPORT_PROJECTION_V1.0.0"
READINESS_MODEL_VERSION = "FORMATION_FM5_READINESS_GATE_V1.0.0"

_FORBIDDEN_INPUT_KEY_TOKENS = (
    "market",
    "odds",
    "price",
    "bookmaker",
    "breakeven",
    "clv",
    "vig",
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _forbidden_input_paths(value: Any, *, prefix: str = "") -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in key_text for token in _FORBIDDEN_INPUT_KEY_TOKENS):
                hits.append(path)
            hits.extend(_forbidden_input_paths(child, prefix=path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_forbidden_input_paths(child, prefix=f"{prefix}[{index}]"))
    return hits


def _base_envelope(
    baseline_raw_projection: dict[str, Any],
    readiness: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "baseline_projection_model_version": baseline_raw_projection.get("model_version"),
        "readiness_model_version": readiness.get("model_version"),
        "sport_first": True,
        "market_fields_consumed": False,
        "odds_consumed": False,
        "decision_weight": 0.0,
        "production_enabled": False,
        "automatic_production_activation_allowed": False,
        "market_evaluation_enabled": False,
        "canonical_raw_projection_changed": False,
        "historical_prediction_rewrite_allowed": False,
        "baseline_raw_projection": copy.deepcopy(baseline_raw_projection),
    }


def _blocked(
    baseline_raw_projection: dict[str, Any],
    readiness: dict[str, Any],
    blockers: list[str],
) -> dict[str, Any]:
    out = _base_envelope(baseline_raw_projection, readiness)
    out.update(
        {
            "status": "FM5_BLOCKED",
            "blockers": sorted(set(str(value) for value in blockers if value)),
            "ready_component": None,
            "sporting_adjustment": None,
            "fm5_raw_projection_research": None,
            "research_candidate_available": False,
            "production_promotion_allowed": False,
        }
    )
    return out


def _adjustment_contract_errors(
    adjustment: dict[str, Any],
    readiness: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    source_component = str(adjustment.get("source_component") or "")
    ready_components = {
        str(value) for value in (readiness.get("ready_components") or []) if value
    }

    if not source_component:
        errors.append("SOURCE_COMPONENT_REQUIRED")
    elif source_component not in ready_components:
        errors.append("SOURCE_COMPONENT_NOT_READINESS_APPROVED")

    if adjustment.get("prior_only") is not True:
        errors.append("PRIOR_ONLY_REQUIRED")
    if adjustment.get("oos_gate_passed") is not True:
        errors.append("OOS_GATE_PASS_REQUIRED")
    if adjustment.get("market_fields_used") is not False:
        errors.append("MARKET_FIELDS_USED_MUST_BE_FALSE")
    if adjustment.get("manual_research_approval") is not True:
        errors.append("MANUAL_RESEARCH_APPROVAL_REQUIRED")

    forbidden_paths = _forbidden_input_paths(
        {
            key: value
            for key, value in adjustment.items()
            if key not in {
                "market_fields_used",
                "market_independent",
                "no_market_input",
            }
        }
    )
    if forbidden_paths:
        errors.append(
            "FORBIDDEN_MARKET_FIELD_KEYS:"
            + ",".join(sorted(forbidden_paths)[:10])
        )

    home = _num(adjustment.get("adjusted_home_goal_rate"))
    away = _num(adjustment.get("adjusted_away_goal_rate"))
    if home is None or home <= 0:
        errors.append("VALID_ADJUSTED_HOME_GOAL_RATE_REQUIRED")
    if away is None or away <= 0:
        errors.append("VALID_ADJUSTED_AWAY_GOAL_RATE_REQUIRED")

    feature_as_of = _dt(adjustment.get("feature_as_of"))
    kickoff = _dt(adjustment.get("fixture_kickoff"))
    if feature_as_of is None:
        errors.append("FEATURE_AS_OF_REQUIRED")
    if kickoff is None:
        errors.append("FIXTURE_KICKOFF_REQUIRED")
    if feature_as_of is not None and kickoff is not None and feature_as_of >= kickoff:
        errors.append("FEATURE_TIMESTAMP_NOT_STRICTLY_BEFORE_KICKOFF")

    if not adjustment.get("source_model_version"):
        errors.append("SOURCE_MODEL_VERSION_REQUIRED")
    if not adjustment.get("source_artifact_id"):
        errors.append("SOURCE_ARTIFACT_ID_REQUIRED")

    return sorted(set(errors))


def build_research_candidate(
    baseline_raw_projection: dict[str, Any],
    readiness: dict[str, Any],
    sporting_adjustment: dict[str, Any] | None,
    *,
    availability_confidence: float | None = None,
) -> dict[str, Any]:
    baseline = _dict(baseline_raw_projection)
    gate = _dict(readiness)

    blockers: list[str] = []
    if baseline.get("status") != "MODELED_LIMITED":
        blockers.append("BASELINE_RAW_PROJECTION_NOT_AVAILABLE")
    if gate.get("model_version") != READINESS_MODEL_VERSION:
        blockers.append("FM5_READINESS_GATE_NOT_MATERIALIZED")
    if gate.get("market_prices_consumed") is not False:
        blockers.append("READINESS_GATE_MARKET_LEAKAGE")
    if gate.get("production_enabled") is not False:
        blockers.append("READINESS_GATE_UNEXPECTED_PRODUCTION_STATE")
    if gate.get("integration_review_allowed") is not True:
        blockers.append("FM5_INTEGRATION_REVIEW_NOT_ALLOWED")
    if not gate.get("ready_components"):
        blockers.append("NO_FM5_READY_COMPONENTS")
    if sporting_adjustment is None:
        blockers.append("SPORTING_ADJUSTMENT_NOT_MATERIALIZED")

    if blockers:
        return _blocked(baseline, gate, blockers)

    adjustment = _dict(sporting_adjustment)
    contract_errors = _adjustment_contract_errors(adjustment, gate)
    if contract_errors:
        return _blocked(baseline, gate, contract_errors)

    home = float(adjustment["adjusted_home_goal_rate"])
    away = float(adjustment["adjusted_away_goal_rate"])
    probs = soccer_model._probabilities(home, away)
    scores = soccer_model._quality_scores(
        home,
        away,
        probs,
        availability_confidence,
    )

    total = home + away
    weaker = min(home, away)
    if total >= 2.85 and weaker >= 1.05:
        scoring_path = "TWO-WAY OPEN GAME"
    elif total >= 2.85 and max(home, away) >= 2.0:
        scoring_path = "FAVORITE-CARRY OVER"
    elif total <= 2.05 and weaker <= 0.75:
        scoring_path = "MUTUAL SUPPRESSION UNDER"
    else:
        scoring_path = "MIXED / NO STRONG SCORING PATH"

    candidate = {
        "status": "MODELED_LIMITED",
        "model_version": MODEL_VERSION,
        "projection_model": "FM5_VALIDATED_SPORTING_ADJUSTMENT_RESEARCH",
        "sport_first": True,
        "market_independent": True,
        "source_component": adjustment.get("source_component"),
        "source_model_version": adjustment.get("source_model_version"),
        "source_artifact_id": adjustment.get("source_artifact_id"),
        "feature_as_of": adjustment.get("feature_as_of"),
        "fixture_kickoff": adjustment.get("fixture_kickoff"),
        "prior_only": True,
        "oos_gate_passed": True,
        "raw_home_goal_rate": round(home, 4),
        "raw_away_goal_rate": round(away, 4),
        "raw_total_goals": round(total, 4),
        "raw_home_xg": "NOT VERIFIED",
        "raw_away_xg": "NOT VERIFIED",
        "raw_home_win_prob": round(probs["home_win"], 6),
        "raw_draw_prob": round(probs["draw"], 6),
        "raw_away_win_prob": round(probs["away_win"], 6),
        "raw_btts_yes_prob": round(probs["btts_yes"], 6),
        "raw_over_1_5_prob": round(probs["over_1_5"], 6),
        "raw_over_2_5_prob": round(probs["over_2_5"], 6),
        "raw_over_3_5_prob": round(probs["over_3_5"], 6),
        "top_scorelines": probs["top_scores"],
        "scoring_path": scoring_path,
        "screen_scores": scores,
        "_total_dist": probs["total_dist"],
    }

    out = _base_envelope(baseline, gate)
    out.update(
        {
            "status": "FM5_RESEARCH_CANDIDATE_AVAILABLE",
            "blockers": [],
            "ready_component": adjustment.get("source_component"),
            "sporting_adjustment": copy.deepcopy(adjustment),
            "fm5_raw_projection_research": candidate,
            "research_candidate_available": True,
            "production_promotion_allowed": False,
        }
    )
    return out


def public_view(report: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(report)
    candidate = out.get("fm5_raw_projection_research")
    if isinstance(candidate, dict):
        out["fm5_raw_projection_research"] = {
            key: value for key, value in candidate.items() if not key.startswith("_")
        }
    return out
