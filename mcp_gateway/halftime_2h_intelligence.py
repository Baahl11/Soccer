from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone as dt_timezone
from typing import Any

import httpx

from mcp_gateway import automation as base
from mcp_gateway import period_rate_registry

SCHEMA_VERSION = "1.0.0"
REGISTRY_URL = (
    "https://raw.githubusercontent.com/Baahl11/Soccer/"
    "soccer-edge-state/soccer_edge_state/analysis/two_h_halftime_conditioned.json"
)
CACHE_TTL = timedelta(hours=6)


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _load_registry() -> dict[str, Any] | None:
    now = datetime.now(dt_timezone.utc)
    cached = base._cache_get("halftime_2h_registry", "latest", CACHE_TTL, now)
    if isinstance(cached, dict) and cached.get("schema_version"):
        return cached
    try:
        response = httpx.get(REGISTRY_URL, timeout=5.0, follow_redirects=True)
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("full_history_state_registry"), dict):
        return None
    base._cache_set("halftime_2h_registry", "latest", payload, now)
    return payload


def _halftime_score(fixture: dict[str, Any]) -> tuple[int, int] | None:
    score = fixture.get("score") if isinstance(fixture.get("score"), dict) else {}
    ht = score.get("halftime") if isinstance(score.get("halftime"), dict) else {}
    try:
        return int(ht.get("home")), int(ht.get("away"))
    except (TypeError, ValueError):
        return None


def _state_bucket(home: int, away: int) -> str:
    result = "DRAW" if home == away else "HOME_LEAD" if home > away else "AWAY_LEAD"
    total = home + away
    total_bucket = "HT0" if total == 0 else "HT1" if total == 1 else "HT2_PLUS"
    return f"{result}|{total_bucket}"


def _over_probability(lam: float, line: float) -> float:
    threshold = int(math.floor(line)) + 1
    cdf = sum(math.exp(-lam) * (lam ** k) / math.factorial(k) for k in range(threshold))
    return max(0.0, min(1.0, 1.0 - cdf))


def build(event: dict[str, Any], registry: dict[str, Any] | None, period_registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    if event.get("stage") != "HT" or str(fixture.get("status") or "").upper() != "HT":
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_HALFTIME_EVENT",
            "actionable": False,
            "decision_weight": 0.0,
        }

    score = _halftime_score(fixture)
    if score is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": "HT",
            "status": "NOT_MODELED_THIS_TICK",
            "reason": "HALFTIME_SCORE_NOT_VERIFIED",
            "actionable": False,
            "decision_weight": 0.0,
        }

    base_model = period_rate_registry.model_fixture(fixture, "2H", period_registry)
    state_registry = registry.get("full_history_state_registry") if isinstance(registry, dict) else None
    buckets = state_registry.get("buckets") if isinstance(state_registry, dict) and isinstance(state_registry.get("buckets"), dict) else {}
    bucket = _state_bucket(*score)
    bucket_row = buckets.get(bucket) if isinstance(buckets.get(bucket), dict) else None
    multiplier = _num(bucket_row.get("multiplier_vs_global")) if isinstance(bucket_row, dict) else None
    state_n = int(bucket_row.get("n") or 0) if isinstance(bucket_row, dict) else 0

    if not isinstance(base_model, dict) or multiplier is None or multiplier <= 0:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": "HT",
            "status": "NOT_MODELED_THIS_TICK",
            "reason": "HALFTIME_STATE_OR_2H_BASE_REGISTRY_NOT_AVAILABLE",
            "ht_score": f"{score[0]}-{score[1]}",
            "state_bucket": bucket,
            "actionable": False,
            "decision_weight": 0.0,
        }

    baseline_lambda = float(base_model["total_lambda"])
    conditioned_lambda = max(0.10, min(7.0, baseline_lambda * multiplier))
    p05 = _over_probability(conditioned_lambda, 0.5)
    p15 = _over_probability(conditioned_lambda, 1.5)
    p25 = _over_probability(conditioned_lambda, 2.5)

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": "HT",
        "status": "LIVE_RESEARCH_MODELED_AT_HALFTIME",
        "model": "PREGAME_2H_BASELINE_X_HALFTIME_STATE_MULTIPLIER_v0.1",
        "model_timing": "HALFTIME_CONDITIONED_RESEARCH_ONLY",
        "ht_score": f"{score[0]}-{score[1]}",
        "state_bucket": bucket,
        "state_prior_n": state_n,
        "baseline_2h_lambda": round(baseline_lambda, 6),
        "state_multiplier": round(multiplier, 6),
        "conditioned_2h_lambda": round(conditioned_lambda, 6),
        "p_over_0_5": round(p05, 6),
        "p_over_1_5": round(p15, 6),
        "p_over_2_5": round(p25, 6),
        "fair_over_0_5_decimal": round(1.0 / p05, 4) if 0 < p05 < 1 else None,
        "fair_over_1_5_decimal": round(1.0 / p15, 4) if 0 < p15 < 1 else None,
        "fair_over_2_5_decimal": round(1.0 / p25, 4) if 0 < p25 < 1 else None,
        "conditioning_verified": {
            "halftime_score": True,
            "lead_state": True,
            "halftime_goal_bucket": True,
            "red_cards": False,
            "shots": False,
            "shots_on_target": False,
            "halftime_xg": False,
        },
        "market_status": "NO_LIVE_2H_PRICE_ATTACHED",
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "promotion_blockers": [
            "RED_CARD_STATE_NOT_VERIFIED_IN_HT_PATH",
            "HALFTIME_SHOTS_SOT_NOT_VERIFIED_IN_HT_PATH",
            "LIVE_2H_MARKET_PRICE_NOT_ATTACHED",
            "HALFTIME_MODEL_NOT_PRODUCTION_CALIBRATED",
        ],
        "calibration_gate": {
            "minimum_oos_for_live_research_review": 200,
            "minimum_oos_for_actionable_review": 400,
            "requires": [
                "beats pregame 2H baseline on Brier/log-loss/MAE",
                "verified current halftime score",
                "red-card state before production review",
                "live 2H exact-line price and true CLV history",
                "stable performance by state bucket and competition",
            ],
        },
        "policy": (
            "HALFTIME SCORE/GAME-STATE CONDITIONING ONLY IN V1; NEVER RELABEL PREGAME 2H AS LIVE; "
            "ZERO DECISION WEIGHT; NO BET_LEAN_GALAXY; MISSING RED-CARD/SHOTS/SOT INPUTS EXPLICIT"
        ),
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = _load_registry()
    period_registry = period_rate_registry.load_registry()
    ht_events = modeled = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") != "HT":
            continue
        ht_events += 1
        intel = build(event, registry, period_registry)
        event["two_h_halftime_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_MODELED_AT_HALFTIME":
            modeled += 1
    return {
        "halftime_events": ht_events,
        "modeled_halftime_events": modeled,
        "halftime_state_registry_loaded": bool(registry),
        "period_rate_registry_loaded": bool(period_registry),
        "provider_requests_added": 0,
    }
