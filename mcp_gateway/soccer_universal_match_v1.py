"""Sport-first fixture intelligence: independent sporting and opportunity states.

Presentation-only. Neither this module nor the public contract turns a research
source into a licensed public model, a price, an actionable BET, or an OOS claim.
All missing fields remain unknown; an explicit canonical PASS is never hidden.
"""
from __future__ import annotations

from typing import Any

VERSION = "SOCCER_UNIVERSAL_INTELLIGENCE_v1.1.0"
CANCELLED = frozenset(("CANC", "ABD", "AWD", "WO"))
POSTPONED = frozenset(("PST", "SUSP"))
COMPLETED = frozenset(("FT", "AET", "PEN"))

def event_state(row: dict[str, Any]) -> str | None:
    fx = obj(row.get("fixture"))
    code = str(fx.get("fixture_status") or fx.get("status") or "").upper()
    if code in CANCELLED:
        return "CANCELLED"
    if code in POSTPONED:
        return "POSTPONED"
    if code in COMPLETED:
        return "COMPLETED"
    return None


CANONICAL = frozenset(("BET", "LEAN", "WATCH", "PASS"))
PUBLIC_STATES = frozenset(("MODELED_LIMITED", "PARTIAL", "SPORT_EVIDENCE_ONLY",
                           "NOT_ANALYZED", "DATA_INSUFFICIENT"))
OPPORTUNITY_LABELS = {
    "BET": "RECOMENDACION_DE_VALOR",
    "LEAN": "TENDENCIA_FAVORABLE",
    "WATCH": "EN_OBSERVACION",
    "PASS": "SIN_VALOR",
}

def obj(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}

def positive_int(value: Any) -> int:
    # Avoid inventing an evidence count from a malformed string/boolean.
    return value if type(value) is int and value >= 0 else 0

def sport_from_registry(row: dict[str, Any]) -> dict[str, Any]:
    """Produce a safe, explanatory sport status for EVERY registry fixture."""
    coverage = obj(row.get("coverage"))
    screening = obj(row.get("screening"))
    screen_queue = obj(row.get("screen_queue"))
    inventory = obj(coverage.get("evidence_inventory"))
    queue_modeled = (
        screen_queue.get("status") == "MODELED_LIMITED"
        and positive_int(screen_queue.get("attempt_count")) > 0
    )
    raw = queue_modeled or screening.get("raw_projection_recorded") is True or (
        coverage.get("raw_sport_projection_present") is True
    )
    features = positive_int(inventory.get("feature_snapshots"))
    models = positive_int(inventory.get("model_runs"))
    lineup = positive_int(inventory.get("lineup_snapshots"))
    availability = positive_int(inventory.get("availability_snapshots"))
    sport = positive_int(coverage.get("sport_evidence_count"))
    analysis = positive_int(coverage.get("analysis_rows"))
    current_event = event_state(row)
    if current_event in ("CANCELLED", "POSTPONED"):
        status = current_event
        reason = ("Encuentro cancelado; se conservan los datos históricos, "
                  "pero no hay pronóstico vigente." if current_event == "CANCELLED"
                  else "Encuentro aplazado o suspendido; no emitir recomendación vigente.")
    elif raw:
        status = "MODELED_LIMITED"
        reason = "Proyeccion RAW registrada. Revisar datos, muestra, fecha y validacion."
    elif models or features or analysis:
        status = "PARTIAL"
        reason = "Hay evidencia persistida, pero no una proyeccion RAW verificable."
    elif sport or lineup or availability:
        status = "SPORT_EVIDENCE_ONLY"
        reason = "Hay datos deportivos, pero no un modelo RAW completo."
    elif screen_queue.get("status") == "DATA_INSUFFICIENT" and positive_int(screen_queue.get("attempt_count")) > 0:
        status = "DATA_INSUFFICIENT"
        reason = "Se ejecutó el screening deportivo, pero no alcanzó para una proyección."
    elif screen_queue.get("status") == "RETRY_REQUIRED":
        status = "DATA_INSUFFICIENT"
        reason = "Screening pendiente de reintento; datos aún no verificados."
    elif positive_int(coverage.get("market_evidence_count")):
        status = "DATA_INSUFFICIENT"
        reason = "Hay precios, pero faltan fundamentos deportivos verificables."
    else:
        status = "NOT_ANALYZED"
        reason = "Partido registrado sin screening deportivo persistido."
    missing = list(screening.get("missing_evidence_codes") or [])
    missing = [x for x in missing if isinstance(x, str)][:16]
    if not raw and "RAW_SPORT_NOT_RECORDED" not in missing:
        missing.insert(0, "RAW_SPORT_NOT_RECORDED")
    return {
        "status": status, "explanation": reason,
        "fixture_event_state": current_event,
        "raw_projection_recorded": raw,
        "modeled_complete": False,  # no unearned OOS or XI certainty
        "verified_feature_snapshot_count": features,
        "sport_evidence_count": sport,
        "model_run_count": models,
        "screen_queue_status": screen_queue.get("status") or "NOT_VERIFIED",
        "screen_attempt_count": screen_queue.get("attempt_count"),
        "missing_evidence_codes": missing,
        "tracks": {
            "SIDE": "RAW_RECORDED" if raw else "NOT_VERIFIED",
            "GOALS_BTTS": "RAW_RECORDED" if raw else "NOT_VERIFIED",
            "CORNERS": "NOT_VERIFIED",
        },
        "probabilities": None,  # public slate carries no unlicensed source estimates
        "source": "POSTGRES_PERSISTED_FIXTURE_EVIDENCE_AND_SCREEN_QUEUE",
        "projection_validated_oos": False,
        "source_redisplay_rights_verified": False,
        "analyst_review_allowed": True,
    }

def opportunity_from_registry(row: dict[str, Any]) -> dict[str, Any]:
    """Never equate unavailable market with a verified negative edge."""
    state = obj(row.get("state"))
    coverage = obj(row.get("coverage"))
    internal = str(state.get("classification") or coverage.get("decision_status") or "").upper()
    internal = internal if internal in CANONICAL else None
    prices = list(coverage.get("verified_price_markets") or [])
    prices = [v for v in prices if isinstance(v, str)][:16]
    cancelled = event_state(row)
    if cancelled in ("CANCELLED", "POSTPONED"):
        status = "EVENTO_NO_VIGENTE"
        reason = "Encuentro cancelado/aplazado: no existe recomendación vigente."
    elif not internal:
        status = "MERCADO_NO_EVALUADO"
        reason = "No existe una decision de mercado canonica verificada."
    elif internal == "PASS" and not prices:
        status = "SPORT_SCREEN_PASS"
        reason = "PASS deportivo registrado, sin evidencia suficiente de precio para declarar 'sin valor'."
    elif internal == "BET":
        status = "VERIFICACION_PENDIENTE"
        reason = "BET canonico historico; vigencia de precio y disponibilidad NO VERIFICADAS ahora."
    else:
        status = OPPORTUNITY_LABELS[internal]
        reason = state.get("reason_display") or "Decision canonica persistida; precio actual NO VERIFICADO."
    return {
        "status": status,
        "canonical_classification": internal,
        "explanation": str(reason)[:500],
        "verified_price_market_count": len(prices),
        "has_current_executable_price": False,  # no such claim from inventory alone
        "probability_edge_pp": None,
        "estimated_ev": None,
        "recommendation_executable": False,
        "market_queried": bool(positive_int(coverage.get("market_evidence_count"))),
        "source": "PERSISTED_CANONICAL_MARKET_DECISION",
        "model_bet_gates_unchanged": True,
    }

def add_to_slate_row(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["sport_analysis"] = sport_from_registry(row)
    result["opportunity"] = opportunity_from_registry(row)
    return result

def detail_layers(packet: dict[str, Any]) -> dict[str, Any]:
    """Deep analyst packet, *only* after existing PRO authorization is checked."""
    sport = obj(packet.get("sport_context"))
    current_event = event_state(packet)
    availability = obj(packet.get("availability"))
    evidence = obj(packet.get("evidence"))
    review = obj(packet.get("analyst_review"))
    decision = obj(packet.get("decision_summary"))
    markets = obj(packet.get("market_context"))
    projection = obj(packet.get("projection_ladder"))
    outcomes = obj(sport.get("outcome_probabilities"))
    probs = {key: outcomes.get(key) for key in ("home", "draw", "away")}
    # No single 1X2 cell may be presented as a complete model.
    has_full_1x2 = all(
        type(probs.get(k)) in (int, float) and 0 <= probs[k] <= 1
        for k in ("home", "draw", "away")
    ) and abs(sum(probs.values()) - 1.0) < 0.025
    raw_model = bool(has_full_1x2)
    has_features = bool(packet.get("evidence_sections"))
    status = (current_event if current_event in ("CANCELLED", "POSTPONED") else
              "MODELED_LIMITED" if raw_model else
              "PARTIAL" if has_features or positive_int(review.get("persisted_evidence_count"))
              or positive_int(review.get("analysis_rows")) else "NOT_ANALYZED")
    missing = [x for x in review.get("missing_sections", []) if isinstance(x, str)][:30]
    sport_layer = {
        "status": status,
        "fixture_event_state": current_event,
        "reason": ("Encuentro cancelado/aplazado: los registros pasados son solo contexto."
                   if current_event in ("CANCELLED", "POSTPONED") else
                   "Proyeccion deportiva RAW guardada; no equivale a modelo OOS calibrado."
                   if raw_model else
                   "Sin proyeccion deportiva completa; revisar evidencias persistidas."),
        "tracks": {
            "SIDE": {"status": "RAW_MODELED" if raw_model else "NOT_VERIFIED",
                     "raw_1x2": probs if raw_model else None},
            "GOALS_BTTS": {
                "status": "GOAL_RATE_BASELINE" if sport.get("expected_goals") else "NOT_VERIFIED",
                "goal_rates_not_xg": sport.get("expected_goals"),
            },
            "CORNERS": {"status": "NOT_VERIFIED",
                        "reason": "No se construye una proyeccion de corners sin modelo dedicado."},
        },
        "sporting_reasons": [v for v in evidence.get("sporting_reasons", []) if isinstance(v, str)][:20],
        "verified_evidence_sections": list(packet.get("evidence_sections") or []),
        "missing_inputs": missing,
        "captured_at": sport.get("captured_at"),
        "model_version": obj(packet.get("model_context")).get("model_version"),
        "projection_validated_oos": False,
        "no_prices_used_to_create_raw": True,
    }
    canonical = str(decision.get("classification") or "").upper()
    canonical = canonical if canonical in CANONICAL else None
    verified = [m for m in review.get("verified_price_markets", []) if isinstance(m, str)]
    if current_event in ("CANCELLED", "POSTPONED"):
        out_status = "EVENTO_NO_VIGENTE"
    elif not canonical:
        out_status = "MERCADO_NO_EVALUADO"
    elif canonical == "PASS" and not verified:
        out_status = "SPORT_SCREEN_PASS"
    elif canonical == "BET":
        out_status = "VERIFICACION_PENDIENTE"
    else:
        out_status = OPPORTUNITY_LABELS[canonical]
    return {
        "sport_analysis": sport_layer,
        "opportunity": {
            "status": out_status,
            "canonical_classification": canonical,
            "reason": decision.get("reason_display"),
            "verified_price_markets": verified[:20],
            "market_candidate_count": positive_int(markets.get("candidate_count")),
            "probability_edge_pp": projection.get("probability_edge_pp"),
            "estimated_ev": projection.get("estimated_ev"),
            "price_currently_verified": False,
            "recommendation_executable": False,
            "model_bet_gates_unchanged": True,
        },
    }
