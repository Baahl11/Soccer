from __future__ import annotations

import math
from typing import Any

from mcp_gateway import period_rate_registry
from mcp_gateway.one_h_goals_intelligence import _fair_decimal, _line_supported, _pair_fair, _settlement, _split_line

SCHEMA_VERSION = "1.0.0"


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _observed(event: dict[str, Any], lam: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snap = event.get("derivative_research_market_snapshot") if isinstance(event.get("derivative_research_market_snapshot"), dict) else {}
    rows: list[dict[str, Any]] = []; unsupported: list[dict[str, Any]] = []
    for group in snap.get("groups") or []:
        if not isinstance(group, dict) or group.get("family") != "2H_GOALS": continue
        market_name = str(group.get("market") or "")
        lower = market_name.lower()
        if "second half" not in lower or "over/under" not in lower: continue
        if "team total" in lower: continue
        parsed: dict[tuple[str, float], float] = {}
        for value in group.get("values") or []:
            if not isinstance(value, dict): continue
            selection = str(value.get("selection") or "").upper()
            side = "OVER" if selection.startswith("OVER") else "UNDER" if selection.startswith("UNDER") else None
            line = _num(value.get("line")); price = _num(value.get("decimal_price"))
            if side is None or line is None or price is None or price <= 1 or not _line_supported(line):
                unsupported.append({"bookmaker": group.get("bookmaker"), "market": market_name, "selection": value.get("selection"), "line": value.get("line"), "reason": "UNSUPPORTED_OR_UNPARSED_2H_TOTAL"})
                continue
            parsed[(side, round(line, 2))] = price
        for side, line in sorted(parsed):
            price = parsed[(side, line)]
            dist = _settlement(lam, line, side)
            if dist is None: continue
            fair = _fair_decimal(dist)
            counterpart = parsed.get(("UNDER" if side == "OVER" else "OVER", line))
            p_market_fair = None; market_fair_basis = "NOT_CALCULATED_FOR_QUARTER_SETTLEMENT"
            if counterpart is not None and abs(line * 2 - round(line * 2)) < 1e-8:
                over_price = parsed.get(("OVER", line)); under_price = parsed.get(("UNDER", line))
                if over_price and under_price:
                    fo, fu = _pair_fair(over_price, under_price)
                    p_market_fair = fo if side == "OVER" else fu
                    market_fair_basis = "NO_VIG_CONDITIONAL_NON_PUSH" if abs(line - round(line)) < 1e-8 else "NO_VIG_BINARY_HALF_LINE"
            rows.append({
                "selection": side, "line": line, "split_components": _split_line(line),
                "total_lambda_2h_pregame": round(lam, 6),
                "win_fraction_model": round(dist["win_fraction"], 6), "push_fraction_model": round(dist["push_fraction"], 6), "loss_fraction_model": round(dist["loss_fraction"], 6),
                "fair_decimal_model": round(fair, 4) if fair is not None else None,
                "bookmaker": group.get("bookmaker"), "market": market_name, "decimal_price": round(price, 4), "provider_update": group.get("provider_update"),
                "p_market_fair": round(p_market_fair, 6) if p_market_fair is not None else None, "market_fair_basis": market_fair_basis,
                "raw_ev": round(dist["win_fraction"] * price + dist["push_fraction"] - 1.0, 6),
                "research_only": True, "actionable": False, "decision_weight": 0.0, "classification": "RESEARCH_ONLY",
                "promotion_block": "PREGAME_2H_MODEL_NOT_PRODUCTION_APPROVED",
            })
    rows.sort(key=lambda r: float(r.get("raw_ev") or -999.0), reverse=True)
    return rows[:32], unsupported[:32]


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    model = period_rate_registry.model_fixture(fixture, "2H", registry)
    if not model:
        return {"schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "stage": event.get("stage"), "status": "NOT_MODELED_THIS_TICK", "actionable": False, "decision_weight": 0.0, "reason": "PERIOD_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT"}
    lam = float(model["total_lambda"]); observed, unsupported = _observed(event, lam)
    return {
        "schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "stage": event.get("stage"), "status": "LIVE_RESEARCH_MODELED",
        "model": model["model"], "model_inputs": model, "model_timing": "PREGAME_ONLY_NOT_HALFTIME_CONDITIONED",
        "p_over_0_5": round(1.0 - math.exp(-lam), 6),
        "p_over_1_5": round(1.0 - math.exp(-lam) * (1.0 + lam), 6),
        "p_over_2_5": round(1.0 - math.exp(-lam) * (1.0 + lam + lam * lam / 2.0), 6),
        "observed_market_rows": observed, "observed_market_count": len(observed), "unsupported_market_rows": unsupported,
        "actionable": False, "decision_weight": 0.0, "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {"minimum_oos_for_market_comparison": 100, "minimum_oos_for_actionable_review": 200, "requires": ["dedicated pregame 2H OOS calibration", "historical exact-line prices and true CLV", "stable line-bucket/league calibration", "separate halftime-conditioned model for any live-2H use"]},
        "policy": "PREGAME PERIOD-SPECIFIC SPORT MODEL; NEVER REUSE FT OR 1H PROBABILITY; NEVER CLAIM HALFTIME-CONDITIONED; EXACT OBSERVED 2H TOTAL LINES ONLY; SETTLEMENT-AWARE; ZERO DECISION WEIGHT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = period_rate_registry.load_registry(); modeled = observed_events = observed_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME": continue
        intel = build(event, registry); event["two_h_goals_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_MODELED": modeled += 1
        count = int(intel.get("observed_market_count") or 0)
        if count: observed_events += 1; observed_rows += count
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["goals_second_half_pregame"] = {"status": intel.get("status"), "model_inputs": intel.get("model_inputs"), "model_timing": intel.get("model_timing"), "observed_market_rows": intel.get("observed_market_rows") or [], "actionable": False, "decision_weight": 0.0, "calibration_gate": intel.get("calibration_gate")}
    return {"period_rate_registry_loaded": bool(registry), "modeled_events": modeled, "events_with_observed_2h_total_markets": observed_events, "observed_2h_total_rows": observed_rows, "provider_requests_added": 0, "state_registry_reads_shared_with_1h": True}
