from __future__ import annotations

import math
from typing import Any

from mcp_gateway import period_rate_registry

SCHEMA_VERSION = "1.0.0"
MAX_PERIOD_GOALS = 20


def _num(value: Any) -> float | None:
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except (TypeError, ValueError):
        return None


def _line_supported(line: float) -> bool:
    return abs(line * 4 - round(line * 4)) < 1e-8 and 0.0 <= line <= 6.0


def _split_line(line: float) -> list[float]:
    if not _line_supported(line): return []
    if abs(line * 2 - round(line * 2)) < 1e-8: return [round(line, 2)]
    return [round(math.floor(line * 2) / 2.0, 2), round(math.ceil(line * 2) / 2.0, 2)]


def _component(goals: int, line: float, side: str) -> tuple[float, float, float]:
    if side == "OVER":
        if goals > line: return 1.0, 0.0, 0.0
        if goals < line: return 0.0, 0.0, 1.0
        return 0.0, 1.0, 0.0
    if goals < line: return 1.0, 0.0, 0.0
    if goals > line: return 0.0, 0.0, 1.0
    return 0.0, 1.0, 0.0


def _settlement(lam: float, line: float, side: str) -> dict[str, float] | None:
    components = _split_line(line)
    if not components: return None
    w = p = l = mass = 0.0
    for goals in range(MAX_PERIOD_GOALS + 1):
        prob = math.exp(-lam) * lam**goals / math.factorial(goals)
        mass += prob
        fw = fp = fl = 0.0
        for component in components:
            a, b, c = _component(goals, component, side)
            fw += a / len(components); fp += b / len(components); fl += c / len(components)
        w += prob * fw; p += prob * fp; l += prob * fl
    if mass <= 0: return None
    return {"win_fraction": w / mass, "push_fraction": p / mass, "loss_fraction": l / mass}


def _fair_decimal(dist: dict[str, float]) -> float | None:
    w = dist["win_fraction"]; p = dist["push_fraction"]
    if w <= 0: return None
    return max(1.0, (1.0 - p) / w)


def _pair_fair(over_price: float, under_price: float) -> tuple[float, float]:
    oi = 1.0 / over_price; ui = 1.0 / under_price; total = oi + ui
    return oi / total, ui / total


def _observed(event: dict[str, Any], lam: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snap = event.get("derivative_research_market_snapshot") if isinstance(event.get("derivative_research_market_snapshot"), dict) else {}
    groups = snap.get("groups") or []
    rows: list[dict[str, Any]] = []; unsupported: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict) or group.get("family") != "1H_GOALS": continue
        market_name = str(group.get("market") or "")
        if "over/under first half" not in market_name.lower(): continue
        parsed: dict[tuple[str, float], float] = {}
        for value in group.get("values") or []:
            if not isinstance(value, dict): continue
            selection = str(value.get("selection") or "").upper()
            side = "OVER" if selection.startswith("OVER") else "UNDER" if selection.startswith("UNDER") else None
            line = _num(value.get("line")); price = _num(value.get("decimal_price"))
            if side is None or line is None or price is None or price <= 1 or not _line_supported(line):
                unsupported.append({"bookmaker": group.get("bookmaker"), "market": market_name, "selection": value.get("selection"), "line": value.get("line"), "reason": "UNSUPPORTED_OR_UNPARSED_1H_TOTAL"})
                continue
            parsed[(side, round(line, 2))] = price
        for side, line in sorted(parsed):
            price = parsed[(side, line)]
            dist = _settlement(lam, line, side)
            if dist is None: continue
            fair = _fair_decimal(dist)
            counterpart = parsed.get(("UNDER" if side == "OVER" else "OVER", line))
            p_market_fair = None
            market_fair_basis = "NOT_CALCULATED_FOR_QUARTER_SETTLEMENT"
            if counterpart is not None and abs(line * 2 - round(line * 2)) < 1e-8:
                over_price = parsed.get(("OVER", line)); under_price = parsed.get(("UNDER", line))
                if over_price and under_price:
                    fo, fu = _pair_fair(over_price, under_price)
                    p_market_fair = fo if side == "OVER" else fu
                    market_fair_basis = "NO_VIG_CONDITIONAL_NON_PUSH" if abs(line - round(line)) < 1e-8 else "NO_VIG_BINARY_HALF_LINE"
            rows.append({
                "selection": side, "line": line, "split_components": _split_line(line),
                "total_lambda_1h": round(lam, 6),
                "win_fraction_model": round(dist["win_fraction"], 6),
                "push_fraction_model": round(dist["push_fraction"], 6),
                "loss_fraction_model": round(dist["loss_fraction"], 6),
                "fair_decimal_model": round(fair, 4) if fair is not None else None,
                "bookmaker": group.get("bookmaker"), "market": market_name,
                "decimal_price": round(price, 4), "provider_update": group.get("provider_update"),
                "p_market_fair": round(p_market_fair, 6) if p_market_fair is not None else None,
                "market_fair_basis": market_fair_basis,
                "raw_ev": round(dist["win_fraction"] * price + dist["push_fraction"] - 1.0, 6),
                "research_only": True, "actionable": False, "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY", "promotion_block": "1H_MODEL_NOT_PRODUCTION_APPROVED",
            })
    rows.sort(key=lambda r: float(r.get("raw_ev") or -999.0), reverse=True)
    return rows[:32], unsupported[:32]


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    model = period_rate_registry.model_fixture(fixture, "1H", registry)
    if not model:
        return {"schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "stage": event.get("stage"), "status": "NOT_MODELED_THIS_TICK", "actionable": False, "decision_weight": 0.0, "reason": "PERIOD_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT"}
    lam = float(model["total_lambda"])
    observed, unsupported = _observed(event, lam)
    return {
        "schema_version": SCHEMA_VERSION, "fixture_id": fixture.get("fixture_id"), "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED", "model": model["model"], "model_inputs": model,
        "p_over_0_5": round(1.0 - math.exp(-lam), 6),
        "p_over_1_5": round(1.0 - math.exp(-lam) * (1.0 + lam), 6),
        "p_over_2_5": round(1.0 - math.exp(-lam) * (1.0 + lam + lam * lam / 2.0), 6),
        "observed_market_rows": observed, "observed_market_count": len(observed), "unsupported_market_rows": unsupported,
        "actionable": False, "decision_weight": 0.0, "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {"minimum_oos_for_market_comparison": 100, "minimum_oos_for_actionable_review": 200, "requires": ["dedicated 1H OOS calibration", "historical exact-line prices and true CLV", "stable line-bucket/league calibration", "XI/availability gates if promoted"]},
        "policy": "PERIOD-SPECIFIC SPORT MODEL; NEVER REUSE FT PROBABILITY; EXACT OBSERVED 1H TOTAL LINES ONLY; PUSH/QUARTER SETTLEMENT EXPLICIT; ZERO DECISION WEIGHT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = period_rate_registry.load_registry()
    modeled = observed_events = observed_rows = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") == "POSTGAME": continue
        intel = build(event, registry); event["one_h_goals_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_MODELED": modeled += 1
        count = int(intel.get("observed_market_count") or 0)
        if count: observed_events += 1; observed_rows += count
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["goals_first_half"] = {"status": intel.get("status"), "model_inputs": intel.get("model_inputs"), "observed_market_rows": intel.get("observed_market_rows") or [], "actionable": False, "decision_weight": 0.0, "calibration_gate": intel.get("calibration_gate")}
    return {"period_rate_registry_loaded": bool(registry), "modeled_events": modeled, "events_with_observed_1h_total_markets": observed_events, "observed_1h_total_rows": observed_rows, "provider_requests_added": 0, "state_registry_reads_max": 1}
