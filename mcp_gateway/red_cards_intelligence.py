from __future__ import annotations

from typing import Any

from mcp_gateway import red_cards_rate_registry

SCHEMA_VERSION = "1.0.0"


def _num(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _decimal(value: Any) -> float | None:
    out = _num(value)
    return out if out is not None and 1.0 < out <= 1000.0 else None


def _is_explicit_match_red_market(name: Any) -> bool:
    n = _norm(name)
    if "red" not in n or "card" not in n:
        return False
    if any(token in n for token in ("player", "team", "home", "away", "first half", "1st half", "second half", "2nd half")):
        return False
    return any(token in n for token in ("match", "game", "a red card", "red card"))


def _yes_no(value: Any) -> str | None:
    n = _norm(value)
    if n in {"yes", "y", "1"} or n.endswith(" yes"):
        return "YES"
    if n in {"no", "n", "0"} or n.endswith(" no"):
        return "NO"
    return None


def _pair_fair(yes_price: float, no_price: float) -> tuple[float, float]:
    yi = 1.0 / yes_price
    ni = 1.0 / no_price
    total = yi + ni
    return yi / total, ni / total


def _observed(event: dict[str, Any], p_yes: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    market = event.get("market") if isinstance(event.get("market"), dict) else {}
    provenance = event.get("market_provenance") if isinstance(event.get("market_provenance"), dict) else {}
    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []

    for group in market.get("markets") or []:
        if not isinstance(group, dict):
            continue
        market_name = group.get("market")
        n = _norm(market_name)
        if "red" not in n or "card" not in n:
            continue
        if not _is_explicit_match_red_market(market_name):
            blocked.append({
                "bookmaker": group.get("bookmaker"),
                "market": market_name,
                "reason": "RED_CARD_MARKET_OUTSIDE_MATCH_ANY_RED_YES_NO_V1_SCOPE",
            })
            continue
        prices: dict[str, float] = {}
        for value in group.get("values") or []:
            if not isinstance(value, dict):
                continue
            side = _yes_no(value.get("selection"))
            price = _decimal(value.get("price"))
            if side is None or price is None:
                blocked.append({
                    "bookmaker": group.get("bookmaker"),
                    "market": market_name,
                    "selection": value.get("selection"),
                    "reason": "UNPARSED_RED_CARD_YES_NO_SELECTION_OR_PRICE",
                })
                continue
            prices[side] = price

        fair_yes = fair_no = None
        if prices.get("YES") is not None and prices.get("NO") is not None:
            fair_yes, fair_no = _pair_fair(prices["YES"], prices["NO"])

        for side, p_model, p_market in (
            ("YES", p_yes, fair_yes),
            ("NO", 1.0 - p_yes, fair_no),
        ):
            price = prices.get(side)
            if price is None:
                continue
            rows.append({
                "target": "ANY_RED_CARD_IN_MATCH",
                "selection": side,
                "probability_model_raw": round(p_model, 6),
                "fair_decimal_model_raw": round(1.0 / p_model, 4) if 0 < p_model < 1 else None,
                "bookmaker": group.get("bookmaker"),
                "bookmaker_id": group.get("bookmaker_id"),
                "market": market_name,
                "market_id": group.get("market_id"),
                "decimal_price": round(price, 4),
                "p_breakeven": round(1.0 / price, 6),
                "p_market_fair": round(p_market, 6) if p_market is not None else None,
                "raw_edge_vs_market_fair_pp": round((p_model - p_market) * 100.0, 3) if p_market is not None else None,
                "raw_ev_at_observed_price": round(p_model * price - 1.0, 6),
                "provider_update": group.get("provider_update"),
                "market_source": provenance.get("source") or "NOT_VERIFIED",
                "market_fresh": provenance.get("fresh") is True,
                "research_only": True,
                "actionable": False,
                "decision_weight": 0.0,
                "classification": "RESEARCH_ONLY",
                "promotion_block": "RED_CARD_MODEL_NOT_OOS_CALIBRATED_OR_PRODUCTION_APPROVED",
            })

    rows.sort(key=lambda r: (1 if r.get("market_fresh") else 0, float(r.get("raw_edge_vs_market_fair_pp") or -999)), reverse=True)
    return rows[:16], blocked[:32]


def build(event: dict[str, Any], registry: dict[str, Any] | None) -> dict[str, Any]:
    fixture = event.get("fixture") if isinstance(event.get("fixture"), dict) else {}
    model = red_cards_rate_registry.model_fixture(fixture, registry)
    if not model:
        return {
            "schema_version": SCHEMA_VERSION,
            "fixture_id": fixture.get("fixture_id"),
            "stage": event.get("stage"),
            "status": "NOT_MODELED_THIS_TICK",
            "actionable": False,
            "decision_weight": 0.0,
            "reason": "RED_CARD_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT",
        }

    p_yes = float(model["p_any_red"])
    rows, blocked = _observed(event, p_yes)
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": fixture.get("fixture_id"),
        "stage": event.get("stage"),
        "status": "LIVE_RESEARCH_MODELED",
        "model": model["model"],
        "model_inputs": model,
        "p_any_red": round(p_yes, 6),
        "p_no_red": round(1.0 - p_yes, 6),
        "fair_yes_decimal": round(1.0 / p_yes, 4) if 0 < p_yes < 1 else None,
        "fair_no_decimal": round(1.0 / (1.0 - p_yes), 4) if 0 < p_yes < 1 else None,
        "observed_explicit_red_market_rows": rows,
        "observed_explicit_red_market_count": len(rows),
        "blocked_red_card_markets": blocked,
        "referee_adjustment_applied": bool(model.get("referee_prior_n", 0) >= 20),
        "actionable": False,
        "decision_weight": 0.0,
        "production_status": "LIVE_RESEARCH_NOT_ACTIONABLE",
        "calibration_gate": {
            "minimum_oos_for_market_review": 500,
            "minimum_oos_for_actionable_review": 1000,
            "minimum_referee_adjusted_oos": 200,
            "requires": [
                "stable Brier/log-loss for rare-event calibration",
                "stable calibration by league/probability bucket",
                "verified current referee when referee adjustment is used",
                "explicit sportsbook match-red-card YES/NO settlement",
                "verified price history and true CLV",
            ],
        },
        "policy": "ANY RED CARD IN MATCH YES/NO ONLY; YELLOW CARDS EXCLUDED; LOW-FREQUENCY STRONG SHRINKAGE; EXACT EXPLICIT RED-CARD MARKET ONLY; ZERO DECISION WEIGHT",
    }


def attach(payload: dict[str, Any]) -> dict[str, int | bool]:
    registry = red_cards_rate_registry.load_registry()
    modeled = priced_events = priced_rows = blocked = referee_adjusted = 0
    for event in payload.get("events") or []:
        if not isinstance(event, dict) or event.get("event_type") != "SOCCER_REFRESH" or event.get("stage") in {"POSTGAME", "HT"}:
            continue
        intel = build(event, registry)
        event["red_cards_intelligence"] = intel
        if intel.get("status") == "LIVE_RESEARCH_MODELED":
            modeled += 1
        if intel.get("referee_adjustment_applied"):
            referee_adjusted += 1
        count = int(intel.get("observed_explicit_red_market_count") or 0)
        if count:
            priced_events += 1
            priced_rows += count
        blocked += len(intel.get("blocked_red_card_markets") or [])
        mi = event.get("match_intelligence")
        if isinstance(mi, dict) and isinstance(mi.get("areas"), dict):
            mi["areas"]["red_cards"] = {
                "status": intel.get("status"),
                "p_any_red": intel.get("p_any_red"),
                "p_no_red": intel.get("p_no_red"),
                "observed_explicit_red_market_rows": intel.get("observed_explicit_red_market_rows") or [],
                "blocked_red_card_markets": intel.get("blocked_red_card_markets") or [],
                "referee_adjustment_applied": intel.get("referee_adjustment_applied"),
                "actionable": False,
                "decision_weight": 0.0,
                "calibration_gate": intel.get("calibration_gate"),
            }
    return {
        "red_cards_rate_registry_loaded": bool(registry),
        "modeled_events": modeled,
        "referee_adjusted_events": referee_adjusted,
        "events_with_explicit_red_card_markets": priced_events,
        "observed_explicit_red_card_rows": priced_rows,
        "blocked_red_card_market_groups": blocked,
        "provider_requests_added": 0,
        "state_registry_reads_max": 1,
    }
