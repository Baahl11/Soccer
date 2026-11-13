from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from mcp_gateway import corners_rate_registry

SCHEMA_VERSION = "1.0.0"
MAX_CORNERS = 45


def _num(value: Any) -> float | None:
    try:
        out = float(value); return out if math.isfinite(out) else None
    except (TypeError, ValueError): return None


def _line_supported(line: float) -> bool:
    return abs(line*4-round(line*4)) < 1e-8 and 0 <= line <= 30


def _split(line: float) -> list[float]:
    if not _line_supported(line): return []
    if abs(line*2-round(line*2)) < 1e-8: return [round(line,2)]
    return [round(math.floor(line*2)/2,2), round(math.ceil(line*2)/2,2)]


def _component(total: int, line: float, side: str) -> tuple[float,float,float]:
    if side == "OVER":
        if total > line: return 1,0,0
        if total < line: return 0,0,1
        return 0,1,0
    if total < line: return 1,0,0
    if total > line: return 0,0,1
    return 0,1,0


def _settlement(lam: float, line: float, side: str) -> dict[str,float] | None:
    parts = _split(line)
    if not parts: return None
    w=p=l=mass=0.0
    for total in range(MAX_CORNERS+1):
        prob = math.exp(-lam)*lam**total/math.factorial(total); mass += prob
        fw=fp=fl=0.0
        for part in parts:
            a,b,c=_component(total,part,side); fw+=a/len(parts); fp+=b/len(parts); fl+=c/len(parts)
        w+=prob*fw; p+=prob*fp; l+=prob*fl
    if mass<=0: return None
    return {"win_fraction":w/mass,"push_fraction":p/mass,"loss_fraction":l/mass}


def _fair(dist: dict[str,float]) -> float | None:
    if dist["win_fraction"]<=0: return None
    return max(1.0,(1.0-dist["push_fraction"])/dist["win_fraction"])


def _is_ft_total_corners(name: Any) -> bool:
    n=" ".join(str(name or "").lower().split())
    if "corner" not in n: return False
    if any(token in n for token in ("team total","home team","away team","first half","1st half","second half","2nd half","race","exact","handicap")): return False
    return "over/under" in n or "total" in n or n in {"corners","corner kicks"}


def _selection(value: Any) -> tuple[str|None,float|None]:
    text=" ".join(str(value or "").strip().lower().split())
    m=re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b",text)
    if not m: return None,None
    return m.group(1).upper(),float(m.group(2))


def _pair_fair(op: float, up: float) -> tuple[float,float]:
    oi=1/op; ui=1/up; t=oi+ui; return oi/t,ui/t


def _observed(event: dict[str,Any], lam: float) -> tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    market=event.get("market") if isinstance(event.get("market"),dict) else {}
    provenance=event.get("market_provenance") if isinstance(event.get("market_provenance"),dict) else {}
    fresh=provenance.get("fresh") is True; source=provenance.get("source") or "NOT_VERIFIED"
    rows=[]; unsupported=[]
    for group in market.get("markets") or []:
        if not isinstance(group,dict) or not _is_ft_total_corners(group.get("market")): continue
        parsed={}
        for value in group.get("values") or []:
            if not isinstance(value,dict): continue
            side,line=_selection(value.get("selection")); price=_num(value.get("price"))
            if side is None or line is None or price is None or price<=1 or not _line_supported(line):
                unsupported.append({"bookmaker":group.get("bookmaker"),"market":group.get("market"),"selection":value.get("selection"),"reason":"UNPARSED_OR_UNSUPPORTED_FT_CORNERS_TOTAL"}); continue
            parsed[(side,round(line,2))]=price
        for side,line in sorted(parsed):
            price=parsed[(side,line)]; dist=_settlement(lam,line,side)
            if dist is None: continue
            fair=_fair(dist); counterpart=parsed.get(("UNDER" if side=="OVER" else "OVER",line)); p_market=None; basis="NOT_CALCULATED_FOR_QUARTER_SETTLEMENT"
            if counterpart is not None and abs(line*2-round(line*2))<1e-8:
                op=parsed.get(("OVER",line)); up=parsed.get(("UNDER",line))
                if op and up:
                    fo,fu=_pair_fair(op,up); p_market=fo if side=="OVER" else fu
                    basis="NO_VIG_CONDITIONAL_NON_PUSH" if abs(line-round(line))<1e-8 else "NO_VIG_BINARY_HALF_LINE"
            rows.append({
                "selection":side,"line":line,"split_components":_split(line),"total_lambda_corners":round(lam,6),
                "win_fraction_model":round(dist["win_fraction"],6),"push_fraction_model":round(dist["push_fraction"],6),"loss_fraction_model":round(dist["loss_fraction"],6),
                "fair_decimal_model":round(fair,4) if fair else None,"bookmaker":group.get("bookmaker"),"market":group.get("market"),"decimal_price":round(price,4),"provider_update":group.get("provider_update"),
                "p_market_fair":round(p_market,6) if p_market is not None else None,"market_fair_basis":basis,
                "raw_ev":round(dist["win_fraction"]*price+dist["push_fraction"]-1,6),"market_source":source,"market_fresh":fresh,
                "research_only":True,"actionable":False,"decision_weight":0.0,"classification":"RESEARCH_ONLY","promotion_block":"CORNERS_MODEL_NOT_PRODUCTION_APPROVED",
            })
    rows.sort(key=lambda r:(1 if r.get("market_fresh") else 0,float(r.get("raw_ev") or -999)),reverse=True)
    return rows[:32],unsupported[:32]


def build(event: dict[str,Any], registry: dict[str,Any]|None) -> dict[str,Any]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {}
    model=corners_rate_registry.model_fixture(fixture,registry)
    if not model:
        return {"schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"NOT_MODELED_THIS_TICK","actionable":False,"decision_weight":0.0,"reason":"CORNERS_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT"}
    rows,unsupported=_observed(event,float(model["total_lambda"]))
    return {
        "schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"LIVE_RESEARCH_MODELED","model":model["model"],"model_inputs":model,
        "observed_market_rows":rows,"observed_market_count":len(rows),"unsupported_market_rows":unsupported,"actionable":False,"decision_weight":0.0,"production_status":"LIVE_RESEARCH_NOT_ACTIONABLE",
        "formation_challenger_status":"OFFLINE_VALIDATED_SEPARATELY_NOT_APPLIED_LIVE",
        "calibration_gate":{"minimum_oos_evaluations":150,"minimum_formation_adjusted_evaluations":100,"requires":["lower corners MAE","lower Brier/log-loss relevant lines","stable league lift","verified price/true-CLV evidence","territorial features when available"]},
        "policy":"SPORT-FIRST TEAM/LEAGUE CORNERS MODEL; EXACT OBSERVED FT TOTAL CORNERS ONLY; SETTLEMENT-AWARE; FORMATION CHALLENGER ZERO LIVE WEIGHT; NO BET_LEAN_GALAXY PROMOTION",
    }


def attach(payload: dict[str,Any]) -> dict[str,int|bool]:
    registry=corners_rate_registry.load_registry(); modeled=observed_events=observed_rows=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH" or event.get("stage")=="POSTGAME": continue
        intel=build(event,registry); event["corners_intelligence"]=intel
        if intel.get("status")=="LIVE_RESEARCH_MODELED": modeled+=1
        count=int(intel.get("observed_market_count") or 0)
        if count: observed_events+=1; observed_rows+=count
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict):
            mi["areas"]["corners"]= {"status":intel.get("status"),"model_inputs":intel.get("model_inputs"),"observed_market_rows":intel.get("observed_market_rows") or [],"formation_challenger_status":intel.get("formation_challenger_status"),"actionable":False,"decision_weight":0.0,"calibration_gate":intel.get("calibration_gate")}
    return {"corners_rate_registry_loaded":bool(registry),"modeled_events":modeled,"events_with_observed_ft_corners_markets":observed_events,"observed_ft_corners_rows":observed_rows,"provider_requests_added":0,"state_registry_reads_max":1}


def saved_match_rates(events: Any, fixture: dict[str, Any]) -> dict[str, Any]:
    """Private display projection: read stored FT rates, not quotes or FE11 means."""
    missing = {"status": "NOT_VERIFIED", "reason": "NO_MATCHED_PREGAME_CORNERS_RATES",
               "bet_eligible": False}

    def utc(value: Any) -> datetime | None:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None
        except (ValueError, OverflowError):
            return None

    keys = ("fixture_id", "home_team_id", "away_team_id", "league_id", "season")
    start = utc(fixture.get("kickoff"))
    if (start is None or not isinstance(events, list)
            or not all(type(fixture.get(k)) is int and fixture[k] > 0 for k in keys)
            or fixture["home_team_id"] == fixture["away_team_id"]):
        return missing
    model = "LEAGUE_TEAM_CORNERS_POISSON_LIVE_v0.1"
    now = datetime.now(timezone.utc)
    eligible = []
    for stored in events[:12]:
        if not isinstance(stored, dict):
            continue
        stamp = utc(stored.get("generated_at"))
        stage = stored.get("stage")
        if (stamp is None or stamp >= start or stamp > now or not isinstance(stage, str)
                or stage in {"HT", "LIVE", "POSTGAME"} or stored.get("event_type") != "SOCCER_REFRESH"):
            continue
        event = stored.get("payload")
        if not isinstance(event, dict) or event.get("stage") != stage or event.get("event_type") != "SOCCER_REFRESH":
            continue
        fx, projection = event.get("fixture"), event.get("corners_intelligence")
        if (not isinstance(fx, dict) or fx.get("status") != "NS"
                or any(fx.get(k) != fixture[k] for k in keys) or utc(fx.get("kickoff")) != start
                or not isinstance(projection, dict) or projection.get("fixture_id") != fixture["fixture_id"]
                or projection.get("stage") != stage or projection.get("model") != model
                or projection.get("status") != "LIVE_RESEARCH_MODELED"
                or projection.get("production_status") != "LIVE_RESEARCH_NOT_ACTIONABLE"
                or projection.get("actionable") is not False or projection.get("decision_weight") != 0):
            continue
        inputs = projection.get("model_inputs")
        if not isinstance(inputs, dict) or inputs.get("model") != model:
            continue
        h, a, total = (inputs.get(k) for k in ("home_lambda", "away_lambda", "total_lambda"))
        if not all(type(x) in (int, float) and 1 <= x <= 10 for x in (h, a)):
            continue
        if type(total) not in (int, float) or not math.isfinite(total) or abs(h + a - total) > 1e-6:
            continue
        sample = {k: inputs[k] for k in ("prior_pool_n", "prior_home_home_matches", "prior_away_away_matches")
                  if type(inputs.get(k)) is int and inputs[k] >= 0}
        eligible.append((stamp, h, a, total, sample))
    if not eligible:
        return missing
    stamp, h, a, total, sample = max(eligible, key=lambda x: x[0])
    return {"status": "PERSISTED_RESEARCH_RATES", "fixture_id": fixture["fixture_id"],
            "source": "POSTGRES_REFRESH_EVENTS_FT_CORNERS", "model": model,
            "captured_at_utc": stamp.isoformat(), "kickoff_utc": start.isoformat(),
            "home_lambda": h, "away_lambda": a, "total_lambda": total, "sample": sample,
            "oos_certified_by_this_view": False, "bet_eligible": False}
