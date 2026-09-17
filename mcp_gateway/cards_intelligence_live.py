from __future__ import annotations

import re
from typing import Any

from mcp_gateway import cards_rate_registry
from mcp_gateway.corners_intelligence import _fair, _line_supported, _pair_fair, _settlement, _split

SCHEMA_VERSION="1.0.0"


def _num(v:Any)->float|None:
    try:return float(v)
    except (TypeError,ValueError):return None


def _norm(v:Any)->str:return " ".join(str(v or "").strip().lower().split())


def _market_kind(name:Any)->str:
    n=_norm(name)
    if not any(t in n for t in ("card","booking")):return "NOT_CARD"
    if any(t in n for t in ("player","team total","home team","away team","first half","1st half","second half","2nd half")):return "OTHER_CARD_FAMILY"
    if "yellow" in n and any(t in n for t in ("over/under","total","cards")):return "EXPLICIT_TOTAL_YELLOW"
    return "AMBIGUOUS_CARD_SCORING_RULE"


def _selection(v:Any)->tuple[str|None,float|None]:
    text=_norm(v);m=re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b",text)
    if not m:return None,None
    return m.group(1).upper(),float(m.group(2))


def _observed(event:dict[str,Any],lam:float)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    market=event.get("market") if isinstance(event.get("market"),dict) else {};prov=event.get("market_provenance") if isinstance(event.get("market_provenance"),dict) else {};rows=[];blocked=[]
    for group in market.get("markets") or []:
        if not isinstance(group,dict):continue
        kind=_market_kind(group.get("market"))
        if kind in {"NOT_CARD","OTHER_CARD_FAMILY"}:continue
        if kind=="AMBIGUOUS_CARD_SCORING_RULE":
            blocked.append({"bookmaker":group.get("bookmaker"),"market":group.get("market"),"reason":"SPORTSBOOK_CARD_SCORING_RULE_NOT_EXPLICIT; DO_NOT_MAP_GENERIC_CARDS_TO_YELLOW_CARDS"});continue
        parsed={}
        for value in group.get("values") or []:
            if not isinstance(value,dict):continue
            side,line=_selection(value.get("selection"));price=_num(value.get("price"))
            if side is None or line is None or price is None or price<=1 or not _line_supported(line):
                blocked.append({"bookmaker":group.get("bookmaker"),"market":group.get("market"),"selection":value.get("selection"),"reason":"UNPARSED_OR_UNSUPPORTED_YELLOW_CARD_TOTAL"});continue
            parsed[(side,round(line,2))]=price
        for side,line in sorted(parsed):
            price=parsed[(side,line)];dist=_settlement(lam,line,side)
            if dist is None:continue
            fair=_fair(dist);p_market=None;basis="NOT_CALCULATED_FOR_QUARTER_SETTLEMENT";counterpart=parsed.get(("UNDER" if side=="OVER" else "OVER",line))
            if counterpart is not None and abs(line*2-round(line*2))<1e-8:
                op=parsed.get(("OVER",line));up=parsed.get(("UNDER",line))
                if op and up:
                    fo,fu=_pair_fair(op,up);p_market=fo if side=="OVER" else fu;basis="NO_VIG_CONDITIONAL_NON_PUSH" if abs(line-round(line))<1e-8 else "NO_VIG_BINARY_HALF_LINE"
            rows.append({"target":"TOTAL_YELLOW_CARDS_ONLY","selection":side,"line":line,"split_components":_split(line),"total_yellow_lambda":round(lam,6),"win_fraction_model":round(dist["win_fraction"],6),"push_fraction_model":round(dist["push_fraction"],6),"loss_fraction_model":round(dist["loss_fraction"],6),"fair_decimal_model":round(fair,4) if fair else None,"bookmaker":group.get("bookmaker"),"market":group.get("market"),"decimal_price":round(price,4),"provider_update":group.get("provider_update"),"p_market_fair":round(p_market,6) if p_market is not None else None,"market_fair_basis":basis,"raw_ev":round(dist["win_fraction"]*price+dist["push_fraction"]-1,6),"market_source":prov.get("source") or "NOT_VERIFIED","market_fresh":prov.get("fresh") is True,"red_cards_included":False,"research_only":True,"actionable":False,"decision_weight":0.0,"classification":"RESEARCH_ONLY","promotion_block":"YELLOW_CARD_MODEL_AND_BOOK_RULE_MAPPING_NOT_PRODUCTION_APPROVED"})
    rows.sort(key=lambda r:(1 if r.get("market_fresh") else 0,float(r.get("raw_ev") or -999)),reverse=True);return rows[:32],blocked[:40]


def build(event:dict[str,Any],registry:dict[str,Any]|None)->dict[str,Any]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {};model=cards_rate_registry.model_fixture(fixture,registry)
    if not model:return {"schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"NOT_MODELED_THIS_TICK","actionable":False,"decision_weight":0.0,"reason":"YELLOW_CARD_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT"}
    rows,blocked=_observed(event,float(model["total_yellow_lambda"]))
    return {"schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"LIVE_RESEARCH_MODELED","model":model["model"],"model_inputs":model,"observed_explicit_yellow_market_rows":rows,"observed_explicit_yellow_market_count":len(rows),"blocked_ambiguous_card_markets":blocked,"actionable":False,"decision_weight":0.0,"production_status":"LIVE_RESEARCH_NOT_ACTIONABLE","referee_adjustment_applied":bool(model.get("referee_prior_n",0)>=8),"calibration_gate":{"minimum_oos_evaluations":200,"minimum_referee_adjusted_evaluations":100,"requires":["stable Brier/log-loss","stable league performance","verified current referee","explicit sportsbook yellow-card scoring rules","verified price/true-CLV evidence"]},"policy":"YELLOW CARDS ONLY; REDS SEPARATE; GENERIC CARDS/BOOKINGS NOT PRICED WITHOUT BOOK RULE; EXACT OBSERVED EXPLICIT-YELLOW MARKETS ONLY; ZERO DECISION WEIGHT"}


def attach(payload:dict[str,Any])->dict[str,int|bool]:
    registry=cards_rate_registry.load_registry();modeled=priced_events=priced_rows=blocked=ref_adjusted=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH" or event.get("stage")=="POSTGAME":continue
        intel=build(event,registry);event["cards_intelligence_live"]=intel
        if intel.get("status")=="LIVE_RESEARCH_MODELED":modeled+=1
        if intel.get("referee_adjustment_applied"):ref_adjusted+=1
        count=int(intel.get("observed_explicit_yellow_market_count") or 0)
        if count:priced_events+=1;priced_rows+=count
        blocked+=len(intel.get("blocked_ambiguous_card_markets") or [])
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict):mi["areas"]["cards"]={"status":intel.get("status"),"model_inputs":intel.get("model_inputs"),"observed_explicit_yellow_market_rows":intel.get("observed_explicit_yellow_market_rows") or [],"blocked_ambiguous_card_markets":intel.get("blocked_ambiguous_card_markets") or [],"referee_adjustment_applied":intel.get("referee_adjustment_applied"),"actionable":False,"decision_weight":0.0,"calibration_gate":intel.get("calibration_gate")}
    return {"cards_rate_registry_loaded":bool(registry),"modeled_events":modeled,"referee_adjusted_events":ref_adjusted,"events_with_explicit_yellow_markets":priced_events,"observed_explicit_yellow_rows":priced_rows,"blocked_ambiguous_card_market_groups":blocked,"provider_requests_added":0,"state_registry_reads_max":1}
