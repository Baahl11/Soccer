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

def _team_role(name:Any,fixture:dict[str,Any])->str|None:
    n=_norm(name)
    if "yellow" not in n or not any(t in n for t in ("card","booking")):return None
    if any(t in n for t in ("player","first half","1st half","second half","2nd half")):return None
    home=_norm(fixture.get("home_team") or fixture.get("home_name"));away=_norm(fixture.get("away_team") or fixture.get("away_name"))
    h=("home team" in n or (home and home in n));a=("away team" in n or (away and away in n))
    if h and not a:return "HOME"
    if a and not h:return "AWAY"
    return None

def _selection(v:Any)->tuple[str|None,float|None]:
    m=re.search(r"\b(over|under)\s*([0-9]+(?:\.[0-9]+)?)\b",_norm(v))
    return (m.group(1).upper(),float(m.group(2))) if m else (None,None)

def _observed(event:dict[str,Any],model:dict[str,Any])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {};market=event.get("market") if isinstance(event.get("market"),dict) else {};prov=event.get("market_provenance") if isinstance(event.get("market_provenance"),dict) else {};rows=[];blocked=[]
    for group in market.get("markets") or []:
        if not isinstance(group,dict):continue
        role=_team_role(group.get("market"),fixture)
        if role is None:continue
        lam=float(model["home_yellow_lambda"] if role=="HOME" else model["away_yellow_lambda"]);parsed={}
        for value in group.get("values") or []:
            if not isinstance(value,dict):continue
            side,line=_selection(value.get("selection"));price=_num(value.get("price"))
            if side is None or line is None or price is None or price<=1 or not _line_supported(line):blocked.append({"team_role":role,"bookmaker":group.get("bookmaker"),"market":group.get("market"),"selection":value.get("selection"),"reason":"UNPARSED_OR_UNSUPPORTED_TEAM_YELLOW_CARDS"});continue
            parsed[(side,round(line,2))]=price
        for side,line in sorted(parsed):
            price=parsed[(side,line)];dist=_settlement(lam,line,side)
            if dist is None:continue
            fair=_fair(dist);p_market=None;basis="NOT_CALCULATED_FOR_QUARTER_SETTLEMENT";counter=parsed.get(("UNDER" if side=="OVER" else "OVER",line))
            if counter is not None and abs(line*2-round(line*2))<1e-8:
                op=parsed.get(("OVER",line));up=parsed.get(("UNDER",line))
                if op and up:fo,fu=_pair_fair(op,up);p_market=fo if side=="OVER" else fu;basis="NO_VIG_CONDITIONAL_NON_PUSH" if abs(line-round(line))<1e-8 else "NO_VIG_BINARY_HALF_LINE"
            rows.append({"target":"TEAM_YELLOW_CARDS_ONLY","team_role":role,"team":fixture.get("home_team") if role=="HOME" else fixture.get("away_team"),"selection":side,"line":line,"team_yellow_lambda":round(lam,6),"win_fraction_model":round(dist["win_fraction"],6),"push_fraction_model":round(dist["push_fraction"],6),"loss_fraction_model":round(dist["loss_fraction"],6),"fair_decimal_model":round(fair,4) if fair else None,"bookmaker":group.get("bookmaker"),"market":group.get("market"),"decimal_price":round(price,4),"provider_update":group.get("provider_update"),"p_market_fair":round(p_market,6) if p_market is not None else None,"market_fair_basis":basis,"raw_ev":round(dist["win_fraction"]*price+dist["push_fraction"]-1,6),"market_source":prov.get("source") or "NOT_VERIFIED","market_fresh":prov.get("fresh") is True,"red_cards_included":False,"research_only":True,"actionable":False,"decision_weight":0.0,"classification":"RESEARCH_ONLY","promotion_block":"TEAM_YELLOW_CARDS_NOT_OOS_CALIBRATED"})
    rows.sort(key=lambda r:(1 if r.get("market_fresh") else 0,float(r.get("raw_ev") or -999)),reverse=True);return rows[:32],blocked[:32]

def build(event:dict[str,Any],registry:dict[str,Any]|None)->dict[str,Any]:
    fixture=event.get("fixture") if isinstance(event.get("fixture"),dict) else {};model=cards_rate_registry.model_fixture(fixture,registry)
    if not model:return {"schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"NOT_MODELED_THIS_TICK","actionable":False,"decision_weight":0.0,"reason":"YELLOW_CARD_RATE_REGISTRY_NOT_AVAILABLE_OR_INSUFFICIENT"}
    rows,blocked=_observed(event,model)
    return {"schema_version":SCHEMA_VERSION,"fixture_id":fixture.get("fixture_id"),"stage":event.get("stage"),"status":"LIVE_RESEARCH_MODELED","model":"TEAM_YELLOW_CARD_POISSON_FROM_DISCIPLINE_LAMBDAS_v0.1","home_yellow_lambda":round(float(model["home_yellow_lambda"]),6),"away_yellow_lambda":round(float(model["away_yellow_lambda"]),6),"observed_market_rows":rows,"observed_market_count":len(rows),"blocked_rows":blocked,"actionable":False,"decision_weight":0.0,"production_status":"LIVE_RESEARCH_NOT_ACTIONABLE","calibration_gate":{"minimum_oos_team_rows":250,"minimum_oos_team_rows_for_actionable_review":500,"requires":["team-line Brier/log-loss","verified explicit team-yellow price/true-CLV history","league/venue stability","sportsbook yellow-card rule verified"]},"policy":"TEAM YELLOW CARDS ONLY; REDS SEPARATE; EXACT EXPLICIT TEAM-YELLOW MARKETS; ZERO DECISION WEIGHT"}

def attach(payload:dict[str,Any])->dict[str,int|bool]:
    registry=cards_rate_registry.load_registry();modeled=events=rows=0
    for event in payload.get("events") or []:
        if not isinstance(event,dict) or event.get("event_type")!="SOCCER_REFRESH" or event.get("stage")=="POSTGAME":continue
        intel=build(event,registry);event["team_cards_intelligence"]=intel
        if intel.get("status")=="LIVE_RESEARCH_MODELED":modeled+=1
        c=int(intel.get("observed_market_count") or 0)
        if c:events+=1;rows+=c
        mi=event.get("match_intelligence")
        if isinstance(mi,dict) and isinstance(mi.get("areas"),dict):mi["areas"]["team_cards"]={"status":intel.get("status"),"home_yellow_lambda":intel.get("home_yellow_lambda"),"away_yellow_lambda":intel.get("away_yellow_lambda"),"observed_market_rows":intel.get("observed_market_rows") or [],"actionable":False,"decision_weight":0.0,"calibration_gate":intel.get("calibration_gate")}
    return {"cards_rate_registry_loaded":bool(registry),"modeled_events":modeled,"events_with_observed_team_yellow_markets":events,"observed_team_yellow_rows":rows,"provider_requests_added":0}
