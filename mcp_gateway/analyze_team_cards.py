from __future__ import annotations

import argparse,json,math,os
from collections import defaultdict
from typing import Any

LINES=(1.5,2.5,3.5)

def pois_over(lam:float,line:float)->float:
    t=int(math.floor(line))+1;return max(0.0,min(1.0,1.0-sum(math.exp(-lam)*lam**k/math.factorial(k) for k in range(t))))
def ll(p:float,y:int)->float:
    q=min(max(p,1e-9),1-1e-9);return -(y*math.log(q)+(1-y)*math.log(1-q))
def summ(rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows:return {"n":0,"brier":None,"log_loss":None,"mean_probability":None,"observed_rate":None}
    n=len(rows);return {"n":n,"brier":round(sum((r["p_over"]-r["actual_over"])**2 for r in rows)/n,6),"log_loss":round(sum(ll(r["p_over"],r["actual_over"]) for r in rows)/n,6),"mean_probability":round(sum(r["p_over"] for r in rows)/n,6),"observed_rate":round(sum(r["actual_over"] for r in rows)/n,6)}
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--baseline-report",default="soccer_edge_state/analysis/cards_baseline.json");ap.add_argument("--output",default="soccer_edge_state/analysis/team_cards_validation.json");args=ap.parse_args()
    try:report=json.load(open(args.baseline_report,encoding="utf-8"))
    except Exception:report={}
    rows=[]
    for ev in report.get("evaluations") or []:
        if not isinstance(ev,dict):continue
        for role,lk,ak in (("HOME","home_yellow_lambda","actual_home_yellow"),("AWAY","away_yellow_lambda","actual_away_yellow")):
            try:lam=float(ev[lk]);actual=float(ev[ak])
            except (KeyError,TypeError,ValueError):continue
            for line in LINES:
                p=pois_over(lam,line);rows.append({"fixture_id":ev.get("fixture_id"),"league_id":ev.get("league_id"),"team_role":role,"line":line,"lambda":round(lam,6),"p_over":round(p,6),"actual_over":int(actual>line)})
    groups=defaultdict(list);leagues=defaultdict(list)
    for r in rows:groups[f"{r['team_role']}|{r['line']}"] .append(r);leagues[str(r.get("league_id") or "UNKNOWN")].append(r)
    fixtures=len({r["fixture_id"] for r in rows})
    out={"schema_version":"1.0.0","status":"RESEARCH_ONLY_TEAM_YELLOW_CARDS_VALIDATION","source":"WALK_FORWARD_CARD_BASELINE_HOME_AWAY_LAMBDAS","evaluated_fixtures":fixtures,"evaluated_rows":len(rows),"overall":summ(rows),"by_role_line":{k:summ(v) for k,v in sorted(groups.items())},"by_league":{k:summ(v) for k,v in sorted(leagues.items())},"promotion_gate":{"enabled":False,"minimum_oos_team_rows":250,"minimum_oos_team_rows_for_actionable_review":500,"market_comparison_sample_gate_met":len(rows)>=250,"actionable_review_sample_gate_met":len(rows)>=500,"requires":["stable team-line Brier/log-loss","explicit team-yellow sportsbook price/true-CLV history","league/venue stability","sportsbook yellow-card settlement rule verified"]},"notes":["Yellow cards only; red cards are not included.","Diagnostic lines do not claim market availability.","No BET/LEAN/Galaxy promotion."],"rows":rows[-1200:]}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh:json.dump(out,fh,ensure_ascii=False,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({k:out[k] for k in ("status","evaluated_fixtures","overall","promotion_gate")},indent=2))
if __name__=="__main__":main()
