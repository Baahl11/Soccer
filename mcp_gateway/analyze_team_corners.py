from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Any

LINES=(3.5,4.5,5.5)


def poisson_over(lam: float,line: float)->float:
    threshold=int(math.floor(line))+1
    return max(0.0,min(1.0,1.0-sum(math.exp(-lam)*lam**k/math.factorial(k) for k in range(threshold))))


def logloss(p:float,y:int)->float:
    q=min(max(p,1e-9),1-1e-9); return -(y*math.log(q)+(1-y)*math.log(1-q))


def summarize(rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows:return {"n":0,"brier":None,"log_loss":None,"mean_probability":None,"observed_rate":None}
    n=len(rows); return {"n":n,"brier":round(sum((r["p_over"]-r["actual_over"])**2 for r in rows)/n,6),"log_loss":round(sum(logloss(r["p_over"],r["actual_over"]) for r in rows)/n,6),"mean_probability":round(sum(r["p_over"] for r in rows)/n,6),"observed_rate":round(sum(r["actual_over"] for r in rows)/n,6)}


def main()->None:
    ap=argparse.ArgumentParser(description="Validate team-corners probabilities from walk-forward corners baseline lambdas.")
    ap.add_argument("--baseline-report",default="soccer_edge_state/analysis/corners_baseline.json")
    ap.add_argument("--output",default="soccer_edge_state/analysis/team_corners_validation.json")
    args=ap.parse_args()
    try:
        report=json.load(open(args.baseline_report,encoding="utf-8"))
    except Exception: report={}
    rows=[]
    for ev in report.get("evaluations") or []:
        if not isinstance(ev,dict):continue
        for role,lk,ak in (("HOME","baseline_home_lambda","actual_home_corners"),("AWAY","baseline_away_lambda","actual_away_corners")):
            try: lam=float(ev[lk]); actual=float(ev[ak])
            except (KeyError,TypeError,ValueError):continue
            for line in LINES:
                p=poisson_over(lam,line); rows.append({"fixture_id":ev.get("fixture_id"),"league_id":ev.get("league_id"),"team_role":role,"line":line,"lambda":round(lam,6),"p_over":round(p,6),"actual_over":int(actual>line)})
    by_role_line:dict[str,list[dict[str,Any]]]=defaultdict(list); by_league:dict[str,list[dict[str,Any]]]=defaultdict(list)
    for row in rows:
        by_role_line[f"{row['team_role']}|{row['line']}"] .append(row); by_league[str(row.get("league_id") or "UNKNOWN")].append(row)
    unique_fixtures=len({r["fixture_id"] for r in rows})
    out={"schema_version":"1.0.0","status":"RESEARCH_ONLY_TEAM_CORNERS_VALIDATION","source":"WALK_FORWARD_BASELINE_HOME_AWAY_CORNER_LAMBDAS","evaluated_fixtures":unique_fixtures,"evaluated_rows":len(rows),"overall":summarize(rows),"by_role_line":{k:summarize(v) for k,v in sorted(by_role_line.items())},"by_league":{k:summarize(v) for k,v in sorted(by_league.items())},"promotion_gate":{"enabled":False,"minimum_oos_team_rows":200,"minimum_oos_team_rows_for_actionable_review":400,"market_comparison_sample_gate_met":len(rows)>=200,"actionable_review_sample_gate_met":len(rows)>=400,"requires":["stable team-line Brier/log-loss","verified historical team-corners prices and true CLV","league/venue stability","parent corners model adequate"]},"notes":["Validation uses walk-forward home/away lambdas already produced by the FT corners baseline.","Diagnostic lines do not claim sportsbook availability; live market comparison requires exact observed lines.","No BET/LEAN/Galaxy promotion is permitted by this report."],"rows":rows[-1200:]}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh:json.dump(out,fh,ensure_ascii=False,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({k:out[k] for k in ("status","evaluated_fixtures","overall","promotion_gate")},indent=2))


if __name__=="__main__":main()
