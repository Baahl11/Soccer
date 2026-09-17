from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any

from mcp_gateway import analyze_cards_baseline as cards


def blank()->dict[str,float|int]: return {"n":0,"home_yellow":0.0,"away_yellow":0.0}
def add(row:dict[str,float|int],h:float,a:float)->None:
    row["n"]=int(row["n"])+1; row["home_yellow"]=float(row["home_yellow"])+h; row["away_yellow"]=float(row["away_yellow"])+a

def main()->None:
    ap=argparse.ArgumentParser(description="Build compact finalized-history yellow-card registry for live research.")
    ap.add_argument("--history-dir",default="soccer_edge_state/history"); ap.add_argument("--output",default="soccer_edge_state/analysis/cards_rate_registry.json"); args=ap.parse_args()
    rows=cards.load_rows(args.history_dir); global_row=blank(); leagues=defaultdict(blank); home_teams=defaultdict(blank); away_teams=defaultdict(blank); referees=defaultdict(lambda:{"n":0,"total_yellow":0.0})
    for r in rows:
        h=float(r["home_yellow"]); a=float(r["away_yellow"]); add(global_row,h,a)
        if r.get("league_id") is not None:add(leagues[str(r["league_id"])],h,a)
        add(home_teams[str(r["home_team_id"])],h,a); add(away_teams[str(r["away_team_id"])],h,a)
        ref=str(r.get("referee") or "").strip()
        if ref: referees[ref]["n"]+=1; referees[ref]["total_yellow"]+=h+a
    out={"schema_version":"1.0.0","status":"RESEARCH_YELLOW_CARD_RATE_REGISTRY","verified_postgame_fixtures":len(rows),"league_pseudo_n":25.0,"team_ratio_pseudo_n":8.0,"referee_pseudo_n":12.0,"global":dict(global_row),"leagues":dict(leagues),"home_teams":dict(home_teams),"away_teams":dict(away_teams),"referees":dict(referees),"policy":"YELLOW CARDS ONLY; RED CARDS STORED/RESEARCHED SEPARATELY; NO SPORTSBOOK CARD-POINT RULE IS ASSUMED"}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh:json.dump(out,fh,ensure_ascii=False,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({"status":out["status"],"verified_postgame_fixtures":len(rows),"referees":len(referees)},indent=2))

if __name__=="__main__":main()
