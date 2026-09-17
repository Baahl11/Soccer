from __future__ import annotations

import argparse,json,os
from collections import defaultdict
from typing import Any
from mcp_gateway import analyze_cards_baseline as cards

def blank()->dict[str,int]:return {"n":0,"home_red_events":0,"away_red_events":0,"any_red_events":0}
def add(row:dict[str,int],hr:float,ar:float)->None:
    row["n"]+=1;row["home_red_events"]+=int(hr>0);row["away_red_events"]+=int(ar>0);row["any_red_events"]+=int(hr>0 or ar>0)
def main()->None:
    ap=argparse.ArgumentParser();ap.add_argument("--history-dir",default="soccer_edge_state/history");ap.add_argument("--output",default="soccer_edge_state/analysis/red_card_registry.json");args=ap.parse_args()
    rows=cards.load_rows(args.history_dir);g=blank();leagues=defaultdict(blank);home=defaultdict(blank);away=defaultdict(blank);refs=defaultdict(lambda:{"n":0,"any_red_events":0})
    for r in rows:
        hr=float(r.get("home_red") or 0);ar=float(r.get("away_red") or 0);add(g,hr,ar);add(leagues[str(r.get("league_id"))],hr,ar);add(home[str(r.get("home_team_id"))],hr,ar);add(away[str(r.get("away_team_id"))],hr,ar)
        ref=str(r.get("referee") or "").strip()
        if ref:refs[ref]["n"]+=1;refs[ref]["any_red_events"]+=int(hr>0 or ar>0)
    out={"schema_version":"1.0.0","status":"RESEARCH_RED_CARD_REGISTRY","verified_postgame_fixtures":len(rows),"global":dict(g),"leagues":dict(leagues),"home_teams":dict(home),"away_teams":dict(away),"referees":dict(refs),"league_prior_games":50.0,"team_prior_games":25.0,"referee_prior_games":30.0,"policy":"RED CARDS MODELED SEPARATELY AS LOW-FREQUENCY EVENTS; YELLOW-CARD LAMBDA IS NOT REUSED"}
    os.makedirs(os.path.dirname(args.output),exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh:json.dump(out,fh,ensure_ascii=False,indent=2,sort_keys=True);fh.write("\n")
    print(json.dumps({"status":out["status"],"verified_postgame_fixtures":len(rows),"red_events":g["any_red_events"]},indent=2))
if __name__=="__main__":main()
