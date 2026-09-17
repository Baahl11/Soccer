from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

WINDOWS=(5,10,20)


def fnum(v):
    try:return float(str(v).replace('%',''))
    except (TypeError,ValueError):return None

def dt(v):
    try:return datetime.fromisoformat(str(v).replace('Z','+00:00')) if v else None
    except ValueError:return None

def score(result):
    if not isinstance(result,dict):return None
    g=result.get('goals') or {}; s=(result.get('score') or {}).get('fulltime') or {}
    try:return int(g.get('home',s.get('home'))),int(g.get('away',s.get('away')))
    except (TypeError,ValueError):return None

def half_score(result):
    try:
        h=(result.get('score') or {}).get('halftime') or {}; return int(h.get('home')),int(h.get('away'))
    except (AttributeError,TypeError,ValueError):return None

def tactical(result):
    if not isinstance(result,dict):return {}
    return result.get('tactical_stats') if isinstance(result.get('tactical_stats'),dict) else {}

def team_stats(tac,tid):
    for x in tac.get('teams') or []:
        if isinstance(x,dict) and str(x.get('team_id'))==str(tid):return x
    return {}

def load(history_dir):
    fixtures={}
    for path in sorted(glob.glob(os.path.join(history_dir,'*.jsonl'))):
        for line in open(path,encoding='utf-8'):
            try: tick=json.loads(line)
            except Exception:continue
            for e in tick.get('events') or []:
                fx=e.get('fixture') or {}; fid=fx.get('fixture_id')
                if not fid:continue
                r=fixtures.setdefault(int(fid),{'fixture_id':int(fid)})
                for k in ('kickoff','league_id','league','season','home_team_id','home_team','away_team_id','away_team'):
                    if fx.get(k) is not None:r[k]=fx.get(k)
                if e.get('result'):r['result']=e['result']
    return fixtures

def rows(fixtures):
    out=[]
    for r in fixtures.values():
        sc=score(r.get('result')); ko=dt(r.get('kickoff'))
        if not sc or not ko:continue
        hg,ag=sc; hs=half_score(r.get('result')); tac=tactical(r.get('result'))
        hts=team_stats(tac,r.get('home_team_id')); ats=team_stats(tac,r.get('away_team_id'))
        base={'fixture_id':r['fixture_id'],'kickoff':ko,'league_id':r.get('league_id'),'season':r.get('season'),'match_goals':hg+ag,'btts':int(hg>0 and ag>0),'over25':int(hg+ag>=3),'first_half_goals':sum(hs) if hs else None}
        for side,tid,name,gf,ga,ts,venue in [('home',r.get('home_team_id'),r.get('home_team'),hg,ag,hts,'home'),('away',r.get('away_team_id'),r.get('away_team'),ag,hg,ats,'away')]:
            opp_ts=(ats if side=='home' else hts)
            x=dict(base); x.update({'team_id':tid,'team':name,'venue':venue,'gf':gf,'ga':ga,'win':int(gf>ga),'draw':int(gf==ga),'scored':int(gf>0),'clean_sheet':int(ga==0),'team_corners':fnum(ts.get('corners')),'opp_corners':fnum(opp_ts.get('corners')),'team_shots':fnum(ts.get('total_shots')),'team_sot':fnum(ts.get('shots_on_goal')),'opp_shots':fnum(opp_ts.get('total_shots')),'opp_sot':fnum(opp_ts.get('shots_on_goal')),'possession':fnum(ts.get('possession')),'yellow_cards':fnum(ts.get('yellow_cards'))})
            out.append(x)
    return sorted(out,key=lambda x:(x['kickoff'],x['fixture_id'],x['team_id'] or 0))
def avg(xs,key):
    v=[fnum(x.get(key)) for x in xs];v=[x for x in v if x is not None];return round(sum(v)/len(v),4) if v else None
def rate(xs,key):return avg(xs,key)
def summarize(xs):
    return {'n':len(xs),'win_rate':rate(xs,'win'),'draw_rate':rate(xs,'draw'),'scored_rate':rate(xs,'scored'),'clean_sheet_rate':rate(xs,'clean_sheet'),'btts_rate':rate(xs,'btts'),'over_2_5_rate':rate(xs,'over25'),'avg_goals_for':avg(xs,'gf'),'avg_goals_against':avg(xs,'ga'),'avg_match_goals':avg(xs,'match_goals'),'avg_1h_goals':avg(xs,'first_half_goals'),'avg_team_corners':avg(xs,'team_corners'),'avg_opponent_corners':avg(xs,'opp_corners'),'avg_team_shots':avg(xs,'team_shots'),'avg_team_sot':avg(xs,'team_sot'),'avg_opponent_shots':avg(xs,'opp_shots'),'avg_opponent_sot':avg(xs,'opp_sot'),'avg_possession':avg(xs,'possession'),'avg_yellow_cards':avg(xs,'yellow_cards')}
def main():
    p=argparse.ArgumentParser();p.add_argument('--history-dir',default='soccer_edge_state/history');p.add_argument('--output',default='soccer_edge_state/analysis/trend_intelligence.json');a=p.parse_args()
    rr=rows(load(a.history_dir)); by=defaultdict(list)
    for x in rr:by[x['team_id']].append(x)
    teams=[]
    for tid,xs in by.items():
        item={'team_id':tid,'team':xs[-1]['team'],'matches_available':len(xs),'windows':{},'venue_windows':{}}
        for n in WINDOWS:
            item['windows'][f'last_{n}']=summarize(xs[-n:])
            venue=xs[-1]['venue']; vx=[x for x in xs if x['venue']==venue]
            item['venue_windows'][f'{venue}_last_{n}']=summarize(vx[-n:])
        teams.append(item)
    result={'schema_version':'1.2.0','status':'RESEARCH_ONLY_TREND_INTELLIGENCE','generated_at_utc':datetime.now(timezone.utc).isoformat(),'decision_weight':0.0,'finalized_team_match_rows':len(rr),'global_context':{'avg_team_goals':avg(rr,'gf'),'avg_goals_allowed':avg(rr,'ga'),'avg_team_shots':avg(rr,'team_shots'),'avg_team_sot':avg(rr,'team_sot'),'avg_shots_allowed':avg(rr,'opp_shots'),'avg_sot_allowed':avg(rr,'opp_sot')},'teams':sorted(teams,key=lambda x:(-x['matches_available'],str(x['team']))),'windows':list(WINDOWS),'policy':['Descriptive trends never upgrade BET/LEAN.','Missing statistics remain null and are not imputed.','Venue splits are explicit; no H2H causal claims.','Promotion requires walk-forward incremental lift versus the corresponding baseline model.','Player trends are a separate future layer and require persisted fixture-player statistics.'],'data_quality':{'rows_with_corners':sum(x['team_corners'] is not None for x in rr),'rows_with_shots':sum(x['team_shots'] is not None for x in rr),'rows_with_sot':sum(x['team_sot'] is not None for x in rr),'rows_with_opponent_shots':sum(x['opp_shots'] is not None for x in rr),'rows_with_opponent_sot':sum(x['opp_sot'] is not None for x in rr),'rows_with_possession':sum(x['possession'] is not None for x in rr)}}
    os.makedirs(os.path.dirname(a.output),exist_ok=True);json.dump(result,open(a.output,'w',encoding='utf-8'),ensure_ascii=False,indent=2,default=str);print(json.dumps({k:v for k,v in result.items() if k!='teams'},indent=2))
if __name__=='__main__':main()
