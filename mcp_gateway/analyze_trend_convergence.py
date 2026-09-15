from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

MIN_PRIOR=5
WINDOW=10

def num(v):
    try:return float(v)
    except (TypeError,ValueError):return None

def parse_dt(v):
    try:return datetime.fromisoformat(str(v).replace('Z','+00:00')) if v else None
    except ValueError:return None

def final(result):
    if not isinstance(result,dict):return None
    g=result.get('goals') or {}; ft=(result.get('score') or {}).get('fulltime') or {}
    try:return int(g.get('home',ft.get('home'))),int(g.get('away',ft.get('away')))
    except (TypeError,ValueError):return None

def tactical(result):return result.get('tactical_stats') if isinstance(result,dict) and isinstance(result.get('tactical_stats'),dict) else {}
def team_tac(tac,tid):
    for x in tac.get('teams') or []:
        if isinstance(x,dict) and str(x.get('team_id'))==str(tid):return x
    return {}
def load(history_dir):
    fx={}
    for path in sorted(glob.glob(os.path.join(history_dir,'*.jsonl'))):
        for line in open(path,encoding='utf-8'):
            try:t=json.loads(line)
            except Exception:continue
            gen=parse_dt(t.get('generated_at_local'))
            for e in t.get('events') or []:
                f=e.get('fixture') or {}; fid=f.get('fixture_id')
                if not fid:continue
                r=fx.setdefault(int(fid),{'fixture_id':int(fid),'projections':[]})
                for k in ('kickoff','league_id','season','home_team_id','home_team','away_team_id','away_team'):
                    if f.get(k) is not None:r[k]=f[k]
                if e.get('result'):r['result']=e['result']
                raw=e.get('raw_projection')
                if isinstance(raw,dict) and gen:
                    r['projections'].append((gen,{k:num(raw.get(k)) for k in ('raw_total_goals','raw_over_2_5_prob','raw_btts_yes_prob')}))
    return fx

def team_rows(fixtures):
    by=defaultdict(list)
    matches=[]
    for r in fixtures.values():
        ko=parse_dt(r.get('kickoff')); sc=final(r.get('result'))
        if not ko or not sc:continue
        hg,ag=sc;tac=tactical(r.get('result'));ht=team_tac(tac,r.get('home_team_id'));at=team_tac(tac,r.get('away_team_id'))
        matches.append((ko,r))
        for tid,name,venue,gf,ga,own,opp in ((r.get('home_team_id'),r.get('home_team'),'home',hg,ag,ht,at),(r.get('away_team_id'),r.get('away_team'),'away',ag,hg,at,ht)):
            by[tid].append({'kickoff':ko,'venue':venue,'gf':gf,'ga':ga,'btts':int(gf>0 and ga>0),'over25':int(gf+ga>=3),'corners_for':num(own.get('corners')),'corners_against':num(opp.get('corners')),'shots':num(own.get('total_shots')),'sot':num(own.get('shots_on_goal'))})
    for v in by.values():v.sort(key=lambda x:x['kickoff'])
    return by,sorted(matches,key=lambda x:x[0])
def mean(xs,key):
    v=[num(x.get(key)) for x in xs];v=[x for x in v if x is not None];return sum(v)/len(v) if v else None
def rate(xs,key):return mean(xs,key)
def prior(by,tid,ko,venue):
    allx=[x for x in by.get(tid,[]) if x['kickoff']<ko]
    vx=[x for x in allx if x['venue']==venue]
    return allx[-WINDOW:],vx[-WINDOW:]
def strength(sample,venue_sample):
    n=len(sample);vn=len(venue_sample)
    return min(1.0,n/WINDOW)*(0.7+0.3*min(1.0,vn/5))
def main():
    p=argparse.ArgumentParser();p.add_argument('--history-dir',default='soccer_edge_state/history');p.add_argument('--output',default='soccer_edge_state/analysis/trend_convergence.json');a=p.parse_args()
    fixtures=load(a.history_dir);by,matches=team_rows(fixtures);evaluated=[]
    for ko,r in matches:
        h,hv=prior(by,r.get('home_team_id'),ko,'home'); aw,av=prior(by,r.get('away_team_id'),ko,'away')
        if min(len(h),len(aw))<MIN_PRIOR:continue
        sc=final(r.get('result')); hg,ag=sc; signals=[]
        # Goals environment convergence. Require both sides, not one streak.
        ho=rate(h,'over25');ao=rate(aw,'over25');hb=rate(h,'btts');ab=rate(aw,'btts')
        if None not in (ho,ao) and ho>=.60 and ao>=.60:signals.append(('GOALS_OVER_CONVERGENCE',(ho+ao)/2))
        if None not in (ho,ao) and ho<=.40 and ao<=.40:signals.append(('GOALS_UNDER_CONVERGENCE',1-(ho+ao)/2))
        if None not in (hb,ab) and hb>=.60 and ab>=.60:signals.append(('BTTS_YES_CONVERGENCE',(hb+ab)/2))
        # Corner convergence only when both production and concession data exist.
        hcf=mean(h,'corners_for');aca=mean(aw,'corners_against');acf=mean(aw,'corners_for');hca=mean(h,'corners_against')
        corner_total=None
        if None not in (hcf,aca,acf,hca):
            corner_total=((hcf+aca)/2)+((acf+hca)/2)
            if corner_total>=10:signals.append(('CORNERS_HIGH_CONVERGENCE',min(1.0,corner_total/12)))
            elif corner_total<=8:signals.append(('CORNERS_LOW_CONVERGENCE',min(1.0,(10-corner_total)/4+.5)))
        proj=None
        valid=[x for x in r.get('projections') or [] if x[0] and x[0]<ko]
        if valid:proj=sorted(valid,key=lambda x:x[0])[-1][1]
        disagreements=[]
        if proj:
            po=proj.get('raw_over_2_5_prob');pb=proj.get('raw_btts_yes_prob')
            if po is not None:
                if ho is not None and ao is not None and ho>=.60 and ao>=.60 and po<=.45:disagreements.append('MODEL_UNDER_VS_TREND_OVER')
                if ho is not None and ao is not None and ho<=.40 and ao<=.40 and po>=.55:disagreements.append('MODEL_OVER_VS_TREND_UNDER')
            if pb is not None and hb is not None and ab is not None and hb>=.60 and ab>=.60 and pb<=.45:disagreements.append('MODEL_NO_BTTS_VS_TREND_BTTS')
        reliability=round(min(strength(h,hv),strength(aw,av)),4)
        evaluated.append({'fixture_id':r['fixture_id'],'kickoff':r.get('kickoff'),'home_team':r.get('home_team'),'away_team':r.get('away_team'),'prior_home_n':len(h),'prior_away_n':len(aw),'reliability':reliability,'signals':[{'type':s,'strength':round(v,4)} for s,v in signals],'disagreements':disagreements,'actual':{'home_goals':hg,'away_goals':ag,'over25':int(hg+ag>=3),'btts':int(hg>0 and ag>0)},'diagnostics':{'home_over25_rate':ho,'away_over25_rate':ao,'home_btts_rate':hb,'away_btts_rate':ab,'expected_corner_total_from_trends':corner_total},'model_snapshot':proj})
    sig=defaultdict(lambda:{'n':0,'hits':0})
    for x in evaluated:
        for s in x['signals']:
            k=s['type'];sig[k]['n']+=1
            y=x['actual'];hit=(k=='GOALS_OVER_CONVERGENCE' and y['over25']) or (k=='GOALS_UNDER_CONVERGENCE' and not y['over25']) or (k=='BTTS_YES_CONVERGENCE' and y['btts'])
            if k.startswith('CORNERS_'):hit=False # Outcome corners not reconstructed here; descriptive until verified join exists.
            sig[k]['hits']+=int(bool(hit))
    summary={k:{'n':v['n'],'observed_hit_rate':round(v['hits']/v['n'],4) if v['n'] and not k.startswith('CORNERS_') else None} for k,v in sorted(sig.items())}
    out={'schema_version':'2.0.0','status':'RESEARCH_ONLY_TREND_CONVERGENCE','generated_at_utc':datetime.now(timezone.utc).isoformat(),'decision_weight':0.0,'minimum_prior_matches_per_team':MIN_PRIOR,'window':WINDOW,'evaluated_matches':len(evaluated),'matches_with_any_signal':sum(bool(x['signals']) for x in evaluated),'matches_with_model_disagreement':sum(bool(x['disagreements']) for x in evaluated),'signal_summary':summary,'policy':['Strict walk-forward: only matches before target kickoff enter trends.','Convergence requires compatible evidence from both teams; a single streak is insufficient.','Disagreement is a research/deep-dive flag only and cannot upgrade BET/LEAN.','Corner signals remain descriptive until verified postgame corner outcomes are joined.','Thresholds are prospective heuristics and must not be tuned on this sample.'],'evaluations':evaluated[-500:]}
    os.makedirs(os.path.dirname(a.output),exist_ok=True);json.dump(out,open(a.output,'w',encoding='utf-8'),ensure_ascii=False,indent=2,default=str);print(json.dumps({k:v for k,v in out.items() if k!='evaluations'},indent=2))
if __name__=='__main__':main()
