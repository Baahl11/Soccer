from __future__ import annotations
import argparse,glob,json,os
from collections import defaultdict
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser();p.add_argument('--history-dir',default='soccer_edge_state/history');p.add_argument('--output',default='soccer_edge_state/analysis/player_trends.json');a=p.parse_args();seen={}
 for path in sorted(glob.glob(os.path.join(a.history_dir,'*.jsonl'))):
  for line in open(path,encoding='utf-8'):
   try:t=json.loads(line)
   except Exception:continue
   for e in t.get('events') or []:
    fid=(e.get('fixture') or {}).get('fixture_id')
    candidates=[e.get('postgame_player_stats'),(e.get('result') or {}).get('player_stats') if isinstance(e.get('result'),dict) else None,e.get('player_trends_research')]
    pr=next((x for x in candidates if isinstance(x,dict) and x.get('status')=='RESEARCH_ONLY_PLAYER_FIXTURE_STATS'),None)
    if not pr or not fid:continue
    seen[int(fid)]={'fixture_id':int(fid),'kickoff':(e.get('fixture') or {}).get('kickoff'),'teams':pr.get('teams') or [],'capture_phase':pr.get('capture_phase') or 'PREGAME'}
 rows=defaultdict(list);names={}
 for f in sorted(seen.values(),key=lambda x:str(x.get('kickoff') or '')):
  for team in f['teams']:
   for x in team.get('players') or []:
    pid=x.get('player_id')
    if not pid:continue
    names[pid]=x.get('name');r=dict(x);r['fixture_id']=f['fixture_id'];r['kickoff']=f.get('kickoff');r['team_id']=team.get('team_id');rows[pid].append(r)
 out=[]
 for pid,rs in rows.items():
  item={'player_id':pid,'name':names.get(pid),'matches_captured':len(rs),'windows':{}}
  for n in (5,10,20):
   z=rs[-n:];played=[r for r in z if (r.get('minutes') or 0)>0]
   def avg(k):
    v=[r.get(k) for r in played if r.get(k) is not None];return round(sum(v)/len(v),3) if v else None
   def hit(k,thr):
    v=[r.get(k) for r in played if r.get(k) is not None];return {'n':len(v),'hits':sum(float(q)>=thr for q in v),'rate':round(sum(float(q)>=thr for q in v)/len(v),4) if v else None}
   item['windows'][f'last_{n}']={'n':len(z),'played_n':len(played),'avg_minutes':avg('minutes'),'avg_rating':avg('rating'),'avg_shots':avg('shots'),'avg_sot':avg('shots_on_target'),'goals':sum(r.get('goals') or 0 for r in played),'assists':sum(r.get('assists') or 0 for r in played),'sot_1plus':hit('shots_on_target',1),'sot_2plus':hit('shots_on_target',2),'shots_2plus':hit('shots',2)}
  out.append(item)
 report={'schema_version':'1.1.0','status':'RESEARCH_ONLY_PLAYER_TRENDS','generated_at_utc':datetime.now(timezone.utc).isoformat(),'decision_weight':0.0,'unique_fixtures_captured':len(seen),'postgame_fixtures_captured':sum(x.get('capture_phase')=='POSTGAME' for x in seen.values()),'unique_players':len(out),'players_with_5plus_matches':sum(x['matches_captured']>=5 for x in out),'policy':['Only persisted verified fixture/player captures are used.','Postgame capture is preferred over pregame capture for a duplicate fixture.','Fixture IDs are deduplicated to avoid repeated scheduler snapshots.','Last-5/10/20 are descriptive until sufficient OOS validation exists.','No player trend can change classification, tier, stake, or bet eligibility.'],'players':sorted(out,key=lambda x:(-x['matches_captured'],str(x['name'])))}
 os.makedirs(os.path.dirname(a.output),exist_ok=True);json.dump(report,open(a.output,'w',encoding='utf-8'),ensure_ascii=False,indent=2);print(json.dumps({k:v for k,v in report.items() if k!='players'},indent=2))
if __name__=='__main__':main()
