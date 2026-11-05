import unittest
from datetime import datetime,timezone,timedelta
from sports_edge.refresh_batch import refresh_due_shortlist

NOW=datetime(2026,11,5,17,5,tzinfo=timezone.utc)
def dataset():
    slate={'eligible_slate_verified':True,'eligible_fixtures':[]}
    batch={'source_slate_complete_verified':True,'all_fixtures_attempted':True,'complete_sport_screen':True,
           'shortlists':{'merged_fixture_ids':[1,2,3,4,5,6]},'screened_fixtures':[]}
    for i,minute in [(1,40),(2,40),(3,20),(4,120),(5,-1),(6,40),(7,40)]:
        slate['eligible_fixtures'].append({'fixture_id':i,'status':'NS','kickoff':(NOW+timedelta(minutes=minute)).isoformat()})
        batch['screened_fixtures'].append({'fixture_id':i,'min_sample_met':True})
    return slate,batch
class BatchRefreshTests(unittest.TestCase):
    def test_only_in_window_and_screened(self):
        slate,batch=dataset();calls=[]
        def fake(b,fid,p,**kwargs):
            calls.append(fid);return {'fixture_id':fid,'refresh_stage':'T-40','reason':'TEST','provider_lineups':{},'report_file':'test.json'}
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,refresh_fn=fake)
        self.assertEqual(calls,[3,1,2,6]);self.assertEqual(r['attempted'],4)
        self.assertEqual(r['actionable_bets'],0)
    def test_blocks_if_source_incomplete_without_queries(self):
        slate,batch=dataset();batch['source_slate_complete_verified']=False
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,refresh_fn=lambda *a,**k:self.fail('network call'))
        self.assertEqual(r['status'],'BLOCKED_INCOMPLETE_SPORT_FIRST_SCREEN')
    def test_only_up_to_five(self):
        slate,batch=dataset()
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,max_fixtures=0,refresh_fn=lambda *a,**k:self.fail('call'))
        self.assertEqual(r['attempted'],0)
        with self.assertRaises(ValueError):refresh_due_shortlist(slate,batch,'/tmp',now=NOW,max_fixtures=6)
    def test_wrong_fixture_response_reported_without_enabling_bets(self):
        slate,batch=dataset()
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,refresh_fn=lambda *a,**k:{'fixture_id':999})
        self.assertEqual(r['completed'],0);self.assertEqual(len(r['errors']),4)
        self.assertEqual(r['status'],'PARTIAL_REFRESH_NOT_VERIFIED')
    def test_lineup_provider_complete_not_officially_verified(self):
        slate,batch=dataset()
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,max_fixtures=1,
          refresh_fn=lambda b,fid,p,**k:{'fixture_id':fid,'provider_lineups':{'provider_lineups_complete':True},'reason':'TEST'})
        self.assertTrue(r['reports'][0]['provider_lineups_complete'])
        self.assertFalse(r['reports'][0]['official_lineup_verified'])
        self.assertEqual(r['official_lineup_verified'],0)
    def test_timezone_naive_not_accepted(self):
        slate,batch=dataset()
        with self.assertRaises(ValueError):refresh_due_shortlist(slate,batch,'/tmp',now=datetime(2026,11,5,11,0))
    def test_expired_and_postponed_not_queried(self):
        slate,batch=dataset()
        slate['eligible_fixtures'][0]['status']='PST'
        r=refresh_due_shortlist(slate,batch,'/tmp',now=NOW,refresh_fn=lambda b,fid,p,**k:{'fixture_id':fid})
        self.assertNotIn(1,[a['fixture_id'] for a in r['reports']])
if __name__=='__main__':unittest.main()