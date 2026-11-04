"""Pro API discovery and sport-first checks (all tests offline)."""
import json, subprocess, sys, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
from datetime import datetime, timezone, timedelta
from sports_edge.pro_pipeline import build_pro_slate, sporting_research, api_pro_get, get_cached_league_coverage
from sports_edge.providers import ProviderFailure
from sports_edge.core import market_family

ROOT=Path(__file__).resolve().parents[1]
TODAY='2026-11-05'

def fixture(fid,league=18,status='NS',kickoff='2026-11-05T10:00:00+00:00'):
    return {'fixture':{'id':fid,'date':kickoff,'status':{'short':status},'venue':{'name':'Somewhere'}},
      'league':{'id':league,'name':'Continental Cup','season':2026},
      'teams':{'home':{'id':303,'name':'Machida'},'away':{'id':833,'name':'Shanghai'}}}

class ProSlates(unittest.TestCase):
    def test_complete_direct_and_eligibility(self):
        raw={'results':3,'paging':{'total':1},'response':[fixture(1),fixture(2,status='FT'),fixture(3,league=39)]}
        r=build_pro_slate(raw,TODAY,{18})
        self.assertTrue(r['eligible_slate_verified'])
        self.assertEqual(r['eligible_count'],1)
        self.assertEqual(r['eligible_fixtures'][0]['fixture_id'],1)
        self.assertEqual(r['excluded_count'],1)
    def test_truncated_cannot_be_filtered_to_fake_completeness(self):
        raw={'results':200,'paging':{'total':1},'response':[fixture(1)]}
        r=build_pro_slate(raw,TODAY,{18})
        self.assertFalse(r['eligible_slate_verified'])
        self.assertEqual(r['provider_world_slate_audit']['missing_minimum'],199)
    def test_connector_partial_cannot_claim_all_leagues(self):
        raw={'match_date':TODAY,'total_results':2,'truncated':True,'fixtures':[{
            'fixture_id':1,'kickoff':'2026-11-05T04:00:00-06:00','status':'NS','league_id':18}]}
        r=build_pro_slate(raw,TODAY,{18},source_kind='connected')
        self.assertFalse(r['eligible_slate_verified'])
    def test_duplicate_or_wrong_date_invalidates_completeness(self):
        raw={'results':2,'paging':{'total':1},'response':[fixture(1),fixture(1)]}
        self.assertFalse(build_pro_slate(raw,TODAY,{18})['eligible_slate_verified'])
        raw={'results':1,'paging':{'total':1},'response':[fixture(1,kickoff='2026-11-04T10:00:00+00:00')]}
        self.assertFalse(build_pro_slate(raw,TODAY,{18})['eligible_slate_verified'])
    def test_requires_selected_league(self):
        with self.assertRaises(ValueError): build_pro_slate({'results':0,'response':[]},TODAY,set())
    def test_blocks_keyless_direct_call(self):
        with patch.dict('os.environ',{'API_FOOTBALL_KEY':''}):
            with self.assertRaises(ProviderFailure):api_pro_get('leagues',{'id':18})
    def test_research_does_not_infer_probabilities_from_three_games(self):
        f={'fixture_id':1,'league_id':18,'season':2026,'home_team_id':303,'away_team_id':833,'home_team':'Machida','away_team':'Shanghai'}
        def api_call(endpoint,params):
            if endpoint=='leagues':return {'response':[{'seasons':[{'coverage':{'fixtures':{'lineups':True}}}]}]}
            return {'response':{'fixtures':{'played':{'total':3},'wins':{'total':2},'loses':{'total':1}},'goals':{'for':{'average':{'total':'1.2'}},'against':{'average':{'total':'0.8'}}}}}
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'API_FOOTBALL_CACHE_DIR':td}):
            with patch('sports_edge.pro_pipeline.api_pro_get',side_effect=api_call):
                x=sporting_research(f)
        self.assertFalse(x['min_sample_met']);self.assertIsNone(x['raw_model_probability']);self.assertEqual(x['status'],'WATCH')
    def test_league_coverage_cached_weekly(self):
        with tempfile.TemporaryDirectory() as td, patch.dict('os.environ',{'API_FOOTBALL_CACHE_DIR':td}):
            with patch('sports_edge.pro_pipeline.api_pro_get',return_value={'response':[{'seasons':[]}]}) as api:
                self.assertEqual(get_cached_league_coverage(18,2026)['response'],[{'seasons':[]}])
                self.assertEqual(get_cached_league_coverage(18,2026)['response'],[{'seasons':[]}])
                self.assertEqual(api.call_count,1)
    def test_soccer_double_chance_is_derivative(self):
        self.assertEqual(market_family('soccer','Double Chance'),'derivative')
    def test_actionable_book_not_assumed(self):
        now=datetime.now(timezone.utc)
        start=now+timedelta(hours=3)
        date=start.astimezone(__import__('zoneinfo').ZoneInfo('America/Mexico_City')).date().isoformat()
        quote={'sport':'soccer','event_id':'1','event_name':'Test','starts_at_utc':start.isoformat(),
          'market':'Match Winner','selection':'Home','point':'','price_decimal':2.0,
          'bookmaker':'Other Book','source':'API-Football','source_updated_at':now.isoformat(),'captured_at_utc':now.isoformat()}
        proj={'sport':'soccer','event_id':'1','market':'Match Winner','selection':'Home','point':'',
              'p_raw':.75,'p_shrunk':.67,'p_lower':.65,'calibrated':True,
              'availability_confidence':.99,'material_availability_verified':True}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            for file,obj in [('odds', [quote]),('projs',[proj]),('audit',{'match_date':date,'complete_slate_verified':True})]:
                (d/(file+'.json')).write_text(json.dumps(obj))
            cmd=[sys.executable,'-m','sports_edge','scan','--odds',str(d/'odds.json'),
                 '--projections',str(d/'projs.json'),'--coverage-audit',str(d/'audit.json'),
                 '--out',str(d/'result.json')]
            res=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(res.returncode,0,res.stderr)
            r=json.loads((d/'result.json').read_text())
            self.assertEqual(r['singles'][0]['status'],'WATCH')
            self.assertEqual(r['singles'][0]['reason'],'ACTIONABLE_BOOK_NOT_CONFIRMED')
if __name__=='__main__': unittest.main()