import json, tempfile, unittest
from pathlib import Path
from datetime import datetime, timezone
from sports_edge.free_sources import (
    normalize_the_odds_api, normalize_connected_soccer_odds,
    summarize_connected_soccer_fixtures, reserve_quota, ProviderFailure,
)
from sports_edge.core import evaluate, attach_market_fair_probabilities
NOW=datetime(2026,11,4,12,0,tzinfo=timezone.utc)
class TestZeroCostAdapter(unittest.TestCase):
    def test_free_odds_normalizes_quote_and_real_update(self):
        raw={'retrieved_at_utc':'2026-11-04T12:00:00Z','response':[{
            'id':'nfl123','home_team':'A','away_team':'B','commence_time':'2026-11-05T15:00:00Z',
            'bookmakers':[{'title':'Book X','last_update':'2026-11-04T11:45:00Z',
            'markets':[{'key':'h2h','outcomes':[{'name':'A','price':1.7},{'name':'B','price':2.1}]}]}]}]}
        out=normalize_the_odds_api(raw,'nfl')
        self.assertEqual(len(out),2)
        self.assertEqual(out[0]['source_updated_at'],'2026-11-04T11:45:00Z')
        self.assertEqual(out[1]['bookmaker'],'Book X')
        self.assertEqual(evaluate(out[0],None,NOW)['status'],'WATCH')
    def test_free_quota_reserve_and_daily_stop(self):
        with tempfile.TemporaryDirectory() as d:
            f=str(Path(d)/'quota.json')
            self.assertEqual(reserve_quota(f,1,daily_cap=2,day='2026-11-04')['day_reserved'],1)
            reserve_quota(f,1,daily_cap=2,day='2026-11-04')
            with self.assertRaises(ProviderFailure):
                reserve_quota(f,1,daily_cap=2,day='2026-11-04')
            self.assertEqual(json.loads(Path(f).read_text())['used'],2)
    def test_free_monthly_stop(self):
        with tempfile.TemporaryDirectory() as d:
            f=str(Path(d)/'quota.json')
            reserve_quota(f,1,month_limit=2,month_reserve=1,daily_cap=2,day='2026-11-04')
            with self.assertRaises(ProviderFailure):
                reserve_quota(f,1,month_limit=2,month_reserve=1,daily_cap=2,day='2026-11-05')
    def test_connected_soccer_quote_preserves_timestamp(self):
        raw={'endpoint':'/odds','retrieved_at_utc':'2026-11-04T06:57:23Z','response':[{ 
            'fixture':{'id':1629973,'date':'2026-11-04T10:00:00Z'},
            'update':'2026-11-04T00:31:14Z','bookmakers':[{'name':'William Hill','bets':[
                {'name':'Match Winner','values':[{'value':'Home','odd':'3.30'},{'value':'Draw','odd':'3.40'},{'value':'Away','odd':'2.00'}]}
            ]}]}]}
        out=normalize_connected_soccer_odds(raw,{'1629973':'Pohang vs Gamba'})
        self.assertEqual(len(out),3)
        self.assertEqual(out[0]['source_updated_at'],'2026-11-04T00:31:14Z')
        self.assertEqual(evaluate(out[0],None,NOW)['reason'],'SOURCE_OR_TIMESTAMP_NOT_VERIFIED')
        self.assertAlmostEqual(sum(x['fair_market_probability'] for x in attach_market_fair_probabilities(out)),1.0)
    def test_truncation_never_reports_full_slate(self):
        a=summarize_connected_soccer_fixtures({'total_results':259,'fixtures':[{'league_id':1,'league':'A'}],
                                               'truncated':True,'source':'API-Football'})
        self.assertEqual(a['coverage_warning'],'INCOMPLETE_SLATE')
        self.assertFalse(a['complete_slate_verified'])
    def test_missing_kickoff_blocks_bet_even_with_calibrated_model(self):
        q={'sport':'soccer','event_id':'1','starts_at_utc':None,'market':'1x2','selection':'Home',
           'point':'','price_decimal':1.7,'bookmaker':'X','source':'TEST','source_updated_at':'2026-11-04T11:45:00Z'}
        p={'sport':'soccer','event_id':'1','market':'1x2','selection':'Home','point':'',
           'p_raw':.8,'p_shrunk':.8,'p_lower':.75,'calibrated':True,
           'material_availability_verified':True,'availability_confidence':1.0}
        self.assertEqual(evaluate(q,p,NOW)['reason'],'KICKOFF_NOT_VERIFIED')
if __name__=='__main__':unittest.main()