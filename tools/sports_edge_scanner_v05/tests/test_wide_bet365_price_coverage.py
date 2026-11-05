"""Research coverage: price discovery includes sport-screened low-sample matches.

Low sample NEVER creates BET; it only avoids wasting a good bookmaker
connection by searching two fixtures after screening the entire slate.
"""
from unittest import TestCase
from unittest.mock import patch
from tempfile import TemporaryDirectory
from datetime import datetime,timedelta,timezone
from sports_edge.daily import fetch_research_reference_odds

NOW=datetime.now(timezone.utc)
def row(i,n,good=False):
    return {'fixture_id':i,'status':'WATCH','min_sample_met':good,
            'kickoff_utc':(NOW+timedelta(hours=3)).isoformat(),
            'match':f'Home{i} vs Away{i}',
            'raw_sport_projection':{'status':'MODELED_LIMITED','sample':{'minimum_split_sample':n}}}
def odds(i):
    return {'response':[{'fixture':{'id':i,'date':(NOW+timedelta(hours=3)).isoformat()},
           'update':NOW.isoformat(),
           'bookmakers':[{'name':'Bet365','bets':[{'name':'Match Winner',
               'values':[{'value':'Home','odd':'1.42'}]}]}]}],
           'retrieved_at_utc':NOW.isoformat()}

class SportFirstExpandedPriceCoverage(TestCase):
    def test_broadens_without_promoting_small_sample(self):
        batch={'source_slate_complete_verified':True,'all_fixtures_attempted':True,
              'shortlists':{'merged_fixture_ids':[2]},
              'screened_fixtures':[row(1,1),row(2,5,True),row(3,3),row(4,2)]}
        called=[]
        def fn(i): called.append(i);return odds(i)
        with TemporaryDirectory() as d,patch.dict('os.environ',{'SOCCER_PROXY_BOOKMAKER_NAME':'Bet365','SOCCER_PROXY_BOOKMAKER_ID':'8'}):
            r=fetch_research_reference_odds(batch,d,3,odds_fn=fn)
        self.assertEqual(called,[2,3,4])
        self.assertEqual(r['proxy_quote_fixtures'],3)
        self.assertEqual(r['actionable_bets'],0)
    def test_never_prices_past_fixtures_from_extras(self):
        old=row(3,20)
        old['kickoff_utc']=(NOW-timedelta(minutes=20)).isoformat()
        batch={'source_slate_complete_verified':True,'all_fixtures_attempted':True,
              'shortlists':{'merged_fixture_ids':[2]},
              'screened_fixtures':[row(2,5,True),old,row(4,1)]}
        called=[]
        with TemporaryDirectory() as d,patch.dict('os.environ',{'SOCCER_PROXY_BOOKMAKER_NAME':'Bet365','SOCCER_PROXY_BOOKMAKER_ID':'8'}):
            r=fetch_research_reference_odds(batch,d,3,odds_fn=lambda i:(called.append(i) or odds(i)))
        self.assertEqual(called,[2,4])
        self.assertEqual(r['selected_count'],2)
    def test_zero_limit_really_zero_calls(self):
        batch={'source_slate_complete_verified':True,'all_fixtures_attempted':True,'shortlists':{'merged_fixture_ids':[]},
               'screened_fixtures':[row(1,5,True)]}
        with TemporaryDirectory() as d:
            r=fetch_research_reference_odds(batch,d,0,odds_fn=lambda i:self.fail('should not query'))
        self.assertEqual(r['selected_count'],0)
