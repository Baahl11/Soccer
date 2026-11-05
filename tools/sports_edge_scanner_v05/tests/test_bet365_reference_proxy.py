"""Bet365-only proxy price guardrails: never an authenticated Mexican offer."""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import json, tempfile
from sports_edge.daily import fetch_research_reference_odds
from sports_edge.providers import api_football_odds

NOW=datetime(2026,11,5,16,0,tzinfo=timezone.utc)
ENV={'SOCCER_PROXY_BOOKMAKER_ID':'8','SOCCER_PROXY_BOOKMAKER_NAME':'Bet365'}

def batch():
    return {'source_slate_complete_verified':True,'all_fixtures_attempted':True,
            'shortlists':{'merged_fixture_ids':[101,202]},
            'screened_fixtures':[
                {'fixture_id':101,'match':'A vs B','min_sample_met':True,'status':'WATCH'},
                {'fixture_id':202,'match':'C vs D','min_sample_met':True,'status':'WATCH'}]}

def odds(fid,books):
    def bk(name):
        return {'name':name,'bets':[{'name':'Goals Over/Under',
               'values':[{'value':'Over 2.5','odd':'1.44'}]}]}
    return {'response':[{'fixture':{'id':fid,'date':(NOW+timedelta(hours=3)).isoformat()},
             'update':NOW.isoformat(),'bookmakers':[bk(n) for n in books]}],
            'retrieved_at_utc':NOW.isoformat()}

class Bet365ReferenceProxyTest(TestCase):
    def test_provider_query_includes_exact_bookmaker_id(self):
        seen=[]
        def fake(url,params,key,header,retries):
            seen.append(params.copy())
            return {'response':[],'paging':{'total':1},'errors':{}}
        with patch.dict('os.environ',{'API_FOOTBALL_KEY':'test-only-secret'}),\
             patch('sports_edge.providers._reserve_api_football_research'),\
             patch('sports_edge.providers.get_json',side_effect=fake):
            api_football_odds(101,bookmaker_id=8)
        self.assertEqual(seen,[{'fixture':101,'page':1,'bookmaker':8}])
    def test_no_implicit_second_book(self):
        def fn(fid):
            return odds(fid,['Bet365','Pinnacle','Betano'])
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',ENV):
            out=fetch_research_reference_odds(batch(),folder,2,odds_fn=fn)
            self.assertEqual(out['reference_bookmaker'],'Bet365')
            self.assertEqual(out['reference_bookmaker_id'],8)
            self.assertEqual(out['proxy_quote_fixtures'],2)
            self.assertEqual(out['proxy_quote_count'],2)
            self.assertFalse(out['bet365_direct_betslip_confirmed'])
            for snap in out['snapshots']:
                content=json.loads(Path(snap['file']).read_text())
                self.assertEqual({q['bookmaker'] for q in content['quotes']},{'Bet365'})
                self.assertFalse(content['mexico_offer_verified'])
                self.assertFalse(content['bet365_betslip_price_confirmed'])
    def test_missing_bet365_does_not_fallback_to_pinnacle(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',ENV):
            out=fetch_research_reference_odds(batch(),folder,2,odds_fn=lambda fid:odds(fid,['Pinnacle']))
            self.assertEqual(out['captured_count'],0)
            self.assertEqual(out['proxy_quote_count'],0)
            self.assertEqual(len(out['failures']),2)
            self.assertTrue(all('NO_BET365' in x['reason'] for x in out['failures']))
    def test_reject_partial_proxy_configuration(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'SOCCER_PROXY_BOOKMAKER_NAME':'Bet365','SOCCER_PROXY_BOOKMAKER_ID':''}):
            with self.assertRaisesRegex(Exception,'configured together'):
                fetch_research_reference_odds(batch(),folder,0,odds_fn=lambda fid:None)
    def test_invalid_bookmaker_id_rejected_before_request(self):
        for id in ('0','-8','bad'):
            with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'SOCCER_PROXY_BOOKMAKER_ID':id,'SOCCER_PROXY_BOOKMAKER_NAME':'Bet365'}):
                with self.assertRaisesRegex(Exception,'Invalid proxy bookmaker ID'):
                    fetch_research_reference_odds(batch(),folder,0,odds_fn=lambda fid:None)
    def test_other_book_non_proxy_mode_compatible(self):
        with tempfile.TemporaryDirectory() as folder,patch.dict('os.environ',{'SOCCER_PROXY_BOOKMAKER_NAME':'','SOCCER_PROXY_BOOKMAKER_ID':''}):
            out=fetch_research_reference_odds(batch(),folder,1,odds_fn=lambda fid:odds(fid,['Pinnacle']))
            self.assertEqual(out['captured_count'],1)
            self.assertEqual(out['reference_bookmaker'],'ALL_REFERENCE_BOOKS')
