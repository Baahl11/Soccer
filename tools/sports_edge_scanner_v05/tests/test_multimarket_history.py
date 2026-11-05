"""No leakage, no invented probabilities, no promotion across soccer markets."""
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from sports_edge.history import normalize_finished_history, get_team_history
from sports_edge.canonical_snapshot import fetch_canonical_market_views, match_canonical_fixtures
from sports_edge.market_registry import load_market_registry, FAMILY_FILES
from sports_edge.board import render_board
from sports_edge.engine_bridge import from_existing_engine

TARGET = dict(fixture_id=200, kickoff='2026-11-05T20:00:00+00:00',status='NS',
              home_team_id=11,away_team_id=22,league_id=3,season=2026)


def fx(fid, date, home=11, away=33, status='FT', hg=2, ag=1, league=100):
    return {'fixture':{'id':fid,'date':date,'status':{'short':status}},
            'teams':{'home':{'id':home},'away':{'id':away}},
            'league':{'id':league,'name':'Other League'},'goals':{'home':hg,'away':ag}}

class HistoryEvidence(unittest.TestCase):
    def test_past_games_across_leagues_are_accepted(self):
        x=normalize_finished_history({'response':[
            fx(1,'2026-11-02T17:00:00+00:00',league=500),
            fx(2,'2026-11-01T17:00:00+00:00',home=22,away=11,league=700)]},
            team_id=11,target_fixture=TARGET)
        self.assertEqual([r['fixture_id'] for r in x],[1,2])
        self.assertEqual({r['league_id'] for r in x},{500,700})
    def test_no_future_current_unrelated_or_incomplete(self):
        payload={'response':[
            fx(1,'2026-11-06T00:00:00+00:00'),
            fx(200,'2026-11-04T00:00:00+00:00'),
            fx(3,'2026-11-02T00:00:00+00:00',home=99,away=88),
            fx(4,'2026-11-03T00:00:00+00:00',status='NS'),
            fx(5,'2026-11-03T00:00:00+00:00',hg=None),
            fx(6,'2026-11-03T00:00:00+00:00'),
            fx(6,'2026-11-03T00:00:00+00:00')
        ]}
        self.assertEqual([r['fixture_id'] for r in normalize_finished_history(payload,team_id=11,target_fixture=TARGET)],[6])
    def test_invalid_target_date_refused(self):
        with self.assertRaises(Exception):
            normalize_finished_history({'response':[]},team_id=11,target_fixture={'kickoff':'not-a-date'})
    def test_source_query_all_competitions_and_cache(self):
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', {'API_FOOTBALL_CACHE_DIR':d}):
            called=[]
            def api(endpoint,params):
                called.append((endpoint,params))
                return {'response':[fx(1,'2026-11-02T00:00:00+00:00')]}
            a=get_team_history(11,TARGET,api)
            b=get_team_history(11,TARGET,api)
            self.assertEqual(a,b)
            self.assertEqual(called,[('fixtures',{'team':11,'last':12,'status':'FT'})])

class CanonicalResearchOnly(unittest.TestCase):
    def test_old_pipeline_stale_no_fixture_merge(self):
        now=datetime(2026,11,5,tzinfo=timezone.utc)
        result=fetch_canonical_market_views(lambda:{'views':{'todays_slate':{'total':1,'rows':[{'fixture_id':10,'market':'Match Winner'}]}},
                       'generated_at_utc':'2026-10-13T00:00:00+00:00','production_promotion_allowed':False},now=now)
        self.assertEqual(result['status'],'STALE_OR_UNDATED_CANONICAL_PIPELINE')
        self.assertEqual(match_canonical_fixtures(result,{10})['matching_fixture_rows'],[])
    def test_fresh_exact_id_only_and_watch(self):
        now=datetime(2026,11,5,tzinfo=timezone.utc)
        payload={'views':{'todays_slate':{'total':2,'rows':[{'fixture_id':10,'market':'1X2','price':2.0},
                                                                  {'fixture_id':11,'market':'1H','price':1.8}]},
                          'corners':{'total':1,'rows':[{'fixture_id':11,'market':'corners over'}]}},
                 'generated_at_utc':(now-timedelta(hours=1)).isoformat(),
                 'production_promotion_allowed':False}
        result=match_canonical_fixtures(fetch_canonical_market_views(lambda:payload,now=now),{11})
        self.assertTrue(result['is_fresh'])
        self.assertEqual(len(result['matching_fixture_rows']),2)
        self.assertTrue(all(x['fixture_id']==11 and x['classification']=='WATCH' and not x['bet_eligible'] for x in result['matching_fixture_rows']))
    def test_unexpected_promoted_model_refused(self):
        x=fetch_canonical_market_views(lambda:{'views':{},'production_promotion_allowed':True})
        self.assertEqual(x['status'],'NOT_VERIFIED')
        self.assertIsNotNone(x['error'])

class MarketValidity(unittest.TestCase):
    def test_market_reports_do_not_authorize_bets_even_if_review_eligible(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td,'v4_017_1x2_calibration_validation.json').write_text(json.dumps({
                'status':'CALIBRATION_REVIEW_ELIGIBLE','model_version':'SOCCER_1X2_VALIDATION',
                'production_promotion_allowed':True,'blockers':[]}),encoding='utf-8')
            d=load_market_registry(td)
            self.assertEqual(len(d['families']),len(FAMILY_FILES))
            self.assertEqual(next(x for x in d['families'] if x['family']=='1X2')['research_status'],'CALIBRATION_REVIEW_ELIGIBLE')
            self.assertTrue(all(not x['bet_eligible'] and not x['production_promotion_allowed'] for x in d['families']))
    def test_fouls_without_dedicated_model_are_unverified(self):
        with tempfile.TemporaryDirectory() as td:
            d=load_market_registry(td)
        f=next(x for x in d['families'] if x['family']=='FOULS')
        self.assertEqual(f['research_status'],'NOT_VERIFIED')
        self.assertIn('FOULS',f['blockers'][0])

class BoardMarkets(unittest.TestCase):
    def test_displays_market_states_and_history_without_fake_percentages(self):
        batch={'screened_fixtures':[{'fixture_id':10,'match':'Alfa vs Beta','league_id':3,'raw_sport_projection':{
            'status':'NOT_MODELED'},'multi_competition_history':{'home_count':10,'away_count':8,'status':'VERIFIED_PAST_MATCHES'}}]}
        summ={'match_date':'2026-11-05','sports_screened':1,'history_multi_competition_verified':1,
              'canonical_market_status':'STALE_OR_UNDATED_CANONICAL_PIPELINE'}
        with tempfile.TemporaryDirectory() as td:
            reg=load_market_registry(td)
        html=render_board(batch,summ,market_registry=reg)
        self.assertIn('Faltas cometidas',html)
        self.assertIn('Primera mitad',html)
        self.assertIn('10 local / 8 visitante',html)
        self.assertIn('STALE_OR_UNDATED_CANONICAL_PIPELINE',html)
        self.assertIn('NOT VERIFIED',html)

if __name__=='__main__':unittest.main()

class PeriodScoreHistory(unittest.TestCase):
    def test_halftime_split_is_historical_and_not_fabricated(self):
        from sports_edge.history import normalize_finished_history
        game=fx(15,'2026-11-03T14:00:00+00:00',hg=2,ag=1)
        game['score']={'halftime':{'home':1,'away':0}}
        r=normalize_finished_history({'response':[game]},team_id=11,target_fixture=TARGET)
        self.assertEqual(r[0]['first_half_goals'],{'home':1,'away':0})
        self.assertEqual(r[0]['second_half_goals'],{'home':1,'away':1})
    def test_unavailable_or_contradictory_half_scores_remain_missing(self):
        from sports_edge.history import normalize_finished_history
        game=fx(15,'2026-11-03T14:00:00+00:00',hg=2,ag=1)
        game['score']={'halftime':{'home':3,'away':0}}
        r=normalize_finished_history({'response':[game]},team_id=11,target_fixture=TARGET)
        self.assertIsNone(r[0]['first_half_goals'])
        self.assertIsNone(r[0]['second_half_goals'])