from unittest import TestCase
from datetime import datetime,timezone,timedelta
from sports_edge.market_history_evidence import past_goals_market_evidence

KICK='2026-11-05T17:45:00+00:00'
def fixture():
    h=[];a=[]
    for n in range(8):
        for arr,home,away in ((h,101,1000+n),(a,2000+n,202)):
            arr.append({'fixture_id':n+(100 if home==101 else 200),
                        'kickoff':(datetime(2026,11,5,17,45,tzinfo=timezone.utc)-timedelta(days=n+1)).isoformat(),
                        'status':'FT','home_team_id':home,'away_team_id':away,
                        'goals':{'home':2,'away':1}})
    return {'fixture_id':10,'kickoff':KICK,'home_team_id':101,'away_team_id':202,
            'raw_sport_projection':{'status':'MODELED_LIMITED'},
            'multi_competition_history':{'status':'VERIFIED_PAST_MATCHES',
                'home_fixtures':h,'away_fixtures':a}}

def leg(family='FULL_TIME_GOALS',sel='Under',line='4.5'):
    return {'family':family,'selection':sel,'point':line,'starts_at_utc':KICK}

class MarketHistoryEvidenceTest(TestCase):
    def test_past_only_eight_games_each(self):
        x=past_goals_market_evidence(leg(),fixture())
        self.assertEqual(x['status'],'DESCRIPTIVE_HISTORY_AVAILABLE')
        self.assertEqual(x['sample_minimum'],8)
        self.assertTrue(x['both_team_directional_support'])
        self.assertEqual([z['matching_totals'] for z in x['teams']],[8,8])
        self.assertFalse(x['historical_probability_calibrated'])
        self.assertFalse(x['opponent_quality_adjusted'])
    def test_over_fails_against_actual_3_goals(self):
        x=past_goals_market_evidence(leg(sel='Over'),fixture())
        self.assertEqual([z['matching_totals'] for z in x['teams']],[0,0])
        self.assertFalse(x['both_team_directional_support'])
    def test_future_and_unfinished_not_used(self):
        fx=fixture()
        fx['multi_competition_history']['home_fixtures'][0]['kickoff']=KICK
        fx['multi_competition_history']['away_fixtures'][0]['status']='NS'
        x=past_goals_market_evidence(leg(),fx)
        self.assertEqual(x['sample_minimum'],7)
    def test_mismatched_team_not_used(self):
        fx=fixture();fx['multi_competition_history']['home_fixtures'][0]['home_team_id']=999
        x=past_goals_market_evidence(leg(),fx)
        self.assertEqual(x['teams'][0]['historical_games'],7)
    def test_duplicate_ids_not_double_counted(self):
        fx=fixture();h=fx['multi_competition_history']['home_fixtures']
        h.append(dict(h[0]))
        self.assertEqual(past_goals_market_evidence(leg(),fx)['teams'][0]['historical_games'],8)
    def test_corners_not_mislabeled_as_goals(self):
        x=past_goals_market_evidence(leg('CORNERS'),fixture())
        self.assertEqual(x['status'],'NOT_VERIFIED')
        self.assertEqual(x['teams'],[])
    def test_missing_identity_rejects(self):
        fx=fixture();fx.pop('away_team_id')
        self.assertEqual(past_goals_market_evidence(leg(),fx)['status'],'NOT_VERIFIED')
    def test_market_integer_asian_line_rejects(self):
        x=past_goals_market_evidence(leg(line='4.0'),fixture())
        self.assertEqual(x['status'],'NOT_VERIFIED')
    def test_invalid_goals_reject(self):
        fx=fixture();fx['multi_competition_history']['home_fixtures'][0]['goals']['home']=None
        self.assertEqual(past_goals_market_evidence(leg(),fx)['teams'][0]['historical_games'],7)
    def test_history_alone_not_approval(self):
        x=past_goals_market_evidence(leg(),fixture())
        self.assertNotIn('BET',x)
        self.assertFalse(x['formation_adjusted'])