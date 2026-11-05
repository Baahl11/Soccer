import unittest
from sports_edge.market_context import (parse_historical_stats,parse_historical_formation,
                                         collect_historical_market_context)
from sports_edge.providers import ProviderFailure

class MarketHistoryTests(unittest.TestCase):
    def test_parses_known_team_no_zero_for_missing(self):
        r=parse_historical_stats({'response':[{'team':{'id':3},'statistics':[
            {'type':'Corner Kicks','value':8},{'type':'Fouls','value':None},{'type':'Yellow Cards','value':'3'}]}]},3)
        self.assertEqual(r['corners'],8)
        self.assertIsNone(r['fouls'])
        self.assertEqual(r['yellow_cards'],3)
        self.assertIsNone(r['shots'])
    def test_mismatched_team_refuses(self):
        with self.assertRaises(ProviderFailure):
            parse_historical_stats({'response':[{'team':{'id':333},'statistics':[]}]},3)
    def test_formation_requires_starting_eleven(self):
        obj={'response':[{'team':{'id':3},'formation':'4-3-3','startXI':[{}]*11}]}
        self.assertEqual(parse_historical_formation(obj,3),'4-3-3')
        obj['response'][0]['startXI']=[{}]*10
        self.assertIsNone(parse_historical_formation(obj,3))
    def test_deep_context_separate_market_data_and_no_bet(self):
        batch={'screened_fixtures':[{'fixture_id':300,'match':'X vs Y','home_team':{'team_id':11},
            'away_team':{'team_id':22},'multi_competition_history':{
             'home_fixtures':[{'fixture_id':10,'home_team_id':11,'away_team_id':44}],
             'away_fixtures':[{'fixture_id':11,'home_team_id':22,'away_team_id':55}]} }],
             'shortlists':{'merged_fixture_ids':[300]}}
        calls=[]
        def api(endpoint,params):
            calls.append((endpoint,params))
            if endpoint=='fixtures/statistics':
                return {'response':[{'team':{'id':11},'statistics':[{'type':'Fouls','value':13}]},
                                    {'team':{'id':22},'statistics':[{'type':'Corner Kicks','value':8}]}]}
            return {'response':[]}
        r=collect_historical_market_context(batch,api,max_fixtures=1)
        self.assertEqual(r['historical_stats_requests'],2)
        self.assertEqual(r['historical_lineups_requests'],2)
        self.assertFalse(r['bet_eligible'])
        self.assertEqual(r['reports'][0]['home']['observed_averages']['fouls']['average'],13)
        self.assertIsNone(r['reports'][0]['away']['observed_averages']['fouls']['average'])
    def test_no_history_no_rate(self):
        batch={'screened_fixtures':[{'fixture_id':300,'home_team':{'team_id':11},'away_team':{'team_id':22}}],
              'shortlists':{'merged_fixture_ids':[300]}}
        r=collect_historical_market_context(batch,lambda *a: (_ for _ in ()).throw(AssertionError('not called')))
        self.assertIsNone(r['reports'][0]['home']['observed_averages']['corners']['average'])

if __name__=='__main__':unittest.main()