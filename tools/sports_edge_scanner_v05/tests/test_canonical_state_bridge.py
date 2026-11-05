"""Regression: source identities, correct outcome calibration and bookmaker audit."""
from datetime import datetime,timezone
from unittest import TestCase
from sports_edge.canonical_state_bridge import canonical_research_comparisons,research_probability
from sports_edge.bookmaker_catalogue import inspect_bookmaker_directory,fetch_bookmaker_directory

NOW=datetime(2026,11,5,16,20,tzinfo=timezone.utc)

def sample():
    fixture={'fixture_id':1636211,'home_team':'Anderlecht','away_team':'Red Bull Salzburg',
             'status':'NS','kickoff':'2026-11-05T17:45:00+00:00'}
    slate={'eligible_slate_verified':True,'provider_world_slate_audit':{'complete_slate_verified':True},
           'eligible_fixtures':[fixture]}
    cal={'status':'RESEARCH_CALIBRATOR_FITTED','market_fields_used':False}
    row={'fixture_id':1636211,'home':'Anderlecht','away':'Red Bull Salzburg','status':'NS',
         'kickoff':fixture['kickoff'],'market_family':'FT_TOTALS',
         'market':'Goals Over/Under','selection':'Over','line':2.5,
         'phase16_calibration_status':'RESEARCH_CALIBRATION_APPLIED',
         'phase16_calibration_source':'CURRENT_MODEL_OOS_PLATT:OVER_2_5',
         'p_model_calibrated':.57528369,'p_raw':.666248,
         'phase16_calibrator_artifact':{'source_model_version_matches':True,
                'exact_calibrator_payload':{'calibrator':cal}},
         'price':1.6,'bookmaker':'Betano','provider_update':'2026-11-05T15:13:17+00:00',
         'price_resolution_source':'API_FOOTBALL_ODDS_V3',
         'production_promotion_allowed':False,'p_shrunk':None,'p_lower':None}
    state={'generated_at_utc':'2026-11-05T15:50:00+00:00',
           'model_version':'SOCCER EDGE ENGINE v1.7','match_table_rows':[row]}
    return slate,state

class CanonicalResearchBridgeTest(TestCase):
    def test_exact_over_only_watch(self):
        slate,state=sample();r=canonical_research_comparisons(state,slate,now=NOW)
        self.assertEqual(r['status'],'CURRENT_RESEARCH_ONLY')
        self.assertEqual(r['selection_calibration_research_rows'],1)
        self.assertEqual(r['rows'][0]['research_probability_for_selection'],.57528369)
        self.assertEqual(r['actionable_bets'],0)
        self.assertIn('RESEARCH_MODEL_NOT_APPROVED_FOR_PRODUCTION',r['rows'][0]['blockers'])
    def test_under_uses_complement_not_over(self):
        slate,state=sample();row=state['match_table_rows'][0];row['selection']='Under'
        r=canonical_research_comparisons(state,slate,now=NOW)
        self.assertAlmostEqual(r['rows'][0]['research_probability_for_selection'],1-.57528369)
        self.assertIn('COMPLEMENT',r['rows'][0]['calibration_mapping'])
    def test_other_goal_line_cannot_reuse_model(self):
        slate,state=sample();state['match_table_rows'][0]['line']=3.5
        r=canonical_research_comparisons(state,slate,now=NOW)
        self.assertIsNone(r['rows'][0]['research_probability_for_selection'])
    def test_1x2_class_discrimination_not_ready(self):
        slate,state=sample();row=state['match_table_rows'][0]
        row.update(market_family='1X2',market='Match Winner',selection='Home',line=None,
                   phase16_calibration_source='CURRENT_MODEL_OOS_TEMPERATURE:1X2')
        self.assertIsNone(canonical_research_comparisons(state,slate,now=NOW)['rows'][0]['research_probability_for_selection'])
    def test_no_future_or_wrong_identity(self):
        slate,state=sample();state['match_table_rows'][0]['away']='Wrong club'
        self.assertEqual(canonical_research_comparisons(state,slate,now=NOW)['rows'],[])
        slate,state=sample();state['match_table_rows'][0]['status']='PST'
        self.assertEqual(canonical_research_comparisons(state,slate,now=NOW)['rows'],[])
    def test_stale_state_and_missing_slate(self):
        slate,state=sample();state['generated_at_utc']='2026-10-12T00:00:00Z'
        self.assertEqual(canonical_research_comparisons(state,slate,now=NOW)['status'],'STALE_CANONICAL_SOURCE')
        slate,state=sample();slate['eligible_slate_verified']=False
        self.assertEqual(canonical_research_comparisons(state,slate,now=NOW)['status'],'SLATE_NOT_VERIFIED')
    def test_stale_quote_never_actionable(self):
        slate,state=sample();state['match_table_rows'][0]['provider_update']='2026-11-04T01:00:00Z'
        r=canonical_research_comparisons(state,slate,now=NOW)
        self.assertIsNone(r['rows'][0]['indicative_price'])
        self.assertEqual(r['rows'][0]['status'],'WATCH')
    def test_market_fields_used_in_calibrator_rejects_sport_first(self):
        slate,state=sample();c=state['match_table_rows'][0]['phase16_calibrator_artifact']['exact_calibrator_payload']['calibrator']
        c['market_fields_used']=True
        self.assertIsNone(canonical_research_comparisons(state,slate,now=NOW)['rows'][0]['research_probability_for_selection'])

class BookmakerDirectoryTest(TestCase):
    def test_named_provider_not_offer(self):
        r=inspect_bookmaker_directory({'paging':{'current':1,'total':1},
                                        'response':[{'id':4,'name':'Caliente MX'},{'id':5,'name':'Playdoit'}]})
        self.assertEqual(len(r['caliente_matches']),1)
        self.assertEqual(len(r['playdoit_matches']),1)
        self.assertFalse(r['authenticated_sportsbook_price'])
        self.assertEqual(r['actionable_offers'],0)
    def test_no_names_in_complete_catalogue(self):
        r=inspect_bookmaker_directory({'paging':{'current':1,'total':1},
                                        'response':[{'id':1,'name':'Pinnacle'}]})
        self.assertEqual(r['status'],'NO_NAMED_BOOKMAKER_IN_COMPLETE_CATALOGUE')
    def test_paged_cannot_assert_absence(self):
        r=inspect_bookmaker_directory({'paging':{'current':1,'total':2},'response':[]})
        self.assertEqual(r['status'],'PAGED_CATALOGUE_NOT_COMPLETE')
    def test_provider_failure_does_not_assert_absence(self):
        def bad(*args):raise RuntimeError('outage')
        self.assertEqual(fetch_bookmaker_directory(bad)['status'],'BOOKMAKER_DIRECTORY_REQUEST_FAILED')
