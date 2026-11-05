"""Live reference markets must go through the ORIGINAL decision gate, no bets."""
from __future__ import annotations
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from sports_edge.live_market_bridge import bridge_real_reference_markets

NOW = datetime(2026,11,5,18,0,tzinfo=timezone.utc)


def q(fid='22',raw=.80,sample=12,starts=120,updated=20,market='Goals Over/Under',selection='Over',point='2.5'):
    return {
        'event_id':str(fid), 'event_name': f'Home vs Away {fid}',
        'sport_first_fixture_verified':True,'bookmaker':'Bet365',
        'market':market,'selection':selection,'point':point,
        'family':'FULL_TIME_GOALS','price_decimal':1.45,
        'source':'API-Football','starts_at_utc':(NOW+timedelta(minutes=starts)).isoformat(),
        'source_updated_at':(NOW-timedelta(minutes=updated)).isoformat(),
        'raw_sport_probability':raw,'venue_sample_min':sample,
    }


def batch():
    return {'complete_sport_screen':True,'screened_fixtures':[
        {'fixture_id':22,'raw_sport_projection':{'status':'MODELED_LIMITED','model_version':'SOCCER EDGE ENGINE v1.0'},
         'fixture_status':'NS'},
        {'fixture_id':33,'raw_sport_projection':{'status':'MODELED_LIMITED','model_version':'SOCCER EDGE ENGINE v1.0'},
         'fixture_status':'NS'},
    ]}


class LiveMarketDecisionBridgeTests(unittest.TestCase):
    def test_real_reference_is_passed_to_final_gate_without_promotion(self):
        x=bridge_real_reference_markets({'all_quotes':[q()]},batch(),now=NOW)
        self.assertEqual(x['quotes_sent_to_final_decision_gate'],1)
        self.assertEqual(x['decisions'][0]['decision'],'WATCH')
        self.assertNotIn('BET365_PROVIDER_PRICE_SOURCE_NOT_VERIFIED',x['decisions'][0]['decision_blockers'])
        self.assertIn('MODEL_NOT_APPROVED_FOR_PRODUCTION',x['decisions'][0]['decision_blockers'])
        self.assertEqual(x['final_bets'],0)
        self.assertIsNone(x['decisions'][0]['expected_value'])
    def test_sport_first_incomplete_yields_no_quotes(self):
        b=batch(); b['complete_sport_screen']=False
        x=bridge_real_reference_markets({'all_quotes':[q()]},b,now=NOW)
        self.assertEqual(x['status'],'INCOMPLETE_SPORT_FIRST_SLATE')
        self.assertFalse(x['decisions'])
    def test_wrong_fixture_is_rejected(self):
        x=bridge_real_reference_markets({'all_quotes':[q(fid=99)]},batch(),now=NOW)
        self.assertEqual(x['quotes_sent_to_final_decision_gate'],0)
        self.assertEqual(x['rejection_reasons']['FIXTURE_NOT_IN_SPORT_FIRST_SLATE'],1)
    def test_started_or_stale_prices_never_enter_decision(self):
        x=bridge_real_reference_markets({'all_quotes':[q(starts=-5),q(updated=400)]},batch(),now=NOW)
        self.assertEqual(x['quotes_sent_to_final_decision_gate'],0)
    def test_missing_raw_market_is_not_implied_from_another_family(self):
        x=bridge_real_reference_markets({'all_quotes':[q(raw=None,market='Corners Over/Under',point='9.5')]},batch(),now=NOW)
        self.assertEqual(x['quotes_exact_raw_available'],0)
        self.assertEqual(x['decisions'][0]['sporting_evidence_stage'],'NO_EXACT_SPORT_PROBABILITY')
        self.assertIn('EXACT_MARKET_MODEL_MISSING',x['decisions'][0]['decision_blockers'])
    def test_refuses_untrusted_name_and_unverified_sport(self):
        a=q(); a['bookmaker']='Other'; b=q(fid=33); b['sport_first_fixture_verified']=False
        x=bridge_real_reference_markets({'all_quotes':[a,b]},batch(),now=NOW)
        self.assertEqual(x['quotes_sent_to_final_decision_gate'],0)
    def test_small_sample_does_not_become_high_confidence(self):
        x=bridge_real_reference_markets({'all_quotes':[q(sample=2,raw=.999)]},batch(),now=NOW)
        self.assertEqual(x['decisions'][0]['sporting_evidence_stage'],'INSUFFICIENT_VENUE_SPLIT_SAMPLE')
        self.assertEqual(x['decisions'][0]['decision'],'WATCH')
    def test_candidate_price_without_positive_raw_stays_watch(self):
        x=bridge_real_reference_markets({'all_quotes':[q(raw=.30)]},batch(),now=NOW)
        self.assertEqual(x['raw_at_or_below_breakeven'],1)
        self.assertEqual(x['decisions'][0]['decision'],'WATCH')
        self.assertEqual(x['decisions'][0]['sporting_evidence_stage'],'RAW_NOT_ABOVE_PRICE_RESEARCH')
    def test_two_real_event_pair_is_reviewed_without_joint_probability(self):
        a=q(fid=22); b=q(fid=33,selection='Under',raw=.7)
        pair={'legs':[a,b],'theoretical_combined_decimal':2.1025}
        x=bridge_real_reference_markets({'all_quotes':[a,b],'sporting_screened_plus100_pairs':[pair]},batch(),now=NOW)
        p=x['parlay_pair_reviews'][0]
        self.assertEqual(p['classification'],'WATCH')
        self.assertEqual(len(p['legs']),2)
        self.assertTrue(all(z['observed'] for z in p['legs']))
        self.assertIsNone(p['joint_p_lower'])
    def test_duplicates_are_bounded_and_not_double_counted(self):
        a=q()
        x=bridge_real_reference_markets({'all_quotes':[a,a]},batch(),now=NOW)
        self.assertEqual(x['quotes_sent_to_final_decision_gate'],1)
        self.assertEqual(x['rejection_reasons']['EXACT_DUPLICATE_PRICE_QUOTE'],1)
    def test_guardrail_prevents_any_real_reference_bet_even_on_patch(self):
        with patch('sports_edge.live_market_bridge.evaluate_verified_leg',return_value={'status':'BET','blockers':[]}):
            with self.assertRaises(AssertionError):
                bridge_real_reference_markets({'all_quotes':[q()]},batch(),now=NOW)

if __name__=='__main__': unittest.main()