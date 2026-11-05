import copy
import unittest
from datetime import datetime, timedelta, timezone
from sports_edge.decision_gate import (
    _identity, evaluate_verified_leg, evaluate_verified_parlay,
    adjudicate_offers, inspect_canonical_research)

NOW = datetime(2026, 11, 5, 17, 0, tzinfo=timezone.utc)


def offer(event='301', price=1.42, market='Goals Over/Under', selection='Over', point='2.5', book='Bet365'):
    return {'sport':'soccer','event_id':event,'event_name':f'Fixture {event}',
            'starts_at_utc':(NOW+timedelta(hours=3)).isoformat(),
            'market':market,'selection':selection,'point':point,
            'bookmaker':book,'price_decimal':price,
            'source_updated_at':(NOW-timedelta(minutes=8)).isoformat(),
            'source':'API-Football','bookmaker_id':8,
            'verified_offer_source':'API_FOOTBALL_BET365_ID8',
            'price_observation_verified':True,'verified_offer':False,
            'offer_reference_id':None}


def model(q, p_raw=.89, p=.83, lower=.77):
    return {'sport':'soccer','event_id':q['event_id'],'market':q['market'],
            'selection':q['selection'],'point':q['point'],
            'p_raw':p_raw,'p_shrunk':p,'p_lower':lower,
            'model_version':'TEST_SYNTHETIC_VALIDATED_SOCCER_1.0',
            'model_artifact_id':'TEST_ARTIFACT_NOT_PRODUCTION',
            'promotion_status':'PRODUCTION_APPROVED','production_promotion_allowed':True,
            'calibrated':True,'calibration_out_of_sample':True,'oos_rows':350,
            'raw_sport_built_without_market_input':True,'data_tier':'A',
            'fixture_status':'NS','fixture_status_verified':True,
            'xi_and_goalkeepers_verified':True,'material_availability_verified':True,
            'availability_source':'OFFICIAL_TEAM','availability_confidence':.96,
            'unresolved_material_conflict':False,
            'model_generated_at_utc':(NOW-timedelta(minutes=12)).isoformat()}


def ticket(a, b, price=2.059):
    return {'leg_keys':[list(_identity(a)),list(_identity(b))],
            'price_decimal':price,'bookmaker':'Bet365','bookmaker_id':8,
            'verified_offer_source':'API_FOOTBALL_BET365_ID8',
            'theoretical_from_two_quotes':True,'individual_quotes_verified':True,
            'ticket_reference_id':None,
            'source_updated_at':(NOW-timedelta(minutes=3)).isoformat(),
            'portfolio_exposure_approved':True}


def independent(a, b, p=.62):
    return {'status':'VERIFIED','method':'JOINT_OUTCOME_OOS',
            'approved_for_decisions':True,'oos_joint_rows':500,
            'event_ids':[a['event_id'],b['event_id']], 'joint_p_lower':p}


class DecisionGateTests(unittest.TestCase):
    def setUp(self):
        self.a=offer('301')
        self.b=offer('302', price=1.45, market='Both Teams Score',selection='Yes',point='')
        self.ma=model(self.a)
        self.mb=model(self.b)

    def test_integrated_real_gate_can_emit_two_singles_and_parlay_bet(self):
        out=adjudicate_offers([self.a,self.b],[self.ma,self.mb],
                             parlay_offers=[ticket(self.a,self.b)],
                             joint_reports=[independent(self.a,self.b)],now=NOW)
        self.assertEqual(out['actionable_bets'],2)
        self.assertEqual(out['actionable_parlays'],1)
        self.assertTrue(all(x['status']=='BET' for x in out['singles']))
        self.assertEqual(out['parlays'][0]['status'],'BET')
        self.assertGreater(out['parlays'][0]['conservative_joint_ev'],0)
        self.assertEqual(out['wagers_placed'],0)
        self.assertEqual(out['parlays'][0]['stake_units'],0)

    def test_no_model_is_watch_not_hardcoded_bet(self):
        out=adjudicate_offers([self.a],[],now=NOW)
        self.assertEqual(out['actionable_bets'],0)
        self.assertEqual(out['singles'][0]['status'],'WATCH')
        self.assertIn('EXACT_MARKET_MODEL_MISSING',out['singles'][0]['blockers'])

    def test_current_research_calibrator_stays_watch(self):
        m={**self.ma,'promotion_status':'RESEARCH_CALIBRATION_APPLIED',
           'production_promotion_allowed':False,'calibrated':False}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertEqual(r['status'],'WATCH')
        self.assertIn('MODEL_NOT_APPROVED_FOR_PRODUCTION',r['blockers'])
        self.assertIsNone(r['conservative_ev'])

    def test_bet365_id8_observation_is_eligible_source_when_model_verified(self):
        r=evaluate_verified_leg(self.a,self.ma,now=NOW)
        self.assertEqual(r['status'],'BET')
        self.assertFalse(r['stake_units'])

    def test_untrusted_bet365_source_cannot_pass(self):
        q={**self.a,'bookmaker_id':999,'verified_offer_source':'UNTRUSTED'}
        r=evaluate_verified_leg(q,self.ma,now=NOW)
        self.assertEqual(r['status'],'WATCH')
        self.assertIn('BET365_PROVIDER_PRICE_SOURCE_NOT_VERIFIED',r['blockers'])

    def test_price_staleness_blocks_bet(self):
        q={**self.a,'source_updated_at':(NOW-timedelta(minutes=160)).isoformat()}
        r=evaluate_verified_leg(q,self.ma,now=NOW)
        self.assertIn('BOOKMAKER_PRICE_NOT_FRESH',r['blockers'])

    def test_low_confidence_blocks_bet(self):
        m={**self.ma,'availability_confidence':.75}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertIn('AVAILABILITY_CONFIDENCE_BELOW_PARLAY_STANDARD',r['blockers'])

    def test_cannot_infer_shrinkage_or_lower_confidence_bound(self):
        m={**self.ma,'p_lower':None}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertIn('VALID_RAW_SHRUNK_LOWER_PROBABILITIES_MISSING',r['blockers'])

    def test_unverified_fair_market_probability_cannot_inflate_edge(self):
        q={**self.a,'fair_market_probability':.33}
        r=evaluate_verified_leg(q,self.ma,now=NOW)
        self.assertEqual(r['status'],'WATCH')
        self.assertIn('MARKET_FAIR_PROBABILITY_ORIGIN_UNVERIFIED',r['blockers'])

    def test_joint_probability_cannot_exceed_marginal_lower_bound(self):
        a=evaluate_verified_leg(self.a,self.ma,now=NOW)
        b=evaluate_verified_leg(self.b,self.mb,now=NOW)
        r=evaluate_verified_parlay(a,b,ticket(self.a,self.b),independent(self.a,self.b,p=.9),now=NOW)
        self.assertEqual(r['status'],'WATCH')
        self.assertIn('JOINT_OOS_EVIDENCE_INVALID',r['blockers'])

    def test_model_not_opponent_independent(self):
        m={**self.ma,'raw_sport_built_without_market_input':False}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertIn('RAW_SPORT_INDEPENDENCE_NOT_VERIFIED',r['blockers'])

    def test_mismatched_selection_or_line(self):
        m={**self.ma,'point':'3.5'}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertIn('MODEL_MARKET_OR_FIXTURE_MISMATCH',r['blockers'])

    def test_price_below_edge_becomes_pass_when_verified(self):
        m={**self.ma,'p_shrunk':.69,'p_lower':.67,'p_raw':.73}
        r=evaluate_verified_leg(self.a,m,now=NOW)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['reason'],'INSUFFICIENT_VALUE')

    def test_one_unverified_leg_blocks_parlay(self):
        single=evaluate_verified_leg(self.a,self.ma,now=NOW)
        bad=evaluate_verified_leg(self.b,{**self.mb,'calibrated':False},now=NOW)
        p=evaluate_verified_parlay(single,bad,ticket(self.a,self.b),independent(self.a,self.b),now=NOW)
        self.assertEqual(p['status'],'WATCH')

    def test_synthetic_parlay_without_actual_ticket_not_bet(self):
        a=evaluate_verified_leg(self.a,self.ma,now=NOW)
        b=evaluate_verified_leg(self.b,self.mb,now=NOW)
        r=evaluate_verified_parlay(a,b,None,independent(self.a,self.b),now=NOW)
        self.assertEqual(r['status'],'WATCH')
        self.assertIn('BET365_TWO_QUOTE_PRICE_NOT_VERIFIED',r['blockers'])

    def test_frechet_bound_rejects_insufficient_joint_margin(self):
        a=evaluate_verified_leg(self.a,self.ma,now=NOW)
        b=evaluate_verified_leg(self.b,self.mb,now=NOW)
        r=evaluate_verified_parlay(a,b,ticket(self.a,self.b),None,now=NOW)
        self.assertEqual(r['status'],'PASS')
        self.assertEqual(r['joint_probability_method'],'FRECHET_ANY_DEPENDENCE')

    def test_joint_lower_insufficient_becomes_pass(self):
        a=evaluate_verified_leg(self.a,self.ma,now=NOW)
        b=evaluate_verified_leg(self.b,self.mb,now=NOW)
        r=evaluate_verified_parlay(a,b,ticket(self.a,self.b),independent(self.a,self.b,p=.48),now=NOW)
        self.assertEqual(r['status'],'PASS')

    def test_frechet_bound_approves_two_validated_strong_marginals(self):
        a=evaluate_verified_leg(self.a,model(self.a,lower=.82),now=NOW)
        b=evaluate_verified_leg(self.b,model(self.b,lower=.82),now=NOW)
        r=evaluate_verified_parlay(a,b,ticket(self.a,self.b),None,now=NOW)
        self.assertEqual(r['status'],'BET')
        self.assertEqual(r['joint_probability_method'],'FRECHET_ANY_DEPENDENCE')
        self.assertGreater(r['conservative_joint_ev'],0)
        self.assertTrue(r['theoretical_price_only'])
        self.assertTrue(r['bet365_betslip_recheck_required'])

    def test_no_same_game_parlay(self):
        a=evaluate_verified_leg(self.a,self.ma,now=NOW)
        b={**a,'market':'Both Teams Score','selection':'Yes','point':''}
        q=ticket(self.a,self.b)
        r=evaluate_verified_parlay(a,b,q,independent(self.a,self.b),now=NOW)
        self.assertIn('MUST_BE_DIFFERENT_FIXTURES',r['blockers'])

    def test_incomplete_slate_overrides_bet(self):
        x=adjudicate_offers([self.a,self.b],[self.ma,self.mb],
                           parlay_offers=[ticket(self.a,self.b)],
                           joint_reports=[independent(self.a,self.b)],
                           slate_complete=False,now=NOW)
        self.assertEqual(x['actionable_bets'],0)
        self.assertEqual(x['actionable_parlays'],0)
        self.assertEqual(x['parlays'][0]['status'],'WATCH')

    def test_canonical_research_do_not_turn_into_promoted_model(self):
        original={'is_fresh':True,'status':'FRESH_RESEARCH_ONLY',
                  'matching_fixture_rows':[{'fixture_id':123,'status':'NS','home':'A','away':'B',
                      'market':'Goals Over/Under','selection':'Over','line':2.5,
                      'classification':'WATCH','p_shrunk':None,'p_model_calibrated':.75,
                      'production_promotion_allowed':False,'bet_eligible':False,
                      'source':'CANONICAL_READ_ONLY'}]}
        report=inspect_canonical_research(original)
        self.assertEqual(report['rows_examined'],1)
        self.assertIn('MODEL_RESEARCH_ONLY_NOT_PROMOTED',report['rows'][0]['blockers'])
        self.assertIn('SHRUNK_LOWER_BOUND_NOT_EXPORTED',report['rows'][0]['blockers'])

if __name__=='__main__':
    unittest.main()