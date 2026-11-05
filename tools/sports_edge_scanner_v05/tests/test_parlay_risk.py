import unittest
from sports_edge.parlay_risk import annotate_parlay_radar,frechet_joint_lower,_raw_probability


def data():
    radar={'status':'PRICE_ONLY_WATCH','pairs':[{'status':'WATCH','bookmaker':'Example',
        'joint_breakeven':0.5,'theoretical_combined_decimal':2.0,
        'legs':[{'event_id':'1','family':'FULL_TIME_GOALS','market':'Goal Line','selection':'Over','point':'2.5','price_decimal':1.46},
                {'event_id':'2','family':'FULL_TIME_GOALS','market':'Goal Line','selection':'Under','point':'3.5','price_decimal':1.37}]}]}
    def fx(id,above,small=3):
        return {'fixture_id':id,'availability_verified':False,'raw_sport_projection':{'status':'MODELED_LIMITED','model_version':'v1.0',
            'raw_over_2_5_prob':.713859,'raw_over_3_5_prob':above,'raw_btts_yes_prob':.6,
            'raw_home_win_prob':.5,'raw_draw_prob':.25,'raw_away_win_prob':.25,
            'sample':{'minimum_split_sample':small}}}
    screen={'screened_fixtures':[fx(1,.51),fx(2,.401183)]}
    registry={'families':[{'family':'FT_TOTALS','research_status':'RESEARCH_HOLD','production_promotion_allowed':False,'bet_eligible':False,'blockers':['FT_TOTALS_TRUE_CLV_8_LT_50']}]}
    return radar,screen,registry
class EvidenceOverlayTests(unittest.TestCase):
    def test_first_half_corners_do_not_reuse_full_time_gate(self):
        r,s,m=data()
        r['pairs'][0]['legs'][0].update(family='CORNERS',market='Total Corners (1st Half)')
        leg=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][0]
        self.assertEqual(leg['market_family_gate'],'FIRST_HALF_CORNERS')
        self.assertIsNone(leg['raw_sport_probability'])
    def test_negative_leg_flags_pair_when_other_leg_unmodeled(self):
        r,s,m=data()
        r['pairs'][0]['legs'][0].update(family='CORNERS',market='Total Corners (1st Half)')
        p=annotate_parlay_radar(r,s,m)['pairs'][0]
        self.assertEqual(p['joint_raw_sanity_status'],'AT_LEAST_ONE_LEG_RAW_BELOW_ITS_BREAK_EVEN')
    def test_each_leg_raw_and_breakeven(self):
        r=annotate_parlay_radar(*data());legs=r['pairs'][0]['legs']
        self.assertAlmostEqual(legs[0]['raw_sport_probability'],.713859,5)
        self.assertAlmostEqual(legs[1]['raw_sport_probability'],.598817,5)
        self.assertGreater(legs[0]['raw_minus_individual_breakeven_pp'],0)
        self.assertLess(legs[1]['raw_minus_individual_breakeven_pp'],0)
        self.assertEqual(r['pairs_with_negative_raw_leg'],1)
        self.assertIn('RAW_BELOW_LEG_PRICE_BREAKEVEN',legs[1]['leg_blockers'])
    def test_zero_bets_ev_and_joint_probs(self):
        p=annotate_parlay_radar(*data())['pairs'][0]
        self.assertIsNone(p['joint_lower_probability'])
        self.assertIsNone(p['joint_calibrated_probability'])
        self.assertIsNone(p['conservative_joint_ev'])
        self.assertIsNone(p['risk_rank'])
        self.assertEqual(p['status'],'WATCH')
    def test_other_market_family_no_ghost_probability(self):
        r,s,m=data()
        r['pairs'][0]['legs'][1]['family']='CORNERS'
        r['pairs'][0]['legs'][1]['market']='Total Corners (1st Half)'
        p=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][1]
        self.assertIsNone(p['raw_sport_probability'])
        self.assertEqual(p['market_validation_status'],'NOT_VERIFIED')
    def test_first_half_goals_not_reusing_full_time_model(self):
        r,s,m=data();r['pairs'][0]['legs'][1]['family']='FIRST_HALF_GOALS'
        leg=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][1]
        self.assertIsNone(leg['raw_sport_probability'])
    def test_double_chance_adds_mutually_exclusive_outcomes(self):
        p,reason=_raw_probability({'family':'DOUBLE_CHANCE','selection':'1X','point':''},data()[1]['screened_fixtures'][0]['raw_sport_projection'])
        self.assertAlmostEqual(p,.75)
    def test_unmodeled_exact_threshold_not_interpolated(self):
        p,reason=_raw_probability({'family':'FULL_TIME_GOALS','selection':'Over','point':'4.5'},data()[1]['screened_fixtures'][0]['raw_sport_projection'])
        self.assertIsNone(p)
    def test_no_fixture_mismatch(self):
        r,s,m=data();r['pairs'][0]['legs'][0]['event_id']='9999'
        leg=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][0]
        self.assertIsNone(leg['raw_sport_probability'])
        self.assertIn('FIXTURE_NOT_IN_COMPLETE_SPORT_SCREEN',leg['leg_blockers'])
    def test_alleged_promotion_only_one_flag_not_sufficient(self):
        r,s,m=data();m['families'][0]['production_promotion_allowed']=True
        leg=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][0]
        self.assertIn('MARKET_FAMILY_NOT_APPROVED_FOR_BET',leg['leg_blockers'])
    def test_preserves_underlying_radar_unmodified(self):
        r,s,m=data();annotate_parlay_radar(r,s,m)
        self.assertNotIn('raw_sport_probability',r['pairs'][0]['legs'][0])
    def test_bounds_only_from_explicit_validated_inputs(self):
        self.assertAlmostEqual(frechet_joint_lower(.75,.78),.53)
        self.assertEqual(frechet_joint_lower(.6,.3),0)
        for a,b in [(None,.8),(-1,.8),(.8,float('nan'))]:
            with self.assertRaises(ValueError):frechet_joint_lower(a,b)
    def test_does_not_create_raw_model_for_corner_or_fouls(self):
        r,s,m=data()
        for family in ['CORNERS','FOULS','FIRST_HALF_GOALS','SECOND_HALF_GOALS','CARDS']:
            r['pairs'][0]['legs'][0]['family']=family
            leg=annotate_parlay_radar(r,s,m)['pairs'][0]['legs'][0]
            self.assertIsNone(leg['raw_sport_probability'])
if __name__=='__main__':unittest.main()