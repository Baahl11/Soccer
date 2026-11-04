import unittest
from datetime import datetime,timezone,timedelta
from sports_edge.core import evaluate,generate_parlays,breakeven,ev_per_unit

NOW=datetime(2026,11,4,12,0,tzinfo=timezone.utc)
def q(event='1',price=1.5,book='TestBook'):
    return dict(sport='soccer',event_id=event,event_name=f'Team {event} vs Rival',
                starts_at_utc=(NOW+timedelta(days=1)).isoformat(),market='1x2',selection='Home',
                point='',price_decimal=price,bookmaker=book,source='TEST_NOT_LIVE',
                captured_at_utc=NOW.isoformat(),source_updated_at=NOW.isoformat())
def p(event='1',prob=.8,lower=.76):
    x=q(event)
    return dict(sport=x['sport'],event_id=event,market=x['market'],selection=x['selection'],point='',
                p_raw=prob,p_shrunk=prob,p_lower=lower,calibrated=True,
                material_availability_verified=True,availability_confidence=.99,
                model_version='soccer-test-v0')
class TestCore(unittest.TestCase):
    def test_math(self):
        self.assertAlmostEqual(breakeven(2),.5)
        self.assertAlmostEqual(ev_per_unit(.55,2),.1)
    def test_missing_model_is_watch(self):
        r=evaluate(q(),None,NOW)
        self.assertEqual(r['status'],'WATCH')
    def test_valid_large_edge_is_candidate(self):
        r=evaluate(q(),p(),NOW)
        self.assertEqual(r['status'],'BET')
        self.assertEqual(r['tier'],'S')
    def test_unverified_availability_blocked(self):
        pp=p();pp['material_availability_verified']=False
        self.assertEqual(evaluate(q(),pp,NOW)['status'],'WATCH')
    def test_uncalibrated_is_watch(self):
        pp=p();pp['calibrated']=False
        self.assertEqual(evaluate(q(),pp,NOW)['status'],'WATCH')
    def test_stale_quote_blocked(self):
        qq=q();qq['source_updated_at']=(NOW-timedelta(days=1)).isoformat()
        self.assertEqual(evaluate(qq,p(),NOW)['status'],'WATCH')
    def test_parlay_requires_independence_evidence(self):
        a=evaluate(q('1',1.4),p('1',.82,.8),NOW)
        b=evaluate(q('2',1.5),p('2',.80,.78),NOW)
        self.assertEqual(generate_parlays([a,b]),[])
        found=generate_parlays([a,b],evidence={'1|2':'VERIFIED'})
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['status'],'WATCH')
        self.assertAlmostEqual(found[0]['theoretical_price_decimal'],2.1)
    def test_no_cross_book_combination(self):
        a=evaluate(q('1',1.4),p('1',.82,.8),NOW)
        b=evaluate(q('2',1.5,'Other'),p('2',.8,.78),NOW)
        self.assertEqual(generate_parlays([a,b],evidence={'1|2':'VERIFIED'}),[])
    def test_missing_source_update_blocks_bet(self):
        qq=q();qq.pop('source_updated_at')
        self.assertEqual(evaluate(qq,p(),NOW)['status'],'WATCH')
    def test_devig_three_way_same_book_only(self):
        from sports_edge.core import attach_market_fair_probabilities
        qs=[dict(q('1',o),selection=s) for s,o in [('Home',2),('Draw',3.5),('Away',3.8)]]
        # q() uses 1x2 and all three match the same market/time/book
        out=attach_market_fair_probabilities(qs)
        self.assertAlmostEqual(sum(x['fair_market_probability'] for x in out),1)
    def test_source_update_age_blocks_false_freshness(self):
        qq=q();qq['source_updated_at']=(NOW-timedelta(hours=7)).isoformat()
        self.assertEqual(evaluate(qq,p(),NOW)['status'],'WATCH')
    def test_source_market_family(self):
        from sports_edge.core import market_family
        self.assertEqual(market_family('soccer','Match Winner'),'side')
        self.assertEqual(market_family('soccer','Goals Over/Under First Half'),'derivative')
    def test_same_event_blocked(self):
        a=evaluate(q('1',1.4),p('1',.82,.8),NOW)
        b=evaluate(q('1',1.5),p('1',.8,.78),NOW)
        self.assertEqual(generate_parlays([a,b],evidence={'1|1':'VERIFIED'}),[])
if __name__=='__main__':unittest.main()