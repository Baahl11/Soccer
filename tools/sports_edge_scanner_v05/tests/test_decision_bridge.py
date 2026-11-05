import json
import hmac
import hashlib
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from sports_edge.decision_bridge import adjudicate_daily, INPUT_SCHEMA
from test_decision_gate import NOW, offer, model, ticket, independent
KEY = "TEST_ONLY_NOT_REAL_0123456789abcdef0123456789abcdef"

def sign(payload):
    canonical=json.dumps(payload,sort_keys=True,separators=(",", ":"),ensure_ascii=False).encode("utf-8")
    return {**payload,"signature_hmac_sha256":hmac.new(KEY.encode("utf-8"),canonical,hashlib.sha256).hexdigest()}


class DecisionBridgeTests(unittest.TestCase):
    def setUp(self):
        self.canonical={'status':'FRESH_RESEARCH_ONLY','is_fresh':True,
                        'matching_fixture_rows':[{'fixture_id':123,'status':'NS',
                                'market':'Goals Over/Under','line':2.5,'classification':'WATCH',
                                'p_shrunk':None,'production_promotion_allowed':False,
                                'bet_eligible':False,'source':'CANONICAL_READ_ONLY'}]}

    def test_real_research_canonical_does_not_force_fake_actionable_bet(self):
        r=adjudicate_daily(self.canonical,slate_complete=True,now=NOW)
        self.assertEqual(r['feed_status'],'BET365_PRODUCTION_MODEL_FEED_NOT_CONNECTED')
        self.assertEqual(r['actionable_bets'],0)
        self.assertEqual(r['canonical_research']['rows_examined'],1)
        self.assertIn('MODEL_RESEARCH_ONLY_NOT_PROMOTED',r['canonical_research']['blocker_counts'])

    def test_integration_trusted_feed_produces_real_engine_bet_status(self):
        a,b=offer('901'),offer('902',1.45,'Both Teams Score','Yes','')
        payload={'schema_version':INPUT_SCHEMA,
                 'trusted_producer':'SOCCER_EDGE_CANONICAL_SIGNED_EXPORT',
                 'generated_at_utc':NOW.isoformat(),
                 'offers':[a,b],'models':[model(a),model(b)],
                 'parlay_offers':[ticket(a,b)],
                 'joint_evidence':[independent(a,b)]}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'decision.json';path.write_text(json.dumps(sign(payload)))
            r=adjudicate_daily(self.canonical,slate_complete=True,now=NOW,trusted_file=path,signing_key=KEY)
        self.assertEqual(r['feed_status'],'TRUSTED_PRODUCER_ASSERTION_SUPPLIED')
        self.assertEqual(r['actionable_bets'],2)
        self.assertEqual(r['actionable_parlays'],1)
        self.assertEqual(r['parlays'][0]['status'],'BET')
        self.assertEqual(r['wagers_placed'],0)

    def test_forged_signature_cannot_emit_bet(self):
        q=offer('999')
        payload={'schema_version':INPUT_SCHEMA,'trusted_producer':'SOCCER_EDGE_CANONICAL_SIGNED_EXPORT',
                 'generated_at_utc':NOW.isoformat(),'offers':[q],'models':[model(q)],
                 'parlay_offers':[],'joint_evidence':[]}
        signed=sign(payload)
        signed['offers'][0]['price_decimal']=1.77  # after signing
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'forged.json';path.write_text(json.dumps(signed))
            out=adjudicate_daily(self.canonical,slate_complete=True,now=NOW,trusted_file=path,signing_key=KEY)
        self.assertEqual(out['actionable_bets'],0)
        self.assertEqual(out['feed_status'],'TRUSTED_DECISION_FEED_INVALID')

    def test_untrusted_producer_cannot_pretend_production(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'decision.json'
            path.write_text(json.dumps({'schema_version':INPUT_SCHEMA,
                                        'trusted_producer':'WEB_SCRAPER','generated_at_utc':NOW.isoformat(),
                                        'offers':[],'models':[],'parlay_offers':[],'joint_evidence':[]}))
            r=adjudicate_daily(self.canonical,slate_complete=True,now=NOW,trusted_file=path,signing_key=KEY)
        self.assertEqual(r['actionable_bets'],0)
        self.assertEqual(r['feed_status'],'TRUSTED_DECISION_FEED_INVALID')

if __name__=='__main__':unittest.main()