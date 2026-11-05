"""Immutable regression from actual GitHub Actions v1.4 quote snapshot.

The timestamp is explicitly historical: never a current pick, live price or BET.
"""
import json
import unittest
from datetime import datetime
from pathlib import Path
from sports_edge.live_market_bridge import bridge_real_reference_markets

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'real_reference_replay_2026-11-05.json'

class RealReferenceReplayTests(unittest.TestCase):
    def test_original_quote_snapshot_passes_through_final_gate_without_fake_bet(self):
        obj=json.loads(FIXTURE.read_text(encoding='utf-8'))
        now=datetime.fromisoformat(obj['reference_run_timestamp_utc'])
        out=bridge_real_reference_markets(obj['matrix'], obj['sport_first'], now=now)
        self.assertGreaterEqual(out['quotes_sent_to_final_decision_gate'],4)
        self.assertGreaterEqual(len(out['parlay_pair_reviews']),2)
        self.assertEqual(out['final_bets'],0)
        self.assertGreaterEqual(out['quotes_exact_raw_available'],4)
        for row in out['decisions']:
            self.assertEqual(row['decision'],'WATCH')
            self.assertIn('CALIENTE_PLAYDOIT_EXACT_OFFER_NOT_VERIFIED',row['decision_blockers'])
            self.assertIsNone(row['expected_value'])
            self.assertNotEqual(row['sporting_evidence_stage'],'RAW_ABOVE_PRICE_RESEARCH')
        self.assertTrue(all(p['classification']=='WATCH' and p['joint_ev'] is None for p in out['parlay_pair_reviews']))