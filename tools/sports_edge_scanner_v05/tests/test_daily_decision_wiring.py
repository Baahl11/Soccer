"""Synthetic wiring test: verify run_daily() does not force BET counters to zero.

These are explicit mocks, NOT historical games or betting recommendations.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sports_edge.daily import run_daily
from test_daily import DATE, fixture, payload, sports_research


class DailyDecisionWiringTests(unittest.TestCase):
    def test_full_daily_flow_uses_real_adjudicator_output_not_fixed_zero(self):
        synthetic={
            'schema_version':'SYNTHETIC_TEST_ONLY', 'feed_status':'TEST_SIGNED_PRODUCER',
            'singles':[{'status':'BET','event_id':'301','tier':'A'},
                       {'status':'BET','event_id':'302','tier':'A'}],
            'parlays':[{'status':'BET','events':['301','302']}],
            'actionable_bets':2,'actionable_parlays':1,
            'canonical_research':{'rows_examined':0,'blocker_counts':{}},
            'wagers_placed':0}
        with tempfile.TemporaryDirectory() as d:
            with patch('sports_edge.decision_bridge.adjudicate_daily',return_value=synthetic):
                result=run_daily(DATE,{18},d,
                    lambda day:payload([fixture(1)]),
                    research_fn=sports_research,odds_limit=0)
            self.assertEqual(result['actionable_bets'],2)
            self.assertEqual(result['parlay_actionable_bets'],1)
            self.assertTrue(result['parlay_risk_ranking_available'])
            self.assertEqual(result['wagers_placed'],0)
            decision=json.loads((Path(d)/f'decision_gate_{DATE}.json').read_text())
            self.assertEqual(decision['actionable_bets'],2)
            html=(Path(d)/f'board_{DATE}.html').read_text()
            self.assertIn('Motor final: BET / LEAN / WATCH / PASS',html)
            self.assertIn('TEST_SIGNED_PRODUCER',html)

    def test_real_default_without_certified_source_reports_reason_not_fake_bet(self):
        with tempfile.TemporaryDirectory() as d:
            result=run_daily(DATE,{18},d,
                lambda day:payload([fixture(1)]),
                research_fn=sports_research,odds_limit=0)
            self.assertEqual(result['actionable_bets'],0)
            self.assertEqual(result['parlay_actionable_bets'],0)
            self.assertIn('decision_feed_status',result)
            self.assertTrue((Path(d)/f'decision_gate_{DATE}.json').exists())

if __name__=='__main__':unittest.main()