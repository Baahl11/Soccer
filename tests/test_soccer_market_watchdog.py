import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.soccer_market_watchdog import REPORTS, audit

NOW = datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc)


class MarketAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.ana = self.state / "soccer_edge_state" / "analysis"
        self.ana.mkdir(parents=True)
        self.commits = {}
        self.put("clv_v4_postgres_report.json",
                 {"family_counts": {"1X2": 53, "BTTS": 20,
                                    "FT_TOTALS": 8, "HOME_TT": 248, "AWAY_TT": 266}},
                 "2026-10-09T16:00:00Z")
        self.put("oos_calibration_v4_report.json",
                 {"status":"OOS_CALIBRATION_MATERIALIZED","current_model_rows":870,
                  "anti_leakage":{"historical_predictions_recomputed":False}})
        self.put("player_props_clv_v4_report.json",
                 {"true_clv_rows":0,"signal_rows":451,"production_promotion_allowed":False})
        self.put("player_props_oos_v4_report.json",
                 {"unique_fixtures":21,"player_game_rows":324,"decision_weight":0,
                  "production_promotion_allowed":False})
        self.put("settlement_postgres_v4.json",
                 {"source_diagnostics":{"actionable_classification_events":0},
                  "settled_decisions":0})
        self.put("market_performance_summary.json",{"settlement_decisions":29},
                 "2026-09-24T00:00:00Z")
        self.put("research_derivative_market_audit.json",
                 {"candidate_rows":5000,"classified_rows":4853,"unclassified_rows":147,
                  "unique_fixtures":59,"provider_requests_added":0})
        source_n = {"1X2":53,"BTTS":20,"FT_TOTALS":8,"TEAM_TOTALS":514}
        for family, name in REPORTS.items():
            self.put(name, {"status":"RESEARCH_HOLD",
                            "production_promotion_allowed":False,
                            "blockers":[],
                            "true_clv":{"rows":source_n.get(family,0),"unique_fixtures":0}})

    def put(self, filename, data, committed="2026-10-09T18:00:00Z"):
        (self.ana / filename).write_text(json.dumps(data),encoding="utf-8")
        self.commits[filename]=committed

    def run_audit(self):
        return audit(self.state,self.commits,NOW)

    def test_family_counts_consistent_and_zeros_are_legitimate(self):
        r=self.run_audit()
        self.assertEqual(r["status"],"HEALTHY")
        self.assertEqual(r["families"]["1H"]["true_clv_rows"],0)
        self.assertEqual(r["families"]["TEAM_TOTALS"]["canonical_true_clv_rows"],514)

    def test_static_settlement_without_new_bets_not_false_positive(self):
        r=self.run_audit()
        self.assertEqual(r["settlement"]["status"],
                         "NO_NEW_ACTIONABLE_EVENTS_LEGITIMATE_STATIC_SETTLEMENT")
        self.assertFalse(any("SETTLEMENT_SUMMARY_STALE" in x for x in r["warnings"]))

    def test_player_props_diagnostic_not_persisted(self):
        self.commits["player_props_clv_v4_report.json"]="2026-09-27T09:46:19Z"
        r=self.run_audit()
        self.assertIn("PLAYER_PROPS_CLV_DIAGNOSTIC_NOT_PERSISTED_GT_72H",r["critical"])

    def test_clv_source_and_validator_disagreement(self):
        name=REPORTS["1X2"]
        self.put(name,{"status":"RESEARCH_HOLD",
                       "production_promotion_allowed":False,"true_clv":{"rows":42}})
        r=self.run_audit()
        self.assertTrue(any(x.startswith("CLV_SOURCE_REPORT_COUNT_MISMATCH:1X2")
                            for x in r["critical"]))

    def test_negative_model_not_auto_promoted(self):
        name=REPORTS["2H"]
        self.put(name,{"status":"RESEARCH_HOLD",
                       "production_promotion_allowed":True,
                       "true_clv":{"rows":0}})
        self.assertIn("UNAUTHORIZED_PROMOTION_FLAG:2H",self.run_audit()["critical"])

    def test_taxonomy_drift_detected(self):
        self.put("research_derivative_market_audit.json",
                 {"candidate_rows":5000,"classified_rows":4853,"unclassified_rows":130})
        self.assertIn("DERIVATIVE_MARKET_TAXONOMY_COUNT_INCONSISTENT",
                      self.run_audit()["critical"])


if __name__ == "__main__":
    unittest.main()
