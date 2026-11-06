"""Offline evidence audit tests; no provider calls, no BET promotion."""
import json
import tempfile
import unittest
from pathlib import Path

from sports_edge.market_unblock_tracker import (
    build_market_unblock_report, render_markdown,
)


class MarketUnblockTrackerTest(unittest.TestCase):
    def _write(self, folder, filename, payload):
        (Path(folder) / filename).write_text(json.dumps(payload), encoding="utf-8")

    def test_ordered_audit_and_bet365_provenance_is_not_invented(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "v4_017_1x2_calibration_validation.json", {
                "status": "CALIBRATION_REVIEW_ELIGIBLE",
                "canonical_multiclass_oos": {
                    "evaluated_rows": 670,
                    "baseline": {"multiclass_brier": 0.6718, "multiclass_log_loss": 1.1118},
                    "temperature_scaled": {"multiclass_brier": 0.655, "multiclass_log_loss": 1.0826},
                },
                "true_clv": {"rows": 53, "minimum_rows": 50},
                "blockers": [], "production_promotion_allowed": False,
            })
            self._write(root, "v4_019_team_totals_oos_validation.json", {
                "status": "OOS_REVIEW_ELIGIBLE",
                "oos_sample": {"evaluated_fixtures": 500},
                "true_clv": {"rows": 703, "unique_fixtures": 110},
                "canonical_clv_outcome_context": {
                    "by_line": {
                        "0.5": {"hypothetical_roi_per_priced_settled_observation": 0.10},
                        "1.5": {"hypothetical_roi_per_priced_settled_observation": -0.15},
                        "2.5": {"hypothetical_roi_per_priced_settled_observation": -0.23},
                    },
                },
                "blockers": [],
            })
            self._write(root, "v4_020_1h_oos_validation.json", {
                "status": "RESEARCH_HOLD", "calibration": {
                    "n": 220,
                    "baseline": {"brier": 0.2459, "log_loss": 0.6853},
                    "challenger": {"brier": 0.2484, "log_loss": 0.6907},
                },
                "blockers": ["CHALLENGER_BRIER_NOT_BETTER_THAN_BASELINE"],
            })
            report = build_market_unblock_report(root, generated_at="2026-11-05T00:00:00+00:00")
            self.assertEqual(report["families_audited"], 12)
            self.assertEqual(report["bookmaker"], "Bet365")
            self.assertEqual(report["manual_review_candidates"], ["1X2", "TEAM_TOTALS"])
            by_name = {row["market_family"]: row for row in report["families"]}
            self.assertEqual(by_name["1X2"]["sample_n"], 670)
            self.assertEqual(by_name["1X2"]["baseline_comparison"]["status"], "BOTH_IMPROVED")
            self.assertEqual(by_name["1X2"]["clv"]["historical_rows_unattributed"], 53)
            self.assertIsNone(by_name["1X2"]["clv"]["bet365_verified_rows"])
            self.assertEqual(by_name["TEAM_TOTALS"]["clv"]["historical_rows_unattributed"], 703)
            self.assertEqual(by_name["FIRST_HALF"]["research_decision"], "MODEL_REBUILD_REQUIRED")
            self.assertEqual(by_name["FOULS"]["research_decision"], "NOT_VERIFIED")
            self.assertTrue(all(not r["production_promotion_allowed"] and not r["bet_eligible"]
                                for r in report["families"]))
            self.assertIn("CLV Bet365 verified", render_markdown(report))
            self.assertEqual(by_name["1X2"]["release_gate"], "NOT_APPROVED")
            self.assertIn("CANONICAL_PRODUCTION_APPROVAL_NOT_GRANTED",
                          by_name["1X2"]["release_blockers"])
            self.assertIn("BET365_ID8_ENTRY_CLOSE_PROVENANCE_NOT_VERIFIED",
                          by_name["1X2"]["release_blockers"])
            self.assertIn("NEGATIVE_CANONICAL_HYPOTHETICAL_ROI_LINE_1_5",
                          by_name["TEAM_TOTALS"]["release_blockers"])
            self.assertIn("NEGATIVE_CANONICAL_HYPOTHETICAL_ROI_LINE_2_5",
                          by_name["TEAM_TOTALS"]["release_blockers"])
            self.assertNotIn("NEGATIVE_CANONICAL_HYPOTHETICAL_ROI_LINE_0_5",
                             by_name["TEAM_TOTALS"]["release_blockers"])
            self.assertIn("Release gate: NOT APPROVED", render_markdown(report))

    def test_only_explicit_bookmaker_8_with_provenance_counts(self):
        with tempfile.TemporaryDirectory() as root:
            filename = "v4_018_btts_calibration_validation.json"
            self._write(root, filename, {
                "status": "RESEARCH_HOLD", "true_clv": {
                    "rows": 50, "bookmaker_id": 8,
                    "bookmaker": "Bet365", "source_verified": False,
                },
            })
            first = build_market_unblock_report(root)
            btts = first["families"][2]
            self.assertIsNone(btts["clv"]["bet365_verified_rows"])
            self._write(root, filename, {
                "status": "RESEARCH_HOLD", "true_clv": {
                    "rows": 50, "bookmaker_id": 8,
                    "bookmaker": "Bet365", "source_verified": True,
                },
            })
            second = build_market_unblock_report(root)
            self.assertEqual(second["families"][2]["clv"]["bet365_verified_rows"], 50)
            self.assertFalse(second["families"][2]["bet_eligible"])
            self.assertIn("CANONICAL_PRODUCTION_APPROVAL_NOT_GRANTED",
                          second["families"][2]["release_blockers"])
            self.assertNotIn("BET365_ID8_ENTRY_CLOSE_PROVENANCE_NOT_VERIFIED",
                             second["families"][2]["release_blockers"])

    def test_second_half_fail_and_absent_reports_do_not_halt(self):
        with tempfile.TemporaryDirectory() as root:
            self._write(root, "v4_021_2h_oos_validation.json", {
                "status": "RESEARCH_HOLD",
                "oos_sample": {"walk_forward_evaluated": 604},
                "baseline": {"brier_o1_5": 0.253, "log_loss_o1_5": 0.699},
                "challenger": {"brier_o1_5": 0.254, "log_loss_o1_5": 0.701},
            })
            report = build_market_unblock_report(root)
            by_name = {r["market_family"]: r for r in report["families"]}
            self.assertEqual(by_name["SECOND_HALF"]["sample_n"], 604)
            self.assertEqual(by_name["SECOND_HALF"]["research_decision"], "MODEL_REBUILD_REQUIRED")
            self.assertEqual(by_name["FT_CORNERS"]["research_decision"], "NOT_VERIFIED")
            self.assertEqual(report["bet_eligible_families"], 0)


if __name__ == "__main__":
    unittest.main()
