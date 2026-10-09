import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from scripts.soccer_research_watchdog import monitor


NOW = datetime(2026, 10, 9, 19, 0, tzinfo=timezone.utc)


class ResearchFreshnessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.state = Path(self.tmp.name)
        base = self.state / "soccer_edge_state"
        (base / "analysis").mkdir(parents=True)
        samples = {
            "health.json": {"generated_at_utc": "2026-10-09T18:50:00+00:00",
                            "database_persisted": True},
            "analysis/signal_ledger_summary.json": {
                "last_evidence_at_utc": "2026-10-09T18:45:00+00:00"},
            "analysis/corners_baseline.json": {
                "status": "RESEARCH_ONLY_CORNERS_BASELINE",
                "walk_forward_evaluations": 266,
                "formation_adjusted_evaluations": 44,
                "formation_eligibility_audit": {
                    "formation_adjusted_evaluations": 44,
                    "reason_counts": {"FORMATION_PRESENT_MATCHUP_HISTORY_LT_8": 162}},
                "evaluations": [{"kickoff_local": "2026-10-08T14:00:00+00:00"}]},
            "analysis/formation_matchup_fm4_style_ablation_v1.json": {
                "style_profile": {"source_date_range": {
                    "latest": "2026-10-08T22:00:00+00:00"}}},
            "analysis/team_corners_validation.json": {"schema_version": "1.1.0", "status": "RESEARCH_ONLY_TEAM_CORNERS_VALIDATION"},
            "analysis/v4_022_corners_oos_validation.json": {"production_promotion_allowed": False, "status": "RESEARCH_HOLD"},
        }
        for path, value in samples.items():
            (base / path).write_text(json.dumps(value), encoding="utf-8")

    def run_monitor(self, **overrides):
        return monitor(self.state, now=NOW,
                       research_commit_utc=overrides.get("research_commit_utc", "2026-10-09T08:15:00Z"),
                       history_latest_date=overrides.get("history_latest_date", "2026-10-09"))

    def test_44_of_100_is_not_itself_an_incident(self):
        r = self.run_monitor()
        self.assertEqual(r["status"], "HEALTHY")
        self.assertEqual(r["measurements"]["formation_adjusted_evaluations"], 44)

    def test_stale_research_commit_alerts(self):
        r = self.run_monitor(research_commit_utc="2026-10-07T08:00:00Z")
        self.assertIn("CORNERS_RESEARCH_ARTIFACT_STALE_OR_UNVERIFIED_GT_48H", r["critical"])

    def test_fresh_live_data_old_oos_alerts(self):
        p = self.state / "soccer_edge_state/analysis/corners_baseline.json"
        r = json.loads(p.read_text())
        r["evaluations"][0]["kickoff_local"] = "2026-09-22T16:30:00-06:00"
        p.write_text(json.dumps(r))
        report = self.run_monitor()
        self.assertIn("CORNERS_VERIFIED_EVALUATION_COVERAGE_GAP_GT_7D", report["critical"])

    def test_corrupt_artifact_detected(self):
        p = self.state / "soccer_edge_state/analysis/corners_baseline.json"
        p.write_text("{")
        report = self.run_monitor()
        self.assertTrue(any(s.startswith("REQUIRED_ARTIFACT_INVALID:corners_baseline.json")
                            for s in report["critical"]))

    def test_no_automatic_promotion(self):
        p = self.state / "soccer_edge_state/analysis/v4_022_corners_oos_validation.json"
        p.write_text(json.dumps({"production_promotion_allowed": True, "status": "RESEARCH_HOLD"}))
        report = self.run_monitor()
        self.assertIn("PRODUCTION_PROMOTION_CONTRACT_VIOLATION", report["critical"])


if __name__ == "__main__":
    unittest.main()
