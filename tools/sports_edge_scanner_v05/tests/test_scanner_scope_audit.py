"""Source-scope reconciliation tests: no network, no betting decisions."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.soccer_scanner_scope_audit import audit

NOW = datetime(2026, 11, 12, 20, 55, tzinfo=timezone.utc)


def receipt():
    return {
        "status": "CAPTURED_RESEARCH_ONLY",
        "generated_at_utc": (NOW-timedelta(minutes=15)).isoformat(),
        "eligible_fixture_ids": 9,
        "league_coverage_report_truncated": False,
        "league_coverage_report_total_leagues": 3,
        "league_coverage_report": [
            {"competition_id": 5, "league": "UEFA Nations League", "eligible": 4},
            {"competition_id": 36, "league": "AFCON Qualification", "eligible": 3},
            {"competition_id": 999, "league": "Outside configured scope", "eligible": 2},
        ],
    }


class ScopeAuditTests(unittest.TestCase):
    def test_official_scope_overlap_is_not_a_model_or_market_prediction(self):
        result = audit(receipt(), "2,3,5,36", now=NOW)
        self.assertEqual(result["status"], "SCOPE_OVERLAP")
        self.assertEqual(result["total_source_eligible"], 9)
        self.assertEqual(result["in_configured_leagues"], 7)
        self.assertEqual(result["outside_configured_leagues"], 2)
        self.assertEqual(result["uncovered_competitions"][0]["league_id"], 999)
        self.assertFalse(result["model_eligibility_promoted"])
        self.assertFalse(result["bet_eligible"])
        self.assertFalse(result["market_inputs_used"])
        self.assertEqual(result["provider_requests_added"], 0)

    def test_zero_overlap_is_explicit_and_does_not_claim_no_football(self):
        result = audit(receipt(), "2,39,140", now=NOW)
        self.assertEqual(result["status"], "SCOPE_ZERO_OVERLAP")
        self.assertEqual(result["total_source_eligible"], 9)
        self.assertEqual(result["in_configured_leagues"], 0)
        self.assertEqual(result["outside_configured_leagues"], 9)
        self.assertIn("TWO_DATE_WINDOW", result["source_scope"])

    def test_truncated_inconsistent_or_duplicated_source_fails_closed(self):
        variants = []
        a = receipt()
        a["league_coverage_report_truncated"] = True
        variants.append(a)
        b = receipt()
        b["eligible_fixture_ids"] = 90
        variants.append(b)
        c = receipt()
        c["league_coverage_report"][1]["competition_id"] = 5
        variants.append(c)
        for source in variants:
            self.assertEqual(audit(source, "5,36", now=NOW)["status"], "NOT_VERIFIED")

    def test_stale_or_future_source_or_invalid_config_cannot_be_scope_truth(self):
        stale = receipt()
        stale["generated_at_utc"] = (NOW-timedelta(hours=7)).isoformat()
        future = receipt()
        future["generated_at_utc"] = (NOW+timedelta(hours=1)).isoformat()
        for source in (stale, future):
            self.assertEqual(audit(source, "5,36", now=NOW)["status"], "NOT_VERIFIED")
        for ids in ("", "5,abc", "5,-36", "5,0"):
            self.assertEqual(audit(receipt(), ids, now=NOW)["status"], "NOT_VERIFIED")


if __name__ == "__main__":
    unittest.main()
