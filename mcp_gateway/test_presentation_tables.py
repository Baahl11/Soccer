from __future__ import annotations

import unittest

from mcp_gateway.presentation_tables import render_match_tables, validate_presentation_payload


class PresentationTablesTests(unittest.TestCase):
    def _row(self, classification: str, **overrides):
        row = {
            "fixture_id": 1,
            "country": "Mexico",
            "competition": "Liga MX",
            "data_tier": "A",
            "kickoff": "2026-09-15T02:00:00+00:00",
            "home": "Home FC",
            "away": "Away FC",
            "side_score": 71.0,
            "goals_score": 82.0,
            "two_way_score": 68.0,
            "availability_confidence": 0.93,
            "market": "Goals Over/Under",
            "selection": "Over 2.5",
            "line": 2.5,
            "price": 1.91,
            "bookmaker": "Book",
            "classification": classification,
            "tier": "B",
            "stake_units": 0.4,
            "reason": "SPORTING_SCREEN_PASS",
        }
        row.update(overrides)
        return row

    def test_renderer_has_expected_sections_and_cdmx_time(self):
        text = render_match_tables([
            self._row("BET"),
            self._row("LEAN", fixture_id=2),
            self._row("WATCH", fixture_id=3),
            self._row("PASS", fixture_id=4),
        ])
        self.assertIn("## 🟢 BET", text)
        self.assertIn("## 🟡 LEAN", text)
        self.assertIn("## 🔵 WATCH / RECHECK", text)
        self.assertIn("## ⚪ PASS — resumen por competición", text)
        # 02:00 UTC on Sep 15, 2026 is 20:00 CDMX on Sep 14.
        self.assertIn("| 20:00 |", text)
        self.assertIn("Mexico / Liga MX", text)

    def test_renderer_escapes_markdown_pipes(self):
        text = render_match_tables([
            self._row("WATCH", competition="Cup | Group A", reason="XI | GK pending")
        ])
        self.assertIn("Cup \\| Group A", text)
        self.assertIn("XI \\| GK pending", text)

    def test_pass_summary_groups_by_competition(self):
        text = render_match_tables([
            self._row("PASS", fixture_id=1, data_tier="A"),
            self._row("PASS", fixture_id=2, data_tier="B"),
        ])
        self.assertIn("| Mexico / Liga MX | 2 | 1 | 1 | 0 | 0 |", text)

    def test_schema_validator_accepts_valid_bounded_payload(self):
        payload = {
            "league_coverage_registry": {
                "provider_calls_per_league": 0,
                "galaxy_activation_is_eligibility_gate": False,
                "competition_count": 1,
                "competition_rows_attached": 1,
                "competitions": [
                    {
                        "data_tier": "A",
                        "galaxy_activation_required": False,
                    }
                ],
            },
            "match_table_row_count": 1,
            "match_table_rows_attached": 1,
            "match_table_rows": [self._row("WATCH")],
            "presentation_contract": {"default_format": "MARKDOWN_TABLES"},
        }
        result = validate_presentation_payload(payload)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["error_count"], 0)

    def test_schema_validator_rejects_galaxy_as_gate(self):
        payload = {
            "league_coverage_registry": {
                "provider_calls_per_league": 1,
                "galaxy_activation_is_eligibility_gate": True,
                "competition_count": 0,
                "competition_rows_attached": 0,
                "competitions": [],
            },
            "match_table_row_count": 0,
            "match_table_rows_attached": 0,
            "match_table_rows": [],
            "presentation_contract": {"default_format": "MARKDOWN_TABLES"},
        }
        result = validate_presentation_payload(payload)
        self.assertFalse(result["valid"])
        self.assertGreaterEqual(result["error_count"], 2)


if __name__ == "__main__":
    unittest.main()
