import json
import tempfile
import unittest
from pathlib import Path

from sports_edge.multimarket_export import make_fe11_envelope,export_from_daily_dir
from sports_edge.multi_market_sport_profile import (
    build_multimarket_sport_profile,describe_three_leg_historical_pattern
)

KICKOFF="2026-11-07T18:00:00Z"

def payload():
    def p(fid,team,home):
        prior=[{
            "fixture_id":fid,"kickoff":"2026-11-01T18:00:00Z","status":"FT",
            "home_team_id":team if home else 90,
            "away_team_id":90 if home else team,
            "goals":{"home":2,"away":1},
            "first_half_goals":{"home":1,"away":0},
            "second_half_goals":{"home":1,"away":1},
        }]
        x=build_multimarket_sport_profile(
            prior,team_id=team,target_kickoff=KICKOFF,
            corners_by_fixture={fid:{"home":5,"away":3,"period":"FT_90"}}
        )
        pattern={name:describe_three_leg_historical_pattern(
            x,goals_over=1.5,corners_over=6.5,
            team_handicap=1,handicap_settlement=name)
            for name in ("ASIAN","THREE_WAY")}
        return {"sport_first_multimarket_profile":x,
                "example_goals_corners_handicap_pattern":pattern}
    r={"fixture_id":3001,"home":p(10,11,True),"away":p(20,22,False)}
    screen={"screened_fixtures":[{
        "fixture_id":3001,"kickoff_utc":"2026-11-07T18:00:00+00:00",
        "home_team":{"team_id":11},"away_team":{"team_id":22}
    }]}
    return (
        {"generated_at_utc":"2026-11-06T18:00:00+00:00"},
        screen,
        {"reports":[r],"selected_fixture_count":1},
    )

class FE11ScannerTransportTests(unittest.TestCase):
    def test_real_profile_export_only_sport_first_and_bookmaker_null(self):
        summary,screen,context=payload()
        doc=make_fe11_envelope(summary,screen,context)
        self.assertEqual(doc["source"],"SPORTS_EDGE_SCANNER_V15_HISTORICAL_CONTEXT")
        self.assertEqual(len(doc["reports"]),1)
        row=doc["reports"][0]
        self.assertEqual(row["fixture_id"],3001)
        self.assertEqual(row["home_profile"]["splits"]["all"]["features"]["total_corners_ft"]["mean"],8)
        self.assertIsNone(doc["calibrated_joint_probability"])
        self.assertIsNone(doc["bookmaker_ticket_quote"])
        self.assertFalse(doc["bet_eligible"])
        self.assertNotIn("price_decimal",row)

    def test_absent_or_wrong_team_or_unavailable_history_is_quarantined(self):
        summary,screen,context=payload()
        context["reports"][0]["away"]["sport_first_multimarket_profile"]["team_id"]=888
        doc=make_fe11_envelope(summary,screen,context)
        self.assertEqual(doc["reports"],[])
        self.assertEqual(doc["not_verified"][0]["reason"],"HISTORICAL_PROFILES_INCOMPLETE")
        summary,screen,context=payload()
        screen["screened_fixtures"]=[]
        doc=make_fe11_envelope(summary,screen,context)
        self.assertEqual(doc["reports"],[])
        self.assertEqual(doc["not_verified"][0]["reason"],"FIXTURE_NOT_IN_SPORT_SCREEN")
        summary,screen,context=payload()
        context["reports"][0]["away"]["sport_first_multimarket_profile"]["observations"]=[]
        doc=make_fe11_envelope(summary,screen,context)
        self.assertEqual(doc["reports"],[])

    def test_export_writes_complete_verifiable_payload_and_no_prices(self):
        summary,screen,context=payload()
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)
            for name,data in (("daily_summary",summary),("sport_first",screen),
                              ("historical_market_context",context)):
                (dest/f"{name}_2026-11-07.json").write_text(json.dumps(data))
            result=export_from_daily_dir(dest,"2026-11-07")
            self.assertEqual(result["verified_reports_for_transport"],1)
            out=json.loads((dest/"fe11_multimarket_transport_2026-11-07.json").read_text())
            self.assertEqual(len(out["reports"]),1)
            self.assertIsNone(out["bookmaker_ticket_quote"])
            self.assertFalse(out["bet_eligible"])

    def test_no_extra_provider_requests_and_max_three_only(self):
        summary,screen,context=payload()
        with self.assertRaises(ValueError):
            make_fe11_envelope(summary,screen,context,max_reports=4)
        self.assertEqual(make_fe11_envelope(summary,screen,{"reports":[]})["reports"],[])

if __name__=="__main__":unittest.main()


class ScannerConfiguredLeagueScopeTests(unittest.TestCase):
    """Research scope includes competition IDs verified in the actual slate."""
    def test_default_research_scope_does_not_ignore_active_internationals(self):
        root = Path(__file__).resolve().parents[3]
        workflow = (root / '.github/workflows/sports-edge-scanner-v05-paper.yml').read_text(encoding='utf-8')
        # Provider-confirmed Nov 12 competition IDs; no model readiness inferred.
        scope = '2,3,18,17,39,140,135,78,61,262,848,5,10,36,536'
        self.assertIn("LEAGUE_IDS: ${{ inputs.league_ids || '" + scope + "' }}", workflow)
        self.assertIn('RESEARCH ONLY, not BET approvals', workflow)
        self.assertIn('Esto NO significa cero partidos en el calendario mundial', workflow)
        self.assertIn('ámbito configurado, NO calendario mundial', workflow)
        self.assertIn("SOCCER_ADVANCED_DEEP_DIVES: '3'", workflow)
        self.assertIn("API_FOOTBALL_RESEARCH_DAILY_CAP: '250'", workflow)
        self.assertIn("API_FOOTBALL_RESEARCH_MONTHLY_CAP: '9300'", workflow)
        self.assertIn('if [ "$REPORTS" = "0" ]; then', workflow)
