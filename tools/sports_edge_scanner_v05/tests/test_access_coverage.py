"""Defensive gates tested entirely offline. No API quotas used."""
import unittest, json, os, subprocess, sys, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
from sports_edge.coverage import audit_fixture_response, audit_connector_counts, audit_capabilities

ROOT = Path(__file__).resolve().parents[1]

class CoverageAuditTests(unittest.TestCase):
    def test_connector_truncation_cannot_claim_full_slate(self):
        data={'total_results':276,'returned_results':100,'truncated':True,'fixtures':[
            {'fixture_id':i,'status':'NS','league_id':39,'league':'Premier League'} for i in range(100)]}
        r=audit_fixture_response(data,source_kind='connected')
        self.assertEqual(r['missing_minimum'],176)
        self.assertFalse(r['complete_slate_verified'])
        self.assertFalse(r['league_coverage_verified'])
        self.assertEqual(r['count_observed_not_started'],100)
    def test_incomplete_counter_only_never_invents_missing_games(self):
        r=audit_connector_counts({'total_results':276,'returned_results':100,'truncated':True})
        self.assertEqual(r['missing_minimum'],176)
        self.assertFalse(r['complete_slate_verified'])
    def test_counter_only_not_equal_to_complete_audited_rows(self):
        r=audit_connector_counts({'total_results':2,'returned_results':2,'truncated':False})
        self.assertFalse(r['complete_slate_verified'])
    def test_direct_complete_validated(self):
        r=audit_fixture_response({'results':2,'paging':{'total':1},'response':[
            {'fixture':{'status':{'short':'FT'}},'league':{'id':39,'name':'EPL'}},
            {'fixture':{'status':{'short':'NS'}},'league':{'id':39,'name':'EPL'}}]},source_kind='direct')
        self.assertTrue(r['complete_slate_verified'])
        self.assertEqual(r['observed_status_counts']['FT'],1)
        self.assertEqual(r['count_observed_not_started'],1)
    def test_direct_paging_total_must_not_be_ignored(self):
        r=audit_fixture_response({'results':2,'paging':{'total':2},'response':[
            {'fixture':{'status':{'short':'NS'}},'league':{'id':18,'name':'AFC'}},
            {'fixture':{'status':{'short':'NS'}},'league':{'id':18,'name':'AFC'}}]},source_kind='direct')
        self.assertFalse(r['complete_slate_verified'])
    def test_access_error_does_not_mean_no_matches(self):
        p={'fixture':{'response':[{'fixture':{'id':1}}], 'quota':{'daily_remaining':'89'}},
           'odds':{'response':[{'bookmakers':[]}]},
           'lineups':{'response':[]},
           'injuries':{'response':[]},
           'match_stats':{'response':[{'team':{'id':1}}]},
           'league_coverage':{'error':{'plan':'Free plans do not have access to this season'}},
           'team_stats':{'error':{'plan':'Free plans do not have access to this season'}}}
        r=audit_capabilities(p)
        self.assertEqual(r['endpoints']['team_stats']['state'],'BLOCKED_BY_PLAN')
        self.assertEqual(r['endpoints']['lineups']['interpretation'],'LINEUP_NOT_VERIFIED')
        self.assertEqual(r['endpoints']['injuries']['interpretation'],'NO_PROVIDER_RECORDS_NOT_HEALTH_VERIFICATION')
        self.assertEqual(r['last_observed_daily_remaining'],89)
        self.assertFalse(r['current_season_team_strength_ready'])
    def test_missing_probe_is_not_available(self):
        self.assertEqual(audit_capabilities({})['endpoints']['odds']['state'],'NOT_TESTED')
    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError): audit_connector_counts({'total_results':100,'returned_results':101})

class CliGateTests(unittest.TestCase):
    def test_no_coverage_downgrades_hypothetical_bet(self):
        now=datetime.now(timezone.utc)
        event=now+timedelta(hours=3)
        quote={'sport':'soccer','event_id':'id-1','event_name':'Test','starts_at_utc':event.isoformat(),
          'market':'Match Winner','selection':'Home','point':'','price_decimal':2.0,
          'bookmaker':'TEST','source':'TEST_API','source_updated_at':now.isoformat(),
          'captured_at_utc':now.isoformat()}
        projection={'sport':'soccer','event_id':'id-1','market':'Match Winner','selection':'Home',
            'point':'','p_raw':0.69,'p_shrunk':0.64,'p_lower':0.60,'calibrated':True,
            'availability_confidence':1,'material_availability_verified':True}
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);(d/'q.json').write_text(json.dumps([quote]));(d/'p.json').write_text(json.dumps([projection]));
            (d/'audit.json').write_text(json.dumps({'match_date':'1900-01-01','complete_slate_verified':True}))
            cmd=[sys.executable,'-m','sports_edge','scan','--odds',str(d/'q.json'),
                '--projections',str(d/'p.json'),'--out',str(d/'result.json')]
            run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((d/'result.json').read_text())
            self.assertEqual(result['singles'][0]['status'],'WATCH')
            self.assertEqual(result['singles'][0]['reason'],'SPORT_DATE_SLATE_COVERAGE_NOT_VERIFIED')
            cmd += ['--coverage-audit',str(d/'audit.json')]
            run=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            result=json.loads((d/'result.json').read_text())
            self.assertEqual(result['singles'][0]['status'],'WATCH')
            self.assertEqual(result['singles'][0]['reason'],'SPORT_DATE_SLATE_COVERAGE_NOT_VERIFIED')

if __name__ == '__main__':unittest.main()