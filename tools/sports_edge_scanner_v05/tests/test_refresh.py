import tempfile
import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sports_edge.refresh import pregame_refresh, _lineup_check
from sports_edge.providers import ProviderFailure

NOW = datetime(2026,11,4,18,0,tzinfo=timezone.utc)


def batch():
    return {'source_slate_complete_verified':True,'all_fixtures_attempted':True,
            'shortlists':{'merged_fixture_ids':[17]},
            'screened_fixtures':[{'fixture_id':17,'min_sample_met':True,'match':'A vs B'}]}


def fixture_at(when, status='NS'):
    return {'response':[{'fixture':{'id':17,'date':when.isoformat(), 'status':{'short':status}},
                         'teams':{'home':{'id':7},'away':{'id':8}}}]}


def xi(team_id):
    return {'team':{'id':team_id},
            'startXI':[{'player':{'pos':'G'}}]+[{'player':{'pos':'D'}}]*10}


class RefreshTests(unittest.TestCase):
    def test_provider_complete_xi_not_official_verified(self):
        a=_lineup_check({'response':[xi(7),xi(8)]}, {'home_team_id':7,'away_team_id':8})
        self.assertTrue(a['provider_lineups_complete'])
        self.assertIn('OFFICIAL',a['reason'])

    def test_missing_goalkeeper_not_complete(self):
        broken=xi(8)
        broken['startXI'][0]['player']['pos']='D'
        self.assertFalse(_lineup_check({'response':[xi(7),broken]}, {'home_team_id':7,'away_team_id':8})['provider_lineups_complete'])

    def test_no_market_before_pregame_window(self):
        calls=[]
        def getter(endpoint,params):
            calls.append(endpoint)
            return fixture_at(NOW+timedelta(hours=5))
        with tempfile.TemporaryDirectory() as d:
            result=pregame_refresh(batch(),17,d,now=NOW,reference_odds=True,
                                   request_fn=getter,odds_fn=lambda _:self.fail('No premature market fetch'))
            self.assertEqual(calls,['fixtures'])
            self.assertEqual(result['reason'],'PREGAME_WINDOW_NOT_REACHED')
            self.assertFalse(result['actionable_bet'])
            self.assertTrue(Path(result['report_file']).exists())

    def test_started_fixture_cannot_research_market(self):
        with tempfile.TemporaryDirectory() as d:
            result=pregame_refresh(batch(),17,d,now=NOW,
                                   request_fn=lambda *_:fixture_at(NOW-timedelta(minutes=5),'2H'),
                                   odds_fn=lambda _:self.fail('Cannot fetch after kickoff'),reference_odds=True)
            self.assertEqual(result['reason'],'MATCH_STARTED_OR_NOT_ELIGIBLE')

    def test_t40_reports_lineup_confidence_without_bet(self):
        endpoint_calls=[]
        def getter(endpoint,params):
            endpoint_calls.append(endpoint)
            if endpoint=='fixtures':return fixture_at(NOW+timedelta(minutes=40))
            if endpoint=='fixtures/lineups':return {'response':[xi(7),xi(8)]}
            if endpoint=='injuries':return {'response':[]}
            raise AssertionError(endpoint)
        with tempfile.TemporaryDirectory() as d:
            result=pregame_refresh(batch(),17,d,now=NOW,request_fn=getter)
            self.assertEqual(result['refresh_stage'],'T-40')
            self.assertTrue(result['provider_lineups']['provider_lineups_complete'])
            self.assertFalse(result['official_lineup_verified'])
            self.assertEqual(result['provider_injuries']['provider_records'],0)
            self.assertFalse(result['provider_injuries']['injuries_cleared'])
            self.assertEqual(endpoint_calls,['fixtures','fixtures/lineups','injuries'])

    def test_t20_missing_lineup_still_watch(self):
        def getter(endpoint,params):
            return fixture_at(NOW+timedelta(minutes=20)) if endpoint=='fixtures' else {'response':[]}
        with tempfile.TemporaryDirectory() as d:
            result=pregame_refresh(batch(),17,d,now=NOW,request_fn=getter)
            self.assertEqual(result['refresh_stage'],'T-20')
            self.assertFalse(result['provider_lineups']['provider_lineups_complete'])
            self.assertEqual(result['classification'],'WATCH')

    def test_must_have_sport_first_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ProviderFailure):
                pregame_refresh({'source_slate_complete_verified':True,'all_fixtures_attempted':True,
                                 'shortlists':{'merged_fixture_ids':[17]},'screened_fixtures':[]},
                                17,d,now=NOW,request_fn=lambda *_:self.fail('Must block'))

if __name__=='__main__':unittest.main()