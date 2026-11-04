"""Full-slate daily tests: no network and NO betting outputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

from sports_edge.daily import preliminary_signals, screen_complete_slate, fetch_research_reference_odds, run_daily
from sports_edge.pro_pipeline import build_pro_slate, get_cached_team_stats
from sports_edge.providers import ProviderFailure

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-11-05'


def fixture(fid, league=18, home_id=123, away_id=321, status='NS'):
    return {'fixture': {'id': fid, 'date': '2026-11-05T10:00:00+00:00', 'status': {'short': status}},
            'league': {'id': league, 'name': 'AFC Cup', 'season': 2026},
            'teams': {'home': {'id': home_id, 'name': f'Home{fid}'},
                      'away': {'id': away_id, 'name': f'Away{fid}'}}}


def payload(fixtures, reported=None):
    return {'results': len(fixtures) if reported is None else reported,
            'paging': {'total': 1}, 'response': fixtures}


def sports_research(item):
    return {'stage': 'SPORT_FIRST_RESEARCH', 'fixture_id': item['fixture_id'],
            'match': f"{item['home_team']} vs {item['away_team']}",
            'status': 'WATCH', 'calibrated': False,
            'min_sample_met': True, 'raw_model_probability': None,
            'market_shrunk_probability': None,
            'home_team': {'played': 10, 'gf_avg': 2.0, 'ga_avg': 0.5},
            'away_team': {'played': 10, 'gf_avg': 1.1, 'ga_avg': 1.8}}


class DailyTest(unittest.TestCase):
    def test_signals_not_model_probability(self):
        signals = preliminary_signals(sports_research({'fixture_id':1,'home_team':'A','away_team':'B'}))
        self.assertIsNone(signals['corners_signal'])
        self.assertEqual(signals['side_direction'], 'home')
        self.assertFalse(signals['probability_generated'])
        self.assertNotIn('p_model', signals)

    def test_all_eligible_not_just_favorites(self):
        raw = payload([fixture(1), fixture(2), fixture(3, league=99), fixture(4, status='FT')])
        slate = build_pro_slate(raw, DATE, {18})
        called = []
        def research(item):
            called.append(item['fixture_id'])
            return sports_research(item)
        result = screen_complete_slate(slate, research_fn=research)
        self.assertEqual(called, [1,2])
        self.assertEqual(result['eligible_count'], 2)
        self.assertTrue(result['complete_sport_screen'])
        self.assertEqual(result['shortlists']['corners_fixture_ids'], [])
        self.assertEqual(result['classification'], 'WATCH')
        self.assertEqual(result['calibrated_probabilities_produced'], 0)

    def test_missing_data_still_attempted_and_watch(self):
        slate=build_pro_slate(payload([fixture(1),fixture(2)]),DATE,{18})
        def research(f):
            if f['fixture_id']==2: raise ProviderFailure('No season stats')
            return sports_research(f)
        r=screen_complete_slate(slate,research_fn=research)
        self.assertTrue(r['all_fixtures_attempted'])
        self.assertFalse(r['complete_sport_screen'])
        self.assertEqual(r['not_verified_fixture_ids'],[2])
        self.assertEqual(r['screened_fixtures'][1]['status'],'WATCH')
        self.assertEqual(r['shortlists']['merged_fixture_ids'],[1])

    def test_incomplete_global_fixture_list_hard_stops(self):
        slate=build_pro_slate(payload([fixture(1)], reported=100),DATE,{18})
        with self.assertRaises(ProviderFailure):
            screen_complete_slate(slate, research_fn=lambda _: self.fail('Unexpected sports research'))

    def test_odds_are_after_screen_and_watch_only(self):
        slate=build_pro_slate(payload([fixture(1),fixture(2)]),DATE,{18})
        batch=screen_complete_slate(slate,research_fn=sports_research)
        calls=[]
        def mock_odds(fid):
            calls.append(fid)
            return {'retrieved_at_utc':'2026-11-04T18:00:00+00:00', 'response':[
                {'fixture':{'id':fid,'date':'2026-11-05T10:00:00+00:00'},
                 'update':'2026-11-04T16:00:00+00:00',
                 'bookmakers':[{'name':'Third-Party Book','bets':[{'name':'Match Winner',
                                'values':[{'value':'Home','odd':'1.55'}]}]}]}]}
        with tempfile.TemporaryDirectory() as d:
            r=fetch_research_reference_odds(batch,d, max_fixtures=1,odds_fn=mock_odds)
            self.assertEqual(len(calls),1)
            self.assertEqual(r['actionable_bets'],0)
            self.assertEqual(r['classification'],'WATCH')
            document=json.loads(Path(r['snapshots'][0]['file']).read_text())
            self.assertEqual(document['quotes'][0]['source_updated_at'],'2026-11-04T16:00:00+00:00')
            self.assertFalse(document['actionable_at_caliente_or_playdoit'])
            self.assertNotEqual(document['quotes'][0]['source_updated_at'],document['checked_at_utc'])

    def test_no_reference_quotes_at_zero_limit(self):
        batch=screen_complete_slate(build_pro_slate(payload([fixture(1)]),DATE,{18}),research_fn=sports_research)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(fetch_research_reference_odds(batch,d,0,odds_fn=lambda _:self.fail('No API call'))['selected_count'],0)

    def test_wrong_event_odds_rejected(self):
        batch=screen_complete_slate(build_pro_slate(payload([fixture(1)]),DATE,{18}),research_fn=sports_research)
        def other_event(_):
            return {'response':[{'fixture':{'id':99},'bookmakers':[{'name':'Other','bets':[
                    {'name':'Match Winner','values':[{'value':'Home','odd':'1.9'}]}]}]}]}
        with tempfile.TemporaryDirectory() as d:
            out=fetch_research_reference_odds(batch,d,1,odds_fn=other_event)
            self.assertEqual(out['captured_count'],0)
            self.assertEqual(len(out['failures']),1)

    def test_daily_blocks_partial_without_sport_or_odds_calls(self):
        with tempfile.TemporaryDirectory() as d:
            r=run_daily(DATE,{18},d,lambda _day:payload([fixture(1)],reported=10),
                        research_fn=lambda _:self.fail('Should never sport-screen'),
                        odds_limit=1,odds_fn=lambda _:self.fail('Should never fetch odds'))
            self.assertEqual(r['status'],'BLOCKED_INCOMPLETE_SLATE')
            self.assertEqual(r['actionable_bets'],0)
            self.assertTrue((Path(d)/f'slate_{DATE}.json').exists())

    def test_daily_complete_preserves_all_stages(self):
        with tempfile.TemporaryDirectory() as d:
            r=run_daily(DATE,{18},d,lambda _day:payload([fixture(1),fixture(2,status='FT')]),
                        research_fn=sports_research,odds_limit=0)
            self.assertEqual(r['status'],'PREFINAL_WATCH_RESEARCH')
            self.assertEqual(r['eligible_games'],1)
            self.assertEqual(r['sports_screened'],1)
            self.assertEqual(r['odds_snapshots_captured'],0)
            self.assertEqual(r['actionable_bets'],0)
            self.assertTrue((Path(d)/f'sport_first_{DATE}.json').exists())

    def test_daily_cli_offline_partial_exit(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'incomplete.json'
            path.write_text(json.dumps(payload([fixture(1)],reported=3)))
            result=subprocess.run([sys.executable,'-m','sports_edge','pro-soccer-daily',
                                   '--date',DATE,'--leagues','18','--raw-file',str(path),
                                   '--output-dir',str(Path(d)/'out')],cwd=ROOT,
                                  capture_output=True,text=True)
            self.assertEqual(result.returncode,3,result.stderr)
            self.assertTrue((Path(d)/'out'/f'daily_summary_{DATE}.json').exists())

    def test_team_stats_cache(self):
        answer={'response':{'fixtures':{'played':{'total':12}},'goals':{'for':{'average':{'total':'1.9'}}}}}
        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ',{'API_FOOTBALL_CACHE_DIR':d}):
            with patch('sports_edge.pro_pipeline.api_pro_get', return_value=answer) as fn:
                a=get_cached_team_stats(303,18,2026)
                b=get_cached_team_stats(303,18,2026)
                c=get_cached_team_stats(833,18,2026)
                self.assertEqual(fn.call_count,2)
                self.assertEqual(a,b)
                self.assertEqual(c,answer)

if __name__=='__main__': unittest.main()