"""Exact-source, same-fixture, raw-before-market, fail-closed guards."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from sports_edge.engine_bridge import from_existing_engine


FX = dict(fixture_id=1234,status='NS',home_team_id=100,away_team_id=200,league_id=3,season=2026)


def stats(team, gf, ga):
    return {'response': {'team': {'id':team},'league': {'id':3,'season':2026},
        'fixtures': {'played': {'home':7,'away':7,'total':14}},
        'goals': {'for': {'average':{'home':gf,'away':gf,'total':gf}},
                  'against': {'average': {'home':ga,'away':ga,'total':ga}}}}}


class EngineBridgeTests(unittest.TestCase):
    def test_no_source_is_watch_not_fake_model(self):
        with patch.dict('os.environ', {'SOCCER_ENGINE_SOURCE_DIR':''}):
            r=from_existing_engine(FX,stats(100,2.1,1.0),stats(200,1.2,1.5))
        self.assertEqual(r['status'],'NOT_CONNECTED')
        self.assertFalse(r['bet_eligible'])

    def test_mismatched_team_blocks_projection(self):
        with tempfile.TemporaryDirectory() as d:
            r=from_existing_engine(FX,stats(777,2.1,1.0),stats(200,1.2,1.5),engine_root=d)
        self.assertEqual(r['reason'],'EXISTING_ENGINE_SOURCE_NOT_FOUND')

    def test_exact_existing_model_only_no_market_odds(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mcp_gateway'/'soccer_model.py';path.parent.mkdir(parents=True)
            path.write_text('''\ndef build_raw_projection(fx, sporting, availability):\n  assert "odds" not in sporting and "market" not in sporting\n  assert sporting["home_stats"]["team"]["id"] == 100\n  return {"status":"MODELED_LIMITED","model_version":"EXISTING_TEST_MODEL","projection_model":"GOAL_RATE_POISSON_LIMITED_v0.1","raw_home_win_prob":0.5,"raw_draw_prob":0.25,"raw_away_win_prob":0.25,"sample":{"minimum_split_sample":7},"_total_dist":{0:0.2}}\n''')
            r=from_existing_engine(FX,stats(100,2.1,1.0),stats(200,1.2,1.5),engine_root=d)
        self.assertEqual(r['status'],'MODELED_LIMITED')
        self.assertEqual(r['raw_home_win_prob'],0.5)
        self.assertNotIn('_total_dist',r)
        self.assertFalse(r['calibrated'])
        self.assertFalse(r['bet_eligible'])
        self.assertFalse(r['market_odds_used'])
        self.assertEqual(r['fixture_id'],1234)

    def test_rejects_mismatched_team_if_engine_is_installed(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mcp_gateway'/'soccer_model.py';path.parent.mkdir(parents=True)
            path.write_text('def build_raw_projection(*args): raise AssertionError("should not run")\n')
            r=from_existing_engine(FX,stats(777,2.1,1.0),stats(200,1.2,1.5),engine_root=d)
        self.assertEqual(r['status'],'NOT_MODELED')
        self.assertEqual(r['reason'],'TEAM_STATS_IDENTITY_MISMATCH_OR_UNVERIFIED')

    def test_does_not_model_postgame(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mcp_gateway'/'soccer_model.py';path.parent.mkdir(parents=True)
            path.write_text('def build_raw_projection(*args): raise AssertionError("should not run")\n')
            r=from_existing_engine({**FX,'status':'FT'},stats(100,2.1,1.0),stats(200,1.2,1.5),engine_root=d)
        self.assertEqual(r['reason'],'FIXTURE_NOT_PREMATCH')

    def test_small_split_sample_never_bet(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'mcp_gateway'/'soccer_model.py';path.parent.mkdir(parents=True)
            path.write_text('def build_raw_projection(*args): return {"status":"MODELED_LIMITED","sample":{"minimum_split_sample":2}}\n')
            r=from_existing_engine(FX,stats(100,2.1,1.0),stats(200,1.2,1.5),engine_root=d)
        self.assertEqual(r['sample_warning'],'LOW_LEAGUE_SPLIT_SAMPLE_UNDER_5')
        self.assertFalse(r['bet_eligible'])

if __name__=='__main__':unittest.main()