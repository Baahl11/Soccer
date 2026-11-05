import unittest
from sports_edge.board import render_board
class BoardTest(unittest.TestCase):
 def test_board_blocks_wager_promotions_and_escapes_names(self):
  batch={'screened_fixtures':[{'fixture_id':1,'league_id':3,'match':'A <script>alert(1)</script> vs B','raw_sport_projection':{'status':'MODELED_LIMITED','projection_model':'GOAL_RATE_POISSON_LIMITED_v0.1','sample':{'minimum_split_sample':2},'sample_warning':'LOW_LEAGUE_SPLIT_SAMPLE_UNDER_5','raw_home_win_prob':0.5,'raw_draw_prob':0.2,'raw_away_win_prob':0.3,'raw_over_2_5_prob':0.6,'raw_over_3_5_prob':0.5,'raw_btts_yes_prob':0.7}}]}
  summary={'match_date':'2026-11-05','generated_at_utc':'2026-11-04T16:00:00+00:00','sports_screened':1,'raw_existing_engine_projections':1}
  html=render_board(batch,summary)
  self.assertIn('50.0%',html)
  self.assertIn('MUESTRA INSUFICIENTE',html)
  self.assertIn('&lt;script&gt;',html)
  self.assertNotIn('<script>alert(1)</script>',html)
  self.assertIn('0</strong><small>BET',html)
 def test_no_model_produces_no_fabricated_probabilities(self):
  html=render_board({'screened_fixtures':[{'fixture_id':3,'match':'A vs B'}]}, {'sports_screened':1,'match_date':'2026-11-05'})
  self.assertIn('NOT VERIFIED',html)
  self.assertNotIn('100.0%',html)
if __name__=='__main__':unittest.main()