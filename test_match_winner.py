"""
Unit tests for the match winner prediction functionality.
"""

import unittest
from match_winner import WinnerPredictor, predict_match_winner, MatchOutcome
import numpy as np

class TestMatchWinnerPrediction(unittest.TestCase):
    """Test the match winner prediction functionality."""
    
    def test_basic_1x2_probabilities(self):
        """Test that basic 1X2 probabilities are calculated correctly."""
        predictor = WinnerPredictor()
        
        # Test case: higher home_xg should result in higher home_win probability
        home_win, draw, away_win = predictor._calculate_1x2_probabilities(2.0, 1.0)
        self.assertGreater(home_win, away_win)
        
        # Equal xG should give advantage to home team but with significant draw probability
        home_win, draw, away_win = predictor._calculate_1x2_probabilities(1.5, 1.5)
        self.assertGreater(home_win, away_win)
        self.assertGreater(draw, 0.25)  # Draw probability should be substantial

        # Much higher away xG should overcome home advantage
        home_win, draw, away_win = predictor._calculate_1x2_probabilities(1.0, 3.0)
        self.assertGreater(away_win, home_win)
        
    def test_refine_probabilities(self):
        """Test probability refinement using team form and head-to-head data."""
        predictor = WinnerPredictor()
        
        # Initial probabilities
        home_win_prob, draw_prob, away_win_prob = 0.50, 0.25, 0.25
        
        # Good home form, poor away form
        home_form = {"form_trend": 0.7, "matches_played": 10}
        away_form = {"form_trend": -0.3, "matches_played": 10}
        h2h = {"matches_played": 5, "home_win_pct": 0.6, "draw_pct": 0.2, "away_win_pct": 0.2}
        
        refined = predictor._refine_probabilities(
            home_win_prob, draw_prob, away_win_prob,
            home_form, away_form, h2h
        )
        
        # Home probability should increase
        self.assertGreater(refined[MatchOutcome.HOME_WIN.value], home_win_prob)
        self.assertLess(refined[MatchOutcome.AWAY_WIN.value], away_win_prob)
        
        # Total should still be 1.0
        total_prob = sum(refined.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
        
    def test_context_adjustments(self):
        """Test context-based probability adjustments."""
        predictor = WinnerPredictor()
        
        # Initial probabilities
        probabilities = {
            MatchOutcome.HOME_WIN.value: 0.50,
            MatchOutcome.DRAW.value: 0.25,
            MatchOutcome.AWAY_WIN.value: 0.25
        }
        
        # Test weather effects - rainy conditions should increase draw probability
        context_rainy = {"is_rainy": True}
        adjusted_rainy = predictor._apply_context_adjustments(probabilities, context_rainy)
        self.assertGreater(adjusted_rainy[MatchOutcome.DRAW.value], probabilities[MatchOutcome.DRAW.value])
        
        # Test injuries
        context_injuries = {"home_injuries_impact": 2.0}  # Significant home team injuries
        adjusted_injuries = predictor._apply_context_adjustments(probabilities, context_injuries)
        self.assertLess(adjusted_injuries[MatchOutcome.HOME_WIN.value], probabilities[MatchOutcome.HOME_WIN.value])
        self.assertGreater(adjusted_injuries[MatchOutcome.AWAY_WIN.value], probabilities[MatchOutcome.AWAY_WIN.value])
        
    def test_confidence_calculation(self):
        """Test the confidence score calculation."""
        predictor = WinnerPredictor()
        
        # Clear winner case
        clear_winner_probabilities = {
            MatchOutcome.HOME_WIN.value: 0.70,
            MatchOutcome.DRAW.value: 0.20,
            MatchOutcome.AWAY_WIN.value: 0.10
        }
        
        # Close case
        close_probabilities = {
            MatchOutcome.HOME_WIN.value: 0.35,
            MatchOutcome.DRAW.value: 0.34,
            MatchOutcome.AWAY_WIN.value: 0.31
        }
        
        home_form = {"matches_played": 10, "consistency": 0.8}
        away_form = {"matches_played": 10, "consistency": 0.8}
        
        # Premier League should have higher confidence than a lower league
        premier_league_id = 39  # Premier League
        lower_league_id = 999  # Made-up lower league ID
        
        # Calculate confidence scores
        clear_winner_conf, _ = predictor._calculate_confidence(
            clear_winner_probabilities, home_form, away_form, premier_league_id
        )
        close_conf, _ = predictor._calculate_confidence(
            close_probabilities, home_form, away_form, premier_league_id
        )
        lower_league_conf, _ = predictor._calculate_confidence(
            clear_winner_probabilities, home_form, away_form, lower_league_id
        )
        
        # Clear winner should have higher confidence than close case
        self.assertGreater(clear_winner_conf, close_conf)
        
        # Premier League should have higher confidence than lower league
        self.assertGreater(clear_winner_conf, lower_league_conf)
        
    def test_predict_winner_integration(self):
        """Test the full match winner prediction integration."""
        # Mock data
        home_team_id = 33  # Manchester United
        away_team_id = 42  # Arsenal
        home_xg = 1.7
        away_xg = 1.2
        home_form = {
            "form_trend": 0.5,
            "matches_played": 10,
            "consistency": 0.7
        }
        away_form = {
            "form_trend": 0.3,
            "matches_played": 10,
            "consistency": 0.6
        }
        h2h = {
            "matches_played": 10,
            "home_win_pct": 0.5,
            "draw_pct": 0.3,
            "away_win_pct": 0.2
        }
        league_id = 39  # Premier League
        context_factors = {
            "weather_data_available": True,
            "lineup_data_available": True
        }
        
        # Get prediction
        result = predict_match_winner(
            home_team_id, away_team_id, home_xg, away_xg,
            home_form, away_form, h2h, league_id, context_factors
        )
        
        # Validate result structure
        self.assertIn("most_likely_outcome", result)
        self.assertIn("probabilities", result)
        self.assertIn("home_win", result["probabilities"])
        self.assertIn("draw", result["probabilities"])
        self.assertIn("away_win", result["probabilities"])
        self.assertIn("confidence", result)
        
        # With these inputs, home win should be most likely
        self.assertEqual(result["most_likely_outcome"], MatchOutcome.HOME_WIN.value)
        
        # Probabilities should sum to 100%
        total_prob = (result["probabilities"]["home_win"] + 
                     result["probabilities"]["draw"] + 
                     result["probabilities"]["away_win"])
        self.assertAlmostEqual(total_prob, 100.0, places=1)
        
        # Confidence score should be in range 0-1
        self.assertGreaterEqual(result["confidence"]["score"], 0.0)
        self.assertLessEqual(result["confidence"]["score"], 1.0)

if __name__ == "__main__":
    unittest.main()
