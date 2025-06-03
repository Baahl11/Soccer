"""
Comprehensive tests for ELO enhanced predictions

This module provides extensive tests for the ELO enhancement functionality,
including edge cases and various match scenarios.
"""

import unittest
import logging
from typing import Dict, Any
import numpy as np
from team_elo_rating import get_elo_ratings_for_match, TeamEloRating
from prediction_integration import enrich_prediction_with_contextual_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestELOEnhancementComprehensive(unittest.TestCase):
    """Comprehensive test cases for the ELO enhancement functionality"""
    
    def setUp(self):
        """Set up test data"""
        # Test cases with different team strength scenarios
        self.test_scenarios = [
            {
                'name': 'Close matchup',
                'home_team_id': 39,  # Manchester City
                'away_team_id': 40,  # Liverpool
                'league_id': 39      # Premier League
            },
            {
                'name': 'Strong home advantage',
                'home_team_id': 33,  # Manchester United
                'away_team_id': 62,  # Everton
                'league_id': 39      # Premier League
            },
            {
                'name': 'Strong away advantage',
                'home_team_id': 65,  # Nottingham Forest
                'away_team_id': 42,  # Arsenal
                'league_id': 39      # Premier League
            },
            {
                'name': 'Different league',
                'home_team_id': 529,  # Barcelona
                'away_team_id': 541,  # Real Madrid
                'league_id': 140     # La Liga
            }
        ]
        
        # Create base prediction templates with varying characteristics
        self.base_predictions = {
            'standard': {
                'predicted_home_goals': 1.8,
                'predicted_away_goals': 1.2,
                'total_goals': 3.0,
                'prob_over_2_5': 0.65,
                'prob_btts': 0.7,
                'prob_1': 0.5,
                'prob_X': 0.25,
                'prob_2': 0.25,
                'confidence': 0.6,
                'method': 'statistical'
            },
            'high_scoring': {
                'predicted_home_goals': 2.7,
                'predicted_away_goals': 1.8,
                'total_goals': 4.5,
                'prob_over_2_5': 0.85,
                'prob_btts': 0.8,
                'prob_1': 0.6,
                'prob_X': 0.15,
                'prob_2': 0.25,
                'confidence': 0.7,
                'method': 'statistical'
            },
            'low_scoring': {
                'predicted_home_goals': 0.9,
                'predicted_away_goals': 0.7,
                'total_goals': 1.6,
                'prob_over_2_5': 0.25,
                'prob_btts': 0.45,
                'prob_1': 0.4,
                'prob_X': 0.4,
                'prob_2': 0.2,
                'confidence': 0.55,
                'method': 'statistical'
            },
            'high_confidence': {
                'predicted_home_goals': 2.0,
                'predicted_away_goals': 0.7,
                'total_goals': 2.7,
                'prob_over_2_5': 0.6,
                'prob_btts': 0.5,
                'prob_1': 0.7,
                'prob_X': 0.2,
                'prob_2': 0.1,
                'confidence': 0.85,
                'method': 'statistical'
            },
            'low_confidence': {
                'predicted_home_goals': 1.4,
                'predicted_away_goals': 1.4,
                'total_goals': 2.8,
                'prob_over_2_5': 0.55,
                'prob_btts': 0.65,
                'prob_1': 0.33,
                'prob_X': 0.34,
                'prob_2': 0.33,
                'confidence': 0.4,
                'method': 'statistical'
            }
        }
    
    def test_elo_ratings_validity(self):
        """Test that ELO ratings are within valid ranges"""
        for scenario in self.test_scenarios:
            elo_data = get_elo_ratings_for_match(
                scenario['home_team_id'],
                scenario['away_team_id'],
                scenario['league_id']
            )
            
            # Check that ratings are in a sensible range (typically 1000-2000)
            self.assertGreater(elo_data['home_elo'], 800)
            self.assertLess(elo_data['home_elo'], 2200)
            self.assertGreater(elo_data['away_elo'], 800)
            self.assertLess(elo_data['away_elo'], 2200)
            
            # Check that probabilities are valid
            self.assertGreaterEqual(elo_data['elo_win_probability'], 0)
            self.assertLessEqual(elo_data['elo_win_probability'], 1)
            self.assertGreaterEqual(elo_data['elo_draw_probability'], 0)
            self.assertLessEqual(elo_data['elo_draw_probability'], 1)
            self.assertGreaterEqual(elo_data['elo_loss_probability'], 0)
            self.assertLessEqual(elo_data['elo_loss_probability'], 1)
            
            # Probabilities should sum to approximately 1
            prob_sum = (elo_data['elo_win_probability'] + 
                       elo_data['elo_draw_probability'] + 
                       elo_data['elo_loss_probability'])
            self.assertAlmostEqual(prob_sum, 1.0, places=2)
    
    def test_all_scenarios(self):
        """Test all scenario combinations"""
        for scenario in self.test_scenarios:
            for pred_name, prediction in self.base_predictions.items():
                with self.subTest(f"Scenario: {scenario['name']}, Prediction: {pred_name}"):
                    enriched = enrich_prediction_with_contextual_data(
                        prediction.copy(),
                        home_team_id=scenario['home_team_id'],
                        away_team_id=scenario['away_team_id'],
                        league_id=scenario['league_id']
                    )
                    
                    # Check that ELO enhancement was applied
                    self.assertIn('elo_ratings', enriched)
                    self.assertIn('elo_probabilities', enriched)
                    self.assertIn('elo_insights', enriched)
                    self.assertIn('elo_enhanced_metrics', enriched)
                    self.assertIn('blended_probabilities', enriched)
                    
                    # Check that ELO was added to adjustments
                    self.assertIn('adjustments_applied', enriched)
                    self.assertIn('elo', enriched['adjustments_applied'])
                    
                    # Test blended probabilities sum to 1
                    blend_sum = sum([
                        enriched['blended_probabilities']['home_win'],
                        enriched['blended_probabilities']['draw'],
                        enriched['blended_probabilities']['away_win']
                    ])
                    self.assertAlmostEqual(blend_sum, 1.0, places=2)
                    
                    # Test that competitiveness rating is in range
                    competitiveness = enriched['elo_enhanced_metrics']['competitiveness_rating']
                    self.assertGreaterEqual(competitiveness, 1)
                    self.assertLessEqual(competitiveness, 10)
    
    def test_elo_influence_on_probabilities(self):
        """Test that ELO ratings have appropriate influence on probabilities"""
        scenario = self.test_scenarios[0]  # Use the first scenario
        prediction = self.base_predictions['standard'].copy()
        
        # Original probabilities
        orig_home_win = prediction['prob_1']
        orig_draw = prediction['prob_X']
        orig_away_win = prediction['prob_2']
        
        enriched = enrich_prediction_with_contextual_data(
            prediction,
            home_team_id=scenario['home_team_id'],
            away_team_id=scenario['away_team_id'],
            league_id=scenario['league_id']
        )
        
        # ELO probabilities
        elo_home_win = enriched['elo_probabilities']['win']
        elo_draw = enriched['elo_probabilities']['draw']
        elo_away_win = enriched['elo_probabilities']['loss']
        
        # Blended probabilities
        blended_home_win = enriched['blended_probabilities']['home_win']
        blended_draw = enriched['blended_probabilities']['draw']
        blended_away_win = enriched['blended_probabilities']['away_win']
        
        # Blended probabilities should be between original and ELO probabilities
        self.assertTrue(
            min(orig_home_win, elo_home_win) <= blended_home_win <= max(orig_home_win, elo_home_win) or
            abs(blended_home_win - min(orig_home_win, elo_home_win)) < 0.05 or
            abs(blended_home_win - max(orig_home_win, elo_home_win)) < 0.05
        )
        
        self.assertTrue(
            min(orig_draw, elo_draw) <= blended_draw <= max(orig_draw, elo_draw) or
            abs(blended_draw - min(orig_draw, elo_draw)) < 0.05 or
            abs(blended_draw - max(orig_draw, elo_draw)) < 0.05
        )
        
        self.assertTrue(
            min(orig_away_win, elo_away_win) <= blended_away_win <= max(orig_away_win, elo_away_win) or
            abs(blended_away_win - min(orig_away_win, elo_away_win)) < 0.05 or
            abs(blended_away_win - max(orig_away_win, elo_away_win)) < 0.05
        )
    
    def test_confidence_adjustment(self):
        """Test that confidence is appropriately adjusted based on ELO"""
        # Test case where model and ELO agree (should increase confidence)
        scenario = self.test_scenarios[1]  # Strong home advantage
        prediction = self.base_predictions['standard'].copy()
        prediction['confidence'] = 0.6
        prediction['prediction'] = 'Home'  # Predict home win
        
        enriched = enrich_prediction_with_contextual_data(
            prediction,
            home_team_id=scenario['home_team_id'],
            away_team_id=scenario['away_team_id'],
            league_id=scenario['league_id']
        )
        
        # Check that enhanced confidence exists
        self.assertIn('enhanced_confidence', enriched)
        
        # If ELO also favors home team, confidence should increase
        elo_diff = enriched['elo_ratings']['elo_diff']
        if elo_diff > 0:
            self.assertGreater(
                enriched['enhanced_confidence']['score'],
                prediction['confidence']
            )
        
        # Test case where model and ELO disagree (should decrease confidence)
        prediction = self.base_predictions['standard'].copy()
        prediction['confidence'] = 0.6
        prediction['prediction'] = 'Away'  # Predict away win
        
        enriched = enrich_prediction_with_contextual_data(
            prediction,
            home_team_id=scenario['home_team_id'],
            away_team_id=scenario['away_team_id'],
            league_id=scenario['league_id']
        )
        
        # If ELO favors home team but prediction is away, confidence should decrease
        elo_diff = enriched['elo_ratings']['elo_diff']
        if elo_diff > 0:
            self.assertLess(
                enriched['enhanced_confidence']['score'],
                prediction['confidence']
            )
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with invalid team IDs
        prediction = self.base_predictions['standard'].copy()
        
        # Should not raise exceptions even with invalid team IDs
        try:
            enriched = enrich_prediction_with_contextual_data(
                prediction,
                home_team_id=99999,  # Invalid ID
                away_team_id=99998,  # Invalid ID
                league_id=39
            )
            # Original prediction should be returned
            self.assertNotIn('elo_ratings', enriched)
        except Exception as e:
            self.fail(f"enrich_prediction_with_contextual_data raised exception with invalid IDs: {e}")
        
        # Test with None team IDs
        try:
            enriched = enrich_prediction_with_contextual_data(
                prediction,
                home_team_id=None,
                away_team_id=None,
                league_id=None
            )
            # Original prediction should be returned
            self.assertNotIn('elo_ratings', enriched)
        except Exception as e:
            self.fail(f"enrich_prediction_with_contextual_data raised exception with None IDs: {e}")
    
    def test_expected_goal_diff(self):
        """Test that expected goal difference values are reasonable"""
        for scenario in self.test_scenarios:
            elo_data = get_elo_ratings_for_match(
                scenario['home_team_id'],
                scenario['away_team_id'],
                scenario['league_id']
            )
            
            # Check that expected goal difference is present and is a float
            self.assertIn('elo_expected_goal_diff', elo_data)
            self.assertIsInstance(elo_data['elo_expected_goal_diff'], float)
            
            # Expected goal difference should typically be in a reasonable range
            self.assertGreater(elo_data['elo_expected_goal_diff'], -3)
            self.assertLess(elo_data['elo_expected_goal_diff'], 3)
            
            # The sign of the expected goal difference should match the ELO difference
            elo_diff = elo_data['elo_diff']
            exp_goal_diff = elo_data['elo_expected_goal_diff']
            
            # If elo_diff is positive, exp_goal_diff should be positive and vice versa
            # Allow for small differences around zero
            if abs(elo_diff) > 50:  # Only check if the ELO difference is significant
                self.assertEqual(np.sign(elo_diff), np.sign(exp_goal_diff))

if __name__ == '__main__':
    unittest.main()
