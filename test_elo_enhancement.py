"""
Tests for ELO enhanced predictions

This module tests the ELO enhancement functionality to ensure it's properly
integrating with the prediction system.
"""

import unittest
import logging
from typing import Dict, Any
from team_elo_rating import get_elo_ratings_for_match
from prediction_integration import enrich_prediction_with_contextual_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestELOEnhancement(unittest.TestCase):
    """Test cases for the ELO enhancement functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.home_team_id = 39  # Manchester City
        self.away_team_id = 40  # Liverpool
        self.league_id = 39     # Premier League
        
        # Create base prediction
        self.base_prediction = {
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
        }
    
    def test_get_elo_ratings(self):
        """Test getting ELO ratings for a match"""
        elo_data = get_elo_ratings_for_match(self.home_team_id, self.away_team_id, self.league_id)
        
        # Check that we got the expected keys
        expected_keys = [
            'home_elo', 'away_elo', 'elo_diff', 
            'elo_win_probability', 'elo_draw_probability', 'elo_loss_probability',
            'elo_expected_goal_diff'
        ]
        
        for key in expected_keys:
            self.assertIn(key, elo_data)
        
        # Check that probabilities sum to 1 (allowing for floating point errors)
        prob_sum = elo_data['elo_win_probability'] + elo_data['elo_draw_probability'] + elo_data['elo_loss_probability']
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
    
    def test_full_elo_enhancement_integration(self):
        """Test the full ELO enhancement integration with the prediction system"""
        enriched = enrich_prediction_with_contextual_data(
            self.base_prediction,
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            league_id=self.league_id
        )
        
        # Check that basic ELO data was added
        self.assertIn('elo_ratings', enriched)
        self.assertIn('home_elo', enriched['elo_ratings'])
        self.assertIn('away_elo', enriched['elo_ratings'])
        self.assertIn('elo_diff', enriched['elo_ratings'])
        
        # Check ELO probabilities
        self.assertIn('elo_probabilities', enriched)
        self.assertIn('win', enriched['elo_probabilities'])
        self.assertIn('draw', enriched['elo_probabilities'])
        self.assertIn('loss', enriched['elo_probabilities'])
        
        # Check expected goal difference
        self.assertIn('elo_expected_goal_diff', enriched)
        
        # Check that ELO is listed in adjustments_applied
        self.assertIn('adjustments_applied', enriched)
        self.assertIn('elo', enriched['adjustments_applied'])
        
        # Check if probabilities sum to 1
        prob_sum = (
            enriched['elo_probabilities']['win'] + 
            enriched['elo_probabilities']['draw'] + 
            enriched['elo_probabilities']['loss']
        )
        self.assertAlmostEqual(prob_sum, 1.0, places=2)
        
        # Check that blended probabilities were added and sum to 1
        if 'blended_probabilities' in enriched:
            blend_sum = (
                enriched['blended_probabilities']['home_win'] + 
                enriched['blended_probabilities']['draw'] + 
                enriched['blended_probabilities']['away_win']
            )
            self.assertAlmostEqual(blend_sum, 1.0, places=2)

if __name__ == '__main__':
    unittest.main()
