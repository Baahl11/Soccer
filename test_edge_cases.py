"""
Test suite for edge case handling in the ELO rating system.

This module tests various edge cases and special situations to ensure
the system handles them correctly and reliably.
"""

import unittest
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import math
import random

from edge_case_handler import EdgeCaseHandler, EdgeCaseConfig
from team_elo_rating import TeamEloRating
from auto_updating_elo import AutoUpdatingEloRating

class TestEdgeCaseHandling(unittest.TestCase):
    """Test edge case handling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.edge_handler = EdgeCaseHandler()
        self.elo_rating = TeamEloRating()
        self.auto_elo = AutoUpdatingEloRating()
        
    def test_rating_validation(self):
        """Test rating validation logic"""
        # Test minimum rating
        self.assertEqual(
            self.edge_handler.validate_rating(0),
            self.edge_handler.config.min_elo_rating
        )
        
        # Test maximum rating
        self.assertEqual(
            self.edge_handler.validate_rating(4000),
            self.edge_handler.config.max_elo_rating
        )
        
        # Test invalid types
        self.assertEqual(
            self.edge_handler.validate_rating("invalid"),
            self.edge_handler.config.default_rating
        )
        
        # Test NaN/Inf
        self.assertEqual(
            self.edge_handler.validate_rating(float('nan')),
            self.edge_handler.config.default_rating
        )
        self.assertEqual(
            self.edge_handler.validate_rating(float('inf')),
            self.edge_handler.config.default_rating
        )
        
    def test_rating_updates(self):
        """Test rating update constraints"""
        old_rating = 1500
        
        # Test maximum change constraint
        new_rating = self.edge_handler.handle_rating_update(
            old_rating,
            old_rating + 300
        )
        self.assertEqual(
            new_rating,
            old_rating + self.edge_handler.config.max_rating_change
        )
        
        # Test negative change constraint
        new_rating = self.edge_handler.handle_rating_update(
            old_rating,
            old_rating - 300
        )
        self.assertEqual(
            new_rating,
            old_rating - self.edge_handler.config.max_rating_change
        )
        
    def test_reliability_calculation(self):
        """Test rating reliability factors"""
        # New team should have low reliability
        reliability = self.edge_handler.calculate_rating_reliability(
            matches_played=1,
            days_inactive=0
        )
        self.assertLess(reliability, 0.5)
        
        # Established team should have high reliability
        reliability = self.edge_handler.calculate_rating_reliability(
            matches_played=20,
            days_inactive=0
        )
        self.assertGreater(reliability, 0.9)
        
        # Inactive team should have reduced reliability
        reliability = self.edge_handler.calculate_rating_reliability(
            matches_played=20,
            days_inactive=400
        )
        self.assertLess(reliability, 0.9)
        
    def test_special_case_handling(self):
        """Test special match situation handling"""
        # Test derby match
        match_data = {
            "home_team_city": "Manchester",
            "away_team_city": "Manchester"
        }
        
        adjustments = self.edge_handler.handle_special_cases(1, match_data)
        self.assertGreater(adjustments["uncertainty_factor"], 1.0)
        
        # Test cup match
        match_data = {
            "is_cup_match": True
        }
        
        adjustments = self.edge_handler.handle_special_cases(1, match_data)
        self.assertGreater(adjustments["importance_factor"], 1.0)
        
        # Test long break
        match_data = {
            "days_since_last_match": 90
        }
        
        adjustments = self.edge_handler.handle_special_cases(1, match_data)
        self.assertGreater(adjustments["uncertainty_factor"], 1.0)
        
    def test_prediction_validation(self):
        """Test prediction data validation"""
        # Valid prediction
        prediction = {
            "elo_ratings": {
                "home": 1500,
                "away": 1500
            },
            "probabilities": {
                "win": 0.4,
                "draw": 0.3,
                "loss": 0.3
            },
            "expected_goal_diff": 0.5
        }
        
        validated, warnings = self.edge_handler.validate_prediction(prediction)
        self.assertEqual(len(warnings), 0)
        
        # Invalid probabilities
        prediction["probabilities"] = {
            "win": 0.5,
            "draw": 0.4,
            "loss": 0.3
        }
        
        validated, warnings = self.edge_handler.validate_prediction(prediction)
        self.assertEqual(len(warnings), 1)  # Should warn about prob sum > 1
        self.assertAlmostEqual(
            sum(validated["probabilities"].values()),
            1.0
        )
        
    def test_integration(self):
        """Test integration with main rating system"""
        # Set up test match
        home_id = 1
        away_id = 2
        
        # Get prediction with edge case handling
        prediction = self.auto_elo.get_match_prediction(home_id, away_id)
        
        # Verify prediction structure
        self.assertIn("elo_ratings", prediction)
        self.assertIn("reliability", prediction)
        self.assertIn("home", prediction["reliability"])
        self.assertIn("away", prediction["reliability"])
        
        # Test rating update
        match_data = {
            "home_team_id": home_id,
            "away_team_id": away_id,
            "score_home": 2,
            "score_away": 1
        }
        
        new_home, new_away = self.auto_elo.update_ratings(match_data)
        
        # Verify rating changes are within limits
        self.assertLess(
            abs(new_home - 1500),
            self.edge_handler.config.max_rating_change
        )
        self.assertLess(
            abs(new_away - 1500),
            self.edge_handler.config.max_rating_change
        )

if __name__ == '__main__':
    unittest.main()
