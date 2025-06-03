"""
Test module for voting ensemble corners model.

This module provides test functions to validate the performance of the
voting ensemble model for corner predictions based on academic research.
"""

import unittest
import pytest
import numpy as np
import os
import logging
import json
from unittest.mock import patch, MagicMock

from voting_ensemble_corners import VotingEnsembleCornersModel, predict_corners_with_voting_ensemble
from corners_improved import ImprovedCornersModel, predict_corners_with_negative_binomial

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestVotingEnsembleModel(unittest.TestCase):
    """Test cases for the voting ensemble corners model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = VotingEnsembleCornersModel()
        
        # Sample match data
        self.home_team_id = 33  # Manchester City
        self.away_team_id = 40  # Liverpool
        self.league_id = 39  # Premier League
        
        # Sample team statistics
        self.home_stats = {
            'avg_corners_for': 6.2,
            'avg_corners_against': 3.8,
            'form_score': 65,
            'attack_strength': 1.15,
            'defense_strength': 1.2,
            'avg_shots': 15.3
        }
        
        self.away_stats = {
            'avg_corners_for': 5.7,
            'avg_corners_against': 4.2,
            'form_score': 62,
            'attack_strength': 1.1,
            'defense_strength': 1.05,
            'avg_shots': 14.2
        }
        
        # Weather/context data
        self.context_factors = {
            'is_windy': False,
            'is_rainy': False,
            'is_high_stakes': True
        }
        
    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, VotingEnsembleCornersModel)
        self.assertIsNotNone(self.model.league_factors)
        
    def test_prediction_output_format(self):
        """Test that the model returns properly formatted predictions."""
        result = self.model.predict_corners(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id
        )
        
        # Check required keys are present
        expected_keys = ['total', 'home', 'away', 'over_8.5', 'over_9.5', 
                         'over_10.5', 'corner_brackets']
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key in prediction: {key}")
        
        # Check reasonable value ranges
        self.assertTrue(4 <= result['total'] <= 18, "Total corners outside reasonable range")
        self.assertTrue(2 <= result['home'] <= 12, "Home corners outside reasonable range")
        self.assertTrue(1 <= result['away'] <= 10, "Away corners outside reasonable range")
        
        # Check probabilities are valid
        for key in ['over_8.5', 'over_9.5', 'over_10.5']:
            self.assertTrue(0 <= result[key] <= 1, f"Invalid probability for {key}")
        
        # Check corner brackets
        self.assertIn('corner_brackets', result)
        self.assertIsInstance(result['corner_brackets'], dict)
        
    def test_fallback_mechanism(self):
        """Test that the fallback mechanism works when model fails."""
        with patch.object(self.model, '_extract_features', side_effect=Exception('Simulated error')):
            result = self.model.predict_corners(
                self.home_team_id,
                self.away_team_id,
                self.home_stats,
                self.away_stats,
                self.league_id
            )
            
            # Should return fallback with is_fallback flag
            self.assertIn('is_fallback', result)
            self.assertTrue(result['is_fallback'])
            
    def test_with_and_without_context(self):
        """Test predictions with and without context factors."""
        # Without context
        result_no_context = self.model.predict_corners(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id
        )
        
        # With context
        result_with_context = self.model.predict_corners(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id,
            self.context_factors
        )
        
        # Results should be different with different context
        self.assertNotEqual(result_no_context['total'], result_with_context['total'])
        
    def test_interface_function(self):
        """Test the module interface function."""
        result = predict_corners_with_voting_ensemble(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id,
            self.context_factors
        )
        
        # Verify it returns the expected structure
        self.assertIn('total', result)
        self.assertIn('corner_brackets', result)
        
    def test_negative_binomial_vs_ensemble(self):
        """Compare negative binomial and ensemble model outputs."""
        # Get predictions from both models
        nb_result = predict_corners_with_negative_binomial(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id
        )
        
        ensemble_result = predict_corners_with_voting_ensemble(
            self.home_team_id,
            self.away_team_id,
            self.home_stats,
            self.away_stats,
            self.league_id
        )
        
        # Both should return valid predictions
        self.assertIn('total', nb_result)
        self.assertIn('total', ensemble_result)
        
        # Log both predictions for comparison
        logger.info("Negative Binomial prediction: %.2f total corners", nb_result['total'])
        logger.info("Ensemble model prediction: %.2f total corners", ensemble_result['total'])
        
        # They should be reasonably close but not identical
        self.assertLess(abs(nb_result['total'] - ensemble_result['total']), 5.0, 
                       "Models are producing wildly different predictions")
        
        # Store comparison for reporting
        comparison = {
            'nb_model': {
                'total': nb_result['total'],
                'home': nb_result['home'],
                'away': nb_result['away'],
                'over_9.5': nb_result.get('over_9.5', 'N/A')
            },
            'ensemble_model': {
                'total': ensemble_result['total'],
                'home': ensemble_result['home'],
                'away': ensemble_result['away'],
                'over_9.5': ensemble_result.get('over_9.5', 'N/A')
            }
        }
        
        # Make results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save comparison
        with open('results/model_comparison_test.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
        logger.info("Model comparison saved to results/model_comparison_test.json")

if __name__ == '__main__':
    unittest.main()
