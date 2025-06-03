import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the dependencies that might not exist in test environment
sys.modules['specialized_ensemble'] = MagicMock()
sys.modules['psychological_features'] = MagicMock()
sys.modules['sequence_transformer'] = MagicMock()

# Now import the module to test
from transformer_integration import (
    IntegratedPredictionSystem,
    prepare_match_sequences,
    add_transformer_to_workflow
)


class TestTransformerIntegration(unittest.TestCase):
    """Test suite for the transformer integration module."""
    
    def setUp(self):
        """Setup common test data."""
        # Create sample match data
        self.sample_matches = [
            {
                'date': '2024-01-01',
                'match_date': datetime(2024, 1, 1),
                'home_team_id': 'team1',
                'away_team_id': 'team2',
                'team_id': 'team1',
                'opponent_id': 'team2',
                'home_goals': 2,
                'away_goals': 1,
                'home_xg': 1.8,
                'away_xg': 0.9
            },
            {
                'date': '2024-01-08',
                'match_date': datetime(2024, 1, 8),
                'home_team_id': 'team3',
                'away_team_id': 'team1',
                'team_id': 'team1',
                'opponent_id': 'team3',
                'home_goals': 1,
                'away_goals': 3,
                'home_xg': 1.2,
                'away_xg': 2.5
            }
        ]
        
        # Create a sample DataFrame
        self.df = pd.DataFrame(self.sample_matches)
        
        # Create a temp directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.test_dir.name, 'test_model.pt')
        
        # Create a mock model file
        with open(self.model_path, 'w') as f:
            f.write('mock model file')
    
    def tearDown(self):
        """Clean up after tests."""
        self.test_dir.cleanup()
    
    @patch('transformer_integration.EnsemblePredictor')
    @patch('transformer_integration.PsychologicalFactorExtractor')
    @patch('transformer_integration.SequenceTransformerPredictor')
    def test_integrated_prediction_system_init(self, mock_transformer, mock_psychological, mock_ensemble):
        """Test initialization of IntegratedPredictionSystem."""
        # Setup mocks
        mock_transformer.return_value = MagicMock()
        mock_psychological.return_value = MagicMock()
        mock_ensemble.return_value = MagicMock()
        
        # Initialize system
        system = IntegratedPredictionSystem(
            transformer_model_path=self.model_path,
            feature_dim=22,
            psychological_factors_enabled=True
        )
        
        # Check if components were initialized
        self.assertIsNotNone(system.transformer_predictor)
        self.assertIsNotNone(system.ensemble_predictor)
        self.assertIsNotNone(system.psychological_extractor)
        
        # Check if weights sum to 1
        total_weight = system.ensemble_weight + system.transformer_weight + system.psychological_weight
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        
    @patch('transformer_integration.EnsemblePredictor')
    @patch('transformer_integration.SequenceTransformerPredictor')
    def test_predict_match(self, mock_transformer, mock_ensemble):
        """Test predict_match method."""
        # Setup mocks
        mock_transformer_instance = MagicMock()
        mock_transformer_instance.predict_goals.return_value = (2.1, 1.2)
        mock_transformer.return_value = mock_transformer_instance
        
        mock_ensemble_instance = MagicMock()
        mock_ensemble_instance.predict.return_value = {
            'predicted_home_goals': 2.0,
            'predicted_away_goals': 1.0
        }
        mock_ensemble.return_value = mock_ensemble_instance
        
        # Initialize system
        system = IntegratedPredictionSystem(
            transformer_model_path=self.model_path,
            feature_dim=22,
            psychological_factors_enabled=False
        )
        
        # Test prediction
        match_data = {'home_team': 'team1', 'away_team': 'team2'}
        previous_matches_home = self.sample_matches
        previous_matches_away = self.sample_matches
        
        prediction = system.predict_match(
            match_data=match_data,
            previous_matches_home=previous_matches_home,
            previous_matches_away=previous_matches_away
        )
        
        # Check prediction structure
        self.assertIn('predicted_home_goals', prediction)
        self.assertIn('predicted_away_goals', prediction)
        self.assertIn('sources', prediction)
        self.assertIn('source_contributions', prediction)
        
        # Check sources
        self.assertIn('ensemble', prediction['sources'])
        self.assertIn('transformer', prediction['sources'])
        
        # Check contributions
        self.assertIn('ensemble', prediction['source_contributions'])
        self.assertIn('transformer', prediction['source_contributions'])
        
        # Ensure prediction is not negative
        self.assertGreaterEqual(prediction['predicted_home_goals'], 0)
        self.assertGreaterEqual(prediction['predicted_away_goals'], 0)
    
    def test_prepare_match_sequences(self):
        """Test prepare_match_sequences function."""
        # Test with normal data
        result = prepare_match_sequences(
            match_df=self.df,
            team_id='team1',
            last_n=2
        )
        
        # Check result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['team_id'], 'team1')
        
        # Test with empty data
        empty_df = pd.DataFrame()
        result_empty = prepare_match_sequences(
            match_df=empty_df,
            team_id='team1',
            last_n=2
        )
        self.assertEqual(len(result_empty), 0)
        
        # Test with missing team
        result_missing = prepare_match_sequences(
            match_df=self.df,
            team_id='team999',
            last_n=2
        )
        self.assertEqual(len(result_missing), 0)
        
        # Test with last_n greater than available matches
        result_large_n = prepare_match_sequences(
            match_df=self.df,
            team_id='team1',
            last_n=10
        )
        self.assertEqual(len(result_large_n), 2)
    
    @patch('transformer_integration.predict_match_with_transformer')
    @patch('transformer_integration.integrate_with_specialized_ensemble')
    def test_add_transformer_to_workflow(self, mock_integrate, mock_predict):
        """Test add_transformer_to_workflow function."""
        # Setup mocks
        mock_predict.return_value = {
            'predicted_home_goals': 2.1,
            'predicted_away_goals': 1.2
        }
        
        mock_integrate.return_value = {
            'predicted_home_goals': 2.05,
            'predicted_away_goals': 1.1
        }
        
        # Initial prediction data
        prediction_data = {
            'raw_predicted_home_goals': 2.0,
            'raw_predicted_away_goals': 1.0,
            'other_field': 'value'
        }
        
        # Call function
        result = add_transformer_to_workflow(
            prediction_data=prediction_data,
            transformer_model_path=self.model_path,
            previous_matches_home=self.sample_matches,
            previous_matches_away=self.sample_matches
        )
        
        # Check result structure
        self.assertIn('transformer_component', result)
        self.assertIn('integration_info', result)
        self.assertEqual(result['raw_predicted_home_goals'], 2.05)
        self.assertEqual(result['raw_predicted_away_goals'], 1.1)
        
        # Test with error in predict_match_with_transformer
        mock_predict.side_effect = Exception("Test error")
        
        result_error = add_transformer_to_workflow(
            prediction_data=prediction_data.copy(),
            transformer_model_path=self.model_path,
            previous_matches_home=self.sample_matches,
            previous_matches_away=self.sample_matches
        )
        
        # Check error handling
        self.assertIn('transformer_error', result_error)
        self.assertEqual(result_error['raw_predicted_home_goals'], 2.0)


if __name__ == '__main__':
    unittest.main()
