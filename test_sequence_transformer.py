import unittest
import pandas as pd
import numpy as np
import torch
import os
import json
import tempfile
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import sequence_transformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sequence_transformer import (
    SequenceTransformer,
    PositionalEncoding, 
    MatchSequenceDataset,
    SequenceModelTrainer,
    SequenceTransformerPredictor,
    prepare_team_sequences,
    load_and_process_match_data,
    train_sequence_transformer,
    predict_match_with_transformer,
    integrate_with_specialized_ensemble
)

class TestSequenceTransformer(unittest.TestCase):
    """Test suite for the sequence transformer module."""
    
    def setUp(self):
        """Setup common test data."""
        # Create sample match data
        self.sample_matches = [
            {
                'date': '2024-01-01',
                'home_team_id': 'team1',
                'away_team_id': 'team2',
                'home_goals': 2,
                'away_goals': 1,
                'home_xg': 1.8,
                'away_xg': 0.9,
                'home_shots': 15,
                'away_shots': 8,
                'home_possession': 60,
                'away_possession': 40,
                'home_form': 0.7,
                'away_form': 0.5,
                'home_elo': 1800,
                'away_elo': 1700
            },
            {
                'date': '2024-01-08',
                'home_team_id': 'team3',
                'away_team_id': 'team1',
                'home_goals': 1,
                'away_goals': 3,
                'home_xg': 1.2,
                'away_xg': 2.5,
                'home_shots': 10,
                'away_shots': 18,
                'home_possession': 55,
                'away_possession': 45,
                'home_form': 0.6,
                'away_form': 0.8,
                'home_elo': 1750,
                'away_elo': 1810
            }
        ]
        
        # Create a sample DataFrame
        self.df = pd.DataFrame(self.sample_matches)
        
        # Create a temp directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.test_dir.name, 'test_data.csv')
        self.df.to_csv(self.csv_path, index=False)
        
        # Set default model parameters for tests
        self.feature_dim = 22  # Example value
        self.model_dim = 64
        self.nhead = 4
        self.num_layers = 2
        self.sequence_length = 2
        
    def tearDown(self):
        """Clean up after tests."""
        self.test_dir.cleanup()
    
    def test_positional_encoding(self):
        """Test PositionalEncoding class."""
        d_model = 64
        max_len = 10
        batch_size = 2
        seq_len = 5
        
        # Create positional encoding
        pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Create dummy input
        x = torch.zeros(batch_size, seq_len, d_model)
        
        # Apply positional encoding
        output = pos_encoder(x)
        
        # Check shape
        self.assertEqual(output.shape, (batch_size, seq_len, d_model))
        
        # Check output is different from input (encoding was applied)
        self.assertFalse(torch.all(torch.eq(output, x)))
    
    def test_match_sequence_dataset(self):
        """Test MatchSequenceDataset class."""
        # Prepare sequences
        sequences = [self.sample_matches]
        
        # Create dataset
        dataset = MatchSequenceDataset(sequences, self.sequence_length, 'goals')
        
        # Check length
        self.assertEqual(len(dataset), 1)
        
        # Get item
        features, mask, target = dataset[0]
        
        # Check shapes
        self.assertEqual(features.shape, (self.sequence_length, dataset.feature_dim))
        self.assertEqual(mask.shape, (self.sequence_length,))
        self.assertEqual(target.shape, (2,))  # [home_goals, away_goals]
        
        # Check mask
        self.assertEqual(mask.sum().item(), 2)  # Two matches
        
        # Create dataset for result prediction
        dataset_result = MatchSequenceDataset(sequences, self.sequence_length, 'result')
        _, _, target_result = dataset_result[0]
        
        # Check target is a scalar
        self.assertEqual(target_result.dim(), 0)
    
    def test_sequence_transformer_model(self):
        """Test SequenceTransformer model architecture."""
        # Create model for goals prediction
        model_goals = SequenceTransformer(
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='goals'
        )
        
        # Create model for result prediction
        model_result = SequenceTransformer(
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='result'
        )
        
        # Create dummy input
        batch_size = 2
        x = torch.randn(batch_size, self.sequence_length, self.feature_dim)
        mask = torch.ones(batch_size, self.sequence_length, dtype=torch.bool)
        mask[:, -1] = 0  # Mask last element
        
        # Forward pass
        output_goals = model_goals(x, mask)
        output_result = model_result(x, mask)
        
        # Check shapes
        self.assertEqual(output_goals.shape, (batch_size, 2))  # [home_goals, away_goals]
        self.assertEqual(output_result.shape, (batch_size, 3))  # [home_win, draw, away_win]
    
    def test_prepare_team_sequences(self):
        """Test prepare_team_sequences function."""
        # Add missing team_id and opponent_id columns
        df = self.df.copy()
        df['team_id'] = df['home_team_id']
        df['opponent_id'] = df['away_team_id']
        df['match_date'] = pd.to_datetime(df['date'])
        
        # Generate sequences
        sequences = prepare_team_sequences(
            df,
            sequence_length=1  # Use a small sequence length for testing
        )
        
        # Check output
        self.assertIsInstance(sequences, dict)
        self.assertIn('team1', sequences)
        
        # Check sequence structure
        team1_sequences = sequences['team1']
        self.assertIsInstance(team1_sequences, list)
        self.assertEqual(len(team1_sequences), 1)  # Should have 1 sequence
        
        # Check sequence content
        first_sequence = team1_sequences[0]
        self.assertIsInstance(first_sequence, list)
        self.assertEqual(len(first_sequence), 1)  # Should have 1 match
    
    def test_load_and_process_match_data(self):
        """Test load_and_process_match_data function."""
        # Process match data
        home_view, away_view = load_and_process_match_data(self.csv_path)
        
        # Check output types
        self.assertIsInstance(home_view, pd.DataFrame)
        self.assertIsInstance(away_view, pd.DataFrame)
        
        # Check column transformations
        self.assertIn('team_id', home_view.columns)
        self.assertIn('opponent_id', home_view.columns)
        self.assertIn('is_home', home_view.columns)
        
        # Check is_home values
        self.assertTrue((home_view['is_home'] == 1).all())
        self.assertTrue((away_view['is_home'] == 0).all())
        
        # Check team_goals and opponent_goals are correctly assigned
        if 'team_goals' in home_view.columns:
            # In home_view, team_goals should be home_goals
            home_row = home_view.iloc[0]
            self.assertEqual(home_row['team_goals'], home_row['home_goals'])
            
            # In away_view, team_goals should be away_goals
            away_row = away_view.iloc[0]
            self.assertEqual(away_row['team_goals'], away_row['away_goals'])
    
    def test_sequence_transformer_predictor(self):
        """Test SequenceTransformerPredictor class."""
        # Create a temporary model file
        model_path = os.path.join(self.test_dir.name, 'test_model.pt')
        
        # Create and save a model
        model = SequenceTransformer(
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='goals'
        )
        torch.save(model.state_dict(), model_path)
        
        # Create predictor
        predictor = SequenceTransformerPredictor(
            model_path=model_path,
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='goals',
            sequence_length=self.sequence_length
        )
        
        # Check predictor attributes
        self.assertEqual(predictor.prediction_type, 'goals')
        self.assertEqual(predictor.sequence_length, self.sequence_length)
        
        # Test prepare_sequence
        features, mask = predictor.prepare_sequence(self.sample_matches)
        self.assertEqual(features.shape, (1, self.sequence_length, predictor.model.input_projection.in_features))
        self.assertEqual(mask.shape, (1, self.sequence_length))
        
        # Test predict_goals
        with torch.no_grad():
            home_goals, away_goals = predictor.predict_goals(self.sample_matches)
        self.assertIsInstance(home_goals, float)
        self.assertIsInstance(away_goals, float)
        
        # Create and test result predictor
        model_result = SequenceTransformer(
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='result'
        )
        model_result_path = os.path.join(self.test_dir.name, 'test_model_result.pt')
        torch.save(model_result.state_dict(), model_result_path)
        
        predictor_result = SequenceTransformerPredictor(
            model_path=model_result_path,
            feature_dim=self.feature_dim,
            model_dim=self.model_dim,
            nhead=self.nhead,
            num_layers=self.num_layers,
            prediction_type='result',
            sequence_length=self.sequence_length
        )
        
        # Test predict_result
        with torch.no_grad():
            result_probs = predictor_result.predict_result(self.sample_matches)
        self.assertIn('home_win', result_probs)
        self.assertIn('draw', result_probs)
        self.assertIn('away_win', result_probs)
    
    def test_integrate_with_specialized_ensemble(self):
        """Test integrate_with_specialized_ensemble function."""
        # Create sample predictions
        transformer_prediction = {
            'predicted_home_goals': 2.0,
            'predicted_away_goals': 1.5
        }
        
        ensemble_predictions = {
            'predicted_home_goals': 2.2,
            'predicted_away_goals': 1.3
        }
        
        # Integrate predictions
        integrated = integrate_with_specialized_ensemble(
            transformer_prediction,
            ensemble_predictions
        )
        
        # Check result
        self.assertIn('predicted_home_goals', integrated)
        self.assertIn('predicted_away_goals', integrated)
        self.assertIn('prediction_sources', integrated)
        self.assertIn('integration_weights', integrated)
        
        # Check integration weights
        self.assertEqual(integrated['integration_weights']['specialized_ensemble'], 0.7)
        self.assertEqual(integrated['integration_weights']['sequence_transformer'], 0.3)
        
        # Check integrated values match weighted average
        self.assertAlmostEqual(
            integrated['predicted_home_goals'],
            0.7 * ensemble_predictions['predicted_home_goals'] + 0.3 * transformer_prediction['predicted_home_goals']
        )
        self.assertAlmostEqual(
            integrated['predicted_away_goals'],
            0.7 * ensemble_predictions['predicted_away_goals'] + 0.3 * transformer_prediction['predicted_away_goals']
        )
        
        # Test integration with different prediction types
        transformer_prediction = {
            'result_probabilities': {
                'home_win': 0.6,
                'draw': 0.2,
                'away_win': 0.2
            },
            'prediction_type': 'result'
        }
        
        integrated = integrate_with_specialized_ensemble(
            transformer_prediction,
            ensemble_predictions
        )
        
        # Check both prediction types are present
        self.assertIn('result_probabilities', integrated)
        self.assertIn('predicted_home_goals', integrated)


if __name__ == '__main__':
    unittest.main()
