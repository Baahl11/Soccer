import unittest
import pandas as pd
import numpy as np
import torch
import dgl
import os
import json
import tempfile
import sys
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path to import team_relationship_gnn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from team_relationship_gnn import (
    TeamGraphDataset,
    TeamGNN,
    MatchDataset,
    TeamGNNTrainer,
    TeamGNNPredictor,
    prepare_data_for_gnn,
    train_team_gnn,
    predict_with_team_gnn,
    integrate_gnn_with_transformer
)

class TestTeamRelationshipGNN(unittest.TestCase):
    """Test suite for the team relationship GNN module."""
    
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
                'away_possession': 40
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
                'home_possession': 45,
                'away_possession': 55
            },
            {
                'date': '2024-01-15',
                'home_team_id': 'team2',
                'away_team_id': 'team3',
                'home_goals': 0,
                'away_goals': 0,
                'home_xg': 0.7,
                'away_xg': 1.1,
                'home_shots': 12,
                'away_shots': 9,
                'home_possession': 52,
                'away_possession': 48
            }
        ]
        
        # Create a sample DataFrame
        self.df = pd.DataFrame(self.sample_matches)
        
        # Create a temp directory for test files
        self.test_dir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.test_dir.name, 'test_model.pt')
        self.metadata_path = os.path.join(self.test_dir.name, 'test_model_metadata.json')
        
        # Default model parameters for tests
        self.in_features = 8  # Example value
        self.hidden_features = 32
        self.num_layers = 2
    
    def tearDown(self):
        """Clean up after tests."""
        self.test_dir.cleanup()
    
    def test_team_graph_dataset(self):
        """Test TeamGraphDataset class."""
        # Create dataset
        dataset = TeamGraphDataset(
            match_df=self.df,
            team_id_home_col='home_team_id',
            team_id_away_col='away_team_id',
            result_cols=['home_goals', 'away_goals']
        )
        
        # Check basic properties
        self.assertEqual(len(dataset.teams), 3)  # team1, team2, team3
        self.assertIn('team1', dataset.team_to_idx)
        self.assertIn('team2', dataset.team_to_idx)
        self.assertIn('team3', dataset.team_to_idx)
        
        # Check graph creation
        g = dataset.graph
        self.assertIsInstance(g, dgl.DGLGraph)
        self.assertEqual(g.number_of_nodes(), 3)  # 3 teams
        self.assertEqual(g.number_of_edges(), 6)  # 3 games × 2 directions
        
        # Check edge features
        self.assertIn('features', g.edata)
        
        # Test node features generation
        dataset.update_node_features('mean')
        self.assertIn('h', g.ndata)
        self.assertEqual(g.ndata['h'].shape, (3, 8))  # 3 teams, 8 features
        
        # Test getting team embedding
        embedding = dataset.get_team_embedding('team1')
        self.assertEqual(embedding.shape, (8,))
        
        # Test subgraph extraction
        subg = dataset.get_subgraph_for_teams(['team1', 'team2'])
        self.assertEqual(subg.number_of_nodes(), 2)
    
    def test_team_gnn_model(self):
        """Test TeamGNN model architecture."""
        # Create model with different GNN types
        for gnn_type in ['gcn', 'gat', 'sage']:
            model = TeamGNN(
                in_feats=self.in_features,
                hidden_feats=self.hidden_features,
                num_layers=self.num_layers,
                gnn_type=gnn_type
            )
            
            # Check model structure
            self.assertEqual(len(model.layers), self.num_layers)
            self.assertIsInstance(model.pred_layer, torch.nn.Sequential)
            
            # Create dummy inputs
            g = dgl.graph(([0, 1], [1, 0]), num_nodes=2)
            g.ndata['h'] = torch.randn(2, self.in_features)
            team_indices = torch.tensor([[0, 1]])
            
            # Forward pass
            output = model(g, team_indices)
            
            # Check output shape
            self.assertEqual(output.shape, (1, 2))  # Batch size 1, 2 outputs (home/away goals)
            
            # Test get_team_embedding
            embedding = model.get_team_embedding(g, 0)
            self.assertEqual(embedding.shape, (self.hidden_features,))
    
    def test_match_dataset(self):
        """Test MatchDataset class."""
        # Create team graph dataset
        team_graph = TeamGraphDataset(
            match_df=self.df,
            team_id_home_col='home_team_id',
            team_id_away_col='away_team_id',
            result_cols=['home_goals', 'away_goals']
        )
        
        # Create match dataset
        dataset = MatchDataset(
            team_graph=team_graph,
            match_df=self.df,
            team_id_home_col='home_team_id',
            team_id_away_col='away_team_id',
            result_cols=['home_goals', 'away_goals']
        )
        
        # Check dataset length
        self.assertEqual(len(dataset), 3)  # 3 matches
        
        # Get an item
        team_indices, goals = dataset[0]
        
        # Check shapes
        self.assertEqual(team_indices.shape, (2,))  # [home_idx, away_idx]
        self.assertEqual(goals.shape, (2,))  # [home_goals, away_goals]
        
        # Check values
        self.assertEqual(goals[0], 2.0)  # home_goals of first match
        self.assertEqual(goals[1], 1.0)  # away_goals of first match
    
    @patch('torch.save')
    def test_team_gnn_trainer(self, mock_save):
        """Test TeamGNNTrainer class."""
        # Create model
        model = TeamGNN(
            in_feats=self.in_features,
            hidden_feats=self.hidden_features,
            num_layers=self.num_layers
        )
        
        # Create trainer
        trainer = TeamGNNTrainer(
            model=model,
            learning_rate=0.01,
            device='cpu'
        )
        
        # Create dummy graph and data
        g = dgl.graph(([0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]), num_nodes=3)
        g.ndata['h'] = torch.randn(3, self.in_features)
        g.edata['features'] = torch.randn(6, self.in_features)
        
        # Create dummy dataloader
        class DummyLoader:
            def __init__(self, num_batches=2):
                self.num_batches = num_batches
                
            def __iter__(self):
                for _ in range(self.num_batches):
                    yield torch.tensor([[0, 1], [1, 2]]), torch.tensor([[2.0, 1.0], [0.0, 0.0]])
        
        train_loader = DummyLoader()
        val_loader = DummyLoader(1)
        
        # Test train_epoch
        train_loss = trainer.train_epoch(g, train_loader)
        self.assertIsInstance(train_loss, float)
        
        # Test evaluate
        metrics = trainer.evaluate(g, val_loader)
        self.assertIn('loss', metrics)
        self.assertIn('mse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('home_mse', metrics)
        self.assertIn('away_mse', metrics)
        
        # Test train
        history = trainer.train(
            graph=g,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            patience=1,
            model_save_path=self.model_path
        )
        
        self.assertIn('train_losses', history)
        self.assertIn('val_metrics', history)
        self.assertEqual(len(history['train_losses']), 2)
    
    @patch('team_relationship_gnn.TeamGNN')
    def test_team_gnn_predictor(self, mock_model):
        """Test TeamGNNPredictor class."""
        # Setup mock model
        mock_instance = MagicMock()
        mock_instance.return_value = torch.tensor([[2.1, 0.9]])
        mock_instance.get_team_embedding.return_value = torch.randn(self.hidden_features)
        mock_model.return_value = mock_instance
        
        # Create team graph
        team_graph = TeamGraphDataset(self.df)
        
        # Create model metadata
        metadata = {
            'model_params': {
                'in_features': self.in_features,
                'hidden_features': self.hidden_features,
                'num_layers': self.num_layers,
                'gnn_type': 'gcn'
            },
            'test_metrics': {},
            'num_teams': 3,
            'team_mapping': {'team1': 0, 'team2': 1, 'team3': 2}
        }
        
        # Write metadata file
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Create dummy model file
        with open(self.model_path, 'w') as f:
            f.write('mock model')
        
        # Create predictor
        predictor = TeamGNNPredictor(
            model_path=self.model_path,
            team_graph=team_graph,
            in_features=self.in_features,
            hidden_features=self.hidden_features,
            num_layers=self.num_layers,
            device='cpu'
        )
        
        # Test predict_match
        home_goals, away_goals = predictor.predict_match('team1', 'team2')
        self.assertIsInstance(home_goals, float)
        self.assertIsInstance(away_goals, float)
        
        # Test get_team_style
        style = predictor.get_team_style('team1')
        self.assertIn('attacking_strength', style)
        self.assertIn('defensive_solidity', style)
        self.assertIn('tactical_flexibility', style)
        
        # Test get_matchup_analysis
        analysis = predictor.get_matchup_analysis('team1', 'team2')
        self.assertIn('predicted_score', analysis)
        self.assertIn('team_styles', analysis)
        self.assertIn('comparative_advantage', analysis)
        self.assertIn('key_factors', analysis)
    
    def test_prepare_data_for_gnn(self):
        """Test prepare_data_for_gnn function."""
        # Prepare data
        data = prepare_data_for_gnn(
            match_df=self.df,
            batch_size=2
        )
        
        # Check output structure
        self.assertIn('team_graph', data)
        self.assertIn('graph', data)
        self.assertIn('dataloaders', data)
        self.assertIn('feature_dim', data)
        
        # Check dataloaders
        self.assertIn('train', data['dataloaders'])
        self.assertIn('val', data['dataloaders'])
        self.assertIn('test', data['dataloaders'])
    
    @patch('team_relationship_gnn.TeamGNNTrainer')
    @patch('team_relationship_gnn.TeamGNN')
    @patch('json.dump')
    def test_train_team_gnn(self, mock_json_dump, mock_model, mock_trainer):
        """Test train_team_gnn function."""
        # Setup mock trainer
        mock_trainer_instance = MagicMock()
        mock_trainer_instance.train.return_value = {
            'train_losses': [0.5, 0.4],
            'val_metrics': [{'loss': 0.5}, {'loss': 0.4}]
        }
        mock_trainer_instance.evaluate.return_value = {
            'loss': 0.4,
            'mse': 0.3,
            'mae': 0.2
        }
        mock_trainer.return_value = mock_trainer_instance
        
        # Create directory for model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Train model
        results = train_team_gnn(
            match_df=self.df,
            model_save_path=self.model_path,
            num_epochs=2
        )
        
        # Check results structure
        self.assertIn('training_history', results)
        self.assertIn('test_metrics', results)
        self.assertIn('model_path', results)
        self.assertIn('metadata_path', results)
    
    def test_integrate_gnn_with_transformer(self):
        """Test integrate_gnn_with_transformer function."""
        # Create sample predictions
        gnn_prediction = {
            'predicted_score': {
                'home': 2.1,
                'away': 0.9
            },
            'team_styles': {
                'home': {'attacking_strength': 0.7},
                'away': {'attacking_strength': 0.5}
            },
            'key_factors': ['Ventaja ofensiva del equipo local']
        }
        
        transformer_prediction = {
            'predicted_home_goals': 2.3,
            'predicted_away_goals': 1.1
        }
        
        ensemble_prediction = {
            'predicted_home_goals': 2.0,
            'predicted_away_goals': 1.0
        }
        
        # Integrate predictions
        integrated = integrate_gnn_with_transformer(
            gnn_prediction=gnn_prediction,
            transformer_prediction=transformer_prediction,
            ensemble_prediction=ensemble_prediction,
            gnn_weight=0.2,
            transformer_weight=0.3,
            ensemble_weight=0.5
        )
        
        # Check integrated prediction structure
        self.assertIn('predicted_home_goals', integrated)
        self.assertIn('predicted_away_goals', integrated)
        self.assertIn('integration_weights', integrated)
        self.assertIn('component_predictions', integrated)
        self.assertIn('team_style_analysis', integrated)
        self.assertIn('key_factors', integrated)
        
        # Check weights sum to 1
        weights = integrated['integration_weights']
        sum_weights = weights['gnn'] + weights['transformer'] + weights['ensemble']
        self.assertAlmostEqual(sum_weights, 1.0)
        
        # Check weighted average calculation
        expected_home = 2.1*0.2 + 2.3*0.3 + 2.0*0.5
        expected_away = 0.9*0.2 + 1.1*0.3 + 1.0*0.5
        self.assertAlmostEqual(integrated['predicted_home_goals'], expected_home)
        self.assertAlmostEqual(integrated['predicted_away_goals'], expected_away)


if __name__ == '__main__':
    unittest.main()
