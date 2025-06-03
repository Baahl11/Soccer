import unittest
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from prediction_integration import create_features, StackingModel
from prediction_integration import (
    prepare_data_for_prediction,
    enrich_prediction_with_contextual_data,
    make_integrated_prediction
)

class TestPredictionIntegration(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.fixture_data = {
            'home_team_id': 1,
            'away_team_id': 2,
            'venue': 'Stadium A'
        }
        self.lineup_data = {
            'home_formation': '4-3-3',
            'away_formation': '4-4-2',
            'wingers': 2
        }
        self.stats_json = pd.DataFrame({
            'home_team_id': [1, 1, 2],
            'away_team_id': [2, 2, 1],
            'shots': [10, 8, 5],
            'corners': [5, 3, 2],
            'expected_goals': [1.5, 1.2, 0.8]
        })
        self.odds_data = {
            'home_win_odds': 1.5,
            'draw_odds': 3.0,
            'away_win_odds': 2.5
        }
        self.base_prediction = {
            'predicted_home_goals': 1.8,
            'predicted_away_goals': 1.2,
            'total_goals': 3.0,
            'prob_over_2_5': 0.65,
            'prob_btts': 0.7
        }

    def test_create_features(self):
        features = create_features(self.fixture_data, self.lineup_data, self.stats_json, self.odds_data)
        self.assertIn('total_goals_home', features)
        self.assertIn('total_goals_away', features)
        self.assertIn('home_corners', features)
        self.assertIn('away_corners', features)

    def test_stacking_model(self):
        model = StackingModel()
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'feature2': [4, 5, 6, 7, 8, 9, 10, 11]
        })
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Verify predictions are integers and have correct shape
        self.assertTrue(np.issubdtype(predictions.dtype, np.integer))
        self.assertEqual(len(predictions), len(y))
        self.assertTrue(all(isinstance(pred, (int, np.integer)) for pred in predictions))
        self.assertTrue(all(pred in [0, 1] for pred in predictions))
    
    def test_enrich_prediction_with_weather(self):
        """Test that weather data correctly affects predictions"""
        weather_data = {
            'condition': 'rain',
            'intensity': 'heavy'
        }
        
        enriched = enrich_prediction_with_contextual_data(
            self.base_prediction,
            weather_data=weather_data
        )
        
        # Heavy rain should reduce goals
        self.assertLess(enriched['predicted_home_goals'], self.base_prediction['predicted_home_goals'])
        self.assertLess(enriched['predicted_away_goals'], self.base_prediction['predicted_away_goals'])
        self.assertIn('weather', enriched['adjustments_applied'])
    
    def test_enrich_prediction_with_injuries(self):
        """Test that injury data correctly affects predictions"""
        player_data = {
            'home_injuries': [
                {'importance': 'high'},
                {'importance': 'high'}
            ],
            'away_injuries': [
                {'importance': 'medium'}
            ]
        }
        
        enriched = enrich_prediction_with_contextual_data(
            self.base_prediction,
            player_data=player_data
        )
        
        # Injuries in home team should reduce their expected goals
        self.assertLess(enriched['predicted_home_goals'], self.base_prediction['predicted_home_goals'])
        self.assertIn('injuries', enriched['adjustments_applied'])

if __name__ == '__main__':
    unittest.main()
