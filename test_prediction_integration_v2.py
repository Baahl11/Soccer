import unittest
import pandas as pd
import numpy as np
from prediction_integration import create_features, StackingModel
from features import FeatureExtractor

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

    def test_create_features(self):
        features = create_features(self.fixture_data, self.lineup_data, self.stats_json, self.odds_data)
        self.assertIn('total_goals_home', features)
        self.assertIn('total_goals_away', features)
        self.assertIn('home_corners', features)
        self.assertIn('away_corners', features)

    def test_stacking_model(self):
        model = StackingModel()
        # Create a larger, balanced dataset
        X = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'feature2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        })
        # Ensure balanced classes (5 of each class)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model.fit(X, y)

if __name__ == '__main__':
    unittest.main()
