"""Unit tests for feature engineering classes."""

import unittest
import numpy as np
from datetime import datetime, timedelta
from feature_engineering import FeatureEngineering, ScheduleCongestion, SeasonalAnalyzer
from match_statistics import MatchStatisticsAnalyzer
from contextual_features import ContextualFeatures

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.feature_engineering = FeatureEngineering()
        self.stats_analyzer = MatchStatisticsAnalyzer()
        self.context_analyzer = ContextualFeatures()
        
        # Sample match data
        self.match_data = {
            'shots': 15,
            'shots_on_target': 6,
            'possession': 55,
            'passes_completed': 400,
            'passes_attempted': 450,
            'pressures': 150,
            'pressure_regains': 50,
            'defensive_line_height': 65,
            'stadium_capacity': 40000,
            'attendance': 35000,
            'pitch_dimensions': {'length': 105, 'width': 68},
            'is_home': True,
            'games_at_stadium': 15,
            'home_manager': {'name': 'John Smith', 'attacking_preference': 0.7},
            'away_manager': {'name': 'Mike Jones', 'attacking_preference': 0.4}
        }
        
        # Sample historical matches
        self.team_history = [
            {
                'date': (datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d'),
                'shots': 12 + i,
                'shots_on_target': 5 + i//2,
                'goals': 2,
                'possession': 52 + i,
                'corners': 6,
                'yellow_cards': 2,
                'red_cards': 0,
                'fouls': 12,
                'pressures': 140 + i*2,
                'pressure_regains': 45 + i
            }
            for i in range(10)
        ]
        
        # Weather data
        self.weather_data = {
            'conditions': 'clear',
            'wind_speed': 15,
            'precipitation': 0,
            'temperature': 20
        }
        
    def test_feature_extraction(self):
        # Test main feature extraction
        features = self.feature_engineering.extract_features(
            self.match_data,
            self.team_history,
            datetime.now().strftime('%Y-%m-%d'),
            2023,
            self.weather_data
        )
        
        # Verify key features are present
        self.assertIn('congestion_index', features)
        self.assertIn('shot_intensity_score', features)
        self.assertIn('possession_dominance', features)
        self.assertIn('h2h_intensity', features)
        
        # Verify feature values are within expected ranges
        self.assertGreaterEqual(features['congestion_index'], 0)
        self.assertLessEqual(features['congestion_index'], 2.0)
        self.assertGreaterEqual(features['shot_intensity_score'], 0)
        self.assertLessEqual(features['shot_intensity_score'], 2.0)
        
    def test_match_statistics(self):
        # Test match statistics analysis
        stats = self.stats_analyzer.analyze_match_stats(
            self.match_data,
            self.team_history
        )
        
        # Verify key statistics metrics
        self.assertIn('shot_intensity_score', stats)
        self.assertIn('possession_dominance', stats)
        self.assertIn('pressing_intensity', stats)
        
        # Check value ranges
        self.assertGreaterEqual(stats['shot_intensity_score'], 0)
        self.assertLessEqual(stats['shot_intensity_score'], 1.0)
        self.assertGreaterEqual(stats['possession_dominance'], 0)
        self.assertLessEqual(stats['possession_dominance'], 2.0)
        
    def test_contextual_features(self):
        # Test contextual feature analysis
        context = self.context_analyzer.analyze_context(
            self.match_data,
            self.team_history,
            self.weather_data
        )
        
        # Verify key contextual features
        self.assertIn('crowd_intensity', context)
        self.assertIn('pitch_size_effect', context)
        self.assertIn('weather_impact', context)
        
        # Check value ranges
        self.assertGreaterEqual(context['crowd_intensity'], 0)
        self.assertLessEqual(context['crowd_intensity'], 1.0)
        self.assertGreaterEqual(context['pitch_size_effect'], 0.5)
        self.assertLessEqual(context['pitch_size_effect'], 1.5)
        
    def test_empty_data_handling(self):
        # Test handling of empty or missing data
        features = self.feature_engineering.extract_features(
            {},  # Empty match data
            [],  # Empty history
            datetime.now().strftime('%Y-%m-%d'),
            None,
            None
        )
        
        # Should return default values without errors
        self.assertIsInstance(features, dict)
        
    def test_schedule_congestion(self):
        schedule = ScheduleCongestion()
        metrics = schedule.analyze_schedule_load(
            self.team_history,
            datetime.now().strftime('%Y-%m-%d')
        )
        
        # Verify schedule metrics
        self.assertIn('congestion_index', metrics)
        self.assertIn('fatigue_risk', metrics)
        self.assertIn('recovery_quality', metrics)
        
        # Check value ranges
        self.assertGreaterEqual(metrics['congestion_index'], 0)
        self.assertLessEqual(metrics['congestion_index'], 2.0)
        self.assertGreaterEqual(metrics['fatigue_risk'], 0)
        self.assertLessEqual(metrics['fatigue_risk'], 1.0)
        
    def test_seasonal_analysis(self):
        seasonal = SeasonalAnalyzer()
        metrics, segment = seasonal.analyze_seasonal_effects(
            self.team_history,
            datetime.now().strftime('%Y-%m-%d'),
            2023
        )
        
        # Verify seasonal metrics
        self.assertIn('season_progress', metrics)
        self.assertIn('month_importance', metrics)
        
        # Check value ranges
        self.assertGreaterEqual(metrics['season_progress'], 0)
        self.assertLessEqual(metrics['season_progress'], 1.0)
        self.assertGreaterEqual(metrics['month_importance'], 0)
        self.assertLessEqual(metrics['month_importance'], 1.0)
        
        # Check segment is valid
        self.assertIn(segment, ['early', 'mid', 'late'])
        
if __name__ == '__main__':
    unittest.main()
