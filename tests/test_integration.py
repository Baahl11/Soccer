import unittest
import sys
import os
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Añadir el directorio padre al path para importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fnn_model import FeedforwardNeuralNetwork
from features import FeatureExtractor
from player_injuries import InjuryAnalyzer
from team_form import FormAnalyzer
from team_history import HistoricalAnalyzer
from predictions import make_global_prediction, calculate_1x2_probabilities
from backup_manager import BackupManager

class TestIntegrationSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Setup para todas las pruebas"""
        cls.feature_extractor = FeatureExtractor()
        cls.injury_analyzer = InjuryAnalyzer()
        cls.form_analyzer = FormAnalyzer()
        cls.historical_analyzer = HistoricalAnalyzer()
        cls.backup_manager = BackupManager()

    def setUp(self):
        """Setup para cada prueba individual"""
        self.sample_fixture = {
            "fixture_id": 1234,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "home_team": {"id": 50, "name": "Manchester City"},
            "away_team": {"id": 49, "name": "Chelsea"},
            "league": {"id": 39, "name": "Premier League"}
        }

    def test_feature_extraction(self):
        """Probar extracción de características"""
        features = self.feature_extractor.extract_match_features(
            home_team_id=50,
            away_team_id=49,
            league_id=39,
            match_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        self.assertIsInstance(features, np.ndarray)
        self.assertTrue(len(features) > 0)
        self.assertFalse(np.any(np.isnan(features)))

    def test_injury_analysis(self):
        """Probar análisis de lesiones"""
        impact_score = self.injury_analyzer.get_team_injury_impact(
            team_id=50,
            fixture_id=1234
        )
        
        self.assertIsInstance(impact_score, (float, int))
        self.assertTrue(0 <= impact_score <= 1)

    def test_form_analysis(self):
        """Probar análisis de forma"""
        form_metrics = self.form_analyzer.get_team_form_metrics(
            team_id=50,
            last_matches=5
        )
        
        self.assertIsInstance(form_metrics, dict)
        self.assertTrue(all(isinstance(v, (float, int)) for v in form_metrics.values()))

    def test_historical_analysis(self):
        """Probar análisis histórico"""
        h2h_stats = self.historical_analyzer.get_head_to_head_stats(
            home_team_id=50,
            away_team_id=49,
            last_matches=5
        )
        
        self.assertIsInstance(h2h_stats, dict)
        self.assertTrue(len(h2h_stats) > 0)

    def test_global_prediction(self):
        """Probar predicción global"""
        # Pass fixture_id (int) instead of dict to make_global_prediction
        prediction = make_global_prediction(self.sample_fixture["fixture_id"])
        
        self.assertIsInstance(prediction, dict)
        required_keys = [
            "predicted_home_goals",
            "predicted_away_goals",
            "total_goals",
            "prob_over_2_5",
            "prob_btts",
            "confidence"
        ]
        for key in required_keys:
            self.assertIn(key, prediction)
            self.assertIsInstance(prediction[key], float)

    def test_probability_calculations(self):
        """Probar cálculos de probabilidades"""
        home_goals = 2.1
        away_goals = 1.4
        
        # Use calculate_1x2_probabilities instead of calculate_result_probabilities
        home_win, draw, away_win = calculate_1x2_probabilities(home_goals, away_goals)
        
        self.assertAlmostEqual(home_win + draw + away_win, 1.0, places=2)
        self.assertTrue(0 <= home_win <= 1)
        self.assertTrue(0 <= draw <= 1)
        self.assertTrue(0 <= away_win <= 1)

    def test_backup_system(self):
        """Probar sistema de backup"""
        # Create backup with no additional files (None)
        backup_info = self.backup_manager.create_backup(additional_files=None)
        self.assertIsNotNone(backup_info)
        
        # List backups
        backups = self.backup_manager.list_backups()
        self.assertGreater(len(backups), 0)
        
        # Get latest backup info
        latest_backup = self.backup_manager.get_latest_backup()
        self.assertIsNotNone(latest_backup)
        self.assertIsNotNone(latest_backup)
        self.assertIsNotNone(latest_backup)
        self.assertIsInstance(latest_backup, dict)
        self.assertIsNotNone(latest_backup)
        self.assertIsInstance(latest_backup, dict)
        self.assertTrue(latest_backup is not None)
        self.assertTrue(isinstance(latest_backup, dict))
        self.assertIsNotNone(latest_backup)
        if latest_backup is not None:
            self.assertIn("timestamp", latest_backup)
            self.assertIn("directory", latest_backup)

    def test_model_integration(self):
        """Probar integración del modelo"""
        try:
            model = FeedforwardNeuralNetwork(input_dim=30)
            self.assertIsNotNone(model)
            
            # Probar predicción con datos aleatorios
            dummy_input = np.random.rand(1, 30)
            prediction = model.predict(dummy_input)
            
            self.assertIsInstance(prediction, np.ndarray)
            self.assertEqual(prediction.shape[1], 2)  # [home_goals, away_goals]
            
        except Exception as e:
            self.fail(f"Modelo falló en integración: {str(e)}")

    def test_data_consistency(self):
        """Probar consistencia de datos entre componentes"""
        # Obtener datos de forma
        form_data = self.form_analyzer.get_team_form_metrics(50, last_matches=5)
        
        # Obtener datos históricos
        h2h_data = self.historical_analyzer.get_head_to_head_stats(50, 49, last_matches=5)
          # Verificar que las métricas sean consistentes
        excluded_metrics = ['recent_avg_goals_scored', 'recent_avg_goals_conceded', 'recent_avg_goal_diff']
        self.assertTrue(all(0 <= v <= 1 for k, v in form_data.items() 
                         if isinstance(v, float) and k not in excluded_metrics))
        self.assertTrue(all(v >= 0 for v in h2h_data.values() if isinstance(v, (int, float))))

if __name__ == '__main__':
    unittest.main(verbosity=2)