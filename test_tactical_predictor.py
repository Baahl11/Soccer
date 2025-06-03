"""
Tests unitarios para el sistema de predicción táctica de corners.
"""

import unittest
import logging
from typing import Dict, Any
from tactical_corner_predictor import TacticalCornerPredictor
from corner_data_collector import FormationDataCollector
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTacticalPredictor(unittest.TestCase):
    """Suite de pruebas para el predictor táctico de corners"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.predictor = TacticalCornerPredictor()
        self.formation_collector = FormationDataCollector()
        
        # Datos de prueba básicos
        self.test_match_data = {
            'home_team_id': 50,  # Manchester City
            'away_team_id': 33,  # Manchester United
            'home_formation': '4-3-3',
            'away_formation': '5-3-2',
            'league_id': 39,
            'home_team_name': 'Manchester City',
            'away_team_name': 'Manchester United',
            'home_stats': {
                'avg_corners_for': 6.2,
                'avg_corners_against': 3.8,
                'form_score': 65,
                'attack_strength': 1.15,
                'defense_strength': 1.2,
                'avg_shots': 15.3
            },
            'away_stats': {
                'avg_corners_for': 5.7,
                'avg_corners_against': 4.2,
                'form_score': 62,
                'attack_strength': 1.1,
                'defense_strength': 1.05,
                'avg_shots': 14.1
            }
        }

    def test_formation_features(self):
        """Prueba la extracción de características de formación"""
        features = self.formation_collector.get_formation_features('4-3-3')
        
        self.assertIsInstance(features, dict)
        self.assertIn('wing_attack', features)
        self.assertIn('high_press', features)
        self.assertIn('possession', features)
        
        # Verificar valores específicos
        self.assertTrue(0 <= features['wing_attack'] <= 1)
        self.assertTrue(0 <= features['high_press'] <= 1)
        self.assertTrue(0 <= features['possession'] <= 1)

    def test_formation_advantage(self):
        """Prueba el cálculo de ventaja táctica"""
        advantage = self.predictor.calculate_formation_advantage('4-3-3', '5-3-2')
        
        self.assertIsInstance(advantage, dict)
        self.assertIn('attack_advantage', advantage)
        self.assertIn('midfield_control', advantage)
        self.assertIn('defensive_solidity', advantage)
        
        # Verificar rangos
        for value in advantage.values():
            self.assertTrue(-1 <= value <= 1)

    def test_predict_with_formations(self):
        """Prueba la predicción incluyendo análisis de formaciones"""
        prediction = self.predictor.predict_with_formations(self.test_match_data)
        
        # Verificar estructura básica
        self.assertIn('home_corners', prediction)
        self.assertIn('away_corners', prediction)
        self.assertIn('total_corners', prediction)
        self.assertIn('tactical_analysis', prediction)
        
        # Verificar validez de predicciones
        self.assertGreaterEqual(prediction['home_corners'], 0)
        self.assertGreaterEqual(prediction['away_corners'], 0)
        self.assertEqual(
            prediction['total_corners'],
            prediction['home_corners'] + prediction['away_corners']
        )

    def test_formation_normalization(self):
        """Prueba la normalización de formatos de formación"""
        test_cases = [
            ('4-4-2', '4-4-2'),
            ('442', '4-4-2'),
            ('4-2-3-1', '4-2-3-1'),
            ('4231', '4-2-3-1'),
            ('3-5-2', '3-5-2'),
            ('352', '3-5-2')
        ]
        
        for input_formation, expected in test_cases:
            normalized = self.formation_collector.normalize_formation(input_formation)
            self.assertEqual(normalized, expected)

    def test_error_handling(self):
        """Prueba el manejo de errores en casos extremos"""
        # Caso 1: Formación inválida
        invalid_data = self.test_match_data.copy()
        invalid_data['home_formation'] = 'invalid'
        prediction = self.predictor.predict_with_formations(invalid_data)
        self.assertIn('home_corners', prediction)  # Debería usar valores por defecto
        
        # Caso 2: Datos estadísticos faltantes
        incomplete_data = self.test_match_data.copy()
        del incomplete_data['home_stats']
        prediction = self.predictor.predict_with_formations(incomplete_data)
        self.assertIn('error', prediction)
        
        # Caso 3: IDs de equipo inválidos
        invalid_team_data = self.test_match_data.copy()
        invalid_team_data['home_team_id'] = -1
        prediction = self.predictor.predict_with_formations(invalid_team_data)
        self.assertIn('error', prediction)

    def test_tactical_indices(self):
        """Prueba el cálculo de índices tácticos"""
        indices = self.predictor.calculate_tactical_indices('4-3-3')
        
        self.assertIsInstance(indices, dict)
        expected_indices = ['wing_attack_index', 'high_press_index', 'possession_index']
        for index in expected_indices:
            self.assertIn(index, indices)
            self.assertTrue(0 <= indices[index] <= 1)

    def test_full_prediction_pipeline(self):
        """Prueba el pipeline completo de predicción"""
        # Preparar datos enriquecidos
        match_data = self.test_match_data.copy()
        match_data['date'] = datetime.now().isoformat()
        match_data['weather'] = {'condition': 'clear', 'temperature': 20}
        
        # Ejecutar predicción completa
        prediction = self.predictor.predict_with_formations(match_data)
        
        # Verificar estructura completa
        required_fields = [
            'home_corners', 'away_corners', 'total_corners',
            'tactical_analysis', 'confidence_score',
            'prediction_details'
        ]
        for field in required_fields:
            self.assertIn(field, prediction)
            
        # Verificar análisis táctico
        tactical = prediction['tactical_analysis']
        self.assertIn('style_comparison', tactical)
        self.assertIn('key_advantages', tactical)
        self.assertIn('suggested_approach', tactical)
        
        # Verificar validez de predicciones numéricas
        self.assertTrue(0 <= prediction['confidence_score'] <= 1)
        self.assertTrue(isinstance(prediction['home_corners'], (int, float)))
        self.assertTrue(isinstance(prediction['away_corners'], (int, float)))

if __name__ == '__main__':
    unittest.main()
