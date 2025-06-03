# test_tactical_analysis.py
import unittest
import sys
import os
import numpy as np
from datetime import datetime
import logging
from unittest.mock import MagicMock, patch

# Añadir el directorio padre al path para importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Crear mocks para dependencias
class MockFeatureEngineering:
    def analyze_possession_patterns(self, team_matches):
        return {
            'avg_possession': 0.55,
            'final_third_possession': 0.48,
            'progression_ratio': 0.7,
            'possession_recovery_rate': 0.4,
            'buildup_success_rate': 0.6,
            'pressure_resistance_rate': 0.7
        }
    
    def analyze_pressing_effectiveness(self, team_matches):
        return {
            'pressing_intensity': 0.65,
            'pressing_success_rate': 0.55,
            'press_resistance': 0.6,
            'defensive_aggression': 0.7,
            'pressing_coordination': 0.6,
            'press_efficiency': 0.65,
            'avg_press_distance': 38.5
        }
    
    def analyze_attacking_patterns(self, team_matches):
        return {
            'direct_play_ratio': 0.4,
            'wing_play_ratio': 0.55,
            'width_variance': 0.45,
            'transition_speed': 0.7
        }
    
    def analyze_defensive_patterns(self, team_matches):
        return {
            'defensive_line_height': 45.5,
            'defensive_action_rate': 0.6,
            'high_press_tendency': 0.55,
            'interception_rate': 0.4,
            'clearance_tendency': 0.3
        }

# Patch del módulo feature_engineering
sys.modules['feature_engineering'] = MagicMock()
sys.modules['feature_engineering'].AdvancedFeatureEngineering = MockFeatureEngineering

# Ahora importamos el módulo que usa la dependencia mockeada
from tactical_analysis import FormationAnalyzer

# Mock para data.get_lineup_data
def mock_get_lineup_data(fixture_id):
    return {
        'response': [
            {
                'team': {'id': 50, 'name': 'Team A'},
                'formation': '4-3-3',
                'startXI': [{'player': {'pos': 'G'}} for _ in range(11)]
            },
            {
                'team': {'id': 49, 'name': 'Team B'},
                'formation': '4-4-2',
                'startXI': [{'player': {'pos': 'G'}} for _ in range(11)]
            }
        ]
    }

# Patch de data.get_lineup_data
sys.modules['data'] = MagicMock()
sys.modules['data'].get_lineup_data = mock_get_lineup_data

# Ahora importamos la función que usa data.get_lineup_data
from tactical_analysis import analyze_tactical_formation_matchup

class TestTacticalAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up para todas las pruebas"""
        cls.formation_analyzer = FormationAnalyzer()
        
        # Desactivar logging durante las pruebas
        logging.disable(logging.CRITICAL)

    @classmethod
    def tearDownClass(cls):
        """Cleanup después de todas las pruebas"""
        # Reactivar logging
        logging.disable(logging.NOTSET)
    
    def test_formation_identification(self):
        """Probar identificación de formaciones"""
        # Caso: Formación explícita en datos
        lineup_with_formation = {
            'formation': '4-3-3'
        }
        formation = self.formation_analyzer.identify_formation(lineup_with_formation)
        self.assertEqual(formation, '4-3-3')
        
        # Caso: Sin datos de alineación
        formation = self.formation_analyzer.identify_formation(None)
        self.assertEqual(formation, '4-4-2')  # Formación por defecto
        
        # Caso: Con datos de jugadores
        lineup_with_players = {
            'startXI': [
                {'player': {'pos': 'G'}},
                {'player': {'pos': 'D'}},
                {'player': {'pos': 'D'}},
                {'player': {'pos': 'D'}},
                {'player': {'pos': 'D'}},
                {'player': {'pos': 'M'}},
                {'player': {'pos': 'M'}},
                {'player': {'pos': 'M'}},
                {'player': {'pos': 'F'}},
                {'player': {'pos': 'F'}},
                {'player': {'pos': 'F'}}
            ]
        }
        formation = self.formation_analyzer.identify_formation(lineup_with_players)
        self.assertEqual(formation, '4-3-3')
    
    def test_formation_matchup_analysis(self):
        """Probar análisis de enfrentamiento de formaciones"""
        # Probar distintas combinaciones de formaciones
        matchup = self.formation_analyzer.analyze_formation_matchup('4-3-3', '4-4-2')
        self.assertIsInstance(matchup, dict)
        self.assertIn('tactical_advantage', matchup)
        self.assertIn('advantage_score', matchup)
        self.assertIn('zones_analysis', matchup)
        
        # Probar con formaciones que dan ventaja táctica
        matchup = self.formation_analyzer.analyze_formation_matchup('3-5-2', '4-4-2')
        self.assertEqual(matchup['tactical_advantage'], 'home')
        self.assertGreater(matchup['advantage_score'], 0)
        
        # Probar formaciones no reconocidas
        matchup = self.formation_analyzer.analyze_formation_matchup('invalid', 'also-invalid')
        self.assertEqual(matchup['home_formation'], '4-4-2')  # Debe usar la formación por defecto
        self.assertEqual(matchup['away_formation'], '4-4-2')
    
    def test_tactical_formation_integration(self):
        """Probar integración con análisis táctico completo"""
        result = analyze_tactical_formation_matchup(50, 49, 1234)
        
        # Verificar que devuelve los campos esperados
        self.assertIn('formation_matchup', result)
        self.assertIn('tactical_insights', result)
        self.assertIn('advantage', result)
        
        # Verificar formaciones detectadas
        self.assertEqual(result['formation_matchup']['home_formation'], '4-3-3')
        self.assertEqual(result['formation_matchup']['away_formation'], '4-4-2')

    def test_zone_analysis(self):
        """Probar análisis de zonas del campo"""
        # Probar zonas para formaciones con ventajas conocidas
        zones = self.formation_analyzer._analyze_formation_zones('3-5-2', '4-3-3')
        
        self.assertIn('central_midfield', zones)
        self.assertIn('wings', zones)
        
        # 3-5-2 debe tener ventaja en el centro contra 4-3-3
        self.assertEqual(zones['central_midfield']['advantage'], 'home')
        
        # 4-3-3 debe tener ventaja en las bandas contra 3-5-2
        self.assertEqual(zones['wings']['advantage'], 'away')

if __name__ == '__main__':
    unittest.main(verbosity=2)
