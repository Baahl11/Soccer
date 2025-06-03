#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test API Integration

Este script realiza pruebas de integración con la API de fútbol
para verificar que las solicitudes y el procesamiento de datos
funcionan correctamente.
"""

import unittest
import os
import sys
import json
import logging
import numpy as np
from unittest.mock import patch, MagicMock

# Configurar logging para pruebas
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Importar módulos para pruebas
from fnn_model import FeedforwardNeuralNetwork
from config import API_KEY

class MockResponse:
    """Clase mock para simular respuestas HTTP."""
    
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code
        
    def json(self):
        return self.json_data

class TestFootballAPIIntegration(unittest.TestCase):
    """Pruebas para la integración con la API de fútbol."""
    
    @classmethod
    def setUpClass(cls):
        """Configurar recursos compartidos para todas las pruebas."""
        # Configurar API key de prueba si no está disponible
        cls.api_key = API_KEY if API_KEY != "your-api-key-here" else os.environ.get("FOOTBALL_API_KEY", "test_api_key")
        
        # Crear instancia de modelo para pruebas
        cls.model = FeedforwardNeuralNetwork(input_dim=14, api_key=cls.api_key)
        
        # Preparar datos de prueba
        cls.home_team_id = 33       # Barcelona
        cls.away_team_id = 541      # Real Madrid
        cls.league_id = 140         # La Liga
        cls.season = "2024"         # Temporada
        cls.features = np.array([[0.65, 0.72, 12, 4, 3, 2, 1.8, 0.9, 2.1, 1.1, 0.6, 0.4, 5, 3]])
        
        # Cargar datos mock para pruebas
        cls.mock_home_stats = {
            "response": {
                "form": "WDWLW",
                "fixtures": {"played": {"total": 20, "home": 10}, "wins": {"total": 12, "home": 8}},
                "goals": {"for": {"total": {"total": 38}}, "against": {"total": {"total": 18}}},
                "clean_sheet": {"total": 8},
                "failed_to_score": {"total": 2}
            }
        }
        
        cls.mock_away_stats = {
            "response": {
                "form": "WWDLD",
                "fixtures": {"played": {"total": 20, "away": 10}, "wins": {"total": 14, "away": 6}},
                "goals": {"for": {"total": {"total": 42}}, "against": {"total": {"total": 16}}},
                "clean_sheet": {"total": 9},
                "failed_to_score": {"total": 1}
            }
        }
        
        cls.mock_h2h = {
            "response": [
                {
                    "teams": {"home": {"id": 33}, "away": {"id": 541}},
                    "goals": {"home": 3, "away": 2}
                },
                {
                    "teams": {"home": {"id": 541}, "away": {"id": 33}},
                    "goals": {"home": 1, "away": 1}
                },
                {
                    "teams": {"home": {"id": 33}, "away": {"id": 541}},
                    "goals": {"home": 2, "away": 0}
                }
            ]
        }
        
        # Preparar respuesta base
        cls.mock_base_prediction = {
            "home_win_probability": 0.45,
            "draw_probability": 0.25,
            "away_win_probability": 0.30,
            "home_goals_expected": 2.1,
            "away_goals_expected": 1.8,
            "most_likely_score": "2-1",
            "exact_score_probabilities": {"2-1": 0.12}
        }
    
    def test_api_key_configuration(self):
        """Prueba que la API key se configura correctamente."""
        self.assertIsNotNone(self.model.api_key, "API key no configurada")
    
    @patch('requests.get')
    def test_enrich_with_api_data(self, mock_get):
        """Prueba que el método enrich_with_api_data funciona correctamente."""
        # Configurar mock para solicitudes HTTP
        def side_effect(url, headers=None, params=None):
            if 'statistics' in url and str(self.home_team_id) in str(params):
                return MockResponse(self.mock_home_stats)
            elif 'statistics' in url and str(self.away_team_id) in str(params):
                return MockResponse(self.mock_away_stats)
            elif 'headtohead' in url:
                return MockResponse(self.mock_h2h)
            return MockResponse({}, 404)
        
        mock_get.side_effect = side_effect
        
        # Probar enriquecimiento de datos
        enriched_data = self.model.enrich_with_api_data(
            self.home_team_id, 
            self.away_team_id, 
            self.league_id, 
            self.season
        )
        
        # Verificar estructura de datos
        self.assertIn('home_team', enriched_data, "Falta información del equipo local")
        self.assertIn('away_team', enriched_data, "Falta información del equipo visitante")
        self.assertIn('head_to_head', enriched_data, "Falta información de head-to-head")
        
        # Verificar datos específicos
        self.assertEqual(enriched_data['home_team'].get('form'), "WDWLW", "Form del equipo local incorrecto")
        self.assertEqual(enriched_data['away_team'].get('form'), "WWDLD", "Form del equipo visitante incorrecto")
        self.assertEqual(enriched_data['head_to_head'].get('total_matches'), 3, "Número de partidos H2H incorrecto")
    
    @patch.object(FeedforwardNeuralNetwork, 'predict_match_outcome')
    @patch.object(FeedforwardNeuralNetwork, 'enrich_with_api_data')
    def test_predict_match_with_api(self, mock_enrich, mock_predict):
        """Prueba que el método predict_match_with_api funciona correctamente."""
        # Configurar mocks
        mock_predict.return_value = self.mock_base_prediction
        
        # Simular datos enriquecidos
        mock_api_data = {
            'home_team': {
                'form': 'WDWLW',
                'avg_goals_scored': 1.9,
                'avg_goals_conceded': 0.9,
                'home_advantage': 1.2,
                'win_percentage': 0.6
            },
            'away_team': {
                'form': 'WWDLD',
                'avg_goals_scored': 2.1,
                'avg_goals_conceded': 0.8,
                'away_performance': 0.9,
                'win_percentage': 0.7
            },
            'head_to_head': {
                'total_matches': 3,
                'home_wins': 2,
                'away_wins': 0,
                'draws': 1,
                'avg_goals': 3.0,
                'h2h_dominance': 0.67
            }
        }
        
        mock_enrich.return_value = mock_api_data
        
        # Probar predicción con API
        prediction = self.model.predict_match_with_api(
            self.features,
            self.home_team_id,
            self.away_team_id,
            self.league_id,
            self.season
        )
        
        # Verificar resultados
        self.assertIsNotNone(prediction, "La predicción no debe ser None")
        self.assertIn('api_data', prediction, "Debe incluir datos de la API")
        mock_predict.assert_called_once()
        mock_enrich.assert_called_once()
    
    def test_calculate_form_factor(self):
        """Prueba que el método de cálculo de factor de forma funciona correctamente."""
        # Probar diferentes secuencias de forma
        self.assertGreater(self.model._calculate_form_factor("WWWWW", "LLLLL"), 0.5, "Factor de forma para WWWWW vs LLLLL debe ser > 0.5")
        self.assertLess(self.model._calculate_form_factor("LLLLL", "WWWWW"), 0.5, "Factor de forma para LLLLL vs WWWWW debe ser < 0.5")
        self.assertAlmostEqual(self.model._calculate_form_factor("DDDDD", "DDDDD"), 0.5, delta=0.1, msg="Factor de forma para DDDDD vs DDDDD debe ser ≈ 0.5")
        
        # Verificar que forma más reciente tiene más peso
        form1 = self.model._calculate_form_factor("WLLLL", "DDDDD")
        form2 = self.model._calculate_form_factor("LLLLW", "DDDDD")
        self.assertGreater(form2, form1, "La forma más reciente debe tener más peso")

def run_tests():
    """Ejecutar las pruebas de integración."""
    # Verificar API key antes de ejecutar pruebas
    api_key = API_KEY if API_KEY != "your-api-key-here" else os.environ.get("FOOTBALL_API_KEY")
    
    if not api_key:
        print("\n⚠️ ADVERTENCIA: No se encontró API key para pruebas con la API real.")
        print("Las pruebas se ejecutarán con datos simulados (mocks).")
        print("Para pruebas completas, configura FOOTBALL_API_KEY en el entorno o en config.py\n")
    
    # Ejecutar pruebas
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests()
