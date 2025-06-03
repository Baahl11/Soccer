"""
Pruebas de integración para el sistema de predicción táctica de corners.
"""

import unittest
import logging
import json
from typing import Dict, Any
from tactical_corner_predictor import TacticalCornerPredictor
from corner_data_collector import FormationDataCollector
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTacticalIntegration(unittest.TestCase):
    """Suite de pruebas de integración para el predictor táctico"""
    
    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.predictor = TacticalCornerPredictor()
        self.formation_collector = FormationDataCollector()
        self.load_test_data()
    
    def load_test_data(self):
        """Carga y prepara datos reales de partidos para pruebas"""
        try:
            # Cargar datos del archivo más reciente
            file_pattern = 'corner_data_39_2023_*.json'
            import glob
            json_files = glob.glob(file_pattern)
            
            if not json_files:
                logger.error("No test data files found")
                self.test_matches = []
                return
                
            latest_file = max(json_files)
            with open(latest_file, 'r') as f:
                data = json.load(f)
                
                # Preparar datos de prueba
                self.test_matches = []
                for match in data[:5]:  # Usar los primeros 5 partidos
                    # Asegurar campos requeridos
                    match_data = {
                        'home_team_id': match.get('home_team_id', 50),
                        'away_team_id': match.get('away_team_id', 33),
                        'league_id': match.get('league_id', 39),
                        'home_formation': match.get('home_formation', '4-3-3'),
                        'away_formation': match.get('away_formation', '4-4-2'),
                        'home_team_name': match.get('home_team_name', 'Home Team'),
                        'away_team_name': match.get('away_team_name', 'Away Team'),
                        'total_corners': match.get('total_corners', 10)
                    }
                    self.test_matches.append(match_data)
                    
                logger.info(f"Loaded {len(self.test_matches)} test matches from {latest_file}")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
            self.test_matches = []
    
    def test_real_match_predictions(self):
        """Prueba predicciones con datos reales de partidos"""
        if not self.test_matches:
            self.skipTest("No test data available")
        
        results = []
        for match in self.test_matches:
            prediction = self.predictor.predict_with_formations(match)
            
            # Verificar estructura básica de la predicción
            self.assertIn('total', prediction)
            self.assertIn('adjusted_total', prediction)
            self.assertIn('tactical_analysis', prediction)
            
            # Guardar resultados para análisis
            results.append({
                'match': f"{match['home_team_name']} vs {match['away_team_name']}",
                'predicted': prediction['adjusted_total'],
                'actual': match.get('total_corners', 0),
                'formation_impact': prediction['formation_impact']
            })
            
            # Verificar que las predicciones están en rangos razonables
            self.assertGreater(prediction['adjusted_total'], 0)
            self.assertLess(prediction['adjusted_total'], 25)
        
        # Guardar resultados para análisis posterior
        with open('integration_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Completed predictions for {len(results)} matches")
    
    def test_formation_impact_consistency(self):
        """Verifica la consistencia del impacto de formaciones"""
        test_formations = [
            ('4-3-3', '5-3-2'),  # Ofensivo vs Defensivo
            ('4-2-3-1', '4-4-2'),  # Moderado vs Balanceado
            ('3-5-2', '5-4-1')   # Ofensivo vs Ultra defensivo
        ]
        
        impacts = {}
        for home_f, away_f in test_formations:
            advantage = self.formation_collector.calculate_matchup_advantage(home_f, away_f)
            impacts[f"{home_f} vs {away_f}"] = advantage
            
            # Verificar consistencia lógica
            if '3-5-2' in home_f or '4-3-3' in home_f:
                self.assertGreater(advantage, 0, 
                    f"Offensive formation {home_f} should have positive advantage")
            
            if '5-4-1' in away_f or '5-3-2' in away_f:
                self.assertGreater(advantage, -0.5,
                    f"Ultra defensive formation impact should be limited")
        
        logger.info(f"Formation impacts calculated: {json.dumps(impacts, indent=2)}")
    
    def test_end_to_end_workflow(self):
        """Prueba el flujo completo desde datos hasta predicción"""
        if not self.test_matches:
            self.skipTest("No test data available")
            
        match = self.test_matches[0]
        
        # 1. Enriquecer datos con formaciones
        enriched = self.formation_collector.enrich_match_data(match)
        self.assertIn('formation_advantage', enriched)
        
        # 2. Realizar predicción táctica
        prediction = self.predictor.predict_with_formations(
            enriched,
            context_factors={'importance': 7}
        )
        self.assertIn('tactical_analysis', prediction)
        
        # 3. Verificar consistencia de ajustes
        base_total = prediction['total']
        adjusted_total = prediction['adjusted_total']
        formation_impact = prediction['formation_impact']
        
        # El ajuste debe ser proporcional al impacto de formación
        expected_adjustment = base_total * (1 + formation_impact * 0.1)
        self.assertAlmostEqual(adjusted_total, expected_adjustment, places=1)
        
        logger.info(
            f"End-to-end test completed for {match['home_team_name']} vs "
            f"{match['away_team_name']}\n"
            f"Base prediction: {base_total}\n"
            f"Adjusted prediction: {adjusted_total}\n"
            f"Formation impact: {formation_impact}"
        )

if __name__ == '__main__':
    unittest.main()
