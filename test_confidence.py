"""
Prueba de implementación de cálculo dinámico de confianza.
"""

import logging
import sys
from typing import Dict, Any

from confidence import (
    calculate_confidence_score, get_h2h_matches_count, 
    get_team_matches_count, get_league_coverage_score,
    get_team_stability_score, get_historical_accuracy
)

# Configurar logging para la prueba
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def test_confidence_calculation():
    """Test the confidence calculation with various scenarios."""
    
    # Caso 1: Predicción con buenos datos
    good_data_prediction = {
        "home_team_id": 1,  # Manchester City
        "away_team_id": 2,  # Liverpool
        "league_id": 1,     # Premier League
        "method": "neural_network",
        "model_predictions": [1.8, 1.75, 1.85]  # Baja varianza = alta confianza
    }
    
    # Caso 2: Predicción con datos limitados
    limited_data_prediction = {
        "home_team_id": 100,  # Equipo con menos datos
        "away_team_id": 200,  # Equipo con menos datos
        "league_id": 15,      # Liga con menor cobertura
        "method": "statistical"
    }
    
    # Caso 3: Predicción con datos parciales
    partial_data_prediction = {
        "home_team_id": 1,    # Equipo conocido
        "away_team_id": 150,  # Equipo menos conocido
        "league_id": 1,       # Liga conocida
        "method": "partial_data",
        "model_predictions": [1.5, 2.0, 1.0]  # Alta varianza = baja confianza
    }
    
    # Calcular y mostrar resultados
    logger.info("Test de cálculo de confianza dinámica")
    logger.info("-" * 50)
    
    confidence1 = calculate_confidence_score(good_data_prediction)
    logger.info(f"Confianza para predicción con buenos datos: {confidence1}")
    
    confidence2 = calculate_confidence_score(limited_data_prediction)
    logger.info(f"Confianza para predicción con datos limitados: {confidence2}")
    
    confidence3 = calculate_confidence_score(partial_data_prediction)
    logger.info(f"Confianza para predicción con datos parciales: {confidence3}")
    
    logger.info("-" * 50)
    logger.info(f"Diferencia en confianza: {confidence1 - confidence2:.2f}")
    
    # Verificar componentes individuales
    logger.info("\nComponentes individuales para el caso con buenos datos:")
    h2h = get_h2h_matches_count(1, 2)
    team1 = get_team_matches_count(1)
    team2 = get_team_matches_count(2)
    league = get_league_coverage_score(1)
    stability1 = get_team_stability_score(1)
    stability2 = get_team_stability_score(2)
    accuracy = get_historical_accuracy(1, 1, 2)
    
    logger.info(f"Partidos H2H: {h2h}")
    logger.info(f"Partidos equipo 1: {team1}")
    logger.info(f"Partidos equipo 2: {team2}")
    logger.info(f"Cobertura de liga: {league}")
    logger.info(f"Estabilidad equipo 1: {stability1}")
    logger.info(f"Estabilidad equipo 2: {stability2}")
    logger.info(f"Precisión histórica: {accuracy}")

if __name__ == "__main__":
    test_confidence_calculation()
