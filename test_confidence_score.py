# test_confidence_score.py
"""
Script simple para probar la funcionalidad de cálculo de confianza dinámica.
"""

import logging
import sys
from confidence import calculate_confidence_score

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_confidence_calculation():
    """
    Prueba básica del cálculo de confianza con diferentes escenarios.
    """
    # Caso 1: Predicción con buena información
    prediction_good = {
        "home_team_id": 50,
        "away_team_id": 42,
        "league_id": 39,  # Liga con buena cobertura
        "model_predictions": [2.1, 2.2, 2.0, 2.15],  # Poca variabilidad
        "method": "ensemble"
    }
    
    # Caso 2: Predicción con información limitada
    prediction_limited = {
        "home_team_id": 500,  # Equipo con menos datos
        "away_team_id": 600,  # Equipo con menos datos
        "league_id": 100,     # Liga con menor cobertura
        "model_predictions": [1.8, 2.5, 1.0, 3.0],  # Alta variabilidad
        "method": "basic"
    }
    
    # Caso 3: Predicción sin league_id
    prediction_no_league = {
        "home_team_id": 50,
        "away_team_id": 42,
        "model_predictions": [2.1, 2.0],
        "method": "basic"
    }
    
    try:
        # Probar cada caso
        confidence_good = calculate_confidence_score(prediction_good)
        logger.info(f"Confianza para predicción con buena información: {confidence_good:.2f}")
        
        confidence_limited = calculate_confidence_score(prediction_limited)
        logger.info(f"Confianza para predicción con información limitada: {confidence_limited:.2f}")
        
        confidence_no_league = calculate_confidence_score(prediction_no_league)
        logger.info(f"Confianza para predicción sin league_id: {confidence_no_league:.2f}")
        
        # Verificar que la confianza está en el rango esperado y sigue la lógica apropiada
        assert 0.4 <= confidence_good <= 0.9, "La confianza debe estar en el rango 0.4-0.9"
        assert 0.4 <= confidence_limited <= 0.9, "La confianza debe estar en el rango 0.4-0.9"
        assert confidence_good > confidence_limited, "La predicción con buena información debería tener mayor confianza"
        
        logger.info("Todas las pruebas pasaron correctamente.")
        return True
    except Exception as e:
        logger.error(f"Error en las pruebas: {e}")
        return False

if __name__ == "__main__":
    test_confidence_calculation()
