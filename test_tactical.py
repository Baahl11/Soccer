"""
Script para probar si el análisis táctico se está incluyendo correctamente
"""

import json
from predictions import make_global_prediction
from tactical_integration import get_simplified_tactical_analysis
import traceback

def test_tactical_analysis():
    """Prueba que el análisis táctico se incluya en el resultado"""
    try:
        print("==== Inicio de prueba de análisis táctico ====")
        
        # ID de fixture inválido para obtener una predicción por defecto
        fixture_id = 12345
        
        # Usar IDs de equipos reales
        home_team_id = 541  # Real Madrid
        away_team_id = 529  # Barcelona
        
        print(f"Obteniendo análisis táctico para: {home_team_id} vs {away_team_id}")
        
        # Obtener análisis táctico directamente
        tactical_analysis = get_simplified_tactical_analysis(home_team_id, away_team_id)
        
        # Verificar si tenemos análisis táctico
        if tactical_analysis:
            print("✓ Se obtuvo análisis táctico")
            print(f"Claves del análisis táctico: {list(tactical_analysis.keys())}")
        else:
            print("✗ No se pudo obtener análisis táctico")
        
        print("==== Fin de prueba de análisis táctico ====")
        
        return tactical_analysis
    
    except Exception as e:
        print(f"Error durante la prueba: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_tactical_analysis()
