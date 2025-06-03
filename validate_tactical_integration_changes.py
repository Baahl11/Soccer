#!/usr/bin/env python3
"""
Script para validar los cambios realizados al módulo tactical_integration
y asegurarse de que se integra correctamente en los datos JSON.
"""

import sys
import json
import logging
import traceback
from typing import Dict, Any, List

# Configurar logging para que se muestre en la consola
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Archivos de salida
VERIFICATION_REPORT = "tactical_integration_check.json"

def validate_tactical_integration_changes():
    """
    Valida que los cambios realizados en tactical_integration funcionen correctamente
    sin necesidad de ejecutar el servidor web.
    """
    try:
        # Imports necesarios
        from tactical_integration import enrich_prediction_with_tactical_analysis, get_simplified_tactical_analysis
        import random
        
        logger.info("Imports completados exitosamente")
        
        # Crear datos de prueba
        home_matches = [
            {"score": {"home": 2, "away": 1}, "formation": "4-3-3", "possession": 55},
            {"score": {"home": 1, "away": 1}, "formation": "4-3-3", "possession": 52},
            {"score": {"home": 3, "away": 0}, "formation": "4-3-3", "possession": 60}
        ]
        
        away_matches = [
            {"score": {"home": 0, "away": 2}, "formation": "4-4-2", "possession": 48},
            {"score": {"home": 1, "away": 2}, "formation": "4-4-2", "possession": 45},
            {"score": {"home": 2, "away": 1}, "formation": "4-5-1", "possession": 40}
        ]
        
        # Crear una predicción básica
        prediction = {
            "fixture_id": 12345,
            "home_team": {"id": 1, "name": "Equipo Local", "formation": "4-3-3"},
            "away_team": {"id": 2, "name": "Equipo Visitante", "formation": "4-4-2"},
            "date": "2025-05-20T18:00:00Z",
            "predicted_home_goals": 1.8,
            "predicted_away_goals": 1.2,
            "prob_over_2_5": 0.55,
            "prob_btts": 0.62
        }
        
        # Probar la función enrich_prediction_with_tactical_analysis
        logger.info("Probando enrich_prediction_with_tactical_analysis...")
        try:
            enriched_prediction = enrich_prediction_with_tactical_analysis(prediction, home_matches, away_matches)
            
            if "tactical_analysis" not in enriched_prediction:
                logger.error("No se agregó el campo tactical_analysis")
                return False
            
            # Verificar campos del análisis táctico
            tactical = enriched_prediction["tactical_analysis"]
            expected_fields = [
                "style_comparison", "key_advantages", "suggested_approach", 
                "tactical_style", "matchup_analysis", "key_battles"
            ]
            
            missing_fields = [field for field in expected_fields if field not in tactical]
            if missing_fields:
                logger.warning(f"Campos faltantes: {', '.join(missing_fields)}")
                
            logger.info(f"Análisis táctico contiene {len(tactical)} campos")
            logger.info(f"Campos presentes: {', '.join(tactical.keys())}")
            
            # Verificar que sea un objeto completo, no sólo errores
            if "error" in tactical and len(tactical) <= 2:
                logger.error(f"tactical_analysis contiene sólo errores: {tactical}")
                return False
            
            logger.info("Prueba 1 completada con éxito")
        except Exception as e:
            logger.error(f"Error en prueba 1: {e}")
            return False
        
        # Probar get_simplified_tactical_analysis
        logger.info("Probando get_simplified_tactical_analysis...")
        try:
            # Usamos valores aleatorios para los IDs
            home_team_id = 123
            away_team_id = 456
            
            simplified = get_simplified_tactical_analysis(home_team_id, away_team_id, home_matches, away_matches)
            
            if not simplified or not isinstance(simplified, dict):
                logger.error(f"get_simplified_tactical_analysis no devolvió un diccionario: {simplified}")
                return False
            
            # Verificar campos importantes
            expected_simple_fields = ["tactical_style", "key_battles", "summary"]
            missing_simple = [f for f in expected_simple_fields if f not in simplified]
            if missing_simple:
                logger.warning(f"Campos faltantes en análisis simplificado: {', '.join(missing_simple)}")
            
            logger.info(f"Análisis simplificado contiene {len(simplified)} campos")
            logger.info(f"Campos presentes: {', '.join(simplified.keys())}")
            
            # Verificar estructura del análisis simplificado
            if "tactical_style" in simplified:
                if not isinstance(simplified["tactical_style"], dict):
                    logger.error("tactical_style no es un diccionario")
                    return False
                if "home" not in simplified["tactical_style"] or "away" not in simplified["tactical_style"]:
                    logger.error("tactical_style no contiene home y away")
            
            logger.info("Prueba 2 completada con éxito")
        except Exception as e:
            logger.error(f"Error en prueba 2: {e}")
            return False
        
        # Guardar resultados de la validación
        results = {
            "validation_date": "2025-05-18",
            "tests_passed": 2,
            "status": "success",
            "summary": "Las funciones de tactical_integration funcionan correctamente",
            "enriched_sample": enriched_prediction,
            "simplified_sample": simplified
        }
        
        with open(VERIFICATION_REPORT, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados guardados en {VERIFICATION_REPORT}")
        return True
    except ImportError as e:
        logger.error(f"Error de importación: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        logger.exception(f"Error general durante la validación: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n🔍 VALIDANDO CAMBIOS EN TACTICAL_INTEGRATION...\n")
    
    if validate_tactical_integration_changes():
        print("\n✅ VALIDACIÓN EXITOSA: Los cambios en tactical_integration funcionan correctamente.")
        sys.exit(0)
    else:
        print("\n❌ VALIDACIÓN FALLIDA: Se encontraron problemas en los cambios.")
        sys.exit(1)
