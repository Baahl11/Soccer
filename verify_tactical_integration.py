#!/usr/bin/env python3
"""
Script para verificar la correcta integración del análisis táctico en la API de predicciones.
"""

import json
import requests
import logging
from typing import Dict, Any

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
API_BASE_URL = "http://localhost:5000"  # Ajustar según la configuración de tu servidor

def verify_tactical_integration():
    """
    Verifica que el análisis táctico se integre correctamente en los resultados de la API.
    """
    endpoint = f"{API_BASE_URL}/api/upcoming_predictions"
    params = {
        "league_id": 39,  # Premier League
        "season": 2024,
        "limit": 3,
        "include_additional_data": "true",
        "visualizations": "true"
    }
    
    try:
        logger.info("Enviando solicitud a la API...")
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code} al acceder a la API: {response.text}")
            return False
        
        data = response.json()
        
        if "match_predictions" not in data:
            logger.error("No se encontró 'match_predictions' en la respuesta")
            return False
        
        predictions = data["match_predictions"]
        logger.info(f"Se encontraron {len(predictions)} predicciones")
        
        # Verificar cada predicción
        total_checked = 0
        tactical_count = 0
        tactical_fields = 0
        
        for i, pred in enumerate(predictions):
            total_checked += 1
            logger.info(f"Verificando predicción {i+1}: {pred.get('home_team', 'Desconocido')} vs {pred.get('away_team', 'Desconocido')}")
            
            # Verificar si existe el análisis táctico
            if "tactical_analysis" in pred:
                tactical_count += 1
                tactical = pred["tactical_analysis"]
                
                # Verificar campos esperados en el análisis táctico
                fields_to_check = [
                    "style_comparison", "key_advantages", "suggested_approach", 
                    "tactical_style", "matchup_analysis", "key_battles", 
                    "tactical_indices", "tactical_traits"
                ]
                
                existing_fields = [field for field in fields_to_check if field in tactical]
                tactical_fields += len(existing_fields)
                
                logger.info(f"Análisis táctico encontrado con {len(existing_fields)}/{len(fields_to_check)} campos esperados")
                logger.info(f"Campos presentes: {', '.join(existing_fields)}")
                
                # Verificar si hay campos faltantes
                missing_fields = [field for field in fields_to_check if field not in tactical]
                if missing_fields:
                    logger.warning(f"Campos faltantes: {', '.join(missing_fields)}")
            else:
                logger.warning("No se encontró análisis táctico en esta predicción")
            
            # Verificar campos añadidos
            if "prediction_summary" in pred:
                logger.info(f"Resumen de predicción: {pred['prediction_summary']}")
            
            if "key_factors" in pred:
                logger.info(f"Factores clave encontrados: {len(pred.get('key_factors', []))}")
            
            # Verificar datos ELO
            if "elo_ratings" in pred:
                logger.info(f"Datos ELO encontrados: {pred['elo_ratings']}")
                if "elo_expected_goal_diff" in pred:
                    logger.info(f"Diferencia esperada de goles ELO: {pred['elo_expected_goal_diff']}")
            
            logger.info("-" * 40)
        
        # Resultados
        success_rate = (tactical_count / total_checked) * 100 if total_checked > 0 else 0
        logger.info(f"Resumen de verificación:")
        logger.info(f"Total predicciones revisadas: {total_checked}")
        logger.info(f"Predicciones con análisis táctico: {tactical_count} ({success_rate:.1f}%)")
        logger.info(f"Promedio de campos tácticos: {tactical_fields/tactical_count:.1f} por predicción" if tactical_count > 0 else "N/A")
        
        # Guardar una copia de los resultados para análisis
        with open("tactical_integration_verification.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Resultados guardados en tactical_integration_verification.json")
        
        return success_rate >= 75  # Éxito si al menos el 75% de las predicciones tienen análisis táctico
        
    except Exception as e:
        logger.exception(f"Error al verificar la integración táctica: {e}")
        return False

if __name__ == "__main__":
    successful = verify_tactical_integration()
    if successful:
        print("\n✅ VERIFICACIÓN EXITOSA: El análisis táctico está correctamente integrado en la API.")
    else:
        print("\n❌ VERIFICACIÓN FALLIDA: Existen problemas con la integración del análisis táctico.")
