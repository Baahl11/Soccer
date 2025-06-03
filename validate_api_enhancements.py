#!/usr/bin/env python3
"""
Script para comprobar y validar que las modificaciones en la API funcionen correctamente.
Este script verifica tanto la integración del análisis táctico como los campos adicionales.
"""

import json
import requests
import sys
import logging
from typing import Dict, Any, List

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuración
API_BASE_URL = "http://localhost:5000"  # Ajustar según configuración
OUTPUT_FILE = "fix_verification_updated.json"

def validate_enhanced_api():
    """
    Valida que las mejoras a la API funcionen correctamente.
    """
    endpoint = f"{API_BASE_URL}/api/upcoming_predictions"
    params = {
        "league_id": 39,  # Premier League
        "season": 2024,
        "limit": 3,
        "include_additional_data": "true"
    }
    
    try:
        # Hacer solicitud a la API
        logger.info(f"Consultando {endpoint}...")
        response = requests.get(endpoint, params=params)
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        
        # Verificar estructura básica
        if "match_predictions" not in data:
            logger.error("Estructura de respuesta inválida: no se encontró 'match_predictions'")
            return False
        
        predictions = data["match_predictions"]
        if not predictions or len(predictions) == 0:
            logger.error("No se encontraron predicciones en la respuesta")
            return False
        
        logger.info(f"Se encontraron {len(predictions)} predicciones para analizar")
        
        # Variables para seguimiento
        all_issues = []
        successful_validations = 0
        
        # Verificar cada predicción
        for i, pred in enumerate(predictions):
            logger.info(f"Verificando predicción {i+1}: {pred.get('home_team', 'Desconocido')} vs {pred.get('away_team', 'Desconocido')}")
            
            issues = []
            required_fields = [
                "fixture_id", "home_team", "away_team", "date", 
                "predicted_home_goals", "predicted_away_goals", "total_goals",
                "elo_ratings", "elo_probabilities", "elo_expected_goal_diff"
            ]
            
            # Verificar campos requeridos
            missing_fields = [field for field in required_fields if field not in pred]
            if missing_fields:
                issues.append(f"Campos requeridos faltantes: {', '.join(missing_fields)}")
            
            # Verificar campos nuevos
            new_fields = ["prediction_summary", "key_factors", "last_updated"]
            missing_new_fields = [field for field in new_fields if field not in pred]
            if missing_new_fields:
                issues.append(f"Campos nuevos faltantes: {', '.join(missing_new_fields)}")
            
            # Verificar análisis táctico
            if "tactical_analysis" not in pred:
                issues.append("No se encontró análisis táctico")
            else:
                tactical = pred["tactical_analysis"]
                # Verificar estructura táctica
                expected_tactical_fields = [
                    "tactical_style", "key_battles", "matchup_analysis", 
                    "tactical_indices", "tactical_traits"
                ]
                missing_tactical = [field for field in expected_tactical_fields if field not in tactical]
                if missing_tactical:
                    issues.append(f"Campos tácticos faltantes: {', '.join(missing_tactical)}")
            
            # Verificar ELO
            if "elo_ratings" in pred:
                elo_ratings = pred["elo_ratings"]
                if not isinstance(elo_ratings, dict):
                    issues.append("elo_ratings no es un diccionario")
                else:
                    expected_elo_fields = ["home_elo", "away_elo", "elo_diff"]
                    missing_elo = [field for field in expected_elo_fields if field not in elo_ratings]
                    if missing_elo:
                        issues.append(f"Campos ELO faltantes: {', '.join(missing_elo)}")
            
            # Si hay null donde no debería haberlos
            if "elo_expected_goal_diff" in pred and pred["elo_expected_goal_diff"] is None:
                issues.append("elo_expected_goal_diff es null")
            
            # Verificar formatos y tipos
            if "prediction_summary" in pred and not isinstance(pred["prediction_summary"], str):
                issues.append("prediction_summary no es un string")
            
            if "key_factors" in pred and not isinstance(pred["key_factors"], list):
                issues.append("key_factors no es una lista")
            
            # Registrar resultados
            if issues:
                all_issues.append({"prediction": i+1, "issues": issues})
                for issue in issues:
                    logger.warning(f"  ❌ {issue}")
            else:
                successful_validations += 1
                logger.info("  ✅ Validación exitosa: Todos los campos presentes y correctos")
        
        # Guardar resultados para análisis
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump({
                "validation_date": data.get("date", "2025-05-18"),
                "predictions_analyzed": len(predictions),
                "successful_validations": successful_validations,
                "success_rate": f"{(successful_validations / len(predictions) * 100):.1f}%" if predictions else "0%",
                "issues": all_issues,
                "sample_data": predictions[0] if predictions else None
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Resultados guardados en {OUTPUT_FILE}")
        
        # Determinar éxito general
        success_rate = successful_validations / len(predictions) if predictions else 0
        if success_rate >= 0.7:  # 70% de éxito como mínimo para considerar que funciona
            return True
        else:
            logger.error(f"Tasa de éxito insuficiente: {success_rate:.1%}")
            return False
    
    except Exception as e:
        logger.exception(f"Error durante la validación: {e}")
        return False

if __name__ == "__main__":
    print("\n🔍 INICIANDO VALIDACIÓN DE MEJORAS EN API...\n")
    
    # Ejecutar validación
    if validate_enhanced_api():
        print("\n✅ VALIDACIÓN EXITOSA: Las mejoras a la API funcionan correctamente.")
        sys.exit(0)
    else:
        print("\n❌ VALIDACIÓN FALLIDA: Se encontraron problemas en las mejoras de la API.")
        print(f"   Consulta el archivo {OUTPUT_FILE} para más detalles.")
        sys.exit(1)
