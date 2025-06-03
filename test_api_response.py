#!/usr/bin/env python3
"""
Script para probar la respuesta de la API y verificar la integración del análisis táctico y odds.
"""

import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_api_response(base_url="http://localhost:8080"):
    """Prueba la respuesta de la API y verifica la estructura del JSON."""
    endpoint = f"{base_url}/api/upcoming_predictions"
    params = {
        "league_id": 218,  # Cambiado a otra liga
        "season": 2024,
        "limit": 2,  # Solo 2 partidos para hacer la prueba más rápida
        "include_additional_data": "true"
    }
    
    try:
        logger.info(f"Enviando solicitud a: {endpoint}")
        logger.info(f"Parámetros: {params}")
        
        response = requests.get(endpoint, params=params, timeout=90)
        
        if response.status_code != 200:
            logger.error(f"Error {response.status_code}: {response.text}")
            return False
        
        data = response.json()
        
        if "match_predictions" not in data:
            logger.error("No se encontró 'match_predictions' en la respuesta")
            return False
        
        predictions = data["match_predictions"]
        logger.info(f"Se encontraron {len(predictions)} predicciones")
        
        # Verificar cada predicción
        all_good = True
        for i, prediction in enumerate(predictions):
            logger.info(f"\n--- Verificando predicción {i+1} ---")
            logger.info(f"Partido: {prediction.get('home_team')} vs {prediction.get('away_team')}")
            
            # Verificar análisis táctico
            if 'tactical_analysis' in prediction:
                tactical = prediction['tactical_analysis']
                logger.info(f"✅ tactical_analysis encontrado en nivel principal")
                logger.info(f"   Campos: {list(tactical.keys())}")
                
                # Verificar contenido del análisis táctico
                if isinstance(tactical, dict) and len(tactical) > 1:
                    if "style_comparison" in tactical:
                        logger.info(f"   Comparación de estilos: {tactical['style_comparison'][:100]}...")
                    if "key_advantages" in tactical:
                        logger.info(f"   Ventajas clave: {len(tactical.get('key_advantages', []))} items")
                    if "tactical_style" in tactical:
                        logger.info(f"   Estilos tácticos disponibles: {list(tactical.get('tactical_style', {}).keys())}")
                else:
                    logger.warning(f"   ⚠️ Análisis táctico parece estar vacío o con errores")
            else:
                logger.error(f"❌ No se encontró tactical_analysis en la predicción {i+1}")
                all_good = False
            
            # Verificar análisis de odds
            if 'odds_analysis' in prediction:
                odds = prediction['odds_analysis']
                logger.info(f"✅ odds_analysis encontrado en nivel principal")
                logger.info(f"   Campos: {list(odds.keys())}")
                
                # Verificar si es simulado
                if odds.get("simulated", False):
                    logger.info(f"   ⚠️ Los odds son simulados")
                else:
                    logger.info(f"   ✅ Los odds parecen ser reales")
            else:
                logger.error(f"❌ No se encontró odds_analysis en la predicción {i+1}")
                all_good = False
            
            # Verificar que no estén en additional_data
            additional_data = prediction.get("additional_data", {})
            if "tactical_analysis" in additional_data:
                logger.warning(f"   ⚠️ tactical_analysis también encontrado en additional_data (debería estar solo en nivel principal)")
            if "odds_analysis" in additional_data:
                logger.warning(f"   ⚠️ odds_analysis también encontrado en additional_data (debería estar solo en nivel principal)")
        
        if all_good:
            logger.info(f"\n🎉 ÉXITO: Todas las predicciones contienen tactical_analysis y odds_analysis en el nivel principal")
        else:
            logger.error(f"\n❌ FALLO: Algunas predicciones no contienen la estructura esperada")
        
        # Guardar respuesta completa para inspección
        with open("last_api_response.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Respuesta completa guardada en 'last_api_response.json'")
        
        return all_good
        
    except requests.RequestException as e:
        logger.error(f"Error de conexión: {e}")
        return False
    except json.JSONDecodeError as e:
        logger.error(f"Error al decodificar JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Probando respuesta de la API...")
    success = test_api_response()
    
    if success:
        print("\n✅ Las predicciones contienen los análisis tácticos y de odds correctamente")
    else:
        print("\n❌ Hay problemas con la estructura de las predicciones")
