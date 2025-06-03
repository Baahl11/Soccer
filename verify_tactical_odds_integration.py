#!/usr/bin/env python3
"""
Script completo para verificar la integración del análisis táctico y odds en el endpoint.
Este script debe ejecutarse con el servidor en funcionamiento.

Uso:
    python -m verify_tactical_odds_integration

Requiere:
    - Servidor Flask en ejecución en localhost:5000
    - Paquete requests instalado
"""

import requests
import json
import sys
import logging
from pprint import pprint

# Configuración básica de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def verify_endpoint_integration(base_url="http://localhost:5000", league_id=71, season=2024, limit=3):
    """
    Realiza una verificación completa del endpoint /api/upcoming_predictions
    para asegurar que tanto el análisis táctico como las odds están
    correctamente integrados en la estructura JSON.
    
    Args:
        base_url: URL base del servidor
        league_id: ID de la liga para hacer la consulta
        season: Temporada para hacer la consulta
        limit: Número máximo de predicciones a solicitar
        
    Returns:
        bool: True si todas las verificaciones pasan, False en caso contrario
    """
    logger.info(f"Iniciando verificación completa del endpoint con league_id={league_id}, season={season}")
    
    # URL del endpoint con parámetros necesarios
    url = f"{base_url}/api/upcoming_predictions?league_id={league_id}&season={season}&include_additional_data=true&limit={limit}"
    
    try:
        logger.info(f"Realizando petición a {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'match_predictions' not in data:
            logger.error("La respuesta no contiene el campo 'match_predictions'")
            return False
            
        predictions = data['match_predictions']
        
        if not predictions:
            logger.error("No se encontraron predicciones en la respuesta")
            return False
            
        logger.info(f"Se encontraron {len(predictions)} predicciones para analizar")
        
        # Estadísticas para el informe
        stats = {
            "total": len(predictions),
            "with_tactical": 0,
            "with_odds": 0,
            "tactical_fields_count": [],
            "odds_in_additional_data": 0,
            "issues": []
        }
        
        # Analizar cada predicción
        for i, pred in enumerate(predictions):
            fixture_id = pred.get('fixture_id', f"desconocido-{i}")
            home_team = pred.get('home_team', 'Desconocido')
            away_team = pred.get('away_team', 'Desconocido')
            
            logger.info(f"Verificando predicción {i+1}/{len(predictions)}: {home_team} vs {away_team} (ID: {fixture_id})")
            
            # Verificar análisis táctico
            if 'tactical_analysis' in pred:
                stats["with_tactical"] += 1
                
                # Verificar estructura del análisis táctico
                ta = pred['tactical_analysis']
                if isinstance(ta, dict):
                    stats["tactical_fields_count"].append(len(ta.keys()))
                    
                    # Verificar campos básicos esperados
                    expected_fields = ['style_comparison', 'key_advantages', 'suggested_approach', 'tactical_style']
                    missing = [f for f in expected_fields if f not in ta]
                    
                    if missing:
                        issue = f"Predicción {fixture_id}: Faltan campos en tactical_analysis: {', '.join(missing)}"
                        logger.warning(issue)
                        stats["issues"].append(issue)
                else:
                    issue = f"Predicción {fixture_id}: tactical_analysis no es un diccionario"
                    logger.warning(issue)
                    stats["issues"].append(issue)
            else:
                issue = f"Predicción {fixture_id}: No se encontró tactical_analysis en el nivel principal"
                logger.warning(issue)
                stats["issues"].append(issue)
            
            # Verificar análisis de odds
            if 'odds_analysis' in pred:
                stats["with_odds"] += 1
                
                # Verificar estructura del análisis de odds
                oa = pred['odds_analysis']
                if not isinstance(oa, dict):
                    issue = f"Predicción {fixture_id}: odds_analysis no es un diccionario"
                    logger.warning(issue)
                    stats["issues"].append(issue)
            else:
                issue = f"Predicción {fixture_id}: No se encontró odds_analysis en el nivel principal"
                logger.warning(issue)
                stats["issues"].append(issue)
                
            # Verificar que odds_analysis NO esté dentro de additional_data
            if 'additional_data' in pred and 'odds_analysis' in pred.get('additional_data', {}):
                stats["odds_in_additional_data"] += 1
                issue = f"Predicción {fixture_id}: odds_analysis encontrado duplicado en additional_data"
                logger.warning(issue)
                stats["issues"].append(issue)
                
        # Generar informe de resultados
        logger.info("Resumen de verificación:")
        logger.info(f"Total predicciones revisadas: {stats['total']}")
        logger.info(f"Predicciones con análisis táctico: {stats['with_tactical']} ({stats['with_tactical']/stats['total']*100:.1f}%)")
        logger.info(f"Predicciones con análisis de odds: {stats['with_odds']} ({stats['with_odds']/stats['total']*100:.1f}%)")
        
        if stats["tactical_fields_count"]:
            avg_fields = sum(stats["tactical_fields_count"]) / len(stats["tactical_fields_count"])
            logger.info(f"Promedio de campos tácticos: {avg_fields:.1f} por predicción")
            
        if stats["odds_in_additional_data"] > 0:
            logger.warning(f"Se encontraron {stats['odds_in_additional_data']} predicciones con odds duplicados en additional_data")
        
        # Verificar si hay problemas
        if stats["with_tactical"] == stats["total"] and stats["with_odds"] == stats["total"] and stats["odds_in_additional_data"] == 0:
            logger.info("✅ VERIFICACIÓN EXITOSA: Todos los elementos están correctamente integrados.")
            return True
        else:
            logger.error("❌ VERIFICACIÓN FALLIDA: Se encontraron problemas en la integración.")
            if stats["issues"]:
                logger.error("Problemas encontrados:")
                for issue in stats["issues"]:
                    logger.error(f"  - {issue}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión: {e}")
        logger.error("¿Está el servidor en ejecución en la URL especificada?")
        return False
    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
        return False

def save_example_json(base_url="http://localhost:5000", league_id=71, season=2024):
    """
    Guarda un ejemplo de la estructura JSON para documentación.
    """
    try:
        url = f"{base_url}/api/upcoming_predictions?league_id={league_id}&season={season}&include_additional_data=true&limit=1"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'match_predictions' in data and data['match_predictions']:
            with open('example_prediction.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info("✅ Ejemplo de JSON guardado en 'example_prediction.json'")
            return True
        else:
            logger.error("No se encontraron predicciones para guardar como ejemplo")
            return False
    except Exception as e:
        logger.error(f"Error guardando ejemplo de JSON: {e}")
        return False

def main():
    """Función principal del script de verificación."""
    print("=" * 80)
    print("VERIFICADOR DE INTEGRACIÓN DE ANÁLISIS TÁCTICO Y ODDS")
    print("=" * 80)
    print("\nEste script verifica que el análisis táctico y las odds estén correctamente")
    print("integrados en la estructura JSON del endpoint /api/upcoming_predictions.")
    print("\nAsegúrate de que el servidor Flask esté en ejecución antes de continuar.")
    
    # Probar la conexión al servidor
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        print(f"\n✅ Servidor detectado en http://localhost:5000 (estado: {response.status_code})")
    except:
        print("\n❌ No se pudo conectar al servidor en http://localhost:5000")
        print("   Asegúrate de que el servidor esté en ejecución e intenta de nuevo.")
        return 1
        
    # Realizar verificación completa
    print("\nRealizando verificación completa del endpoint...")
    success = verify_endpoint_integration()
    
    if success:
        print("\n✅ VERIFICACIÓN EXITOSA: Tanto el análisis táctico como las odds están")
        print("   correctamente integrados en la estructura JSON.")
        
        # Guardar ejemplo para documentación
        save_example_json()
        
        return 0
    else:
        print("\n❌ VERIFICACIÓN FALLIDA: Se encontraron problemas en la integración.")
        print("   Revisa los mensajes de error arriba para más detalles.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
