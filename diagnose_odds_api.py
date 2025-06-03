"""
Diagnóstico de API de Odds - Script de Verificación

Este script realiza un diagnóstico completo de la conexión con la API de odds
y verifica posibles problemas en la integración.

Uso:
    python diagnose_odds_api.py

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import requests
import logging
import json
import time
from pathlib import Path
import os
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='odds_api_diagnostic.log',
    filemode='w'
)

logger = logging.getLogger('odds_diagnostic')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Cargar configuración
try:
    config_path = Path(__file__).parent / 'config.py'
    if not config_path.exists():
        raise FileNotFoundError(f"Archivo de configuración no encontrado en {config_path}")
    
    # Importamos las variables de configuración
    import sys
    from dotenv import load_dotenv
    load_dotenv()
    
    sys.path.append(str(Path(__file__).parent))
    from config import API_FOOTBALL_KEY, API_BASE_URL, ODDS_ENDPOINTS
    API_KEY = API_FOOTBALL_KEY  # Usar la variable correcta del config
except ImportError as e:
    logger.error(f"Error importando configuración: {str(e)}")
    # Variables de respaldo para permitir ejecución
    API_KEY = os.environ.get('API_FOOTBALL_KEY', 'clave_no_encontrada')
    API_BASE_URL = os.environ.get('API_BASE_URL', 'https://v3.football.api-sports.io')
    ODDS_ENDPOINTS = {"pre_match": "/odds"}

# IDs de partidos para pruebas
# Usar la función que busca partidos con odds reales disponibles
try:
    from optimize_odds_integration import test_odds_integration
    logger.info("Buscando partidos con odds reales disponibles...")
    results = test_odds_integration()
    TEST_FIXTURES = list(results.keys())[:3]  # Usar los primeros 3 encontrados
    logger.info(f"Partidos encontrados con odds: {TEST_FIXTURES}")
except Exception as e:
    logger.warning(f"Error buscando partidos con odds: {e}")
    # Fallback a IDs fijos como antes
    TEST_FIXTURES = [1208383, 1208384, 1208385]  # IDs de partidos de prueba

def check_api_status():
    """Verifica estado general de la API de odds"""
    logger.info("Verificando estado de la API de odds")
    try:
        # Endpoint de estado/verificación
        endpoint = f"{API_BASE_URL}/status"
        headers = {
            "x-rapidapi-key": API_KEY,
            "x-rapidapi-host": API_BASE_URL.replace("https://", "")
        }
        
        logger.info(f"Conectando con: {endpoint}")
        response = requests.get(endpoint, headers=headers, timeout=10)
        
        logger.info(f"Código de respuesta: {response.status_code}")
        logger.info(f"Cabeceras: {response.headers}")
        
        # Intentamos parsear la respuesta JSON
        if response.status_code == 200:
            try:
                data = response.json()
                logger.info(f"Respuesta: {json.dumps(data, indent=2)[:500]}...")
                return True, "API operativa"
            except json.JSONDecodeError:
                logger.warning(f"Respuesta no es JSON válido: {response.text[:200]}...")
                return False, "Respuesta no es JSON válido"
        else:
            logger.error(f"Error en API: {response.text[:200]}...")
            return False, f"Error {response.status_code}: {response.text[:100]}..."
    except Exception as e:
        logger.error(f"Excepción conectando con API: {str(e)}")
        return False, f"Error de conexión: {str(e)}"

def check_fixture_odds(fixture_id):
    """Verifica la obtención de odds para un partido específico usando el pipeline de producción"""
    logger.info(f"Verificando odds para partido ID: {fixture_id}")
    try:
        # Usar el mismo sistema que la producción
        from optimize_odds_integration import get_fixture_odds
        
        start_time = time.time()
        normalized_odds = get_fixture_odds(fixture_id, use_cache=False, force_refresh=True)
        elapsed = time.time() - start_time
        
        logger.info(f"Tiempo de respuesta: {elapsed:.2f} segundos")
        
        if normalized_odds:
            simulated = normalized_odds.get("simulated", True)
            source = normalized_odds.get("source", "Unknown")
            logger.info(f"¿Simulado?: {simulated}")
            logger.info(f"Fuente: {source}")
            logger.info(f"Datos obtenidos exitosamente")
            return True, {"success": True, "simulated": simulated, "source": source, "time": elapsed}
        else:
            logger.warning(f"No se obtuvieron datos para partido {fixture_id}")
            return False, {"success": False, "error": "Sin datos de odds", "time": elapsed}
            
    except Exception as e:
        logger.error(f"Excepción obteniendo odds: {str(e)}")
        return False, {"success": False, "error": str(e)}

def check_rate_limits():
    """Verifica si estamos experimentando limitación de tasa (rate limiting)"""
    logger.info("Verificando posible rate limiting")
    results = []
    
    # Hacemos 5 solicitudes rápidas para ver si hay limitación
    for i in range(5):
        logger.info(f"Solicitud rápida #{i+1}")
        success, result = check_fixture_odds(TEST_FIXTURES[0])
        results.append(result)
        time.sleep(0.5)  # Espera mínima entre solicitudes
    
    # Analizamos resultados
    errors = sum(1 for r in results if not r.get("success", False))
    times = [r.get("time", 0) for r in results if "time" in r]
    avg_time = sum(times) / len(times) if times else 0
    
    if errors >= 3:
        logger.warning(f"Posible rate limiting detectado: {errors}/5 errores")
        return False, "Posible limitación de tasa (rate limiting)"
    else:
        logger.info(f"Límite de tasa adecuado. Tiempo medio: {avg_time:.2f}s")
        return True, f"Sin limitación. Tiempo medio: {avg_time:.2f}s"

def verify_simulated_status():
    """Verifica si todas las odds están marcadas como simuladas usando el pipeline de producción"""
    logger.info("Verificando si todas las odds están marcadas como simuladas")
    simulated_count = 0
    real_count = 0
    total = len(TEST_FIXTURES)
    
    for fixture_id in TEST_FIXTURES:
        success, result = check_fixture_odds(fixture_id)
        if success and result.get("success", False):
            if result.get("simulated", True):
                simulated_count += 1
                logger.info(f"Partido {fixture_id}: SIMULADO ({result.get('source', 'Unknown')})")
            else:
                real_count += 1
                logger.info(f"Partido {fixture_id}: REAL ({result.get('source', 'Unknown')})")
        else:
            logger.warning(f"Partido {fixture_id}: ERROR - {result.get('error', 'Unknown')}")
    
    simulation_percentage = (simulated_count / total) * 100 if total > 0 else 100
    
    if simulated_count == total:
        logger.warning(f"ALERTA: El 100% de las odds ({simulated_count}/{total}) están marcadas como simuladas")
        return False, f"Todas las odds ({simulated_count}/{total}) son simuladas"
    elif real_count > 0:
        logger.info(f"✅ {real_count}/{total} predicciones usan odds reales ({(real_count/total)*100:.1f}%)")
        logger.info(f"⚠️  {simulated_count}/{total} predicciones usan odds simuladas ({simulation_percentage:.1f}%)")
        return True, f"{real_count}/{total} predicciones usan odds reales ({(real_count/total)*100:.1f}%)"
    else:
        logger.warning(f"PROBLEMA: No se pudieron obtener datos válidos para ningún partido")
        return False, f"No se obtuvieron datos válidos para ningún partido"

def run_full_diagnostic():
    """Ejecuta diagnóstico completo y genera informe"""
    logger.info("="*60)
    logger.info("INICIANDO DIAGNÓSTICO COMPLETO DE API DE ODDS")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API URL: {API_BASE_URL}")
    logger.info(f"Proveedor configurado: API-Football")
    logger.info(f"API Key: ...{API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    logger.info("="*60)
    
    results = {}
    
    # Verificar estado general
    logger.info("\n1. VERIFICACIÓN DE ESTADO DE API")
    results["api_status"] = check_api_status()
    
    # Verificar obtención de odds específicas
    logger.info("\n2. VERIFICACIÓN DE ODDS POR PARTIDO")
    fixture_results = {}
    for fixture_id in TEST_FIXTURES:
        fixture_results[fixture_id] = check_fixture_odds(fixture_id)
    results["fixture_odds"] = fixture_results
    
    # Verificar limitación de tasa
    logger.info("\n3. VERIFICACIÓN DE RATE LIMITING")
    results["rate_limits"] = check_rate_limits()
    
    # Verificar porcentaje de simulación
    logger.info("\n4. ANÁLISIS DE ODDS SIMULADAS")
    results["simulated_status"] = verify_simulated_status()
    
    # Generar informe
    logger.info("\n"+"="*60)
    logger.info("RESUMEN DE DIAGNÓSTICO")
    logger.info("="*60)
    
    # Estado API
    api_ok, api_msg = results["api_status"]
    logger.info(f"Estado de API: {'✓' if api_ok else '✗'} {api_msg}")
      # Odds por partido
    fixture_ok = all(r[0] for r in fixture_results.values())
    successful_fixtures = sum(1 for r in fixture_results.values() if r[0])
    logger.info(f"Obtención de odds: {'✓' if fixture_ok else '✗'} ({successful_fixtures}/{len(fixture_results)} exitosos)")
    for fixture_id, (ok, result) in fixture_results.items():
        status = "✓" if ok else "✗"
        if ok:
            simulated_status = "SIMULADO" if result.get('simulated', True) else "REAL"
            source = result.get('source', 'Unknown')
            logger.info(f"  - Partido {fixture_id}: {status} {simulated_status} ({source})")
        else:
            logger.info(f"  - Partido {fixture_id}: {status} {result.get('error', 'Error desconocido')}")
    
    # Rate limiting
    rate_ok, rate_msg = results["rate_limits"]
    logger.info(f"Rate Limiting: {'✓' if rate_ok else '✗'} {rate_msg}")
    
    # Simulación
    sim_ok, sim_msg = results["simulated_status"]
    logger.info(f"Odds simuladas: {'✓' if sim_ok else '✗'} {sim_msg}")
    
    # Recomendaciones
    logger.info("\nRECOMENDACIONES:")
    if not api_ok:
        logger.info("- Verificar credenciales de API y configuración de URL")
    if not fixture_ok:
        logger.info("- Revisar formato de solicitud de odds y parámetros")
    if not rate_ok:
        logger.info("- Implementar sistema de reintentos con backoff exponencial")
        logger.info("- Considerar cache local para reducir llamadas API")
    if not sim_ok:
        logger.info("- Verificar proveedor alternativo de odds")
        logger.info("- Revisar integración con proveedor actual")
    
    # Guardar resultados en archivo
    results_file = Path(__file__).parent / "odds_api_diagnostic_results.json"
    with open(results_file, "w") as f:
        # Convertimos los resultados a un formato serializable
        serializable_results = {
            "timestamp": datetime.now().isoformat(),
            "api_status": {
                "success": results["api_status"][0],
                "message": results["api_status"][1]
            },
            "fixture_odds": {
                str(fixture_id): {
                    "success": result[0],
                    "details": result[1]
                } for fixture_id, result in fixture_results.items()
            },
            "rate_limits": {
                "success": results["rate_limits"][0],
                "message": results["rate_limits"][1]
            },
            "simulated_status": {
                "success": results["simulated_status"][0],
                "message": results["simulated_status"][1]
            }
        }
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"\nInforme guardado en: {results_file}")
    return results

if __name__ == "__main__":
    try:
        run_full_diagnostic()
        logger.info("\nDiagnóstico completado. Revise el archivo de log para detalles.")
    except Exception as e:
        logger.error(f"Error en diagnóstico: {str(e)}")
