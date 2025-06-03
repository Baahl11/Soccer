"""
Verificación Final de Integración de Odds

Este script ejecuta una verificación completa de la integración de odds
y genera un informe de resultados.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import requests

# Importar scripts optimizados
sys.path.append('.')
import optimize_odds_integration
import update_api_credentials
import config

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='odds_final_verification.log',
    filemode='w'
)

logger = logging.getLogger('final_verification')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

def verify_config():
    """Verificar que la configuración esté correctamente actualizada"""
    logger.info("Verificando configuración...")
    
    # Verificar que existen las configuraciones necesarias
    required_attrs = ["API_KEY", "API_BASE_URL", "ODDS_ENDPOINTS", "ODDS_BOOKMAKERS_PRIORITY"]
    
    all_present = True
    for attr in required_attrs:
        if hasattr(config, attr):
            logger.info(f"✅ Configuración encontrada: {attr}")
        else:
            logger.error(f"❌ Configuración faltante: {attr}")
            all_present = False
    
    # Verificar que no hay variables redundantes
    deprecated_attrs = ["ODDS_API_KEY", "ODDS_API_URL", "ODDS_API_PROVIDER"]
    
    for attr in deprecated_attrs:
        if hasattr(config, attr):
            logger.warning(f"⚠️ Configuración redundante encontrada: {attr}")
            all_present = False
    
    return all_present

def verify_api_key():
    """Verificar que la API key es válida y tiene acceso a odds"""
    logger.info("Verificando API key...")
    
    # Verificar si la API key existe
    if not config.API_KEY or config.API_KEY == "your-api-key-here":
        logger.error("❌ API key no configurada")
        return False
    
    # Verificar si puede acceder a los endpoints de odds
    try:
        odds_endpoint = f"{config.API_BASE_URL}/odds"
        headers = {
            "x-rapidapi-key": config.API_KEY,
            "x-rapidapi-host": config.API_BASE_URL.replace("https://", "")
        }
        params = {
            "league": "39",  # Premier League
            "season": "2023"
        }
        
        logger.info(f"Probando endpoint {odds_endpoint}...")
        response = requests.get(odds_endpoint, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if "response" in data and data["response"]:
                logger.info(f"✅ API key válida y con acceso a odds")
                return True
            else:
                logger.warning(f"⚠️ API key válida pero no retorna datos de odds")
                return False
        else:
            logger.error(f"❌ Error API: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error de conexión: {str(e)}")
        return False

def test_odds_integration():
    """Probar la integración de odds completa"""
    logger.info("Probando integración completa...")
    
    try:
        # Probar el método de test_odds_integration
        results = optimize_odds_integration.test_odds_integration()
        
        # Contar odds reales vs simuladas
        total = len(results)
        simulated = sum(1 for r in results.values() if r.get("simulated", True))
        real = total - simulated
        
        logger.info(f"Total de partidos probados: {total}")
        logger.info(f"Partidos con odds reales: {real}")
        logger.info(f"Partidos con odds simuladas: {simulated}")
        
        # Verificar la fuente de los datos
        sources = {}
        for fixture_id, result in results.items():
            source = result.get("source", "Desconocido")
            sources[source] = sources.get(source, 0) + 1
            
        for source, count in sources.items():
            logger.info(f"Fuente '{source}': {count} partidos")
        
        # Escribir resultados detallados a archivo JSON
        with open('odds_integration_final_results.json', 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_fixtures": total,
                "real_odds_count": real,
                "simulated_odds_count": simulated,
                "real_odds_percentage": (real / total * 100) if total > 0 else 0,
                "sources_distribution": sources,
                "detailed_results": {str(k): v for k, v in results.items()}
            }, f, indent=2)
            
        return real > 0  # Éxito si al menos hay un partido con odds reales
    except Exception as e:
        logger.error(f"❌ Error en prueba de integración: {str(e)}")
        return False

def main():
    """Ejecutar verificación completa"""
    logger.info("="*60)
    logger.info("INICIANDO VERIFICACIÓN FINAL DE INTEGRACIÓN DE ODDS")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Resultados de las verificaciones
    results = {
        "config_ok": verify_config(),
        "api_key_ok": verify_api_key(),
        "integration_ok": test_odds_integration()
    }
    
    # Verificar resultados
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE VERIFICACIÓN")
    logger.info("="*60)
    
    all_ok = True
    for test, result in results.items():
        status = "✅ CORRECTO" if result else "❌ FALLIDO"
        logger.info(f"{test}: {status}")
        if not result:
            all_ok = False
    
    # Resultado final
    if all_ok:
        logger.info("\n✅✅✅ VERIFICACIÓN COMPLETA EXITOSA")
        logger.info("La integración de odds está funcionando correctamente")
    else:
        logger.info("\n⚠️ VERIFICACIÓN INCOMPLETA")
        logger.info("Hay problemas que requieren atención. Revise el log para detalles.")
    
    return all_ok

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\nVerificación completada exitosamente.")
        else:
            print("\nVerificación completada con advertencias o errores.")
            print("Revise el archivo de log para más detalles.")
    except Exception as e:
        logger.error(f"Error en verificación: {str(e)}")
        print(f"\nError durante la verificación: {str(e)}")
        print("Revise el archivo de log para más detalles.")
