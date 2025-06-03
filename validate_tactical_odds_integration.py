"""
Script de Validación de Integración Táctica y Odds

Este script valida que la integración del analizador táctico mejorado y
el gestor de odds avanzado estén funcionando correctamente.

Uso:
    python validate_tactical_odds_integration.py

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='tactical_odds_validation.log',
    filemode='w'
)

logger = logging.getLogger('validation')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Importar módulos necesarios para la validación
try:
    from improved_tactical_odds_integration import process_predictions_batch
    from tactical_analyzer_enhanced import TacticalAnalyzerEnhanced
    from odds_manager import OddsManager
except ImportError as e:
    logger.error(f"Error importando módulos: {e}")
    logger.error("Verifica que los archivos improved_tactical_odds_integration.py, "
                 "tactical_analyzer_enhanced.py y odds_manager.py existen.")
    sys.exit(1)

# Rutas
BASE_DIR = Path(__file__).parent
SAMPLE_DATA_PATH = BASE_DIR / "data" / "sample_predictions.json"
OUTPUT_PATH = BASE_DIR / "data" / "validated_predictions.json"
TEAM_DB_PATH = BASE_DIR / "data" / "tactical_teams.json"


def validate_tactical_analyzer():
    """Valida que el analizador táctico mejorado esté funcionando correctamente"""
    logger.info("Validando TacticalAnalyzerEnhanced...")
    
    analyzer = TacticalAnalyzerEnhanced()
    
    # Validar que la base de datos de equipos se ha cargado
    if not analyzer.team_db:
        logger.error("La base de datos de equipos no se ha cargado correctamente")
        return False
    
    logger.info(f"Base de datos táctica cargada con {len(analyzer.team_db)} equipos")
    
    # Probar análisis con equipos conocidos (Real Madrid vs Barcelona)
    analysis = analyzer.get_tactical_analysis(541, 529, "Real Madrid", "Barcelona")
    
    if not analysis:
        logger.error("No se pudo obtener análisis para Real Madrid vs Barcelona")
        return False
    
    # Verificar que el análisis no es genérico
    home_style = analysis.get('tactical_style', {}).get('home', {})
    away_style = analysis.get('tactical_style', {}).get('away', {})
    
    if home_style.get('possession') == 'medio' and away_style.get('possession') == 'medio':
        logger.error("El análisis para Real Madrid vs Barcelona parece ser genérico")
        return False
    
    logger.info("Análisis táctico para Real Madrid vs Barcelona obtenido correctamente:")
    logger.info(f"- Estilo Real Madrid: {home_style}")
    logger.info(f"- Estilo Barcelona: {away_style}")
    
    # Probar con equipos desconocidos (IDs 9999, 8888)
    unknown_analysis = analyzer.get_tactical_analysis(9999, 8888, "Equipo Desconocido", "Otro Equipo")
    
    if not unknown_analysis:
        logger.error("No se pudo obtener análisis para equipos desconocidos")
        return False
    
    logger.info("Análisis táctico para equipos desconocidos obtenido (con fallback)")
    
    return True


def validate_odds_manager():
    """Valida que el gestor de odds esté funcionando correctamente"""
    logger.info("Validando OddsManager...")
    
    odds_manager = OddsManager()
    
    # Probar obtención de odds para un partido (fixture_id 12345)
    odds = odds_manager.get_odds_for_fixture(
        fixture_id=12345,
        home_team_id=541,
        away_team_id=529,
        use_cache=True
    )
    
    if not odds:
        logger.error("No se pudieron obtener odds para el partido de prueba")
        return False
    
    # Verificar simulación y caché
    if not "simulated" in odds:
        logger.error("El campo 'simulated' no está presente en las odds")
        return False
    
    logger.info(f"Odds para partido de prueba: simulated={odds.get('simulated')}")
    
    # Probar caché
    cached_odds = odds_manager.get_odds_for_fixture(
        fixture_id=12345,
        home_team_id=541,
        away_team_id=529,
        use_cache=True
    )
    
    if not cached_odds:
        logger.error("No se pudieron obtener odds desde la caché")
        return False
    
    logger.info("Odds obtenidas de la caché correctamente")
    
    return True


def validate_integration():
    """Valida la integración completa procesando predicciones de muestra"""
    logger.info("Validando integración completa...")
    
    # Cargar datos de muestra
    if not SAMPLE_DATA_PATH.exists():
        logger.error(f"No se encuentra archivo de muestra en {SAMPLE_DATA_PATH}")
        return False
    
    try:
        with open(SAMPLE_DATA_PATH, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        logger.info(f"Datos de muestra cargados desde {SAMPLE_DATA_PATH}")
        
        # Procesar predicciones
        predictions = sample_data.get('predictions', [])
        
        if not predictions:
            logger.error("No hay predicciones en los datos de muestra")
            return False
        
        logger.info(f"Procesando {len(predictions)} predicciones...")
        
        processed = process_predictions_batch(predictions)
        
        if not processed:
            logger.error("No se pudieron procesar las predicciones")
            return False
        
        # Guardar resultados procesados
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump({'predictions': processed}, f, indent=2)
        
        logger.info(f"Resultados guardados en: {OUTPUT_PATH}")
        
        # Análisis de calidad
        tactical_generic = 0
        odds_simulated = 0
        total = len(processed)
        
        for pred in processed:
            tactical = pred.get('tactical_analysis', {})
            odds = pred.get('odds_analysis', {})
            
            # Verificar si el análisis táctico es genérico
            home_style = tactical.get('tactical_style', {}).get('home', {})
            if home_style.get('possession') == 'medio' and home_style.get('pressing') == 'medio':
                tactical_generic += 1
                
            # Verificar si las odds son simuladas
            if odds.get('simulated', True):
                odds_simulated += 1
        
        # Cálculo de porcentajes
        tactical_generic_pct = (tactical_generic / total) * 100 if total > 0 else 0
        odds_simulated_pct = (odds_simulated / total) * 100 if total > 0 else 0
        
        # Resultados
        logger.info("\n=== RESULTADOS DE VALIDACIÓN ===")
        logger.info(f"Total de predicciones: {total}")
        logger.info(f"Análisis tácticos genéricos: {tactical_generic} ({tactical_generic_pct:.1f}%)")
        logger.info(f"Odds simuladas: {odds_simulated} ({odds_simulated_pct:.1f}%)")
        
        # Verificar mejora
        if tactical_generic_pct < 100 and odds_simulated_pct < 100:
            logger.info("VALIDACIÓN EXITOSA: Se detectó mejora en los datos")
            return True
        else:
            logger.warning("VALIDACIÓN INCOMPLETA: No se detectó mejora significativa")
            return False
        
    except Exception as e:
        logger.error(f"Error durante la validación: {e}")
        return False


def run_validation():
    """Ejecuta todas las validaciones"""
    logger.info("="*60)
    logger.info("INICIANDO VALIDACIÓN DE INTEGRACIÓN TÁCTICA Y ODDS")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    tactical_ok = validate_tactical_analyzer()
    odds_ok = validate_odds_manager()
    integration_ok = validate_integration()
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE VALIDACIÓN")
    logger.info("="*60)
    logger.info(f"Analizador Táctico: {'✓ OK' if tactical_ok else '❌ FALLÓ'}")
    logger.info(f"Gestor de Odds: {'✓ OK' if odds_ok else '❌ FALLÓ'}")
    logger.info(f"Integración Completa: {'✓ OK' if integration_ok else '❌ FALLÓ'}")
    logger.info("="*60)
    
    if tactical_ok and odds_ok and integration_ok:
        logger.info("✅ VALIDACIÓN COMPLETADA CON ÉXITO")
        return True
    else:
        logger.warning("⚠️ VALIDACIÓN COMPLETADA CON PROBLEMAS")
        return False


if __name__ == "__main__":
    try:
        success = run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error en validación: {str(e)}")
        sys.exit(1)
