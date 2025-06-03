"""
Este script corrige directamente app.py para forzar la inclusión de tactical_analysis y odds_analysis.
Debe ejecutarse después de modificar las importaciones, pero antes de ejecutar el servidor.
"""
import re
import logging
import sys

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_app_py():
    """
    Modifica app.py para asegurar que tactical_analysis y odds_analysis se incluyen
    correctamente y en el nivel correcto del JSON.
    """
    app_file_path = "app.py"
    
    try:
        # Leer el archivo app.py
        with open(app_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Verificar que la importación correcta está presente
        if "from tactical_integration import" in content:
            logger.info("Actualizando importación para usar fixed_tactical_integration...")
            content = content.replace(
                "from tactical_integration import get_simplified_tactical_analysis, enrich_prediction_with_tactical_analysis",
                "from fixed_tactical_integration import get_simplified_tactical_analysis, enrich_prediction_with_tactical_analysis"
            )
            logger.info("✅ Importación actualizada")
        elif "from fixed_tactical_integration import" in content:
            logger.info("✅ La importación correcta ya está presente")
        else:
            logger.warning("⚠️ No se encontró ninguna importación de tactical_integration. Añadiendo...")
            content = content.replace(
                "from odds_analyzer import OddsAnalyzer",
                "from odds_analyzer import OddsAnalyzer\nfrom fixed_tactical_integration import get_simplified_tactical_analysis, enrich_prediction_with_tactical_analysis"
            )
            logger.info("✅ Importación añadida")
            
        # 2. Asegurarnos de que el código para añadir tactical_analysis existe y está desbloqueado
        tactical_pattern = r'# Añadir análisis táctico\s+try:.*?pred\["tactical_analysis"\] = tactical_analysis.*?except Exception as te:'
        odds_pattern = r'# Add odds analysis.*?try:.*?pred\["odds_analysis"\] = value_opps.*?except Exception as e:'
        
        if not re.search(tactical_pattern, content, re.DOTALL):
            logger.warning("⚠️ No se encontró el código para añadir análisis táctico o está bloqueado")
            
            # Buscar lugar adecuado para insertar
            additional_data_pattern = r'pred\["additional_data"\] = additional_data'
            match = re.search(additional_data_pattern, content)
            
            if match:
                insert_pos = match.start()
                
                # Código a insertar
                tactical_code = """
                                # Añadir análisis táctico
                                try:
                                    # Obtener datos del análisis táctico
                                    tactical_analysis = get_simplified_tactical_analysis(
                                        home_team_id, 
                                        away_team_id
                                    )
                                    
                                    logger.info(f"Añadiendo análisis táctico para: {home_team_id} vs {away_team_id}")
                                    
                                    # Si tenemos datos históricos suficientes para un análisis más detallado
                                    home_matches = additional_data.get("home_team_form", {}).get("matches", [])
                                    away_matches = additional_data.get("away_team_form", {}).get("matches", [])
                                    
                                    # Añadir análisis táctico al nivel principal de la predicción
                                    if home_matches and away_matches and len(home_matches) > 0 and len(away_matches) > 0:
                                        pred = enrich_prediction_with_tactical_analysis(pred, home_matches, away_matches)
                                    else:
                                        pred["tactical_analysis"] = tactical_analysis
                                        
                                    logger.info(f"Análisis táctico añadido con éxito: {list(tactical_analysis.keys())}")
                                except Exception as te:
                                    logger.warning(f"Error al obtener análisis táctico para equipos {home_team_id} vs {away_team_id}: {te}")
                                    # Añadir un análisis táctico básico
                                    pred["tactical_analysis"] = {
                                        "style_comparison": "Datos no disponibles",
                                        "key_advantages": ["No hay datos suficientes"],
                                        "suggested_approach": "Análisis no disponible",
                                        "tactical_style": {"home": {}, "away": {}}
                                    }
"""
                
                # Insertar el código
                content = content[:insert_pos] + tactical_code + content[insert_pos:]
                logger.info("✅ Código para añadir análisis táctico insertado")
                
        if not re.search(odds_pattern, content, re.DOTALL):
            logger.warning("⚠️ No se encontró el código para añadir odds o está bloqueado")
            
            # Buscar lugar adecuado para insertar
            additional_data_pattern = r'pred\["additional_data"\] = additional_data'
            match = re.search(additional_data_pattern, content)
            
            if match:
                insert_pos = match.start()
                
                # Código a insertar
                odds_code = """
                                # Añadir análisis de odds
                                try:
                                    odds_analyzer = OddsAnalyzer()
                                    value_opps = odds_analyzer.get_value_opportunities(fixture_id, base_prediction)
                                    
                                    logger.info(f"Añadiendo análisis de odds para fixture: {fixture_id}")
                                    
                                    if value_opps:
                                        # Mover odds al nivel principal del JSON
                                        pred["odds_analysis"] = value_opps
                                        
                                        # Eliminar posible duplicado en additional_data
                                        if "odds_analysis" in additional_data:
                                            del additional_data["odds_analysis"]
                                            
                                        logger.info("✅ Análisis de odds añadido con éxito")
                                    else:
                                        logger.warning("⚠️ No se pudo obtener análisis de odds")
                                        pred["odds_analysis"] = {
                                            "value_opportunities": [],
                                            "market_sentiment": {
                                                "description": "Datos no disponibles",
                                                "implied_probabilities": {"home_win": 0, "draw": 0, "away_win": 0}
                                            }
                                        }
                                except Exception as e:
                                    logger.warning(f"Error getting odds analysis for fixture {fixture_id}: {e}")
                                    pred["odds_analysis"] = {
                                        "error": f"Error al obtener odds: {str(e)}",
                                        "value_opportunities": []
                                    }
"""
                
                # Insertar el código
                content = content[:insert_pos] + odds_code + content[insert_pos:]
                logger.info("✅ Código para añadir análisis de odds insertado")
                
        # 3. Guardar los cambios
        with open(app_file_path + ".fixed", 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"✅ Los cambios se han guardado en {app_file_path}.fixed")
        logger.info(f"Para aplicar los cambios:")
        logger.info(f"   1. Verificar que {app_file_path}.fixed es correcto")
        logger.info(f"   2. Renombrar {app_file_path}.fixed a {app_file_path}")
        logger.info(f"   3. Reiniciar el servidor")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error al modificar app.py: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando corrección de app.py para integración de tactical_analysis y odds_analysis")
    
    success = fix_app_py()
    
    if success:
        logger.info("✅ Proceso de corrección completado con éxito")
        sys.exit(0)
    else:
        logger.error("❌ El proceso de corrección falló")
        sys.exit(1)
