"""
Este script crea un parche para el servidor app.py que asegura que tactical_analysis y odds_analysis
aparecen en el nivel principal del JSON.
"""
import sys
import logging
import json
from flask import Flask, jsonify

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_patched_server():
    """
    Crea una versión parcheada del servidor que asegura la inclusión de tactical_analysis y odds_analysis.
    """
    try:
        # Importar el módulo original
        import app
        
        # Guardar una referencia a la aplicación original
        original_app = app.app
        
        # Clonar la ruta /api/upcoming_predictions con nuestro código modificado
        @original_app.route("/api/upcoming_predictions_fixed", methods=["GET"])
        def patched_upcoming_predictions():
            """
            Versión parcheada del endpoint que asegura la inclusión de tactical_analysis y odds_analysis.
            """
            try:
                # Llamar al endpoint original para obtener su respuesta
                response = app.upcoming_predictions()
                
                # Si es un error, devolverlo tal cual
                if not isinstance(response, tuple) and response.status_code != 200:
                    return response
                
                # Obtener el JSON original
                if isinstance(response, tuple):
                    json_data = response[0].json
                else:
                    json_data = response.json
                
                # Verificar si hay predicciones
                if "match_predictions" not in json_data:
                    return jsonify({"error": "No hay predicciones disponibles"}), 400
                
                # Obtener las predicciones originales
                original_predictions = json_data["match_predictions"]
                
                # Lista para las predicciones modificadas
                modified_predictions = []
                
                # Modificar cada predicción para añadir tactical_analysis y odds_analysis
                for pred in original_predictions:
                    # Obtener IDs de equipo
                    home_team_id = pred.get("home_team_id")
                    away_team_id = pred.get("away_team_id")
                    fixture_id = pred.get("fixture_id")
                    
                    # Añadir análisis táctico si no existe
                    if "tactical_analysis" not in pred:
                        try:
                            from fixed_tactical_integration import get_simplified_tactical_analysis
                            logger.info(f"Añadiendo análisis táctico para equipos {home_team_id} vs {away_team_id}")
                            tactical_analysis = get_simplified_tactical_analysis(home_team_id, away_team_id)
                            pred["tactical_analysis"] = tactical_analysis
                        except Exception as te:
                            logger.error(f"Error al añadir análisis táctico: {te}")
                            # Añadir un análisis táctico básico
                            pred["tactical_analysis"] = {
                                "style_comparison": "Datos no disponibles",
                                "key_advantages": ["No hay datos suficientes"],
                                "suggested_approach": "Análisis no disponible",
                                "tactical_style": {"home": {}, "away": {}}
                            }
                    
                    # Añadir análisis de odds si no existe
                    if "odds_analysis" not in pred:
                        try:
                            from odds_analyzer import OddsAnalyzer
                            logger.info(f"Añadiendo análisis de odds para fixture {fixture_id}")
                            odds_analyzer = OddsAnalyzer()
                            value_opps = odds_analyzer.get_value_opportunities(fixture_id, pred)
                            
                            if value_opps:
                                pred["odds_analysis"] = value_opps
                            else:
                                # Añadir un análisis de odds básico
                                pred["odds_analysis"] = {
                                    "value_opportunities": [],
                                    "market_sentiment": {
                                        "description": "Datos no disponibles",
                                        "implied_probabilities": {"home_win": 0, "draw": 0, "away_win": 0}
                                    }
                                }
                        except Exception as oe:
                            logger.error(f"Error al añadir análisis de odds: {oe}")
                            pred["odds_analysis"] = {
                                "error": f"Error al obtener odds: {str(oe)}",
                                "value_opportunities": []
                            }
                    
                    # Verificar si odds_analysis está duplicado en additional_data
                    if "additional_data" in pred and "odds_analysis" in pred.get("additional_data", {}):
                        logger.info(f"Eliminando odds_analysis duplicado de additional_data para fixture {fixture_id}")
                        del pred["additional_data"]["odds_analysis"]
                    
                    # Añadir la predicción modificada a la lista
                    modified_predictions.append(pred)
                
                # Crear respuesta modificada
                modified_response = {
                    "match_predictions": modified_predictions
                }
                
                logger.info(f"✅ Se han procesado {len(modified_predictions)} predicciones con tactical_analysis y odds_analysis")
                
                return jsonify(modified_response)
                
            except Exception as e:
                logger.error(f"❌ Error en el endpoint parcheado: {e}")
                return jsonify({"error": f"Error interno: {str(e)}"}), 500
        
        # Añadir una ruta de información sobre el parche
        @original_app.route("/patch_info", methods=["GET"])
        def patch_info():
            return jsonify({
                "status": "active",
                "description": "Parche para asegurar que tactical_analysis y odds_analysis aparecen en el nivel principal del JSON",
                "endpoints_patched": [
                    {
                        "original": "/api/upcoming_predictions",
                        "patched": "/api/upcoming_predictions_fixed"
                    }
                ],
                "instructions": "Usa el endpoint parcheado '/api/upcoming_predictions_fixed' en lugar del original"
            })
            
        logger.info("✅ Parche aplicado con éxito")
        logger.info("✅ Nuevos endpoints disponibles:")
        logger.info("   - /api/upcoming_predictions_fixed (versión parcheada)")
        logger.info("   - /patch_info (información sobre el parche)")
        
        # Devolver la aplicación original (ahora con nuestras rutas añadidas)
        return original_app
    
    except Exception as e:
        logger.error(f"❌ Error al crear el servidor parcheado: {e}")
        return None

if __name__ == "__main__":
    logger.info("Iniciando aplicación de parche al servidor Flask...")
    
    patched_app = create_patched_server()
    
    if patched_app:
        logger.info("✅ Servidor parcheado listo para ejecutarse")
        logger.info("Para probar:")
        logger.info("   1. Ejecuta: python -m patch_server")
        logger.info("   2. Accede a: http://localhost:5000/api/upcoming_predictions_fixed?league_id=71&season=2024&include_additional_data=true")
        logger.info("   3. Compara con: http://localhost:5000/api/upcoming_predictions?league_id=71&season=2024&include_additional_data=true")
    else:
        logger.error("❌ No se pudo parchear el servidor")
        sys.exit(1)
    
    # Si se ejecuta directamente, iniciar el servidor parcheado
    try:
        logger.info("Iniciando servidor en http://127.0.0.1:5000...")
        patched_app.run(host='127.0.0.1', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Error al iniciar el servidor parcheado: {e}")
        sys.exit(1)
