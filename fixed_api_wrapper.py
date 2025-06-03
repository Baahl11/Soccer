"""
Módulo que implementa un proxy completo que intercepta y corrige la estructura JSON.
Este wrapper garantiza que tactical_analysis y odds_analysis estén siempre en el nivel principal,
y actúa como un proxy completo para el servidor original (basado en json_interceptor.py).
"""

from flask import Flask, request, jsonify
import json
import logging
import requests
from fixed_tactical_integration import create_default_tactical_analysis, get_simplified_tactical_analysis
from odds_analyzer import OddsAnalyzer

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
odds_analyzer = OddsAnalyzer()

def enrich_prediction(prediction):
    """
    Asegura que la predicción tenga la estructura correcta con tactical_analysis y odds_analysis
    en el nivel principal, usando funcionalidades del json_interceptor.py.
    """
    fixture_id = prediction.get("fixture_id")
    home_team_id = prediction.get("home_team_id")
    away_team_id = prediction.get("away_team_id")
    
    # 1. Procesar tactical_analysis
    # Si tactical_analysis está en additional_data, moverlo al nivel principal
    if "additional_data" in prediction and "tactical_analysis" in prediction["additional_data"]:
        if "tactical_analysis" not in prediction:  # Evitar sobrescribir si ya existe
            prediction["tactical_analysis"] = prediction["additional_data"]["tactical_analysis"]
        del prediction["additional_data"]["tactical_analysis"]
    
    # Si tactical_analysis no existe, crearlo con datos enriquecidos
    if "tactical_analysis" not in prediction:
        try:
            logger.info(f"Añadiendo análisis táctico para equipos {home_team_id} vs {away_team_id}")
            
            # Intentar obtener un análisis táctico detallado primero
            if home_team_id and away_team_id:
                tactical_analysis = get_simplified_tactical_analysis(home_team_id, away_team_id)
                prediction["tactical_analysis"] = tactical_analysis
                logger.info(f"✅ Análisis táctico añadido con campos: {list(tactical_analysis.keys()) if tactical_analysis else []}")
                
                # Verificar si el análisis tiene todos los campos necesarios
                required_fields = ["tactical_style", "key_battles", "strengths", "weaknesses"]
                missing_fields = [field for field in required_fields if field not in tactical_analysis]
                
                if missing_fields:
                    logger.warning(f"Faltan campos en el análisis táctico: {missing_fields}. Creando análisis con datos básicos.")
                    # Usar create_default_tactical_analysis para generar datos más completos
                    prediction["tactical_analysis"] = create_default_tactical_analysis(home_team_id, away_team_id)
            else:
                # Sin IDs de equipo, crear estructura mínima
                prediction["tactical_analysis"] = {
                    "style_comparison": "Insufficient data for tactical analysis",
                    "key_advantages": [],
                    "suggested_approach": "No data available for tactical suggestions"
                }
        except Exception as e:
            logger.error(f"Error al añadir análisis táctico: {e}")
            prediction["tactical_analysis"] = {
                "style_comparison": "Error generating tactical analysis",
                "key_advantages": [],
                "error": str(e)
            }
    
    # 2. Procesar odds_analysis
    # Si odds_analysis está en additional_data, moverlo al nivel principal
    if "additional_data" in prediction and "odds_analysis" in prediction["additional_data"]:
        if "odds_analysis" not in prediction:  # Evitar sobrescribir si ya existe
            prediction["odds_analysis"] = prediction["additional_data"]["odds_analysis"]
        del prediction["additional_data"]["odds_analysis"]
    
    # Si odds_analysis no existe, crear uno más detallado usando odds_analyzer
    if "odds_analysis" not in prediction and fixture_id and fixture_id > 0:
        try:
            logger.info(f"Generando análisis de odds para fixture {fixture_id}")
            # Intentar obtener valor real de odds si está disponible
            value_opps = odds_analyzer.get_value_opportunities(fixture_id, prediction)
            
            if value_opps and isinstance(value_opps, dict) and len(value_opps) > 0:
                prediction["odds_analysis"] = value_opps
                logger.info(f"✅ Análisis de odds obtenido correctamente")
            else:
                # Crear análisis simulado con estructura completa
                prediction["odds_analysis"] = {
                    "market_analysis": {
                        "efficiency": 0.92,
                        "margin": 1.08
                    },
                    "value_opportunities": [
                        {
                            "market": "1X2",
                            "selection": "1" if prediction.get("predicted_home_goals", 0) > prediction.get("predicted_away_goals", 0) else "2",
                            "fair_odds": 1.9,
                            "market_odds": 2.1,
                            "value": 0.1,
                            "recommendation": "Considerar"
                        }
                    ],
                    "simulated": True  # Marcar como datos simulados
                }
        except Exception as e:
            logger.error(f"Error al añadir análisis de odds: {e}")
            prediction["odds_analysis"] = {
                "market_analysis": {
                    "efficiency": 0.0,
                    "margin": 1.0
                }
            }
    
    return prediction

@app.route("/api/fixed_predictions", methods=["GET"])
def fixed_predictions():
    """
    Endpoint wrapper que llama al endpoint original y asegura que la estructura sea correcta
    """
    try:
        # Obtener todos los parámetros de la petición original
        params = request.args.to_dict()
        
        # Llamar al endpoint original
        response = requests.get("http://localhost:5000/api/upcoming_predictions", params=params)
        
        if response.status_code != 200:
            return jsonify({"error": "Error calling original endpoint", "details": response.text}), response.status_code
        
        # Obtener los datos JSON originales
        original_data = response.json()
        
        if "match_predictions" not in original_data:
            return jsonify({"error": "Invalid response from original endpoint"}), 500
        
        # Enriquecer cada predicción con datos tácticos y de odds
        logger.info(f"Enriqueciendo {len(original_data['match_predictions'])} predicciones")
        enriched_predictions = [enrich_prediction(pred) for pred in original_data["match_predictions"]]
        
        # Devolver el resultado con estructura corregida
        pretty = request.args.get("pretty", 0, type=int)
        if pretty == 1:
            return app.response_class(
                json.dumps({"match_predictions": enriched_predictions}, indent=2, ensure_ascii=False),
                mimetype='application/json'
            )
        else:
            return jsonify({"match_predictions": enriched_predictions})
            
    except Exception as e:
        logger.exception("Error in fixed_predictions endpoint")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting wrapper API on http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
