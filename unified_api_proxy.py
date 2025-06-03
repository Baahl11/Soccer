"""
Unified API Proxy
----------------
Este módulo combina las mejores características de json_interceptor.py y fixed_api_wrapper.py
para crear una solución completa que garantiza la correcta estructura de datos JSON,
asegurando que tactical_analysis y odds_analysis siempre estén en el nivel principal
de cada predicción.

Características:
- Implementa un proxy completo para todos los endpoints del API original
- Proporciona un endpoint dedicado /api/fixed_predictions para compatibilidad
- Enriquece automáticamente las predicciones con análisis táctico y de cuotas detallados
- Asegura la consistencia de la estructura JSON en todas las respuestas
- Incluye manejo robusto de errores y validación de campos

Uso:
1. Iniciar el servidor original: python -m app
2. Iniciar este proxy: python -m unified_api_proxy
3. Acceder a través de http://localhost:8080/api/upcoming_predictions o http://localhost:8080/api/fixed_predictions
"""

from flask import Flask, request, jsonify
import json
import logging
import sys
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
                
        # Verificar campos requeridos de forma más exhaustiva
                required_fields = ["tactical_style", "key_battles", "strengths", "weaknesses", 
                                   "tactical_recommendation", "expected_formations"]
                missing_fields = [field for field in required_fields if field not in tactical_analysis]
                
                if missing_fields:
                    logger.warning(f"Faltan campos en el análisis táctico: {missing_fields}. Complementando datos...")
                    # Complementar con datos predeterminados pero manteniendo lo existente
                    default_analysis = create_default_tactical_analysis(home_team_id, away_team_id)
                    
                    # Añadir solo los campos faltantes para enriquecer el análisis existente
                    for field in missing_fields:
                        if field in default_analysis:
                            tactical_analysis[field] = default_analysis[field]
                    
                    prediction["tactical_analysis"] = tactical_analysis
            else:                # Sin IDs de equipo, crear estructura completa predeterminada
                # Usar IDs genéricos para garantizar estructura consistente
                prediction["tactical_analysis"] = create_default_tactical_analysis(home_team_id or 0, away_team_id or 0)
                prediction["tactical_analysis"]["generated"] = "default"
        except Exception as e:
            logger.error(f"Error al añadir análisis táctico: {e}")            # Proporcionar estructura mínima en caso de error
            prediction["tactical_analysis"] = {
                "style_comparison": f"Error al generar análisis táctico: {str(e)}",
                "key_advantages": [],
                "tactical_style": {
                    "home": {"possession_style": "Unknown", "defensive_line": "Unknown"},
                    "away": {"possession_style": "Unknown", "defensive_line": "Unknown"}
                },
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
                },
                "value_opportunities": [],
                "market_sentiment": {
                    "description": f"Error al obtener datos de mercado: {str(e)}",
                    "implied_probabilities": {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}
                },
                "error": str(e),
                "simulated": True
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

@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    """
    Proxy para reenviar todas las solicitudes al servidor original.
    
    Para solicitudes al endpoint /api/upcoming_predictions, intercepta y enriquece el JSON.
    """
    # Construir URL original
    original_url = f"http://localhost:5000/{path}"
    
    # Reenviar todos los parámetros y headers
    headers = {key: value for (key, value) in request.headers if key != 'Host'}
    params = request.args.to_dict()
    
    logger.info(f"Proxy: {request.method} {original_url}")
    
    try:
        # Hacer la solicitud al servidor original
        if request.method == 'GET':
            resp = requests.get(original_url, headers=headers, params=params)
        elif request.method == 'POST':
            resp = requests.post(original_url, headers=headers, json=request.get_json())
        elif request.method == 'PUT':
            resp = requests.put(original_url, headers=headers, json=request.get_json())
        elif request.method == 'DELETE':
            resp = requests.delete(original_url, headers=headers)
        else:
            return jsonify({"error": "Método no soportado"}), 400
        
        # Si no es el endpoint que nos interesa o hay un error, reenviar la respuesta tal cual
        if path != 'api/upcoming_predictions' or resp.status_code != 200:
            response = app.make_response(resp.content)
            response.status_code = resp.status_code
            response.headers.extend(resp.headers.items())
            return response
        
        # Interceptar y enriquecer el JSON para /api/upcoming_predictions
        try:
            json_data = resp.json()
            
            if "match_predictions" in json_data:
                match_predictions = json_data["match_predictions"]
                
                logger.info(f"Interceptando {len(match_predictions)} predicciones para enriquecer")
                
                # Enriquecer cada predicción
                enriched_predictions = [enrich_prediction(pred) for pred in match_predictions]
                json_data["match_predictions"] = enriched_predictions
                
                logger.info(f"✅ Proceso de enriquecimiento completado para {len(enriched_predictions)} predicciones")
                
                # Devolver el JSON enriquecido
                pretty = request.args.get("pretty", 0, type=int)
                if pretty == 1:
                    return app.response_class(
                        json.dumps(json_data, indent=2, ensure_ascii=False),
                        mimetype='application/json'
                    )
                return jsonify(json_data)
            else:
                logger.warning("No se encontraron match_predictions en la respuesta")
                response = app.make_response(resp.content)
                response.status_code = resp.status_code
                response.headers.extend(resp.headers.items())
                return response
                
        except Exception as e:
            logger.error(f"Error al procesar el JSON: {e}")
            response = app.make_response(resp.content)
            response.status_code = resp.status_code
            response.headers.extend(resp.headers.items())
            return response
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al conectar con el servidor original: {e}")
        return jsonify({"error": f"Error de conexión: {str(e)}"}), 500

@app.route('/proxy_info', methods=['GET'])
def proxy_info():
    """Información sobre el proxy unificado"""
    return jsonify({
        "name": "Unified API Proxy",
        "version": "2.1",
        "status": "active",
        "description": "Proxy unificado que garantiza la correcta estructura JSON en las predicciones",
        "original_server": "http://localhost:5000",
        "endpoints": {
            "proxy": "Todos los endpoints del servidor original",
            "fixed_predictions": "/api/fixed_predictions (endpoint dedicado)"
        },
        "features": [
            "Enriquecimiento automático de tactical_analysis",
            "Enriquecimiento automático de odds_analysis",
            "Corrección de estructura JSON",
            "Manejo robusto de errores",
            "Validación de campos requeridos",
            "Datos tácticos enriquecidos",
            "Análisis de cuotas completo"
        ]
    })

def check_original_server():
    """Comprueba si el servidor original está en ejecución"""
    try:
        response = requests.get("http://localhost:5000/health", timeout=2)
        if response.status_code == 200:
            logger.info("✅ Servidor original en ejecución correctamente")
            return True
    except:
        pass
    
    logger.warning("⚠️ No se puede conectar al servidor original. Algunas funciones pueden no estar disponibles.")
    return False

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("Iniciando Unified API Proxy en http://localhost:8080...")
    logger.info("Este proxy combina la funcionalidad de json_interceptor.py y fixed_api_wrapper.py")
    logger.info("-"*80)
    
    # Comprobar si el servidor original está en ejecución
    check_original_server()
    
    logger.info("")
    logger.info("Para usar el proxy unificado:")
    logger.info("  1. Asegúrate de que el servidor original está en ejecución (python -m app)")
    logger.info("  2. Puedes usar cualquiera de estos endpoints:")
    logger.info("     - http://localhost:8080/api/upcoming_predictions (proxy completo)")
    logger.info("     - http://localhost:8080/api/fixed_predictions (endpoint dedicado)")
    logger.info("  3. Ambos garantizan que tactical_analysis y odds_analysis estén siempre presentes")
    logger.info("     en el nivel principal de cada predicción")
    logger.info("="*80)
    
    try:
        app.run(host='127.0.0.1', port=8080, debug=True)
    except Exception as e:
        logger.error(f"Error al iniciar el proxy: {e}")
        sys.exit(1)
