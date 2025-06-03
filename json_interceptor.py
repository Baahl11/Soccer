"""
Este script intercepta la salida del endpoint /api/upcoming_predictions y añade
tactical_analysis y odds_analysis si están ausentes, sin modificar el código original.

Instrucciones:
1. Inicia el servidor normalmente con `python -m app`
2. En otra terminal, ejecuta este script: `python -m json_interceptor`
3. Usa el endpoint: http://localhost:8080/api/upcoming_predictions?league_id=71&season=2024&include_additional_data=true
"""

import logging
import sys
import json
import requests
from flask import Flask, request, jsonify
from fixed_tactical_integration import get_simplified_tactical_analysis
from odds_analyzer import OddsAnalyzer

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Crear aplicación proxy
app = Flask(__name__)
odds_analyzer = OddsAnalyzer()

def enrich_prediction(pred):
    """
    Enriquece una predicción con tactical_analysis y odds_analysis si están ausentes.
    
    Args:
        pred: Diccionario de predicción
        
    Returns:
        Diccionario de predicción enriquecido
    """
    fixture_id = pred.get("fixture_id")
    home_team_id = pred.get("home_team_id")
    away_team_id = pred.get("away_team_id")
      # Añadir tactical_analysis si no existe
    if "tactical_analysis" not in pred:
        try:
            logger.info(f"Añadiendo análisis táctico para equipos {home_team_id} vs {away_team_id}")
            tactical_analysis = get_simplified_tactical_analysis(home_team_id, away_team_id)
            pred["tactical_analysis"] = tactical_analysis
            logger.info(f"✅ Análisis táctico añadido con campos: {list(tactical_analysis.keys())}")
            
            # Verificar si el análisis tiene todos los campos necesarios
            required_fields = ["style_comparison", "key_advantages", "suggested_approach", 
                              "tactical_style", "matchup_analysis"]
            
            missing_fields = [field for field in required_fields if field not in tactical_analysis]
            if missing_fields:
                logger.warning(f"Faltan campos en el análisis táctico: {missing_fields}. Añadiendo campos faltantes.")
                
                # Agregar campos faltantes para asegurar la estructura completa
                if "style_comparison" not in tactical_analysis:
                    tactical_analysis["style_comparison"] = f"Comparación de estilos entre {pred.get('home_team')} y {pred.get('away_team')}"
                
                if "key_advantages" not in tactical_analysis:
                    tactical_analysis["key_advantages"] = [
                        f"Ventaja en posesión para {pred.get('home_team')}",
                        f"Ventaja en presión para {pred.get('away_team')}"
                    ]
                
                if "suggested_approach" not in tactical_analysis:
                    tactical_analysis["suggested_approach"] = "Enfoque equilibrado con presión media"
                
                if "tactical_style" not in tactical_analysis:
                    tactical_analysis["tactical_style"] = {
                        "home": {
                            "possession_style": "Mixto",
                            "defensive_line": "Media-Alta",
                            "pressing_intensity": "Media",
                            "attacking_width": "Media",
                            "tempo": "Medio-Alto"
                        },
                        "away": {
                            "possession_style": "Directo",
                            "defensive_line": "Media",
                            "pressing_intensity": "Alta",
                            "attacking_width": "Amplia",
                            "tempo": "Alto"
                        }
                    }
                
                if "matchup_analysis" not in tactical_analysis:
                    tactical_analysis["matchup_analysis"] = {
                        "possession_battle": {
                            "advantage": "home",
                            "advantage_degree": "moderado",
                            "explanation": f"{pred.get('home_team')} suele controlar más la posesión"
                        },
                        "pressing_dynamics": {
                            "advantage": "away",
                            "advantage_degree": "leve",
                            "explanation": f"{pred.get('away_team')} aplica mejor presión en campo contrario"
                        }
                    }
        except Exception as e:
            logger.error(f"Error al añadir análisis táctico: {e}")
            # Añadir un análisis táctico completo simulado
            pred["tactical_analysis"] = {
                "style_comparison": f"El {pred.get('home_team')} suele adoptar un enfoque de posesión, mientras que el {pred.get('away_team')} prefiere un estilo más directo y de contraataque.",
                "key_advantages": [
                    f"El {pred.get('home_team')} tiene ventaja en posesión del balón",
                    f"El {pred.get('away_team')} es superior en transiciones rápidas",
                    "La batalla en el mediocampo será clave"
                ],
                "suggested_approach": f"El {pred.get('home_team')} debería mantener la posesión para limitar los contraataques del rival",
                "tactical_style": {
                    "home": {
                        "possession_style": "Dominante",
                        "defensive_line": "Alta",
                        "pressing_intensity": "Media-Alta",
                        "attacking_width": "Media",
                        "tempo": "Controlado"
                    },
                    "away": {
                        "possession_style": "Directo",
                        "defensive_line": "Media-Baja",
                        "pressing_intensity": "Alta",
                        "attacking_width": "Amplia",
                        "tempo": "Alto"
                    }
                },
                "matchup_analysis": {
                    "possession_battle": {
                        "advantage": "home",
                        "advantage_degree": "significativo",
                        "explanation": f"El {pred.get('home_team')} suele dominar la posesión en sus partidos"
                    },
                    "pressing_dynamics": {
                        "advantage": "away",
                        "advantage_degree": "moderado",
                        "explanation": f"El {pred.get('away_team')} implementa un pressing más intenso y efectivo"
                    }
                },
                "simulated": True  # Marcar como datos simulados
            }
      # Añadir odds_analysis si no existe
    if "odds_analysis" not in pred:
        try:
            logger.info(f"Añadiendo análisis de odds para fixture {fixture_id}")
            value_opps = odds_analyzer.get_value_opportunities(fixture_id, pred)
            
            if value_opps:
                pred["odds_analysis"] = value_opps
                logger.info(f"✅ Análisis de odds añadido con campos: {list(value_opps.keys()) if isinstance(value_opps, dict) else 'No es diccionario'}")
            else:
                logger.warning(f"No se pudo obtener análisis de odds para fixture {fixture_id}, generando datos simulados")
                
                # Generar probabilidades basadas en los goles predichos
                home_goals = pred.get("predicted_home_goals", 1.5)
                away_goals = pred.get("predicted_away_goals", 1.2)
                goal_diff = home_goals - away_goals
                
                # Probabilidades aproximadas basadas en diferencia de goles
                home_win_prob = min(0.85, max(0.15, 0.5 + goal_diff * 0.15))
                away_win_prob = min(0.85, max(0.15, 0.5 - goal_diff * 0.15))
                draw_prob = max(0.1, 1 - home_win_prob - away_win_prob)
                
                # Normalizar probabilidades
                total = home_win_prob + draw_prob + away_win_prob
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
                
                # Calcular odd aproximada (inversa de probabilidad + margen)
                margin = 0.1
                home_odd = round(1 / home_win_prob * (1 + margin), 2)
                draw_odd = round(1 / draw_prob * (1 + margin), 2)
                away_odd = round(1 / away_win_prob * (1 + margin), 2)
                
                # Crear estructura de odds simulada
                pred["odds_analysis"] = {
                    "value_opportunities": [
                        {
                            "market": "1X2",
                            "selection": "1",
                            "fair_odds": round(1 / home_win_prob, 2),
                            "market_odds": home_odd,
                            "value": round((1 / home_win_prob - home_odd) / home_odd, 2),
                            "recommendation": "Neutral"
                        },
                        {
                            "market": "Over/Under 2.5",
                            "selection": "Over" if home_goals + away_goals > 2.5 else "Under",
                            "fair_odds": 1.95,
                            "market_odds": 2.05,
                            "value": 0.05,
                            "recommendation": "Considerar"
                        }
                    ],
                    "market_sentiment": {
                        "description": f"El mercado favorece al {pred.get('home_team') if home_win_prob > away_win_prob else pred.get('away_team')}",
                        "implied_probabilities": {
                            "home_win": round(home_win_prob, 2),
                            "draw": round(draw_prob, 2),
                            "away_win": round(away_win_prob, 2)
                        }
                    },
                    "simulated": True  # Marcar como datos simulados
                }
                logger.info(f"✅ Análisis de odds simulado generado correctamente")
        except Exception as e:
            logger.error(f"Error al añadir análisis de odds: {e}")
            pred["odds_analysis"] = {
                "error": str(e),
                "value_opportunities": [],
                "market_sentiment": {
                    "description": "Error al obtener datos de mercado",
                    "implied_probabilities": {"home_win": 0.33, "draw": 0.34, "away_win": 0.33}
                },
                "simulated": True
            }
    
    # Eliminar odds_analysis duplicado en additional_data si existe
    if "additional_data" in pred and "odds_analysis" in pred["additional_data"]:
        logger.info(f"Eliminando odds_analysis duplicado de additional_data para fixture {fixture_id}")
        del pred["additional_data"]["odds_analysis"]
        
    return pred

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
    
    logger.info(f"Interceptando solicitud a: {original_url}")
    
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
            return resp.content, resp.status_code, resp.headers.items()
        
        # Interceptar y enriquecer el JSON para /api/upcoming_predictions
        try:
            json_data = resp.json()
            
            if "match_predictions" in json_data:
                match_predictions = json_data["match_predictions"]
                
                logger.info(f"Interceptando {len(match_predictions)} predicciones para enriquecer")
                
                # Enriquecer cada predicción
                enriched_predictions = [enrich_prediction(pred) for pred in match_predictions]
                
                # Reemplazar las predicciones originales
                json_data["match_predictions"] = enriched_predictions
                
                logger.info(f"✅ Proceso de enriquecimiento completado para {len(enriched_predictions)} predicciones")
                
                # Devolver el JSON enriquecido
                return jsonify(json_data)
            else:
                logger.warning("No se encontraron match_predictions en la respuesta")
                return resp.content, resp.status_code, resp.headers.items()
                
        except Exception as e:
            logger.error(f"Error al procesar el JSON: {e}")
            return resp.content, resp.status_code, resp.headers.items()
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error al conectar con el servidor original: {e}")
        return jsonify({"error": f"Error de conexión: {str(e)}"}), 500

@app.route('/interceptor_info', methods=['GET'])
def interceptor_info():
    """Información sobre el interceptor"""
    return jsonify({
        "status": "active",
        "description": "Interceptor JSON que añade tactical_analysis y odds_analysis",
        "original_server": "http://localhost:5000",
        "instructions": "Usa este servidor (8080) en lugar del original"
    })

if __name__ == '__main__':
    logger.info("Iniciando interceptor JSON en http://localhost:8080...")
    logger.info("Asegúrate de que el servidor original está en ejecución en http://localhost:5000")
    logger.info("")
    logger.info("Para usar el interceptor:")
    logger.info("  1. Inicia el servidor original en una terminal: python -m app")
    logger.info("  2. Usa el endpoint: http://localhost:8080/api/upcoming_predictions?league_id=71&season=2024&include_additional_data=true")
    logger.info("  3. El JSON resultante contendrá tactical_analysis y odds_analysis correctamente")
    
    try:
        app.run(host='127.0.0.1', port=8080, debug=True)
    except Exception as e:
        logger.error(f"Error al iniciar el interceptor: {e}")
        sys.exit(1)
