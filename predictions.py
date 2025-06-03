import os
import pickle
from typing import Dict, Any, Optional, List, Union, Tuple, cast
import logging
import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from fnn_model import FeedforwardNeuralNetwork
import joblib
# Import from the features module itself
from features import FeatureExtractor
from player_injuries import InjuryAnalyzer
from team_form import FormAnalyzer, get_team_form, get_head_to_head_analysis
from team_history import HistoricalAnalyzer
from datetime import datetime
import json
from data import get_fixture_data, get_lineup_data, get_fixture_statistics
from models import load_model, predict_corners, EnsemblePredictor
from business_rules import adjust_prediction_based_on_lineup, adjust_prediction_for_weather
from calibration import PredictionConfidenceEvaluator
from data_integration import DataIntegrator

# Add missing cache and league stats functions

_cache_store = {}

def get_from_cache(key: str) -> Optional[Dict[str, Any]]:
    return _cache_store.get(key)

def save_to_cache(key: str, value: Dict[str, Any], expire_hours: int = 24) -> None:
    _cache_store[key] = value

def get_league_stats(league_id: int, season: int) -> Dict[str, Any]:
    # Placeholder: Implement actual retrieval logic or return empty dict
    # Return empty dict to satisfy return type and avoid None
    return {}

import logging
import os
import joblib
from scipy.stats import poisson, norm
import random
import numpy as np
from utils import (
    calculate_over_probability,
    get_referee_strictness,
    get_fixture_referee,
    get_league_averages
)

logger = logging.getLogger(__name__)

# Cargamos el modelo pre-entrenado o creamos uno nuevo
try:
    # Cargar el modelo desde el archivo .h5
    if os.path.exists('models/fnn_model.h5'):
        # Crear una instancia de FeedforwardNeuralNetwork
        fnn_model = FeedforwardNeuralNetwork(input_dim=14)
        # El modelo se cargará en app.py usando load_model_from_path
        logger.info("Instancia de FNN creada para usar con modelo .h5")
    elif os.path.exists('models/fnn_model.pkl'):
        # Intentar cargar desde .pkl como fallback
        model_dict = joblib.load('models/fnn_model.pkl')
        # Si model_dict es un diccionario, convertirlo a un modelo real
        if isinstance(model_dict, dict) and 'input_dim' in model_dict:
            fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict.get('input_dim', 14))
            # Copiar parámetros si están disponibles
            if 'weights' in model_dict:
                try:
                    fnn_model.model.set_weights(model_dict['weights'])
                except Exception as e:
                    logger.warning(f"No se pudieron establecer los pesos desde .pkl: {e}")
            logger.info("Modelo FNN convertido desde .pkl")
        else:
            logger.warning("El archivo .pkl no contiene un modelo válido. Inicializando nuevo modelo.")
            fnn_model = FeedforwardNeuralNetwork(input_dim=14)
    else:
        # Inicializar el modelo con 14 características para coincidir con extract_features_from_form
        fnn_model = FeedforwardNeuralNetwork(input_dim=14)
        logger.info("Se inicializó un nuevo modelo FeedforwardNeuralNetwork")
    
    # Cargar el scaler
    feature_scaler = None
    if os.path.exists('models/scaler.pkl'):
        feature_scaler = joblib.load('models/scaler.pkl')
        logger.info("Feature scaler cargado correctamente")
except Exception as e:
    logger.warning(f"No se pudo cargar el modelo pre-entrenado: {e}. Inicializando nuevo modelo.")
    # Inicializar el modelo con 14 características para coincidir con extract_features_from_form
    fnn_model = FeedforwardNeuralNetwork(input_dim=14)
    logger.info("Se inicializó un nuevo modelo FeedforwardNeuralNetwork debido a un error")
    feature_scaler = None

# Inicializar extractores de características y analizadores
feature_extractor = FeatureExtractor()
injury_analyzer = InjuryAnalyzer()
form_analyzer = FormAnalyzer()
historical_analyzer = HistoricalAnalyzer()

def safe_round(value: Any, decimals: int = 2) -> float:
    """Realiza redondeo seguro para diferentes tipos de valores"""
    try:
        # Manejar tipos numpy
        if isinstance(value, (np.ndarray, np.generic)):
            try:
                # Intentar convertir a escalar nativo
                value = value.item() if value.size == 1 else float(value)
            except (AttributeError, ValueError):
                # Si falla, intentar conversión directa
                value = float(np.asarray(value).flatten()[0])
        return float(round(float(value), decimals))
    except Exception:
        return 0.0

def get_default_prediction() -> Dict[str, Any]:
    """Return a basic prediction with reasonable defaults based on league averages."""
    return {
        "predicted_home_goals": 1.35,  # Media histórica de goles locales
        "predicted_away_goals": 1.15,  # Media histórica de goles visitantes
        "total_goals": 2.50,
        "prob_over_2_5": 0.48,
        "prob_btts": 0.52,
        "method": "historical_average",
        "confidence": 0.5,
        "corners": {
            "total": 9.8,
            "home": 5.3,
            "away": 4.5,
            "over_8.5": 0.55,
            "over_9.5": 0.45,
            "over_10.5": 0.35
        },
        "cards": {
            "total": 3.5,
            "home": 1.8,
            "away": 1.7,
            "over_2.5": 0.65,
            "over_3.5": 0.45,
            "over_4.5": 0.25
        },
        "fouls": {
            "total": 22.5,
            "home": 11.0,
            "away": 11.5,
            "over_19.5": 0.70,
            "over_21.5": 0.55,
            "over_23.5": 0.40
        }
    }

def predict_from_historical_data(fixture_id: int, weather_conditions: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Makes predictions using historical data and model predictions
    """
    try:
        # Get fixture data
        fixture_data = get_fixture_data(fixture_id)
        if not fixture_data:
            logger.warning(f"No fixture data found for ID {fixture_id}")
            return get_default_prediction()

        # Extract team IDs and league info
        teams = fixture_data.get('teams', {})
        league = fixture_data.get('league', {})
        home_team_id = teams.get('home', {}).get('id')
        away_team_id = teams.get('away', {}).get('id')
        league_id = league.get('id')
        season = league.get('season')

        if not (home_team_id and away_team_id and league_id and season):
            logger.warning(f"Missing required data for fixture {fixture_id}")
            return get_default_prediction()

        try:
            # Get historical form data with league context
            home_form = get_team_form(home_team_id, league_id, season)
            away_form = get_team_form(away_team_id, league_id, season)
            
            if not (home_form and away_form):
                logger.warning("Missing form data for one or both teams")
                return get_default_prediction()
            
            # Calculate prediction from form data
            home_xg, away_xg = calculate_statistical_prediction(
                home_form, away_form, 
                get_head_to_head_analysis(home_team_id, away_team_id),
                home_team_id, away_team_id
            )

            # Calculate additional metrics
            total_goals = home_xg + away_xg
            prob_over = 1 - norm.cdf(2.5, total_goals, 1.25)
            prob_btts = min(0.95, max(0.05, (home_xg * away_xg) / 3.0))
            
            result = {
                "predicted_home_goals": round(float(home_xg), 2),
                "predicted_away_goals": round(float(away_xg), 2),
                "total_goals": round(float(total_goals), 2),
                "prob_over_2_5": round(float(prob_over), 2),
                "prob_btts": round(float(prob_btts), 2),
                "method": "historical_data",
                "confidence": 0.65
            }
            
            # Handle calibration and confidence evaluation
            try:
                evaluator = PredictionConfidenceEvaluator()
                confidence_info = evaluator.evaluate_prediction_confidence(
                    result,
                    ensemble_predictions={},
                    data_quality_metrics={},
                    historical_metrics={},
                    calibration_metrics={}
                )
                
                result['confidence_level'] = confidence_info.get('confidence_level', 'medium')
                result['confidence_score'] = confidence_info.get('confidence_score', 0.65)
                result['confidence_factors'] = confidence_info.get('confidence_factors', {})
                
                return result
                
            except Exception as calib_error:
                logger.warning(f"Confidence evaluation error: {calib_error}")
                return result
                
        except Exception as form_error:
            logger.warning(f"Error processing form data: {form_error}")
            return get_default_prediction()
            
    except Exception as e:
        logger.error(f"Error predicting from historical data: {e}")
        return get_default_prediction()

def make_global_prediction(fixture_id: int, weather_data: Optional[Dict[str, Any]] = None, odds_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a global prediction for a fixture."""
    try:
        # Get fixture data to extract teams
        fixture_data = get_fixture_data(fixture_id)
        if not fixture_data or 'response' not in fixture_data or not fixture_data['response']:
            logger.warning(f"No fixture data found for ID {fixture_id}")
            return get_default_prediction()
            
        response = fixture_data['response'][0]
        teams = response.get('teams', {})
        league = response.get('league', {})
        
        home_team_id = teams.get('home', {}).get('id')
        away_team_id = teams.get('away', {}).get('id')
        league_id = league.get('id')
        season = league.get('season')
        
        if not home_team_id or not away_team_id or not league_id or not season:
            logger.warning(f"Missing team or league information for fixture {fixture_id}")
            return get_default_prediction()
            
        # Get team form data
        home_form = get_team_form(home_team_id, league_id, season)
        away_form = get_team_form(away_team_id, league_id, season)
        h2h = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Add team IDs to form data for feature extraction
        if isinstance(home_form, dict):
            home_form['team_id'] = home_team_id
        if isinstance(away_form, dict):
            away_form['team_id'] = away_team_id
        
        # Try using neural network model directly with history-extracted features
        try:
            if fnn_model is not None and feature_scaler is not None:
                # Extract features for the model
                features = extract_features_from_form(home_form, away_form, h2h)
                
                # Scale features
                features_scaled = feature_scaler.transform(features.reshape(1, -1))
                
                # Get predictions
                predictions = fnn_model.predict(features_scaled)
                raw_home_xg = float(predictions[0, 0])
                raw_away_xg = float(predictions[0, 1])
                
                # Model predictions can be very low due to differences
                # between training data and real data
                # If model gives predictions ≈ 0, we'll use statistical data
                if raw_home_xg < 0.2 or raw_away_xg < 0.2:
                    logger.warning(f"Model predictions too low ({raw_home_xg:.2f}, {raw_away_xg:.2f}) for fixture {fixture_id}")
                    use_model = False
                else:
                    use_model = True
                
                if use_model:
                    # Use model predictions directly
                    home_xg = max(0.5, raw_home_xg)  # Minimum of 0.5 goals
                    away_xg = max(0.3, raw_away_xg)  # Minimum of 0.3 goals
                    
                    # Apply additional home advantage
                    home_xg *= 1.1
                    
                    # Ensure reasonable values
                    home_xg = min(3.5, home_xg)
                    away_xg = min(3.0, away_xg)
                    
                    logger.info(f"Neural network prediction for fixture {fixture_id}: home={home_xg:.2f}, away={away_xg:.2f} (raw: {raw_home_xg:.2f}, {raw_away_xg:.2f}) [Teams: {home_team_id} vs {away_team_id}]")
                    method = "neural_network"
                else:
                    # Fallback to statistical method
                    home_xg, away_xg = calculate_statistical_prediction(home_form, away_form, h2h, home_team_id, away_team_id)
                    method = "historical_data"
            else:
                # If no model available, use statistical method
                home_xg, away_xg = calculate_statistical_prediction(home_form, away_form, h2h, home_team_id, away_team_id)
                method = "historical_data"
        except Exception as model_error:
            logger.warning(f"Error using neural network model: {model_error}. Using statistical method.")
            home_xg, away_xg = calculate_statistical_prediction(home_form, away_form, h2h, home_team_id, away_team_id)
            method = "historical_data"
        
        # Calculate additional metrics
        total_goals = home_xg + away_xg
        prob_over = 1 - norm.cdf(2.5, total_goals, 1.25)
        prob_btts = min(0.95, max(0.05, (home_xg * away_xg) / 3.0))
        
        logger.info(f"Final prediction for fixture {fixture_id}: home={home_xg:.2f}, away={away_xg:.2f} [Teams: {home_team_id} vs {away_team_id}]")
        
        # Get additional stats predictions
        match_data = {
            "fixture_id": fixture_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league_id": league_id,
            "season": season
        }        # Obtener predicciones de estadísticas adicionales
        stats_prediction = predict_match_stats(fixture_id, home_team_id, away_team_id, league_id, season)
          # Obtener datos de Elo si están disponibles
        elo_data = {}
        try:
            from team_elo_rating import get_elo_ratings_for_match
            elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
        except Exception as elo_error:
            logger.warning(f"Error getting Elo data: {elo_error}")
        
        result = {
            "predicted_home_goals": round(float(home_xg), 2),
            "predicted_away_goals": round(float(away_xg), 2),
            "total_goals": round(float(total_goals), 2),
            "prob_over_2_5": round(float(prob_over), 2),
            "prob_btts": round(float(prob_btts), 2),
            "method": method,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        }
        
        # PIPELINE FIX: Calculate 1x2 probabilities BEFORE confidence calculation
        # This ensures dynamic confidence receives real probabilities instead of defaults
        home_win_prob, draw_prob, away_win_prob = calculate_1x2_probabilities(
            result.get("predicted_home_goals", 1.0),
            result.get("predicted_away_goals", 1.0)
        )
        result["home_win_prob"] = round(home_win_prob, 3)
        result["draw_prob"] = round(draw_prob, 3)
        result["away_win_prob"] = round(away_win_prob, 3)
        
        # Also add alternative names for consistency with normalize_prediction_structure
        result["home_win_probability"] = round(home_win_prob, 3)
        result["draw_probability"] = round(draw_prob, 3)
        result["away_win_probability"] = round(away_win_prob, 3)
        
        # Calculate dynamic confidence based on match characteristics
        # NOW with real 1X2 probabilities available
        try:
            from app import calculate_dynamic_confidence
            # Add additional data needed for confidence calculation
            result["league_id"] = league_id
            result["fixture_id"] = fixture_id 
            dynamic_confidence = calculate_dynamic_confidence(result)
            result["confidence"] = dynamic_confidence
        except Exception as conf_error:
            logger.warning(f"Error calculating dynamic confidence: {conf_error}")
            # Fallback with some variation to avoid identical values
            base_conf = 0.65
            variation = (abs((home_team_id or 0) + (away_team_id or 0)) % 20) / 100
            result["confidence"] = round(base_conf + variation - 0.1, 2)
          # Add remaining prediction components
        result.update({
            # Agregar predicciones de estadísticas
            "corners": stats_prediction.get("corners", {
                "total": 10.0,
                "home": 5.5,
                "away": 4.5,
                "over_8.5": 0.65,
                "over_9.5": 0.55,
                "over_10.5": 0.45
            }),
            "cards": stats_prediction.get("cards", {
                "total": 3.5,
                "home": 1.5,
                "away": 2.0,
                "over_2.5": 0.70,
                "over_3.5": 0.50,
                "over_4.5": 0.30
            }),
            "fouls": stats_prediction.get("fouls", {
                "total": 22.0,
                "home": 10.0,
                "away": 12.0,
                "over_19.5": 0.65,
                "over_21.5": 0.55,
                "over_23.5": 0.40
            }),
            
            # Agregar datos de clasificaciones Elo
            "elo_ratings": {
                "home_elo": elo_data.get('home_elo', 1500),
                "away_elo": elo_data.get('away_elo', 1500),
                "elo_diff": elo_data.get('elo_diff', 0)
            },
            "elo_probabilities": {
                "win": elo_data.get('elo_win_probability', 0.5),
                "draw": elo_data.get('elo_draw_probability', 0.25),
                "loss": elo_data.get('elo_loss_probability', 0.25)
            }
        })
        
        # Añadir diferencia de goles esperada basada en Elo si está disponible
        if 'expected_goal_diff' in elo_data:
            result['elo_expected_goal_diff'] = elo_data['expected_goal_diff']
        
        # Integrate calibration and confidence evaluation
        try:
            evaluator = PredictionConfidenceEvaluator()
            confidence_info = evaluator.evaluate_prediction_confidence(
                result,
                ensemble_predictions={},
                data_quality_metrics={},
                historical_metrics={},
                calibration_metrics={}
            )
            
            result['confidence_level'] = confidence_info.get('confidence_level', 'medium')
            result['confidence_score'] = confidence_info.get('confidence_score', 0.65)
            result['confidence_factors'] = confidence_info.get('confidence_factors', {})
              # Apply weather adjustment if weather_data is provided
            if weather_data:
                weather_adjusted = adjust_prediction_for_weather(result, weather_data)
                result.update(weather_adjusted)
            
            # NOTE: 1x2 probabilities now calculated earlier in pipeline before confidence
            # This ensures dynamic confidence receives real probabilities instead of defaults
            
            return result
            
        except Exception as calib_error:
            logger.warning(f"Confidence evaluation error: {calib_error}")
            return result
            
    except Exception as e:
        logger.error(f"Error predicting from historical data: {e}")
        return get_default_prediction()

def extract_features_from_form(home_form: Dict[str, Any], away_form: Dict[str, Any], h2h: Dict[str, Any]) -> np.ndarray:
    """Extraer características para el modelo a partir de datos de forma de los equipos."""
    try:
        # Obtener IDs de equipos para crear características específicas de cada equipo
        home_team_id = home_form.get('team_id', 0)
        away_team_id = away_form.get('team_id', 0)
        
        # Si los IDs no están disponibles, usar valores constantes
        if not home_team_id:
            home_team_id = 1001
        if not away_team_id:
            away_team_id = 2002
            
        logger.info(f"Extracting features for teams: {home_team_id} vs {away_team_id}")
        
        # Características principales de los equipos
        home_goals_per_match = float(home_form.get('goals_scored', 8)) / 5.0
        home_goals_conceded = float(home_form.get('goals_conceded', 5)) / 5.0
        home_win_percentage = float(home_form.get('win_percentage', 50)) / 100.0
        home_clean_sheets = float(home_form.get('clean_sheets', 2)) / 5.0
        
        away_goals_per_match = float(away_form.get('goals_scored', 8)) / 5.0
        away_goals_conceded = float(away_form.get('goals_conceded', 5)) / 5.0
        away_win_percentage = float(away_form.get('win_percentage', 50)) / 100.0
        away_clean_sheets = float(away_form.get('clean_sheets', 2)) / 5.0
        
        # Características Head-to-Head
        h2h_ratio = 0.5  # Valor neutral por defecto
        h2h_total_matches = float(h2h.get('total_matches', 0))
        
        if h2h_total_matches > 0:
            h2h_team1_wins = float(h2h.get('team1_wins', 0))
            h2h_team2_wins = float(h2h.get('team2_wins', 0))
            h2h_draws = float(h2h.get('draws', 0))
            if h2h_team1_wins + h2h_team2_wins > 0:
                h2h_ratio = h2h_team1_wins / (h2h_team1_wins + h2h_team2_wins)
        else:
            h2h_team1_wins = 0
            h2h_team2_wins = 0
            h2h_draws = 0
        
        h2h_avg_goals = float(h2h.get('average_goals_per_match', 2.5))
        
        # Inicializar el vector de características reducido a las 14 que espera el modelo
        features = np.zeros(14)
        
        # Características básicas (8 características)
        features[0] = home_goals_per_match
        features[1] = home_goals_conceded
        features[2] = home_win_percentage
        features[3] = home_clean_sheets
        
        features[4] = away_goals_per_match
        features[5] = away_goals_conceded
        features[6] = away_win_percentage
        features[7] = away_clean_sheets
        
        # Características combinadas (6 características adicionales)
        features[8] = (home_goals_per_match + away_goals_conceded) / 2.0  # Expectativa de gol local
        features[9] = (away_goals_per_match + home_goals_conceded) / 2.0  # Expectativa de gol visitante
        features[10] = h2h_ratio  # Ratio de victorias en h2h
        features[11] = h2h_avg_goals  # Promedio de goles en h2h
        features[12] = home_goals_per_match  # Goles local por partido
        features[13] = away_goals_per_match  # Goles visitante por partido
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features from form: {e}")
        # Retornar características por defecto con el tamaño correcto
        return np.zeros(14)  # Vector de 14 características como espera el modelo

def extract_features(fixture_data: Dict[str, Any]) -> np.ndarray:
    """
    Extrae características para el modelo a partir de los datos del partido.
    
    Args:
        fixture_data: Diccionario con datos del partido
        
    Returns:
        Array numpy con las características procesadas
    """
    try:
        # Extraer IDs de equipos
        home_team_id = fixture_data.get("home_team", {}).get("id", 0)
        away_team_id = fixture_data.get("away_team", {}).get("id", 0)
        fixture_id = fixture_data.get("fixture_id", 0)
        league_id = fixture_data.get("league", {}).get("id", 0)
        
        # Asegurar que la fecha sea string o usar valor por defecto
        match_date = fixture_data.get("date")
        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")
        elif not isinstance(match_date, str):
            match_date = str(match_date)
        
        # Obtener métricas de lesiones
        home_injuries = injury_analyzer.get_team_injury_impact(home_team_id, fixture_id)
        away_injuries = injury_analyzer.get_team_injury_impact(away_team_id, fixture_id)
        
        # Obtener métricas de forma reciente
        home_form = form_analyzer.get_team_form_metrics(home_team_id, last_matches=5)
        away_form = form_analyzer.get_team_form_metrics(away_team_id, last_matches=5)
        
        # Obtener datos históricos (enfrentamientos directos)
        h2h_stats = historical_analyzer.get_head_to_head_stats(home_team_id, away_team_id, last_matches=5)
        
        # Construir vector de características manualmente
        # Usar valores de home_form, away_form, h2h_stats y lesiones
        features_list = []
        
        # Ejemplo: agregar algunas características básicas
        features_list.append(home_injuries)
        features_list.append(away_injuries)
        
        # Agregar valores de home_form
        if isinstance(home_form, dict):
            features_list.extend([float(v) for v in home_form.values()])
        else:
            features_list.extend([0.0]*5)  # Ajustar según número esperado
        
        # Agregar valores de away_form
        if isinstance(away_form, dict):
            features_list.extend([float(v) for v in away_form.values()])
        else:
            features_list.extend([0.0]*5)
        
        # Agregar valores de h2h_stats
        if isinstance(h2h_stats, dict):
            features_list.extend([float(v) for v in h2h_stats.values()])
        else:
            features_list.extend([0.0]*5)
        
        # Convertir a numpy array
        features = np.array(features_list, dtype=np.float64)
        
        # Aplicar escalado si hay un scaler disponible
        if feature_scaler is not None:
            features = feature_scaler.transform(features.reshape(1, -1))[0]
            
        return features
        
    except Exception as e:
        logger.error(f"Error al extraer características: {e}")
        # Devolver vector de características con ceros en caso de error
        return np.zeros(30, dtype=np.float64)  # Ajustar según el número real de características

def calculate_statistical_prediction(home_form: Dict[str, Any], away_form: Dict[str, Any], 
                                  h2h: Dict[str, Any], home_team_id: int, away_team_id: int) -> Tuple[float, float]:
    """Calculate prediction using statistical method."""
    try:
        # Base expected goals from historical averages
        home_xg = float(home_form.get('avg_goals_scored', 1.5))
        away_xg = float(away_form.get('avg_goals_scored', 1.2))
        
        # Adjust for defensive strength
        home_defensive_strength = 1.0 - (float(home_form.get('clean_sheets_ratio', 0.3)) * 0.5)
        away_defensive_strength = 1.0 - (float(away_form.get('clean_sheets_ratio', 0.25)) * 0.5)
        
        home_xg = home_xg * away_defensive_strength
        away_xg = away_xg * home_defensive_strength
        
        # Adjust for head-to-head history
        if h2h.get('matches_played', 0) > 0:
            h2h_factor = 0.3
            home_xg = (home_xg * (1 - h2h_factor)) + (float(h2h.get('avg_goals_team1', home_xg)) * h2h_factor)
            away_xg = (away_xg * (1 - h2h_factor)) + (float(h2h.get('avg_goals_team2', away_xg)) * h2h_factor)
        
        # Apply home advantage
        home_xg *= 1.2
        
        # Apply form adjustment
        home_form_factor = float(home_form.get('form_trend', 0.0)) * 0.15
        away_form_factor = float(away_form.get('form_trend', 0.0)) * 0.15
        
        home_xg *= (1 + home_form_factor)
        away_xg *= (1 + away_form_factor)
        
        # Ensure reasonable values
        home_xg = min(4.0, max(0.3, home_xg))
        away_xg = min(3.5, max(0.2, away_xg))
        
        return home_xg, away_xg
        
    except Exception as e:
        logger.error(f"Error in statistical prediction: {e}")
        return 1.5, 1.2

def generate_varied_corners(base_stats: Dict[str, Any]) -> Dict[str, float]:
    """Generate corner predictions based on team attacking styles"""
    try:
        # Get team-specific data
        home_attacking_strength = base_stats.get('home_attacking_strength', 1.0)
        away_attacking_strength = base_stats.get('away_attacking_strength', 1.0)
        league_corners_avg = base_stats.get('league_corners_avg', 9.8)
        
        # Base corners adjusted by team strengths
        base_corners = league_corners_avg * ((home_attacking_strength + away_attacking_strength) / 2.0)
        
        # Add controlled randomness
        import random
        variation = random.uniform(-1.0, 1.0)
        total = max(8.0, min(13.0, base_corners + variation))
        
        # Home/Away split based on attacking strengths
        home_ratio = home_attacking_strength / (home_attacking_strength + away_attacking_strength)
        home_corners = total * home_ratio
        away_corners = total * (1 - home_ratio)
        
        # Calculate over probabilities based on expected total
        std_dev = 2.0
        over_8_5 = calculate_over_probability(total, 8.5, std_dev)
        over_9_5 = calculate_over_probability(total, 9.5, std_dev)
        over_10_5 = calculate_over_probability(total, 10.5, std_dev)
        
        return {
            'total': round(total, 1),
            'home': round(home_corners, 1),
            'away': round(away_corners, 1),
            'over_8.5': round(over_8_5, 2),
            'over_9.5': round(over_9_5, 2),
            'over_10.5': round(over_10_5, 2)
        }
        
    except Exception as e:
        logger.error(f"Error in corner prediction: {e}")
        return generate_default_corners()

def generate_varied_cards(base: float) -> Dict[str, float]:
    """Generate varied card predictions around a base value"""
    import random
    variation = random.uniform(-0.5, 0.5)
    total = max(2.5, min(4.5, base + variation))
    home_ratio = random.uniform(0.45, 0.55)
    
    return {
        'total': round(total, 1),
        'home': round(total * home_ratio, 1),
        'away': round(total * (1 - home_ratio), 1),
        'over_2.5': round(calculate_over_probability(total, 2.5, 1.0), 2),
        'over_3.5': round(calculate_over_probability(total, 3.5, 1.0), 2),
        'over_4.5': round(calculate_over_probability(total, 4.5, 1.0), 2)
    }

def generate_varied_fouls(base: float) -> Dict[str, float]:
    """Generate varied foul predictions around a base value"""
    import random
    variation = random.uniform(-2.0, 2.0)
    total = max(18.0, min(24.0, base + variation))
    home_ratio = random.uniform(0.45, 0.55)
    
    return {
        'total': round(total, 1),
        'home': round(total * home_ratio, 1),
        'away': round(total * (1 - home_ratio), 1),
        'over_19.5': round(calculate_over_probability(total, 19.5, 3.0), 2),
        'over_21.5': round(calculate_over_probability(total, 21.5, 3.0), 2),
        'over_23.5': round(calculate_over_probability(total, 23.5, 3.0), 2)
    }

def predict_match_stats(fixture_id: int, home_team_id: int, away_team_id: int, league_id: int, season: int) -> Dict[str, Any]:
    """Predict match statistics using team-specific data"""
    try:
        # Get team statistics
        home_stats = get_team_statistics(home_team_id, league_id, season)
        away_stats = get_team_statistics(away_team_id, league_id, season)
        
        # Get recent form
        home_form = get_team_form(home_team_id, league_id=league_id, last_matches=5)
        away_form = get_team_form(away_team_id, league_id=league_id, last_matches=5)
        
        # Get league statistics for context
        league_stats = get_league_averages(league_id, season)
        
        # Get referee data
        referee_data = get_fixture_referee(fixture_id)
        referee_strictness = get_referee_strictness(fixture_id)
        
        # Calculate team strengths
        home_attacking = calculate_attacking_strength(home_stats, home_form)
        away_attacking = calculate_attacking_strength(away_stats, away_form)
        
        # Get current match stats if available (for in-game predictions)
        current_stats = None
        if fixture_id:
            try:
                current_stats = get_fixture_statistics(fixture_id)
            except Exception as e:
                logger.debug(f"No in-game stats available for fixture {fixture_id}: {e}")
        
        # Prepare base stats for all predictors
        base_stats = {
            'home_attacking_strength': home_attacking,
            'away_attacking_strength': away_attacking,
            'referee_strictness': referee_strictness,
            'league_cards_avg': league_stats.get('cards_per_game', 3.8),
            'league_corners_avg': league_stats.get('corners_per_game', 9.8),
            'league_fouls_avg': league_stats.get('fouls_per_game', 21.5)
        }
        
        # Use corners model from models.py for corner predictions
        try:
            from models import predict_corners
            corners_prediction = predict_corners(home_stats, away_stats, current_stats)
            
            # Extract relevant data from the corners model output
            corners = {
                'total': round(corners_prediction.get('predicted_corners_mean', 10.0), 1),
                'home': round(corners_prediction.get('home_corners_exp', 5.5), 1),
                'away': round(corners_prediction.get('away_corners_exp', 4.5), 1),
                'over_8.5': round(calculate_over_probability(corners_prediction.get('predicted_corners_mean', 10.0), 8.5, corners_prediction.get('std_dev_corners', 2.5)), 2),
                'over_9.5': round(calculate_over_probability(corners_prediction.get('predicted_corners_mean', 10.0), 9.5, corners_prediction.get('std_dev_corners', 2.5)), 2),
                'over_10.5': round(calculate_over_probability(corners_prediction.get('predicted_corners_mean', 10.0), 10.5, corners_prediction.get('std_dev_corners', 2.5)), 2),
                'confidence': corners_prediction.get('confidence_level', 'Media')
            }
            logger.info(f"Corner prediction generated using CornersModel: {corners['total']} total corners")
        except Exception as e:
            logger.warning(f"Could not use corners model, falling back to generate_varied_corners: {e}")
            corners = generate_varied_corners(base_stats)
        
        # Generate card predictions based on referee data and team discipline
        cards = generate_varied_cards(base_stats.get('league_cards_avg', 3.8) * referee_strictness)
        
        # Generate foul predictions
        fouls = generate_varied_fouls(base_stats.get('league_fouls_avg', 21.5))
        
        return {
            'corners': corners,
            'cards': cards,
            'fouls': fouls
        }
        
    except Exception as e:
        logger.error(f"Error in match stats prediction: {e}")
        return get_default_stats_prediction()

def calculate_attacking_strength(team_stats: Dict[str, Any], form_data: Dict[str, Any]) -> float:
    """Calculate team's attacking strength based on stats and form"""
    try:
        base_strength = 1.0
        
        # Adjust for goals scored
        goals_per_game = team_stats.get('avg_goals_scored', 1.5)
        if goals_per_game > 2.0:
            base_strength *= 1.2
        elif goals_per_game < 1.0:
            base_strength *= 0.8
            
        # Adjust for recent form
        form_goals = form_data.get('goals_scored', 0) / 5.0
        if form_goals > 1.5:
            base_strength *= 1.15
        elif form_goals < 0.8:
            base_strength *= 0.85
            
        return min(1.5, max(0.5, base_strength))
        
    except Exception:
        return 1.0

def generate_default_corners():
    return {
        'total': 9.5,
        'home': 5.0,
        'away': 4.5,
        'over_8.5': 0.55,
        'over_9.5': 0.45,
        'over_10.5': 0.35
    }

def generate_default_cards():
    return {
        'total': 3.5,
        'home': 1.8,
        'away': 1.7,
        'over_2.5': 0.60,
        'over_3.5': 0.40,
        'over_4.5': 0.25
    }

def generate_default_fouls():
    return {
        'total': 20.5,
        'home': 10.5,
        'away': 10.0,
        'over_19.5': 0.55,
        'over_21.5': 0.40,
        'over_23.5': 0.25
    }

def get_default_stats_prediction() -> Dict[str, Any]:
    """Return default stats prediction values."""
    return {
        "corners": {
            "total": 10.0,
            "home": 5.5,
            "away": 4.5,
            "over_8.5": 0.65,
            "over_9.5": 0.55,
            "over_10.5": 0.45
        },
        "cards": {
            "total": 3.5,
            "home": 1.5,
            "away": 2.0,
            "over_2.5": 0.70,
            "over_3.5": 0.50,
            "over_4.5": 0.30
        },
        "fouls": {
            "total": 22.0,
            "home": 10.0,
            "away": 12.0,
            "over_19.5": 0.65,
            "over_21.5": 0.55,
            "over_23.5": 0.40
        }
    }

def calculate_1x2_probabilities(home_goals: float, away_goals: float) -> Tuple[float, float, float]:
    """
    Calcula probabilidades de victoria local, empate y victoria visitante usando Poisson.
    
    Args:
        home_goals: Expectativa de goles del equipo local
        away_goals: Expectativa de goles del equipo visitante
    
    Returns:
        Tupla (p_home_win, p_draw, p_away_win)
    """
    # Simular hasta un máximo de goles razonable
    max_goals = 10
    
    home_win_prob = 0.0
    draw_prob = 0.0
    away_win_prob = 0.0
    
    # Calcular probabilidades para cada posible resultado
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            # Probabilidad de este resultado específico
            p = poisson.pmf(h, home_goals) * poisson.pmf(a, away_goals)
            
            if h > a:
                home_win_prob += p
            elif h == a:
                draw_prob += p
            else:
                away_win_prob += p
    
    return float(home_win_prob), float(draw_prob), float(away_win_prob)

def make_enhanced_prediction(
    fixture_data: Dict[str, Any],
    player_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Make enhanced prediction using additional data.
    
    Args:
        fixture_data: Dictionary containing fixture information
        player_data: Optional dictionary with player statistics
        
    Returns:
        Dictionary containing enhanced prediction results with normalized structure
    """
    try:
        fixture_id = fixture_data.get("fixture_id", 0)
        weather_data = fixture_data.get("weather", {})
        odds_data = fixture_data.get("odds", {})
        
        # Get base prediction
        prediction = make_global_prediction(fixture_id, weather_data, odds_data)
        
        # Enhance prediction with player data if available
        if player_data:
            prediction = adjust_prediction_based_on_lineup(prediction, player_data)
        
        # Update method to indicate enhanced prediction
        prediction["method"] = "enhanced"
        
        # Import and apply normalization
        from app import normalize_prediction_structure
        normalized_prediction = normalize_prediction_structure(prediction)
        
        return normalized_prediction
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        # Import and apply normalization to default prediction too
        try:
            from app import normalize_prediction_structure
            default_prediction = get_default_prediction()
            return normalize_prediction_structure(default_prediction)
        except:
            return get_default_prediction()

def make_prediction(fixture_id: int, include_additional: bool = True) -> Dict[str, Any]:
    """
    Make prediction for a specific fixture with enhanced data handling.
    """
    try:
        # Get base fixture data
        fixture_data = get_fixture_data(fixture_id)
        if not fixture_data:
            raise ValueError(f"No fixture data found for ID {fixture_id}")

        # Get weather data with proper error handling - ensure date is string
        match_date = fixture_data.get("date", "") or datetime.now().strftime("%Y-%m-%d")
        city = fixture_data.get("city", "")
        weather_data = get_weather_data(city, str(match_date))
        
        # Get lineup data
        lineup_data = get_lineup_data(fixture_id) if include_additional else None
        
        # Make base prediction
        prediction = make_global_prediction(fixture_id, weather_data)
        
        if include_additional and lineup_data:
            # Adjust prediction based on lineup data
            prediction = adjust_prediction_based_on_lineup(prediction, lineup_data)
            prediction["method"] = "enhanced"
            prediction["lineup_data_used"] = True
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in prediction process: {e}")
        return {
            "fixture_id": fixture_id,
            "error": str(e),
            "success": False
        }

def get_weather_data(city: str, date: str) -> Dict[str, Any]:
    """
    Get weather data for a given city and date.
    
    Args:
        city: City name
        date: Date in ISO format
        
    Returns:
        Dictionary containing weather information
    """
    try:
        from weather_api import get_weather_forecast
        return get_weather_forecast(city, "", date)
    except Exception as e:
        logger.warning(f"Error getting weather data: {e}")
        return {}

def get_team_statistics(team_id: int, league_id: int, season: int) -> Dict[str, Any]:
    """
    Obtiene estadísticas históricas de un equipo en una liga.
    """
    try:
        # Get current season
        from datetime import datetime
        current_season = datetime.now().year
        
        # Obtener datos de forma del equipo
        team_form = get_team_form(team_id, league_id=league_id, last_matches=5) 
        if not team_form:
            return {}

        # Extraer estadísticas relevantes
        stats = {
            "avg_corners_for": float(team_form.get("avg_corners_for", 5.0)),
            "avg_corners_against": float(team_form.get("avg_corners_against", 4.5)),
            "avg_cards": float(team_form.get("avg_cards", 2.0)),
            "avg_fouls": float(team_form.get("avg_fouls", 11.0)),
            "goals_scored": float(team_form.get("goals_scored", 0)),
            "goals_conceded": float(team_form.get("goals_conceded", 0)),
            "matches_played": float(team_form.get("matches_played", 1))
        }

        # Normalizar por partidos jugados
        matches = max(1, stats["matches_played"])
        stats["avg_goals_scored"] = stats["goals_scored"] / matches
        stats["avg_goals_conceded"] = stats["goals_conceded"] / matches

        return stats

    except Exception as e:
        logger.error(f"Error obteniendo estadísticas del equipo {team_id}: {e}")
        return {}

def calculate_over_probability(expected_value: float, line: float, std_dev: float) -> float:
    """
    Calculate probability of going over a line given expected value and standard deviation
    
    Handles edge cases safely and avoids division by zero or other numerical issues
    """
    # Ensure we have valid inputs
    if std_dev <= 0:
        std_dev = 0.5  # Use a reasonable default if std_dev is invalid
    
    # Safety check for expected_value
    if not isinstance(expected_value, (int, float)) or np.isnan(expected_value):
        expected_value = line  # Fallback to even probability
    
    try:
        from scipy.stats import norm
        # Use norm.sf (survival function) which is 1-cdf and numerically more accurate for tails
        return float(norm.sf(line, loc=expected_value, scale=std_dev))
    except (ImportError, ValueError, ZeroDivisionError, TypeError):
        # Fallback to simpler calculation if scipy not available or errors occur
        # This is a simple approximation based on the normal distribution properties
        if expected_value > line:
            return min(0.95, 0.5 + (expected_value - line)/(4*std_dev))
        else:
            return max(0.05, 0.5 - (line - expected_value)/(4*std_dev))
    except Exception as e:
        logger.warning(f"Unexpected error in calculating probabilities: {e}")
        # Even more basic fallback: use linear interpolation based on expected value
        if expected_value > line + 1.5:
            return 0.80
        elif expected_value > line + 0.5:
            return 0.65
        elif expected_value > line:
            return 0.55
        elif expected_value > line - 0.5:
            return 0.45
        elif expected_value > line - 1.5:
            return 0.35
        else:
            return 0.20

def get_referee_strictness(fixture_id: int, default: float = 1.0) -> float:
    """Get referee strictness factor based on historical data"""
    try:
        referee_data = get_fixture_referee(fixture_id)
        if not referee_data:
            return default
            
        cards_per_game = referee_data.get('cards_per_game', default * 3.5) / 3.5
        return min(1.3, max(0.7, cards_per_game))
    except Exception:
        return default

def get_fixture_referee(fixture_id: int) -> Optional[Dict[str, Any]]:
    """Get referee information for a fixture"""
    try:
        # Try to get referee data from API
        api_data = get_fixture_data(fixture_id)
        if api_data and "referee" in api_data:
            return api_data["referee"]
        return {}  # Return empty dict instead of None
    except Exception as e:
        logger.error(f"Error getting referee data: {e}")
        return {}  # Return empty dict instead of None

def get_partial_prediction(home_team_id: int, away_team_id: int, league_id: int, season: int) -> Dict[str, Any]:
    """Generate partial prediction using available data when full prediction fails"""
    try:
        # Try to get basic team info
        home_stats = get_basic_team_stats(home_team_id, league_id) or {}
        away_stats = get_basic_team_stats(away_team_id, league_id) or {}
        
        # Use league averages as base
        league_avg = get_league_averages(league_id, season) or {
            'goals_per_game': 2.6,
            'corners_per_game': 9.8,
            'cards_per_game': 3.8,
            'fouls_per_game': 21.5
        }
        
        # Scale based on available team strength indicators
        home_strength = home_stats.get('attack_strength', 1.0)
        away_strength = away_stats.get('attack_strength', 1.0)
        
        predicted_home_goals = league_avg['goals_per_game'] * 0.55 * home_strength
        predicted_away_goals = league_avg['goals_per_game'] * 0.45 * away_strength
        
        return {
            'predicted_home_goals': round(predicted_home_goals, 2),
            'predicted_away_goals': round(predicted_away_goals, 2),
            'total_goals': round(predicted_home_goals + predicted_away_goals, 2),
            'prob_over_2_5': round(calculate_over_probability(predicted_home_goals + predicted_away_goals, 2.5, 1.2), 2),
            'prob_btts': round(min(0.95, predicted_home_goals * predicted_away_goals / 2), 2),
            'method': 'partial_data',
            'confidence': 0.4,  # Lower confidence for partial predictions
            'corners': generate_varied_corners({
                'home_attacking_strength': 1.0,
                'away_attacking_strength': 1.0,
                'league_corners_avg': league_avg['corners_per_game']
            }),
            'cards': generate_varied_cards(league_avg['cards_per_game']),
            'fouls': generate_varied_fouls(league_avg['fouls_per_game'])
        }
    except Exception as e:
        logger.error(f"Error generating partial prediction: {e}")
        return generate_minimum_prediction()

def get_league_averages(league_id: int, season: int) -> Dict[str, float]:
    """Get average statistics for a league"""
    try:
        # Try to get cached data first
        cache_key = f"league_averages_{league_id}_{season}"
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        # Get league statistics data from API or database
        league_stats = get_league_stats(league_id, season)
        
        # Set default values
        defaults = {
            "goals_per_game": 2.6,
            "corners_per_game": 9.8,
            "cards_per_game": 3.8,
            "fouls_per_game": 21.5,
            "home_goals_avg": 1.5,
            "away_goals_avg": 1.1,
            "home_win_pct": 0.45,
            "draw_pct": 0.25,
            "away_win_pct": 0.30
        }
        
        # Override defaults with actual values if available
        result = defaults.copy()
        if league_stats:
            for key, value in league_stats.items():
                if key in result and isinstance(value, (int, float)):
                    result[key] = value
        
        # Cache the results
        save_to_cache(cache_key, result, expire_hours=24)
        return result
        
    except Exception as e:
        logger.error(f"Error getting league averages: {e}")
        # Return reasonable defaults for football statistics
        return {
            "goals_per_game": 2.6,
            "corners_per_game": 9.8,
            "cards_per_game": 3.8,
            "fouls_per_game": 21.5,
            "home_goals_avg": 1.5,
            "away_goals_avg": 1.1
        }

def generate_minimum_prediction() -> Dict[str, Any]:
    """Generate minimal prediction with default values when no data is available"""
    return {
        "predicted_home_goals": 1.2,
        "predicted_away_goals": 1.0,
        "total_goals": 2.2,
        "prob_over_2_5": 0.45,
        "prob_btts": 0.40,
        "method": "minimal_data",
        "confidence": 0.35,
        "corners": {
            "total": 9.5,
            "home": 5.0,
            "away": 4.5,
            "over_8.5": 0.55,
            "over_9.5": 0.45,
            "over_10.5": 0.35
        },
        "cards": {
            "total": 3.5,
            "home": 1.8,
            "away": 1.7,
            "over_2.5": 0.60,
            "over_3.5": 0.40,
            "over_4.5": 0.25
        },
        "fouls": {
            "total": 20.5,
            "home": 10.5,
            "away": 10.0,
            "over_19.5": 0.55,
            "over_21.5": 0.40,
            "over_23.5": 0.25
        }
    }

def get_basic_team_stats(team_id: int, league_id: int) -> Dict[str, float]:
    """Get basic team statistics"""
    try:
        # Try to get cached data first
        cache_key = f"basic_stats_{team_id}_{league_id}"
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return cached_data

        # Get team form data
        form_data = get_team_form(team_id, league_id=league_id, last_matches=5)
        
        if not form_data:
            return {
                'attack_strength': 1.0,
                'defense_strength': 1.0,
                'avg_goals_scored': 1.2,
                'avg_goals_conceded': 1.1
            }

        # Calculate attack and defense strength
        games_played = max(1, form_data.get('matches_played', 1))
        goals_scored = form_data.get('goals_scored', 0)
        goals_conceded = form_data.get('goals_conceded', 0)

        stats = {
            'attack_strength': min(1.5, max(0.5, (goals_scored / games_played) / 1.2)),
            'defense_strength': min(1.5, max(0.5, 1.1 / (goals_conceded / games_played + 0.1))),
            'avg_goals_scored': goals_scored / games_played,
            'avg_goals_conceded': goals_conceded / games_played
        }

        # Cache the results
        save_to_cache(cache_key, stats)
        return stats
        
    except Exception as e:
        logger.error(f"Error getting basic team stats: {e}")
        return {
            'attack_strength': 1.0,
            'defense_strength': 1.0,
            'avg_goals_scored': 1.2,
            'avg_goals_conceded': 1.1
        }
