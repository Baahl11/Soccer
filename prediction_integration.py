# prediction_integration.py
"""
Módulo de integración entre la API de fútbol y el sistema de predicción.
Este módulo se encarga de:
1. Obtener datos necesarios para las predicciones desde varias fuentes
2. Preparar estos datos para los algoritmos de predicción
3. Integrar factores externos (clima, lesiones, etc.) en las predicciones
4. Validar coherencia entre diferentes componentes de predicción
5. Enriquecer análisis táctico con datos de cualquier liga

Las funciones de predicción reales están en predictions.py
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import requests
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import joblib
from player_injuries import InjuryAnalyzer
from data_validation import DataValidator
from data import get_fixture_data, get_fixture_players, get_fixture_statistics
from team_form import get_team_form, get_head_to_head_analysis
from weather_api import get_weather_forecast, get_weather_impact
from predictions import (
    calculate_statistical_prediction,
    extract_features_from_form,
    make_global_prediction,
    calculate_1x2_probabilities
)
from team_elo_rating import get_elo_ratings_for_match
from prediction_coherence_validator import CoherenceValidator
from advanced_1x2_system import Advanced1X2System

# Inicializar el sistema avanzado 1X2
advanced_1x2_system = Advanced1X2System()

from enhanced_tactical_analyzer import EnhancedTacticalAnalyzer

# Inicializar los componentes avanzados
coherence_validator = CoherenceValidator()
tactical_analyzer = EnhancedTacticalAnalyzer()

# Create cache directory if it doesn't exist
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StackingModel:
    """Modelo de stacking para predicciones que combina múltiples modelos."""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models = []
        self.scaler = None
        self.meta_model = None
        self.model_loaded = False
        
    def load_models(self):
        """Carga los modelos guardados desde el directorio especificado."""
        try:
            if not os.path.exists(self.models_dir):
                logger.warning(f"Directory {self.models_dir} does not exist")
                return False
                
            # Cargar scaler
            scaler_path = os.path.join(self.models_dir, "scaler.pkl")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                
            # Cargar modelo neural network
            nn_path = os.path.join(self.models_dir, "fnn_model.pkl")
            if os.path.exists(nn_path):
                try:
                    nn_model = joblib.load(nn_path)
                    self.models.append(("nn", nn_model))
                except Exception as e:
                    logger.warning(f"Could not load neural network model: {e}")
            
            # Cargar otros modelos disponibles
            for model_file in os.listdir(self.models_dir):
                if model_file.endswith(".pkl") and model_file not in ["scaler.pkl", "fnn_model.pkl"]:
                    try:
                        model_path = os.path.join(self.models_dir, model_file)
                        model = joblib.load(model_path)
                        model_name = model_file.replace(".pkl", "")
                        self.models.append((model_name, model))
                    except Exception as e:
                        logger.warning(f"Could not load model {model_file}: {e}")
            
            # Modelo meta para stacking (usa estadístico si no hay)
            meta_path = os.path.join(self.models_dir, "meta_model.pkl")
            if os.path.exists(meta_path):
                try:
                    self.meta_model = joblib.load(meta_path)
                except Exception as e:
                    logger.warning(f"Could not load meta model: {e}")
            
            logger.info(f"Loaded {len(self.models)} base models")
            self.model_loaded = len(self.models) > 0
            return self.model_loaded
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def fit(self, X, y):
        """Entrena el modelo de stacking con los datos proporcionados."""
        # Implementación simplificada para pruebas
        self.model_loaded = True
        return self
    
    def predict(self, X):
        """Realiza predicciones usando el modelo de stacking."""
        # Asegurarse de que los modelos estén cargados
        if not self.model_loaded and not self.load_models():
            # Usar valores por defecto si los modelos no están disponibles
            if isinstance(X, pd.DataFrame):
                return np.ones(len(X), dtype=int)
            else:
                return np.array([1], dtype=int)
        
        # Implementación simplificada para pruebas
        if isinstance(X, pd.DataFrame):
            return np.ones(len(X), dtype=int)
        else:
            return np.array([1], dtype=int)


def create_features(
    fixture_data: Dict[str, Any], 
    lineup_data: Dict[str, Any],
    stats_df: pd.DataFrame,
    odds_data: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Crea características para modelos predictivos a partir de datos del partido.
    
    Args:
        fixture_data: Datos del partido
        lineup_data: Datos de alineaciones
        stats_df: DataFrame con estadísticas históricas
        odds_data: Datos de cuotas (opcional)
        
    Returns:
        Diccionario con características relevantes
    """
    features = {}
    
    # Extraer información básica
    home_team_id = fixture_data.get('home_team_id', 0)
    away_team_id = fixture_data.get('away_team_id', 0)
    
    # Filtrar estadísticas para los equipos
    home_stats = stats_df[stats_df['home_team_id'] == home_team_id]
    away_stats = stats_df[stats_df['away_team_id'] == away_team_id]
    
    # Calcular medias de goles
    features['total_goals_home'] = home_stats['shots'].mean() * 0.1 if not home_stats.empty else 1.2
    features['total_goals_away'] = away_stats['shots'].mean() * 0.07 if not away_stats.empty else 0.8
    
    # Calcular medias de córneres
    features['home_corners'] = home_stats['corners'].mean() if not home_stats.empty else 5.0 
    features['away_corners'] = away_stats['corners'].mean() if not away_stats.empty else 3.5
    
    # Características de alineación
    features['home_formation_attacking'] = 1 if lineup_data.get('home_formation', '') in ['4-3-3', '3-4-3'] else 0
    features['away_formation_attacking'] = 1 if lineup_data.get('away_formation', '') in ['4-3-3', '3-4-3'] else 0
    features['wingers'] = lineup_data.get('wingers', 0)
    
    # Características de cuotas si están disponibles
    if odds_data:
        features['home_win_prob'] = 1.0 / odds_data.get('home_win_odds', 2.0)
        features['draw_prob'] = 1.0 / odds_data.get('draw_odds', 3.0)  
        features['away_win_prob'] = 1.0 / odds_data.get('away_win_odds', 4.0)
    else:
        features['home_win_prob'] = 0.5
        features['draw_prob'] = 0.25
        features['away_win_prob'] = 0.25
    
    return features


def prepare_data_for_prediction(fixture_id: int) -> Dict[str, Any]:
    """
    Prepara todos los datos necesarios para predecir un partido específico.
    
    Args:
        fixture_id: ID del partido
        
    Returns:
        Diccionario con datos preparados para predicción
    """
    try:
        # Validar fixture_id
        if not DataValidator.validate_fixture_id(fixture_id):
            logger.error(f"ID de partido inválido: {fixture_id}")
            return {}
            
        # Obtener datos básicos del partido
        fixture_data = get_fixture_data(fixture_id)
        if not fixture_data or 'response' not in fixture_data or not fixture_data['response']:
            logger.error(f"No se encontraron datos para el partido {fixture_id}")
            return {}
            
        # Validar respuesta de la API
        is_valid, error_msg = DataValidator.validate_response(fixture_data)
        if not is_valid:
            logger.error(f"Datos de partido inválidos: {error_msg}")
            return {}
            
        # Extraer información básica
        match_info = fixture_data['response'][0]
        teams = match_info.get('teams', {})
        home_team_id = teams.get('home', {}).get('id')
        away_team_id = teams.get('away', {}).get('id')
        league_info = match_info.get('league', {})
        league_id = league_info.get('id')
        season = league_info.get('season')
        
        if not (home_team_id and away_team_id and league_id and season):
            logger.error(f"Información incompleta para el partido {fixture_id}")
            return {}
            
        # Obtener datos del clima, localización y fecha
        fixture_details = match_info.get('fixture', {})
        venue = fixture_details.get('venue', {})
        city = venue.get('city', '')
        country = venue.get('country', '')
        match_date = fixture_details.get('date', '')
        
        # Obtener datos de alineaciones y jugadores
        player_data = get_fixture_players(fixture_id)
        lineup_data = {
            'home_players': [],
            'away_players': []
        }
        
        if player_data and 'response' in player_data:
            for team_players in player_data['response']:
                team_id = team_players.get('team', {}).get('id')
                if team_id == home_team_id:
                    lineup_data['home_players'] = team_players.get('players', [])
                elif team_id == away_team_id:
                    lineup_data['away_players'] = team_players.get('players', [])
        
        # Obtener datos de forma de equipos
        home_form = get_team_form(home_team_id, league_id, season)
        away_form = get_team_form(away_team_id, league_id, season)
        h2h_data = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Obtener datos de lesiones
        injury_analyzer = InjuryAnalyzer()
        home_injuries = injury_analyzer.get_team_injuries(home_team_id)
        away_injuries = injury_analyzer.get_team_injuries(away_team_id)
        
        # Obtener estadísticas adicionales
        fixture_stats = get_fixture_statistics(fixture_id)
        
        # Preparar objeto de datos integrados
        integrated_data = {
            'fixture_id': fixture_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': teams.get('home', {}).get('name', ''),
            'away_team_name': teams.get('away', {}).get('name', ''),
            'league_id': league_id,
            'season': season,
            'match_date': match_date,
            'venue': {
                'city': city,
                'country': country
            },
            'home_form': home_form,
            'away_form': away_form,
            'head_to_head': h2h_data,
            'home_injuries': home_injuries,
            'away_injuries': away_injuries,
            'lineups': lineup_data,
            'statistics': fixture_stats.to_dict(orient='records') if hasattr(fixture_stats, 'to_dict') else []
        }
        
        return integrated_data
        
    except Exception as e:
        logger.error(f"Error preparando datos para predicción: {e}")
        return {}

def make_advanced_1x2_prediction(fixture_id: int) -> Dict[str, Any]:
    """
    Realiza una predicción avanzada 1X2 usando el sistema avanzado.
    
    Args:
        fixture_id: ID del partido
        
    Returns:
        Predicción avanzada 1X2
    """
    try:
        # Preparar datos integrados
        integrated_data = prepare_data_for_prediction(fixture_id)
        if not integrated_data:
            logger.error(f"No se pudieron preparar datos para el partido {fixture_id}")
            return {}
        
        home_team_id = integrated_data.get('home_team_id')
        away_team_id = integrated_data.get('away_team_id')
        league_id = integrated_data.get('league_id')
        
        # Validar que los IDs no sean None y convertir a int si es necesario
        if home_team_id is None or away_team_id is None or league_id is None:
            logger.error(f"IDs de equipo o liga faltantes para el partido {fixture_id}")
            return {}
        
        home_team_id = int(home_team_id)
        away_team_id = int(away_team_id)
        league_id = int(league_id)
        
        # Usar el sistema avanzado para predecir
        prediction_result = advanced_1x2_system.predict_match_advanced(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            use_calibration=True,
            context_data=integrated_data
        )
        
        return prediction_result
        
    except Exception as e:
        logger.error(f"Error en predicción avanzada 1X2: {e}")
        return {}

# Removed duplicate definition of make_integrated_prediction to resolve obscured function declaration error.

def enrich_prediction_with_contextual_data(
    prediction: Dict[str, Any], 
    weather_data: Optional[Dict[str, Any]] = None,
    player_data: Optional[Dict[str, Any]] = None,
    odds_data: Optional[Dict[str, Any]] = None,
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    league_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Enriquece una predicción básica con datos contextuales como clima, lesiones, cuotas y Elo ratings.
    
    Args:
        prediction: Predicción base
        weather_data: Datos del clima
        player_data: Datos de jugadores y lesiones
        odds_data: Datos de cuotas
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        league_id: ID de la liga
        
    Returns:
        Predicción enriquecida
    """
    try:
        enriched = prediction.copy()
        
        # Ajustar por clima
        if weather_data:
            # Factores de ajuste basados en condiciones climáticas
            condition = weather_data.get('condition', '').lower()
            intensity = weather_data.get('intensity', '').lower()
            
            # Ajustes basados en condiciones específicas
            if condition == 'rain' and intensity in ['moderate', 'heavy']:
                # La lluvia tiende a reducir goles
                enriched['predicted_home_goals'] = max(0, enriched['predicted_home_goals'] * 0.9)
                enriched['predicted_away_goals'] = max(0, enriched['predicted_away_goals'] * 0.9)
                enriched['weather_adjustment'] = 'reduced_goals_rain'
                
            elif condition == 'snow':
                # La nieve reduce goles significativamente
                enriched['predicted_home_goals'] = max(0, enriched['predicted_home_goals'] * 0.8)
                enriched['predicted_away_goals'] = max(0, enriched['predicted_away_goals'] * 0.8)
                enriched['weather_adjustment'] = 'reduced_goals_snow'
                
            elif condition == 'wind' and intensity in ['strong', 'severe']:
                # Viento fuerte afecta la precisión
                enriched['predicted_home_goals'] = max(0, enriched['predicted_home_goals'] * 0.85)
                enriched['predicted_away_goals'] = max(0, enriched['predicted_away_goals'] * 0.85)
                enriched['weather_adjustment'] = 'reduced_accuracy_wind'
                
            elif condition == 'hot' and intensity in ['extreme']:
                # Calor extremo reduce ritmo de juego
                enriched['predicted_home_goals'] = max(0, enriched['predicted_home_goals'] * 0.95)
                enriched['predicted_away_goals'] = max(0, enriched['predicted_away_goals'] * 0.95)
                enriched['weather_adjustment'] = 'reduced_pace_heat'
                
            # Actualizar total de goles y probabilidades
            enriched['total_goals'] = enriched['predicted_home_goals'] + enriched['predicted_away_goals']
            
        # Ajustar por lesiones y suspensiones
        if player_data:
            # Algoritmo simplificado de ajuste por ausencias clave
            home_injuries = player_data.get('home_injuries', [])
            away_injuries = player_data.get('away_injuries', [])
            
            # Calcular impacto de lesiones
            home_key_players_out = sum(1 for player in home_injuries if player.get('importance', '') == 'high')
            away_key_players_out = sum(1 for player in away_injuries if player.get('importance', '') == 'high')
            
            # Ajustar predicciones basadas en ausencias
            if home_key_players_out > 0:
                factor = max(0.8, 1 - (home_key_players_out * 0.05))
                enriched['predicted_home_goals'] = max(0, enriched['predicted_home_goals'] * factor)
                enriched['injuries_adjustment_home'] = f'reduced_strength_{factor:.2f}'
            
            if away_key_players_out > 0:
                factor = max(0.8, 1 - (away_key_players_out * 0.05))
                enriched['predicted_away_goals'] = max(0, enriched['predicted_away_goals'] * factor)
                enriched['injuries_adjustment_away'] = f'reduced_strength_{factor:.2f}'
            
            # Actualizar total de goles
            enriched['total_goals'] = enriched['predicted_home_goals'] + enriched['predicted_away_goals']
        
        # Recalcular probabilidades de over/under con los valores ajustados
        lambda_home = enriched['predicted_home_goals']
        lambda_away = enriched['predicted_away_goals']
        
        # Probability for over 2.5 goals
        enriched['prob_over_2_5'] = 1 - calculate_poisson_probability(lambda_home, lambda_away, 2.5)
        
        # Probability for both teams to score
        enriched['prob_btts'] = (1 - np.exp(-lambda_home)) * (1 - np.exp(-lambda_away))
          # Probabilidades de 1X2 recalculadas
        probs = calculate_1x2_probabilities(lambda_home, lambda_away)
        
        # Convert tuple return value to a properly typed dictionary
        if isinstance(probs, tuple) and len(probs) == 3:
            home_win_val, draw_val, away_win_val = probs
            probs_dict = {
                'home_win': float(home_win_val),
                'draw': float(draw_val),
                'away_win': float(away_win_val)
            }
        elif isinstance(probs, dict):
            probs_dict = probs
        else:
            # Fallback with default values if unexpected format
            probs_dict = {
                'home_win': 0.0,
                'draw': 0.0,
                'away_win': 0.0
            }
            
        enriched['prob_1'] = probs_dict.get('home_win', 0.0)
        enriched['prob_X'] = probs_dict.get('draw', 0.0)
        enriched['prob_2'] = probs_dict.get('away_win', 0.0)
          # Añadir metadatos sobre ajustes
        enriched['adjustments_applied'] = []
        if weather_data and 'weather_adjustment' in enriched:
            enriched['adjustments_applied'].append('weather')
        if player_data and ('injuries_adjustment_home' in enriched or 'injuries_adjustment_away' in enriched):
            enriched['adjustments_applied'].append('injuries')
        if odds_data:
            enriched['adjustments_applied'].append('odds')
          # Agregar datos de Elo rating si los IDs de los equipos están disponibles
        if home_team_id is not None and away_team_id is not None:
            try:                # We'll implement the ELO enhancement directly here to avoid import issues
                # Internal functions for ELO enhancement
                def _get_key_elo_insights(elo_data):
                    """Internal function to get human-readable ELO insights"""
                    insights = {}
                    
                    # Team strength comparison
                    elo_diff = elo_data.get('elo_diff', 0)
                    if abs(elo_diff) < 25:
                        insights['team_comparison'] = "Teams are very evenly matched"
                    elif abs(elo_diff) < 75:
                        stronger = "Home team" if elo_diff > 0 else "Away team"
                        insights['team_comparison'] = f"{stronger} has a slight advantage"
                    elif abs(elo_diff) < 150:
                        stronger = "Home team" if elo_diff > 0 else "Away team"
                        insights['team_comparison'] = f"{stronger} has a clear advantage"
                    else:
                        stronger = "Home team" if elo_diff > 0 else "Away team"
                        insights['team_comparison'] = f"{stronger} is significantly stronger"
                    
                    # Draw likelihood
                    draw_prob = elo_data.get('elo_draw_probability', 0)
                    if draw_prob > 0.33:
                        insights['draw_likelihood'] = "High draw probability"
                    elif draw_prob > 0.25:
                        insights['draw_likelihood'] = "Above average draw probability"
                    else:
                        insights['draw_likelihood'] = "Low draw probability"
                    
                    # Expected goal difference
                    exp_goal_diff = elo_data.get('elo_expected_goal_diff', 0)
                    if abs(exp_goal_diff) < 0.3:
                        insights['goal_expectation'] = "Expect a tight, low-scoring game"
                    elif abs(exp_goal_diff) < 0.7:
                        stronger = "Home team" if exp_goal_diff > 0 else "Away team"
                        insights['goal_expectation'] = f"{stronger} expected to score slightly more"
                    else:
                        stronger = "Home team" if exp_goal_diff > 0 else "Away team"
                        insights['goal_expectation'] = f"{stronger} likely to dominate scoring"
                    
                    return insights
                
                def _enhance_prediction_with_elo(prediction, elo_data):
                    """Internal function to enhance prediction with ELO metrics"""
                    enhanced = prediction.copy()
                    
                    # Add ELO-enhanced metrics
                    elo_diff = elo_data.get('elo_diff', 0)
                    
                    # 1. Expected margin of victory
                    elo_margin = abs(elo_data.get('elo_expected_goal_diff', 0))
                    if elo_margin < 0.5:
                        margin_category = "Very close match"
                    elif elo_margin < 1.0:
                        margin_category = "Narrow margin expected"
                    elif elo_margin < 2.0:
                        margin_category = "Comfortable margin likely"
                    else:
                        margin_category = "Potential one-sided match"
                    
                    # 2. Match competitiveness rating (1-10 scale)
                    if abs(elo_diff) < 50:
                        competitiveness = 9
                    elif abs(elo_diff) < 100:
                        competitiveness = 8
                    elif abs(elo_diff) < 150:
                        competitiveness = 7
                    elif abs(elo_diff) < 200:
                        competitiveness = 5
                    elif abs(elo_diff) < 300:
                        competitiveness = 3
                    else:
                        competitiveness = 1
                    
                    # Enhanced metrics
                    enhanced['elo_enhanced_metrics'] = {
                        'expected_margin': elo_margin,
                        'margin_category': margin_category,
                        'competitiveness_rating': competitiveness
                    }
                    
                    return enhanced
                
                def _blend_predictions_with_elo(prediction, elo_data, blend_weight=0.3):
                    """Internal function to blend model predictions with ELO"""
                    blended = prediction.copy()
                    
                    # Extract model probabilities
                    model_home_win = prediction.get('prob_1', 0.33)
                    model_draw = prediction.get('prob_X', 0.33)
                    model_away_win = prediction.get('prob_2', 0.33)
                    
                    # Extract ELO probabilities
                    elo_home_win = elo_data.get('elo_win_probability', 0.33)
                    elo_draw = elo_data.get('elo_draw_probability', 0.33)
                    elo_away_win = elo_data.get('elo_loss_probability', 0.33)
                    
                    # Blend probabilities
                    blended_home = (1 - blend_weight) * model_home_win + blend_weight * elo_home_win
                    blended_draw = (1 - blend_weight) * model_draw + blend_weight * elo_draw
                    blended_away = (1 - blend_weight) * model_away_win + blend_weight * elo_away_win
                    
                    # Normalize
                    total = blended_home + blended_draw + blended_away
                    blended_home /= total
                    blended_draw /= total
                    blended_away /= total
                    
                    # Update with blended probabilities
                    blended['blended_probabilities'] = {
                        'home_win': round(blended_home, 3),
                        'draw': round(blended_draw, 3),
                        'away_win': round(blended_away, 3),
                        'blend_weight': {
                            'model': round(1 - blend_weight, 2),
                            'elo': round(blend_weight, 2)
                        }
                    }
                    
                    # Update main probabilities
                    blended['prob_1'] = blended_home
                    blended['prob_X'] = blended_draw
                    blended['prob_2'] = blended_away
                    
                    return blended
                
                # Get base ELO data first
                elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
                
                # Add basic ELO data to prediction
                enriched['elo_ratings'] = {
                    'home_elo': elo_data['home_elo'],
                    'away_elo': elo_data['away_elo'],
                    'elo_diff': elo_data['elo_diff']
                }
                
                # Add ELO-based probabilities
                enriched['elo_probabilities'] = {
                    'win': elo_data['elo_win_probability'],
                    'draw': elo_data['elo_draw_probability'],
                    'loss': elo_data['elo_loss_probability']
                }
                
                # Add expected goal difference
                if 'elo_expected_goal_diff' in elo_data:
                    enriched['elo_expected_goal_diff'] = elo_data['elo_expected_goal_diff']
                  # Get human-readable insights
                enriched['elo_insights'] = _get_key_elo_insights(elo_data)
                  # Apply advanced ELO enhancements
                enriched = _enhance_prediction_with_elo(enriched, elo_data)
                
                # Blend statistical model with ELO predictions                # Use different blend weights based on the quality of data
                blend_weight = 0.3  # Default - 30% ELO, 70% model
                
                # If confidence is present, adjust blend weight
                if 'confidence' in prediction:
                    base_confidence = prediction['confidence']
                    # Higher model confidence = lower ELO weight
                    blend_weight = max(0.15, min(0.5, 0.45 - (base_confidence * 0.3)))
                
                enriched = _blend_predictions_with_elo(
                    enriched,
                    elo_data,
                    blend_weight=blend_weight
                )
                
                # Adjust confidence based on ELO data
                if 'confidence' in prediction:
                    base_confidence = prediction['confidence']
                    elo_diff = elo_data['elo_diff']
                    
                    # More ELO difference = higher confidence adjustment (up to 0.2)
                    elo_confidence_factor = min(0.2, abs(elo_diff) / 400)
                    
                    # If prediction agrees with ELO advantage, increase confidence
                    prediction_direction = prediction.get('prediction', '')
                    elo_favors_home = elo_diff > 0
                    
                    if (prediction_direction == 'Home' and elo_favors_home) or \
                       (prediction_direction == 'Away' and not elo_favors_home):
                        # Increase confidence when ELO and prediction agree
                        enhanced_confidence = min(0.95, base_confidence + elo_confidence_factor)
                    else:
                        # Decrease confidence when they disagree
                        enhanced_confidence = max(0.4, base_confidence - (elo_confidence_factor / 2))
                    
                    # Add enhanced confidence metrics
                    enriched['enhanced_confidence'] = {
                        'score': round(enhanced_confidence, 2),
                        'factors': {
                            'base': round(base_confidence, 2),
                            'elo_adjustment': round(enhanced_confidence - base_confidence, 2)
                        }
                    }
                
                # Mark that ELO adjustments have been applied
                if 'adjustments_applied' not in enriched:
                    enriched['adjustments_applied'] = []
                if 'elo' not in enriched['adjustments_applied']:
                    enriched['adjustments_applied'].append('elo')
                
                logger.info(f"Enhanced prediction with ELO data: Home {elo_data['home_elo']} vs Away {elo_data['away_elo']}")
            except Exception as e:
                logger.warning(f"Could not enhance prediction with ELO data: {e}")
        
        return enriched
        
    except Exception as e:
        logger.error(f"Error enriqueciendo predicción: {e}")
        return prediction  # Devolver predicción original si hay error

def calculate_poisson_probability(lambda_home: float, lambda_away: float, goal_threshold: float) -> float:
    """
    Calcula la probabilidad acumulada de Poisson para un umbral de goles.
    
    Args:
        lambda_home: Media de goles del equipo local
        lambda_away: Media de goles del equipo visitante
        goal_threshold: Umbral de goles (por ejemplo, 2.5)
        
    Returns:
        Probabilidad de que los goles totales sean <= threshold
    """
    import scipy.stats as stats
    
    # Distribucion de Poisson para goles totales
    lambda_total = lambda_home + lambda_away
    
    # Probabilidad acumulada P(X ≤ k)
    prob = stats.poisson.cdf(int(goal_threshold), lambda_total)
    
    # Asegurarse de devolver un float, no un array
    return float(prob)


def make_mock_integrated_prediction(fixture_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Creates an integrated prediction for synthetic/mock fixtures (like those from ELO workflow).
    
    Args:
        fixture_data: Synthetic fixture data containing team IDs and match info
        
    Returns:
        Integrated prediction using mock data sources
    """
    try:
        home_team_id = fixture_data.get('home_team_id', 0)
        away_team_id = fixture_data.get('away_team_id', 0)
        home_team_name = fixture_data.get('home_team_name', f'Team {home_team_id}')
        away_team_name = fixture_data.get('away_team_name', f'Team {away_team_id}')
        league_id = fixture_data.get('league_id', 39)
        
        # Create mock team form data
        mock_home_form = {
            'team_id': home_team_id,
            'recent_goals_scored': np.random.uniform(1.0, 2.5),
            'recent_goals_conceded': np.random.uniform(0.8, 2.0),
            'form_points': np.random.uniform(1.2, 2.8),
            'home_advantage': np.random.uniform(0.1, 0.4),
            'recent_stats': {
                'wins': np.random.randint(2, 8),
                'draws': np.random.randint(1, 4),
                'losses': np.random.randint(0, 5),
                'goals_for': np.random.randint(8, 25),
                'goals_against': np.random.randint(3, 20)
            }
        }
        
        mock_away_form = {
            'team_id': away_team_id,
            'recent_goals_scored': np.random.uniform(0.8, 2.2),
            'recent_goals_conceded': np.random.uniform(1.0, 2.3),
            'form_points': np.random.uniform(1.0, 2.5),
            'away_penalty': np.random.uniform(0.1, 0.3),
            'recent_stats': {
                'wins': np.random.randint(1, 7),
                'draws': np.random.randint(1, 5),
                'losses': np.random.randint(1, 6),
                'goals_for': np.random.randint(6, 22),
                'goals_against': np.random.randint(5, 25)
            }
        }
        
        # Create mock head-to-head data
        mock_h2h = {
            'total_matches': np.random.randint(3, 15),
            'home_wins': np.random.randint(1, 8),
            'draws': np.random.randint(0, 4),
            'away_wins': np.random.randint(1, 6),
            'avg_goals_home': np.random.uniform(1.0, 2.5),
            'avg_goals_away': np.random.uniform(0.8, 2.2),
            'recent_trend': np.random.choice(['home_dominant', 'away_dominant', 'balanced'])
        }
        
        # Make statistical prediction using mock data
        raw_prediction = calculate_statistical_prediction(
            mock_home_form, mock_away_form, mock_h2h, home_team_id, away_team_id
        )
        
        # Convert to standard format
        if isinstance(raw_prediction, tuple) and len(raw_prediction) >= 2:
            prediction = {
                'predicted_home_goals': float(raw_prediction[0]),
                'predicted_away_goals': float(raw_prediction[1]),
                'total_goals': float(raw_prediction[0]) + float(raw_prediction[1]),
                'method': 'statistical_mock'
            }
        else:
            prediction = raw_prediction
            if isinstance(prediction, dict):
                prediction['method'] = 'statistical_mock'
        
        # Add mock contextual data
        mock_weather_data = {
            'condition': np.random.choice(['clear', 'cloudy', 'light_rain', 'moderate_rain']),
            'temperature': np.random.uniform(10, 25),
            'wind_speed': np.random.uniform(5, 20),
            'intensity': np.random.choice(['light', 'moderate'])
        }
        
        mock_player_data = {
            'home_injuries': [],
            'away_injuries': [],
            'home_lineup': [],
            'away_lineup': []
        }
        
        # Enrich with contextual data using the same function as real predictions
        enriched_prediction = enrich_prediction_with_contextual_data(
            prediction,
            weather_data=mock_weather_data,
            player_data=mock_player_data,
            odds_data=None,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id
        )
        
        # Add metadata for mock prediction
        enriched_prediction.update({
            'fixture_id': fixture_data.get('fixture_id', 0),
            'home_team': home_team_name,
            'away_team': away_team_name,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'league_id': league_id,
            'season': fixture_data.get('season', 2024),
            'date': fixture_data.get('match_date', datetime.now().isoformat()),
            'prediction_timestamp': datetime.now().isoformat(),
            'data_source': 'mock',
            'mock_data_used': True
        })
        
        # Add basic tactical analysis for mock data
        try:
            tactical_profiles = {
                'home': {
                    'team_id': home_team_id,
                    'team_name': home_team_name,
                    'attacking_style': np.random.choice(['possession', 'counter_attack', 'direct']),
                    'defensive_style': np.random.choice(['high_press', 'mid_block', 'deep_defense']),
                    'formation_preference': np.random.choice(['4-3-3', '4-4-2', '3-5-2', '4-2-3-1']),
                    'strengths': np.random.choice([['pace'], ['physicality'], ['technique'], ['set_pieces']]),
                    'weaknesses': np.random.choice([['defending'], ['finishing'], ['creativity']])
                },
                'away': {
                    'team_id': away_team_id,
                    'team_name': away_team_name,
                    'attacking_style': np.random.choice(['possession', 'counter_attack', 'direct']),
                    'defensive_style': np.random.choice(['high_press', 'mid_block', 'deep_defense']),
                    'formation_preference': np.random.choice(['4-3-3', '4-4-2', '3-5-2', '4-2-3-1']),
                    'strengths': np.random.choice([['pace'], ['physicality'], ['technique'], ['set_pieces']]),
                    'weaknesses': np.random.choice([['defending'], ['finishing'], ['creativity']])
                }
            }
            
            enriched_prediction['tactical_analysis'] = {
                'home_team': tactical_profiles['home'],
                'away_team': tactical_profiles['away'],
                'matchup_analysis': {
                    'key_battle': f"{tactical_profiles['home']['attacking_style']} vs {tactical_profiles['away']['defensive_style']}",
                    'tactical_advantage': np.random.choice(['home', 'away', 'neutral']),
                    'expected_style': np.random.choice(['high_scoring', 'tactical', 'physical'])
                }
            }
        except Exception as e:
            logger.warning(f"Error adding mock tactical analysis: {e}")
        
        logger.info(f"Generated mock integrated prediction for {home_team_name} vs {away_team_name}")
        return enriched_prediction
        
    except Exception as e:
        logger.error(f"Error creating mock integrated prediction: {e}")
        return {}

def is_synthetic_fixture(fixture_id: int) -> bool:
    """
    Determines if a fixture ID represents a synthetic/mock fixture.
    
    Args:
        fixture_id: Fixture ID to check
        
    Returns:
        True if synthetic, False if real
    """
    # Synthetic fixtures from ELO workflow start at 1000000
    return fixture_id >= 1000000

def make_integrated_prediction(fixture_id: int, fixture_data: Optional[Dict[str, Any]] = None, use_advanced_1x2: bool = False) -> Dict[str, Any]:
    """
    Realiza una predicción integrada para un partido, usando datos de múltiples fuentes.
    
    Args:
        fixture_id: ID del partido
        fixture_data: Datos del partido (opcional, para casos de prueba)
        use_advanced_1x2: Flag para usar el sistema avanzado 1X2
        
    Returns:
        Predicción integrada
    """
    try:
        # Para fixtures sintéticos, usar predicción mock
        if is_synthetic_fixture(fixture_id) and fixture_data is not None:
            logger.info(f"Processing synthetic fixture {fixture_id} with mock data")
            return make_mock_integrated_prediction(fixture_data)
        
        # Lógica normal de predicción integrada
        integrated_data = prepare_data_for_prediction(fixture_id)
        if not integrated_data:
            logger.error(f"No se pudieron preparar datos para el partido {fixture_id}")
            return {}
        
        # Si se usa el sistema avanzado 1X2, obtener predicción avanzada
        advanced_prediction = None
        if use_advanced_1x2:
            try:
                home_team_id = integrated_data.get('home_team_id')
                away_team_id = integrated_data.get('away_team_id')
                league_id = integrated_data.get('league_id')
                # Validar y convertir a int para evitar errores de tipo
                if home_team_id is None or away_team_id is None or league_id is None:
                    raise ValueError("Faltan IDs de equipo o liga para predicción avanzada 1X2")
                home_team_id = int(home_team_id)
                away_team_id = int(away_team_id)
                league_id = int(league_id)
                advanced_prediction = advanced_1x2_system.predict_match_advanced(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    league_id=league_id,
                    use_calibration=True,
                    context_data=integrated_data
                )
                logger.info(f"Advanced 1X2 prediction obtained for fixture {fixture_id}")
            except Exception as e:
                logger.warning(f"Error obteniendo predicción avanzada 1X2: {e}")
                advanced_prediction = None
        
        # Hacer predicción base
        home_form = integrated_data.get('home_form', {})
        away_form = integrated_data.get('away_form', {})
        h2h = integrated_data.get('head_to_head', {})
        
        # Añadir IDs de equipos a los datos de forma si no existen
        if home_form and 'team_id' not in home_form:
            home_form['team_id'] = integrated_data.get('home_team_id')
        if away_form and 'team_id' not in away_form:
            away_form['team_id'] = integrated_data.get('away_team_id')
        
        # Usar la clase global_prediction si los datos son suficientes
        try:
            base_prediction = make_global_prediction(fixture_id, weather_data=None, odds_data=None)
        except Exception as e:
            logger.warning(f"Error usando make_global_prediction: {e}. Usando método estadístico.")
            # Ensure team IDs are valid integers, use defaults if missing
            home_team_id = integrated_data.get('home_team_id')
            away_team_id = integrated_data.get('away_team_id')
            
            if home_team_id is None or away_team_id is None:
                logger.warning("Team IDs missing, using default values")
                home_team_id = 0 if home_team_id is None else home_team_id
                away_team_id = 0 if away_team_id is None else away_team_id
                
            # Get statistical prediction
            raw_prediction = calculate_statistical_prediction(home_form, away_form, h2h, home_team_id, away_team_id)
            
            # Convert tuple to dictionary if needed
            if isinstance(raw_prediction, tuple) and len(raw_prediction) >= 2:
                base_prediction = {
                    'predicted_home_goals': float(raw_prediction[0]),
                    'predicted_away_goals': float(raw_prediction[1]),
                    'total_goals': float(raw_prediction[0]) + float(raw_prediction[1]),
                    'method': 'statistical'
                }
            else:
                base_prediction = raw_prediction  # Already a dictionary
        
        # Si hay predicción avanzada, combinar con la base
        if advanced_prediction:
            # Ejemplo simple: blend probabilidades 1X2 con peso configurable
            blend_weight = 0.5  # Peso para la predicción avanzada
            base_probs = {
                'home_win': base_prediction.get('prob_1', 0.33),
                'draw': base_prediction.get('prob_X', 0.33),
                'away_win': base_prediction.get('prob_2', 0.33)
            }
            adv_probs = advanced_prediction.get('probabilities', {
                'home_win': 0.33,
                'draw': 0.33,
                'away_win': 0.33
            })
            
            blended_probs = {
                'home_win': blend_weight * adv_probs.get('home_win', 0) + (1 - blend_weight) * base_probs.get('home_win', 0),
                'draw': blend_weight * adv_probs.get('draw', 0) + (1 - blend_weight) * base_probs.get('draw', 0),
                'away_win': blend_weight * adv_probs.get('away_win', 0) + (1 - blend_weight) * base_probs.get('away_win', 0)
            }
            
            # Normalizar probabilidades
            total_prob = sum(blended_probs.values())
            if total_prob > 0:
                blended_probs = {k: v / total_prob for k, v in blended_probs.items()}
            
            # Actualizar base_prediction con probabilidades mezcladas
            base_prediction['prob_1'] = blended_probs['home_win']
            base_prediction['prob_X'] = blended_probs['draw']
            base_prediction['prob_2'] = blended_probs['away_win']
            
            # Añadir metadatos de integración
            base_prediction['advanced_1x2_used'] = True
            base_prediction['advanced_1x2_prediction'] = advanced_prediction
        else:
            base_prediction['advanced_1x2_used'] = False
        
        # Enriquecer con datos contextuales - Utilizar el nuevo método de clima preciso por ciudad
        weather_data = None
        if integrated_data.get('venue', {}).get('city'):
            try:
                city = integrated_data['venue']['city']
                country = integrated_data['venue']['country']
                match_date = integrated_data['match_date']
                venue_name = integrated_data.get('venue', {}).get('name')
                # Usar la nueva función mejorada para obtener datos meteorológicos precisos por ubicación
                weather_data = get_weather_forecast(city, country, match_date)
            except Exception as e:
                logger.warning(f"Error obteniendo datos de clima precisos: {e}")
                # Intentar con el método de respaldo
                try:
                    weather_data = get_weather_forecast(city, country, match_date)
                except Exception as e2:
                    logger.warning(f"Error en método de respaldo para clima: {e2}")
        
        # Datos de jugadores e incluir lesiones
        player_data = {
            'home_injuries': integrated_data.get('home_injuries', []),
            'away_injuries': integrated_data.get('away_injuries', []),
            'home_lineup': integrated_data.get('lineups', {}).get('home_players', []),
            'away_lineup': integrated_data.get('lineups', {}).get('away_players', [])
        }
          
        # Obtener home y away team IDs e información de la liga
        home_team_id = integrated_data.get('home_team_id')
        away_team_id = integrated_data.get('away_team_id')
        league_id = integrated_data.get('league_id')
        home_team_name = integrated_data.get('home_team_name', '')
        away_team_name = integrated_data.get('away_team_name', '')
          # Obtener perfiles tácticos usando el nuevo analizador táctico avanzado
        tactical_profiles = {
            'home': {},
            'away': {}
        }
        
        try:
            # Obtener datos recientes para análisis táctico
            home_formation = integrated_data.get('lineups', {}).get('home_formation', None)
            away_formation = integrated_data.get('lineups', {}).get('away_formation', None)
            
            # Obtener datos de estadísticas recientes si están disponibles
            home_recent_stats = home_form.get('recent_stats', None) if home_form else None
            away_recent_stats = away_form.get('recent_stats', None) if away_form else None
            
            # Para cualquier liga, generar perfiles tácticos
            if home_team_id and home_team_name:
                tactical_profiles['home'] = tactical_analyzer.get_team_tactical_profile(
                    team_id=home_team_id,
                    team_name=home_team_name,
                    recent_formations=[home_formation] if home_formation else None,
                    recent_stats=home_recent_stats
                )
            
            if away_team_id and away_team_name:
                tactical_profiles['away'] = tactical_analyzer.get_team_tactical_profile(
                    team_id=away_team_id,
                    team_name=away_team_name,
                    recent_formations=[away_formation] if away_formation else None,
                    recent_stats=away_recent_stats
                )
                
            logger.info(f"Perfiles tácticos generados para {home_team_name} y {away_team_name}")
        except Exception as e:
            logger.warning(f"Error al generar perfiles tácticos: {e}")
        
        # Enriquecer predicción con datos contextuales
        enriched_prediction = enrich_prediction_with_contextual_data(
            base_prediction,
            weather_data=weather_data,
            player_data=player_data,
            odds_data=None,  # No incluimos odds en esta versión
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id
        )
        
        # Añadir los perfiles tácticos a la predicción
        if tactical_profiles['home'] or tactical_profiles['away']:
            enriched_prediction['tactical_analysis'] = {
                'home_team': tactical_profiles['home'],
                'away_team': tactical_profiles['away'],
                'matchup_analysis': tactical_analyzer.analyze_tactical_matchup(
                    tactical_profiles['home'], 
                    tactical_profiles['away']
                ) if tactical_profiles['home'] and tactical_profiles['away'] else None
            }
              # Obtener ratings Elo para el partido
        try:
            elo_ratings = get_elo_ratings_for_match(home_team_id, away_team_id)
            
            # Validar coherencia entre ratings Elo y predicciones de goles usando el nuevo validador
            if elo_ratings and isinstance(enriched_prediction, dict):
                # Primero validar, luego ajustar si es necesario
                is_coherent = coherence_validator.is_prediction_coherent_with_elo(
                    enriched_prediction, elo_ratings
                )
                
                if not is_coherent:
                    logger.warning(f"Predicción incoherente con Elo para partido {fixture_id}. Ajustando...")
                    enriched_prediction = coherence_validator.validate_and_adjust_goal_predictions(
                        enriched_prediction, elo_ratings, validate_only=False
                    )
                    enriched_prediction['adjusted_for_coherence'] = True
                else:
                    enriched_prediction['adjusted_for_coherence'] = False
                    
                enriched_prediction['elo_ratings'] = elo_ratings
                
        except Exception as e:
            logger.warning(f"Error al validar coherencia con ratings Elo: {e}")
        
        # Añadir metadatos
        enriched_prediction['fixture_id'] = fixture_id
        enriched_prediction['home_team'] = home_team_name
        enriched_prediction['away_team'] = away_team_name
        enriched_prediction['league_id'] = league_id
        enriched_prediction['season'] = integrated_data.get('season', 0)
        enriched_prediction['date'] = integrated_data.get('match_date', '')
        enriched_prediction['prediction_timestamp'] = datetime.now().isoformat()
        
        return enriched_prediction
        
    except Exception as e:
        logger.error(f"Error haciendo predicción integrada: {e}")
        return {}
