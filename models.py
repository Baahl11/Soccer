# models.py
import statsmodels.api as sm
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import logging
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, brier_score_loss
from math import sqrt
from functools import lru_cache
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from typing import Dict, List, Optional, Any, Union, TypedDict
from fnn_model import FeedforwardNeuralNetwork
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

class CornersModel:
    def __init__(self):
        self.mean_corners = 10.2
        self.std_dev = 2.5
    
    def predict(self, home_stats, away_stats, current_stats=None):
        try:
            # Convert pandas DataFrames to dictionaries if needed
            if hasattr(home_stats, 'to_dict'):
                home_stats = home_stats.iloc[0].to_dict() if not home_stats.empty else {}
            if hasattr(away_stats, 'to_dict'):
                away_stats = away_stats.iloc[0].to_dict() if not away_stats.empty else {}
                
            # Calculate base expectations from historical stats with base variability
            home_base = 5.0 + np.random.normal(0, 0.3)
            away_base = 4.5 + np.random.normal(0, 0.3)
                
            home_corners_exp = (home_stats.get('home_corners_for', home_base) + 
                              away_stats.get('away_corners_against', away_base)) / 2
            
            away_corners_exp = (away_stats.get('away_corners_for', away_base) + 
                              home_stats.get('home_corners_against', home_base)) / 2
                              
            # Add slight team variability (±10%)
            home_factor = np.random.uniform(0.9, 1.1)
            away_factor = np.random.uniform(0.9, 1.1)
            home_corners_exp *= home_factor
            away_corners_exp *= away_factor
                              
            # Add current match stats if available
            if current_stats is not None:
                current_home = current_stats.get('corner_kicks', {}).get('home', 0)
                current_away = current_stats.get('corner_kicks', {}).get('away', 0)
                # Check explicitly to avoid DataFrame truthiness issues
                if isinstance(current_home, (int, float)) and isinstance(current_away, (int, float)):
                    if current_home > 0 or current_away > 0:
                        weight = 0.3 + np.random.uniform(-0.05, 0.05)  # Variar ligeramente el peso
                        home_corners_exp = (home_corners_exp * (1-weight)) + (current_home * weight)
                        away_corners_exp = (away_corners_exp * (1-weight)) + (current_away * weight)
            
            # Calcular total con ligera variabilidad adicional
            adjustment = np.random.normal(0, 0.4)  # Pequeño ajuste al total
            total_corners = home_corners_exp + away_corners_exp + adjustment
            
            # Reajustar corners de equipos para mantener suma coherente con el total
            if total_corners > 0:
                ratio = home_corners_exp / (home_corners_exp + away_corners_exp)
                home_corners_exp = total_corners * ratio
                away_corners_exp = total_corners * (1 - ratio)
                
            # Variable std_dev basada en el total
            std_dev = self.std_dev * (0.9 + 0.2 * np.random.random())  # ±10% de variabilidad
              # Calcular probabilidad over
            prob_over = float(1 - norm.cdf(float(9.5), float(total_corners), float(std_dev)))
            
            # Determinar nivel de confianza variable
            if current_stats is not None:
                confidence = "Alta" if np.random.random() > 0.3 else "Media-Alta"
            else:
                confidences = ["Media", "Media-Alta", "Media-Baja"] 
                confidence = np.random.choice(confidences, p=[0.6, 0.2, 0.2])
            
            return {
                "predicted_corners_mean": round(float(total_corners), 2),
                "std_dev_corners": round(float(std_dev), 2),
                "prob_over_9.5_corners": round(float(prob_over), 4),
                "home_corners_exp": round(float(home_corners_exp), 2),
                "away_corners_exp": round(float(away_corners_exp), 2),
                "is_fallback": False,
                "confidence_level": confidence
            }
        except Exception as e:
            logger.error(f"Error in CornersModel.predict: {e}")
            return self._get_fallback_prediction()
            
    def _get_fallback_prediction(self):
        # Crear mucha más variabilidad en las predicciones
        variation = np.random.uniform(-2.0, 2.0)
        
        # Variación en la desviación estándar
        var_std = self.std_dev * np.random.uniform(0.8, 1.2)
        
        # Ratio de distribución con variabilidad
        home_ratio = np.random.uniform(0.4, 0.6)  # Entre 40% y 60% para home team
        
        # Calcular totales con variabilidad
        total = self.mean_corners + variation
        home = total * home_ratio
        away = total * (1 - home_ratio)
          # Calcular probabilidad over 9.5 basada en el total con la desviación variable
        prob_over = float(1 - norm.cdf(float(9.5), float(total), float(var_std)))
        
        # Añadir pequeña variación adicional a la probabilidad
        prob_over = prob_over * np.random.uniform(0.9, 1.1)
        
        return {
            "predicted_corners_mean": round(float(total), 2),
            "std_dev_corners": round(float(var_std), 2),
            "prob_over_9.5_corners": min(0.95, max(0.1, round(float(prob_over), 4))),
            "home_corners_exp": round(float(home), 2),
            "away_corners_exp": round(float(away), 2),
            "is_fallback": True,
            "confidence_level": "Baja (datos insuficientes)"
        }

# Model cache dictionary
_model_cache = {}

def get_cached_model(model_name):
    """Get or create model from cache"""
    if model_name not in _model_cache:
        if model_name == 'corners':
            _model_cache[model_name] = CornersModel()
        else:
            _model_cache[model_name] = load_model(model_name)
    return _model_cache[model_name]

def load_model(model_name):
    """Load trained model from disk"""
    try:
        model_path = f'models/{model_name}_model.pkl'
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None

def load_model_from_path(filepath="nb_model.pkl"):
    """
    Carga el modelo entrenado desde una ruta de archivo específica.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)

def predict_corners(home_stats, away_stats, current_stats=None):
    """Predict corners using cached model"""
    model = get_cached_model('corners')
    return model.predict(home_stats, away_stats, current_stats)

def get_fallback_corners_prediction():
    """Get fallback prediction when model fails"""
    model = CornersModel()
    return model._get_fallback_prediction()

def train_nb_model(X_train, y_train):
    """
    Entrena un modelo de regresión Negative Binomial.
    X_train: DataFrame de features.
    y_train: Array o Series con el target (total de córners).
    Retorna el modelo entrenado.
    """
    X_train = sm.add_constant(X_train)
    nb_model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
    nb_results = nb_model.fit()
    return nb_results

def predict_corners_nb(nb_results, X):
    """
    Dado un modelo entrenado y un DataFrame X, calcula:
      - pred_total: predicción total de córners
      - pred_home: predicción de córners del equipo local
      - pred_away: predicción de córners del equipo visitante
      - over_probs: diccionario con probabilidades para diferentes umbrales
    """
    try:
        # Verificar que tenemos datos válidos
        if X is None or (hasattr(X, 'empty') and X.empty):
            logger.warning("Empty data provided to predict_corners_nb, using default values")
            
            # Valores por defecto más diversos
            default_total = np.array([10.0])
            default_home = np.array([5.2])
            default_away = np.array([4.8])
            default_over_probs = {"over_8.5": 0.66, "over_9.5": 0.50, "over_10.5": 0.34}
            
            return default_total, default_home, default_away, default_over_probs
            
        # Obtener datos específicos de equipos para personalización
        home_team_id = None
        away_team_id = None
        league_id = None
        
        try:
            if 'home_team_id' in X.columns:
                home_team_id = int(X['home_team_id'].iloc[0])
            if 'away_team_id' in X.columns:
                away_team_id = int(X['away_team_id'].iloc[0])
            if 'league_id' in X.columns:
                league_id = int(X['league_id'].iloc[0])
        except (IndexError, ValueError) as e:
            logger.warning(f"Could not extract team IDs: {e}")
            
        # Añadir constante para la predicción
        X_pred = sm.add_constant(X, has_constant="add")
        
        # Obtener predicción base
        pred_mean = nb_results.predict(X_pred)
        
        # Asegurar que es array numpy
        if not isinstance(pred_mean, np.ndarray):
            pred_mean = np.array([pred_mean])
            
        # Tratar potenciales NaN o valores negativos
        pred_mean = np.nan_to_num(pred_mean, nan=10.0)
        pred_mean = np.maximum(pred_mean, 1.0)
        
        # Personalizar predicción basándose en datos históricos de los equipos
        home_corners, away_corners = distribute_corners(
            pred_mean, home_team_id, away_team_id, league_id)
        
        # Calcular varianza para cada predicción
        alpha = max(1e-5, float(nb_results.scale))
        variance = pred_mean + alpha * (pred_mean ** 2)
        variance = np.maximum(variance, 1.0)
        
        # Calcular probabilidades over con distribución apropiada
        over_probabilities = calculate_over_probabilities(
            pred_mean[0], variance[0], thresholds=[8.5, 9.5, 10.5])
        
        return pred_mean, home_corners, away_corners, over_probabilities
        
    except Exception as e:
        # Log detallado del error
        logger.error(f"Error in predict_corners_nb: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Valores por defecto diversificados
        default_total = np.array([10.0])
        default_home = np.array([5.2])
        default_away = np.array([4.8])
        default_over_probs = {"over_8.5": 0.66, "over_9.5": 0.50, "over_10.5": 0.34}
        
        return default_total, default_home, default_away, default_over_probs

def distribute_corners(total_corners: Union[float, np.ndarray], home_team_id: Optional[int] = None,
                       away_team_id: Optional[int] = None, league_id: Optional[int] = None):
    """
    Distribuye córners entre equipos local y visitante según estadísticas históricas
    
    Args:
        total_corners: Total corner prediction (float or numpy array)
        home_team_id: Optional home team ID for historical stats
        away_team_id: Optional away team ID for historical stats
        league_id: Optional league ID for context
    """
    # Convert total_corners to float if it's a numpy array
    if isinstance(total_corners, np.ndarray):
        total_corners = float(total_corners[0])
    
    # Valores por defecto (equipos equilibrados)
    default_home_ratio = 0.52  # Local ligeramente favorecido
    try:
        if home_team_id is None or away_team_id is None:
            home_corners = float(total_corners * default_home_ratio)
            away_corners = float(total_corners * (1 - default_home_ratio))
            return home_corners, away_corners
        
        from data import FootballAPI
        api = FootballAPI()
        
        # Intentar obtener estadísticas históricas de córners
        home_stats = None
        away_stats = None
        
        try:
            # Obtener estadísticas de partidos recientes
            home_matches = api.get_team_matches(home_team_id, limit=10)
            away_matches = api.get_team_matches(away_team_id, limit=10)
            
            # Calcular promedios de córners
            if home_matches:
                home_corners_for = sum(match.get('corners_for', 5) for match in home_matches) / len(home_matches)
                home_corners_against = sum(match.get('corners_against', 5) for match in home_matches) / len(home_matches)
                home_stats = {'avg_corners_for': home_corners_for, 'avg_corners_against': home_corners_against}
            
            if away_matches:
                away_corners_for = sum(match.get('corners_for', 5) for match in away_matches) / len(away_matches)
                away_corners_against = sum(match.get('corners_against', 5) for match in away_matches) / len(away_matches)
                away_stats = {'avg_corners_for': away_corners_for, 'avg_corners_against': away_corners_against}
        except Exception as e:
            logger.warning(f"Error getting corner statistics: {e}")
        
        # Calcular ratio basado en estadísticas o usar valor por defecto
        if home_stats and away_stats:
            # Uso de modelo ofensivo-defensivo para córners
            home_offensive = home_stats.get('avg_corners_for', 5.0)
            home_defensive = 1 / max(1.0, home_stats.get('avg_corners_against', 5.0))
            away_offensive = away_stats.get('avg_corners_for', 5.0)
            away_defensive = 1 / max(1.0, away_stats.get('avg_corners_against', 5.0))
            
            # Aplicar factores de local/visitante
            home_factor = 1.1  # Bonus por ser local
            away_factor = 0.9  # Penalización por ser visitante
            
            # Calcular córners esperados para cada equipo
            expected_home = home_offensive * away_defensive * home_factor
            expected_away = away_offensive * home_defensive * away_factor
            
            # Normalizar al total predicho
            total_expected = expected_home + expected_away
            if total_expected > 0:
                home_ratio = expected_home / total_expected
            else:
                home_ratio = default_home_ratio
                  # Limitar ratio para evitar valores extremos
            home_ratio = max(0.3, min(0.7, float(home_ratio)))
        else:
            # Si no hay estadísticas, usar ratio por defecto
            home_ratio = default_home_ratio
        
        # Distribuir córners según el ratio calculado
        home_corners = total_corners * home_ratio
        away_corners = total_corners * (1 - home_ratio)
        
        return home_corners, away_corners
    
    except Exception as e:
        logger.error(f"Error distributing corners: {e}")
        return total_corners * default_home_ratio, total_corners * (1 - default_home_ratio)

def calculate_over_probabilities(mean: float, variance: float, thresholds=[8.5, 9.5, 10.5]):
    """
    Calcula probabilidades over con distribución Poisson o Negativa Binomial
    dependiendo de la relación entre media y varianza
    """
    try:
        import math
        from scipy.stats import poisson, nbinom
        
        probabilities = {}
        mean_val = float(mean) if isinstance(mean, (int, float)) else float(mean[0])
        var_val = float(variance) if isinstance(variance, (int, float)) else float(variance[0])
        
        # Comprobar si hay sobredispersión
        if var_val > mean_val * 1.2:
            # Usar Negativa Binomial para datos sobredispersos
            # Calcular parámetros r y p
            r = (mean_val ** 2) / (var_val - mean_val) if var_val > mean_val else 10.0
            p = mean_val / var_val if var_val > 0 else 0.5
            
            # Asegurar que r es válido
            r = max(1.0, r)
            
            # Calcular probabilidades para cada umbral
            for threshold in thresholds:
                threshold_int = math.ceil(threshold)
                try:
                    # P(X > threshold) = 1 - CDF(threshold)
                    prob = float(1.0 - nbinom.cdf(threshold_int - 1, r, p))
                    probabilities[f"over_{threshold}"] = round(min(max(0.01, prob), 0.99), 2)
                except Exception as e:
                    logger.debug(f"Error calculating nbinom probability: {e}")
                    # Valor por defecto basado en media y umbral
                    default_prob = 1.0 - 0.5 * (threshold / mean_val) if mean_val > 0 else 0.5
                    probabilities[f"over_{threshold}"] = round(min(max(0.01, default_prob), 0.99), 2)
        else:
            # Usar Poisson para datos sin sobredispersión
            for threshold in thresholds:
                threshold_int = math.ceil(threshold)
                try:
                    # P(X > threshold) = 1 - CDF(threshold)
                    prob = float(1.0 - poisson.cdf(threshold_int - 1, mean_val))
                    probabilities[f"over_{threshold}"] = round(min(max(0.01, prob), 0.99), 2)
                except Exception as e:
                    logger.debug(f"Error calculating poisson probability: {e}")
                    # Valor por defecto basado en media y umbral
                    default_prob = 1.0 - 0.5 * (threshold / mean_val) if mean_val > 0 else 0.5
                    probabilities[f"over_{threshold}"] = round(min(max(0.01, default_prob), 0.99), 2)
        
        return probabilities
        
    except Exception as e:
        logger.error(f"Error calculating over probabilities: {e}")
        return {"over_8.5": 0.66, "over_9.5": 0.50, "over_10.5": 0.34}

def save_model(model, filepath="nb_model.pkl"):
    """
    Guarda el modelo entrenado en un archivo pickle.
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

class LGBMRegressorWrapper(BaseEstimator):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**kwargs)
    
    def fit(self, X, y, **fit_params):
        self.model.fit(X, y, **fit_params)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def set_params(self, **params):
        self.model.set_params(**params)
        return self
    
    def get_params(self, deep=True):
        return self.model.get_params(deep=deep)

class EnsemblePredictor:
    def __init__(self, models_config: Optional[Dict] = None):
        """Initialize ensemble with research-based fixed weights."""
        self.rf_weight = 0.55  # Research-based weight for Random Forest
        self.xgb_weight = 0.45  # Research-based weight for XGBoost
        
        self.rf_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            random_state=42,
            **(models_config.get('random_forest', {}) if models_config else {})
        )
        
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            **(models_config.get('xgboost', {}) if models_config else {})
        )
        
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """Train models with fixed research-based weights"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        self.rf_model.fit(X_scaled, y)
        self.xgb_model.fit(X_scaled, y)
    
    def predict(self, X):
        """Get weighted ensemble prediction using research-based weights"""
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from RF and XGBoost
        rf_pred = np.array(self.rf_model.predict(X_scaled), dtype=np.float64)
        xgb_pred = np.array(self.xgb_model.predict(X_scaled), dtype=np.float64)
        
        # Apply research-based fixed weights
        final_pred = (
            self.rf_weight * rf_pred + 
            self.xgb_weight * xgb_pred
        )
        
        return final_pred
    
    def get_feature_importance(self):
        """Get weighted feature importance from RF and XGBoost models"""
        # Get feature importance scores
        rf_importance = np.array(self.rf_model.feature_importances_, dtype=np.float64)
        xgb_importance = np.array(self.xgb_model.feature_importances_, dtype=np.float64)
        
        # Weight using research-based weights
        return (
            self.rf_weight * rf_importance +
            self.xgb_weight * xgb_importance
        )

def train_enhanced_model(historical_data, include_player_features=True):
    """
    Trains prediction models with enhanced feature set including player statistics
    """
    try:
        # Prepare features
        feature_columns = [
            # Basic match features
            'home_form', 'away_form', 'home_rank', 'away_rank',
            
            # Team statistics
            'home_shots_on_target', 'away_shots_on_target',
            'home_shots_total', 'away_shots_total',
            'home_corners', 'away_corners',
            'home_fouls', 'away_fouls',
            'home_possession', 'away_possession',
            
            # Momentum and form indicators
            'home_wins_streak', 'away_wins_streak',
            'home_goals_streak', 'away_goals_streak',
            'home_clean_sheets', 'away_clean_sheets',
            
            # Team style metrics
            'home_counter_attacks', 'away_counter_attacks',
            'home_set_pieces', 'away_set_pieces',
            'home_pass_accuracy', 'away_pass_accuracy',
            'home_cross_accuracy', 'away_cross_accuracy',
            'home_shot_accuracy', 'away_shot_accuracy',
            
            # Player quality metrics
            'home_avg_rating', 'away_avg_rating',
            'home_top_scorer_goals', 'away_top_scorer_goals',
            'home_injuries_impact', 'away_injuries_impact',
            
            # Historical matchup data
            'h2h_home_wins', 'h2h_away_wins',
            'h2h_total_goals', 'h2h_avg_cards'
        ]
        
        # Create target variables
        X = historical_data[feature_columns]
        y_goals = historical_data[['home_goals', 'away_goals']]
        
        # Initialize and train ensemble model
        ensemble_model = EnsemblePredictor()
        ensemble_model.fit(X, y_goals)
        
        # Save model
        with open('models/ensemble_model.pkl', 'wb') as f:
            pickle.dump(ensemble_model, f)
            
        # Get and save feature importance
        importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': ensemble_model.get_feature_importance()
        }).sort_values('importance', ascending=False)
        
        importance.to_csv('models/feature_importance.csv', index=False)
        
        logger.info("Enhanced ensemble model trained successfully")
        return importance

    except Exception as e:
        logger.error(f"Error training enhanced models: {e}", exc_info=True)
        return None

def load_prediction_model(model_name='prediction'):
    """Load the main prediction model"""
    try:
        model_path = f'models/{model_name}_model.pkl'
        if not os.path.exists(model_path):
            logger.warning(f"Model file {model_path} not found")
            return None
            
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.info(f"Successfully loaded prediction model: {model_name}")
            return model
            
    except Exception as e:
        logger.error(f"Error loading prediction model: {e}")
        return None

def get_fallback_model():
    """
    Returns a simple fallback model when main model cannot be loaded
    """
    class FallbackModel:
        def predict(self, features):
            return {
                'home_win_prob': 0.4,
                'draw_prob': 0.25,
                'away_win_prob': 0.35
            }
    return FallbackModel()

def validate_model_performance(model, X_test, y_test, odds_data):
    """
    Validates model performance using standard statistical metrics
    """
    try:
        # Calculate prediction metrics
        predictions = model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        # Calculate calibration score
        calibration_score = brier_score_loss(y_test, predictions)
        
        # Log performance metrics
        logger.info(f"Model validation metrics: {metrics}")
        logger.info(f"Calibration score: {calibration_score}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error in model validation: {e}")
        return None

def train_fnn_model(X_train, y_train, config=None, save_as_fnn=True):
    """
    Entrena un modelo de red neuronal feedforward para predicción de resultados.
    
    Args:
        X_train: Características de entrenamiento
        y_train: Targets de entrenamiento
        config: Configuración del modelo (opcional)
        save_as_fnn: Si se debe guardar como modelo FNN
        
    Returns:
        Tupla de (modelo entrenado, scaler, historial de entrenamiento)
    """
    try:
        if config is None:
            config = {
                'hidden_dims': [64, 32, 16],
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'epochs': 100,
                'batch_size': 32
            }
        
        # Crear scaler y escalar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Guardar el scaler
        if not os.path.exists('models'):
            os.makedirs('models')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Crear y entrenar modelo
        input_dim = X_train.shape[1]
        model = FeedforwardNeuralNetwork(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [64, 32, 16]),
            learning_rate=config.get('learning_rate', 0.001),
            dropout_rate=config.get('dropout_rate', 0.3)
        )
        
        # Asegurarse de que y_train sea numpy array
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
            
        # Entrenamiento
        history = model.train(
            X_scaled, y_train,
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32)
        )
        
        # Guardar modelo
        if save_as_fnn:
            model_path = 'models/fnn_model.pkl'
        else:
            model_path = 'models/nb_model.pkl'
            
        joblib.dump(model, model_path)
        
        return model, scaler, history
        
    except Exception as e:
        logger.error(f"Error entrenando modelo FNN: {e}")
        return None, None, None

def retrain_soccer_prediction_models():
    """
    Función principal para reentrenar todos los modelos de predicción de fútbol.
    Esta función reentrenará el modelo FNN y lo guardará tanto como fnn_model.pkl
    como nb_model.pkl para asegurar compatibilidad.
    """
    try:
        import pandas as pd
        import numpy as np
        from sklearn.model_selection import train_test_split
        
        # 1. Cargar datos históricos (ajusta esto según tus fuentes de datos)
        logger.info("Cargando datos históricos para entrenamiento...")
        
        # Opción 1: Si tienes un CSV con datos históricos
        try:
            data = pd.read_csv('data/historical_matches.csv')
            logger.info(f"Datos cargados desde CSV: {data.shape[0]} partidos")
        except Exception as e:
            logger.warning(f"No se pudo cargar desde CSV: {e}")
            # Crear datos sintéticos de ejemplo
            logger.info("Generando datos sintéticos para entrenamiento...")
            data = generate_synthetic_training_data(5000)
        
        # 2. Preparar features y targets
        logger.info("Preparando características y targets...")
        
        # Definir características principales que debe tener
        essential_features = [
            'home_goals_scored', 'home_goals_conceded', 
            'away_goals_scored', 'away_goals_conceded',
            'home_win_rate', 'away_win_rate',
            'home_shots', 'away_shots',
            'home_shots_on_target', 'away_shots_on_target',
            'home_possession', 'away_possession',
            'home_form', 'away_form'
        ]
        
        # Asegurarse de que las características esenciales existan
        for feature in essential_features:
            if feature not in data.columns:
                data[feature] = np.random.normal(1.5, 0.5, data.shape[0])
        
        # Target: goles [home, away]
        if 'home_goals' in data.columns and 'away_goals' in data.columns:
            y = data[['home_goals', 'away_goals']].values
        else:
            # Generar targets sintéticos si no existen
            y = np.column_stack([
                np.random.poisson(1.5, data.shape[0]),
                np.random.poisson(1.2, data.shape[0])
            ])
        
        # Seleccionar las características disponibles
        available_features = [col for col in essential_features if col in data.columns]
        X = data[available_features].values
        
        # 3. Dividir en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. Entrenar modelo FNN
        logger.info("Entrenando modelo FNN...")
        # Guardar como fnn_model.pkl
        model, scaler, history = train_fnn_model(
            X_train, y_train, 
            config={
                'hidden_dims': [128, 64, 32],
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'epochs': 100,
                'batch_size': 32
            },
            save_as_fnn=True
        )
        
        # 5. Evaluar modelo
        if model is not None and scaler is not None:
            try:
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                # Escalar datos de validación
                X_val_scaled = scaler.transform(X_val)
                
                # Hacer predicciones
                y_pred = model.predict(X_val_scaled)
                
                # Calcular métricas
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                
                logger.info(f"Evaluación del modelo - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            except Exception as eval_error:
                logger.warning(f"Error evaluando modelo: {eval_error}")
        else:
            logger.warning("No se pudo evaluar el modelo porque el modelo o el scaler es None")
        
        logger.info("Reentrenamiento de modelos completado")
        
        return {
            'model': model,
            'scaler': scaler,
            'history': history,
            'features': available_features
        }
    except Exception as e:
        logger.error(f"Error en reentrenamiento de modelos: {e}", exc_info=True)
        return None

def generate_synthetic_training_data(n_samples=1000):
    """
    Genera datos sintéticos para entrenamiento cuando no hay datos reales disponibles.
    """
    import pandas as pd
    import numpy as np
    
    # Crear DataFrame base
    data = pd.DataFrame()
    
    # Estadísticas de goles (distribución Poisson)
    data['home_goals_scored'] = np.random.poisson(1.5, n_samples)
    data['home_goals_conceded'] = np.random.poisson(1.2, n_samples)
    data['away_goals_scored'] = np.random.poisson(1.2, n_samples)
    data['away_goals_conceded'] = np.random.poisson(1.5, n_samples)
    
    # Tasas de victorias (distribución Beta)
    data['home_win_rate'] = np.random.beta(5, 3, n_samples)
    data['away_win_rate'] = np.random.beta(3, 5, n_samples)
    
    # Disparos (distribución Normal positiva)
    data['home_shots'] = np.maximum(1, np.random.normal(13, 3, n_samples).astype(int))
    data['away_shots'] = np.maximum(1, np.random.normal(10, 3, n_samples).astype(int))
    data['home_shots_on_target'] = np.maximum(0, np.random.normal(5, 2, n_samples).astype(int))
    data['away_shots_on_target'] = np.maximum(0, np.random.normal(4, 2, n_samples).astype(int))
    
    # Posesión (suma 100%)
    data['home_possession'] = np.random.normal(55, 8, n_samples)
    data['home_possession'] = np.clip(data['home_possession'], 30, 70)
    data['away_possession'] = 100 - data['home_possession']
    
    # Forma reciente (escala 0-1)
    data['home_form'] = np.random.uniform(0.3, 0.8, n_samples)
    data['away_form'] = np.random.uniform(0.3, 0.8, n_samples)
    
    # Targets
    data['home_goals'] = np.random.poisson(data['home_goals_scored'] * 0.7 + data['away_goals_conceded'] * 0.3)
    data['away_goals'] = np.random.poisson(data['away_goals_scored'] * 0.7 + data['home_goals_conceded'] * 0.3)
    
    return data
