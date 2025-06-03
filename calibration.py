# calibration.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_squared_error, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_score
from scipy.stats import poisson, norm
import logging
import joblib
from typing import Dict, Any, Tuple, List, Optional, Union, cast
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class PredictionCalibrator:
    def __init__(self):
        self.calibration_window = 100  # Number of matches to use for calibration
        self.predictions_history = []
        self.actual_results = []
        
    def add_prediction(self, prediction: Dict[str, Any], actual_result: Dict[str, Any]):
        """Store prediction and actual result for calibration"""
        self.predictions_history.append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'actual': actual_result
        })
        
        # Keep only recent matches within calibration window
        if len(self.predictions_history) > self.calibration_window:
            self.predictions_history.pop(0)
            
        # Recalibrate if we have enough data
        if len(self.predictions_history) >= 20:
            self._calibrate_predictions()
            
    def _calibrate_predictions(self):
        """Update calibration based on recent prediction accuracy"""
        try:
            # Extract prediction probabilities and actual results
            pred_probs = []
            actuals = []
            
            for entry in self.predictions_history:
                pred = entry['prediction']
                actual = entry['actual']
                
                pred_probs.append([
                    pred['home_win_prob'],
                    pred['draw_prob'],
                    pred['away_win_prob']
                ])
                
                # Convert result to one-hot encoding
                if actual['home_goals'] > actual['away_goals']:
                    actuals.append(0)  # Home win
                elif actual['home_goals'] < actual['away_goals']:
                    actuals.append(2)  # Away win
                else:
                    actuals.append(1)  # Draw
                    
            pred_probs = np.array(pred_probs)
            actuals = np.array(actuals)
            
            # Calculate calibration metrics
            metrics = self._calculate_calibration_metrics(pred_probs, actuals)
            
            # Save calibration metrics
            self._save_calibration_metrics(metrics)
            
            logger.info(f"Calibration updated with metrics: {metrics}")
            
        except Exception as e:
            logger.error(f"Error in calibration: {e}")
            
    def _calculate_calibration_metrics(self, pred_probs: np.ndarray, actuals: np.ndarray) -> Dict[str, float]:
        """Calculate various calibration metrics"""
        metrics = {}
        
        # Brier score (lower is better)
        metrics['brier_score'] = float(brier_score_loss(actuals, pred_probs, pos_label=2))
        
        # Log loss (lower is better)
        metrics['log_loss'] = float(log_loss(actuals, pred_probs))
        
        # Calibration ratio (closer to 1 is better)
        predicted_wins = np.sum(pred_probs[:, 0])  # Sum of home win probabilities
        actual_wins = np.sum(actuals == 0)  # Count of actual home wins
        if actual_wins > 0:
            metrics['home_win_calibration'] = float(predicted_wins / actual_wins)
        
        return metrics
        
    def calibrate_prediction(self, raw_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply calibration to a new prediction"""
        try:
            # Get current calibration metrics
            metrics = self._load_calibration_metrics()
            
            if not metrics:
                return raw_prediction
                
            # Apply calibration adjustments
            calibrated = raw_prediction.copy()
            
            # Adjust probabilities based on historical calibration
            if 'home_win_calibration' in metrics:
                calibration_factor = 1 / metrics['home_win_calibration']
                calibrated['home_win_prob'] *= calibration_factor
                calibrated['away_win_prob'] *= calibration_factor
                
                # Normalize probabilities to sum to 1
                total = sum([calibrated['home_win_prob'], 
                           calibrated['draw_prob'],
                           calibrated['away_win_prob']])
                           
                calibrated['home_win_prob'] /= total
                calibrated['draw_prob'] /= total
                calibrated['away_win_prob'] /= total
                
            # Round probabilities
            calibrated['home_win_prob'] = round(float(calibrated['home_win_prob']), 3)
            calibrated['draw_prob'] = round(float(calibrated['draw_prob']), 3)
            calibrated['away_win_prob'] = round(float(calibrated['away_win_prob']), 3)
            
            return calibrated
            
        except Exception as e:
            logger.error(f"Error applying calibration: {e}")
            return raw_prediction
            
    def _save_calibration_metrics(self, metrics: Dict[str, float]):
        """Save calibration metrics to file"""
        try:
            with open('models/calibration_metrics.json', 'w') as f:
                json.dump(metrics, f)
        except Exception as e:
            logger.error(f"Error saving calibration metrics: {e}")
            
    def _load_calibration_metrics(self) -> Dict[str, float]:
        """Load calibration metrics from file"""
        try:
            with open('models/calibration_metrics.json', 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading calibration metrics: {e}")
            return {}
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Generate summary of prediction performance"""
        try:
            correct_predictions = 0
            total_predictions = len(self.predictions_history)
            
            if total_predictions == 0:
                return {"accuracy": 0, "sample_size": 0}
                
            for entry in self.predictions_history:
                pred = entry['prediction']
                actual = entry['actual']
                
                # Get highest probability outcome
                pred_probs = [
                    (pred['home_win_prob'], 'home'),
                    (pred['draw_prob'], 'draw'),
                    (pred['away_win_prob'], 'away')
                ]
                predicted_result = max(pred_probs, key=lambda x: x[0])[1]
                
                # Get actual result
                if actual['home_goals'] > actual['away_goals']:
                    actual_result = 'home'
                elif actual['home_goals'] < actual['away_goals']:
                    actual_result = 'away'
                else:
                    actual_result = 'draw'
                    
                if predicted_result == actual_result:
                    correct_predictions += 1
                    
            accuracy = correct_predictions / total_predictions
            
            recent_accuracy = 0
            if total_predictions >= 10:
                recent_correct = sum(1 for entry in self.predictions_history[-10:]
                    if self._is_correct_prediction(entry['prediction'], entry['actual']))
                recent_accuracy = recent_correct / 10
                
            return {
                "overall_accuracy": round(accuracy, 3),
                "recent_accuracy": round(recent_accuracy, 3),
                "sample_size": total_predictions,
                "recent_sample": min(10, total_predictions)
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {"accuracy": 0, "sample_size": 0}
            
    def _is_correct_prediction(self, prediction: Dict[str, Any], actual: Dict[str, Any]) -> bool:
        """Check if prediction matches actual result"""
        # Get predicted outcome (highest probability)
        pred_probs = [
            (prediction['home_win_prob'], 'home'),
            (prediction['draw_prob'], 'draw'),
            (prediction['away_win_prob'], 'away')
        ]
        predicted_result = max(pred_probs, key=lambda x: x[0])[1]
        
        # Get actual result
        if actual['home_goals'] > actual['away_goals']:
            actual_result = 'home'
        elif actual['home_goals'] < actual['away_goals']:
            actual_result = 'away'
        else:
            actual_result = 'draw'
            
        return predicted_result == actual_result

class PredictionConfidenceEvaluator:
    """
    Evaluador de confianza para predicciones deportivas.
    Asigna una puntuación de confianza basada en varios factores.
    """
    
    def __init__(self) -> None:
        """Inicializa los parámetros del evaluador de confianza."""
        # Pesos para diferentes factores que contribuyen a la confianza
        self.confidence_weights = {
            'ensemble_agreement': 0.25,  # Concordancia entre modelos
            'data_quality': 0.30,        # Calidad y completitud de datos
            'historical_accuracy': 0.20,  # Precisión histórica del modelo
            'calibration_score': 0.25    # Calibración del modelo
        }
        
        # Umbrales para determinar niveles de confianza
        self.confidence_thresholds = {
            'high': 0.75,   # Confianza alta
            'medium': 0.50  # Confianza media, por debajo es baja
        }
    
    def evaluate_prediction_confidence(self, 
                                    prediction: Dict[str, Any],
                                    ensemble_predictions: Dict[str, np.ndarray],
                                    data_quality_metrics: Dict[str, float],
                                    historical_metrics: Dict[str, float],
                                    calibration_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Evalúa la confiabilidad de una predicción específica.
        
        Args:
            prediction: Predicción principal del modelo
            ensemble_predictions: Predicciones individuales de cada modelo del ensemble
            data_quality_metrics: Métricas de calidad de datos
            historical_metrics: Métricas históricas de rendimiento
            calibration_metrics: Métricas de calibración
        """
        # 1. Evaluar concordancia del ensemble
        ensemble_agreement = self._calculate_ensemble_agreement(ensemble_predictions)
        
        # 2. Evaluar calidad de datos
        data_quality_score = self._evaluate_data_quality(data_quality_metrics)
        
        # 3. Evaluar precisión histórica
        historical_score = self._evaluate_historical_accuracy(historical_metrics)
        
        # 4. Evaluar calibración
        calibration_score = self._evaluate_calibration(calibration_metrics)
        
        # Calcular score final ponderado
        final_confidence_score = (
            self.confidence_weights['ensemble_agreement'] * ensemble_agreement +
            self.confidence_weights['data_quality'] * data_quality_score +
            self.confidence_weights['historical_accuracy'] * historical_score +
            self.confidence_weights['calibration_score'] * calibration_score
        )
        
        # Asegurar que el score no es NaN ni está fuera de rango
        if np.isnan(final_confidence_score) or not np.isfinite(final_confidence_score):
            final_confidence_score = 0.5  # Valor por defecto moderado
        else:
            # Limitar a rango [0, 1]
            final_confidence_score = max(0.0, min(1.0, final_confidence_score))
        
        # Determinar nivel de confianza
        confidence_level = self._determine_confidence_level(final_confidence_score)
        
        # Factores que aumentan/disminuyen la confianza
        confidence_factors = self._identify_confidence_factors(
            ensemble_agreement,
            data_quality_score,
            historical_score,
            calibration_score
        )
        
        return {
            'confidence_level': confidence_level,
            'confidence_score': final_confidence_score,
            'confidence_factors': confidence_factors,
            'detailed_scores': {
                'ensemble_agreement': ensemble_agreement,
                'data_quality': data_quality_score,
                'historical_accuracy': historical_score,
                'calibration': calibration_score
            }
        }
    
    def _calculate_ensemble_agreement(self, ensemble_predictions: Dict[str, np.ndarray]) -> float:
        """
        Calcula el nivel de acuerdo entre los diferentes modelos del ensemble.
        """
        # Si no hay predicciones de ensemble, asignar valor predeterminado
        if not ensemble_predictions:
            return 0.5  # Valor neutral por defecto
            
        predictions = np.array(list(ensemble_predictions.values()))        # Calcular desviación estándar entre predicciones
        if len(predictions) > 1:
            std_dev = np.std(predictions, axis=0)
            # Normalizar y convertir a score (menor std = mayor acuerdo)
            agreement_score = 1 / (1 + std_dev.mean())
        else:
            # Si solo hay una predicción, el acuerdo es máximo (no hay variación)
            agreement_score = 1.0
        return float(agreement_score)
    
    def _evaluate_data_quality(self, metrics: Dict[str, float]) -> float:
        """
        Evalúa la calidad de los datos usados para la predicción.
        """
        # Si no hay métricas disponibles, asignar un valor predeterminado bajo
        if not metrics:
            return 0.5  # Valor neutral por defecto
        values = [
            metrics.get('completeness', 0.5),
            metrics.get('recency', 0.5),
            metrics.get('reliability', 0.5)
        ]
        # Filtrar valores que no son None o NaN
        filtered_values = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
        quality_score = np.mean(filtered_values) if filtered_values else 0.5
        return float(quality_score)
    
    def _evaluate_historical_accuracy(self, metrics: Dict[str, float]) -> float:
        """
        Evalúa la precisión histórica del modelo.
        """
        # Si no hay métricas, asignar valor predeterminado moderado
        if not metrics:
            return 0.6  # Valor por defecto ligeramente optimista
            
        # Combinar diferentes métricas de precisión
        accuracy_score = np.mean([
            metrics.get('r2_score', 0.5),
            1 - metrics.get('normalized_rmse', 0.5),  # Convertir RMSE a score
            metrics.get('accuracy', 0.6)
        ])
        return float(accuracy_score)
    
    def _evaluate_calibration(self, metrics: Dict[str, float]) -> float:
        """
        Evalúa la calibración del modelo.
        """
        # Si no hay métricas, usar valor predeterminado
        if not metrics:
            return 0.7  # Por defecto, asumir buena calibración
            
        # Usar Brier score y otras métricas de calibración
        calibration_score = 1 - metrics.get('brier_score', 0.3)  # Convertir a score (mayor es mejor)
        return float(calibration_score)
    
    def _determine_confidence_level(self, confidence_score: float) -> str:
        """
        Determina el nivel de confianza basado en el score final.
        """
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _identify_confidence_factors(self,
                                  ensemble_agreement: float,
                                  data_quality: float,
                                  historical_accuracy: float,
                                  calibration: float) -> Dict[str, List[str]]:
        """
        Identifica factores específicos que aumentan o disminuyen la confianza.
        """
        factors = {
            'increasing': [],
            'decreasing': []
        }
        
        # Evaluar factores de ensemble
        if ensemble_agreement > 0.8:
            factors['increasing'].append('Alto acuerdo entre modelos del ensemble')
        elif ensemble_agreement < 0.5:
            factors['decreasing'].append('Bajo acuerdo entre modelos del ensemble')
        
        # Evaluar calidad de datos
        if data_quality > 0.8:
            factors['increasing'].append('Alta calidad y completitud de datos')
        elif data_quality < 0.5:
            factors['decreasing'].append('Baja calidad o datos incompletos')
        
        # Evaluar precisión histórica
        if historical_accuracy > 0.75:
            factors['increasing'].append('Alto rendimiento histórico del modelo')
        elif historical_accuracy < 0.6:
            factors['decreasing'].append('Bajo rendimiento histórico')
        
        # Evaluar calibración
        if calibration > 0.8:
            factors['increasing'].append('Excelente calibración del modelo')
        elif calibration < 0.5:
            factors['decreasing'].append('Problemas de calibración')
            
        return factors

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evalúa el rendimiento del modelo con métricas básicas.
    
    Args:
        y_true: Valores reales (etiquetas)
        y_pred: Valores predichos por el modelo
        
    Returns:
        Diccionario con varias métricas de evaluación
    """
    # Comprobar que los arrays no estén vacíos
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'mae': float('nan'),
            'rmse': float('nan'),
            'normalized_rmse': float('nan'),
            'r2_score': float('nan')
        }
    
    # Calcular métricas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Calcular RMSE normalizado
    range_y = np.max(y_true) - np.min(y_true)
    normalized_rmse = rmse / range_y if range_y > 0 else float('nan')
    
    # Calcular R2 (coeficiente de determinación)
    # Primero calculamos la varianza total
    y_mean = np.mean(y_true)
    total_variance = np.sum((y_true - y_mean) ** 2)
    
    # Luego calculamos la varianza residual
    residual_variance = np.sum((y_true - y_pred) ** 2)
    
    # R2 = 1 - (varianza residual / varianza total)
    r2 = 1 - (residual_variance / total_variance) if total_variance > 0 else float('nan')
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'normalized_rmse': float(normalized_rmse),
        'r2_score': float(r2)
    }

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Figure:
    """
    Genera un gráfico de calibración para analizar cuánto se ajustan las probabilidades 
    predichas a las frecuencias reales.
    
    Args:
        y_true: Valores reales (0/1)
        y_prob: Probabilidades predichas (valores entre 0 y 1)
        n_bins: Número de bins para el histograma de calibración
        
    Returns:
        Figura matplotlib con el gráfico de calibración
    """
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calcular curva de calibración
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Graficar curva de calibración
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label='Modelo')
    
    # Agregar línea de calibración perfecta
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Calibración perfecta')
    
    # Personalizar gráfico
    ax.set_xlabel('Probabilidad predicha')
    ax.set_ylabel('Frecuencia observada')
    ax.set_title('Curva de calibración')
    ax.legend(loc='best')
    ax.grid(True)
    
    return fig

def calibrate_probabilities(y_true: np.ndarray, y_prob: np.ndarray) -> IsotonicRegression:
    """
    Calibra probabilidades usando regresión isotónica.
    
    Args:
        y_true: Valores reales (0/1)
        y_prob: Probabilidades predichas (valores entre 0 y 1)
        
    Returns:
        Modelo de calibración entrenado
    """
    # Crear modelo de calibración
    calibrator = IsotonicRegression(out_of_bounds='clip')
    
    # Entrenar modelo
    calibrator.fit(y_prob, y_true)
    
    return calibrator

def apply_calibration(calibrator: IsotonicRegression, probabilities: np.ndarray) -> np.ndarray:
    """
    Aplica un modelo de calibración a nuevas probabilidades.
    
    Args:
        calibrator: Modelo de calibración entrenado
        probabilities: Probabilidades a calibrar
        
    Returns:
        Probabilidades calibradas
    """
    return calibrator.transform(probabilities)

def cross_validate_calibration(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Dict[str, Union[float, List[float]]]:
    """
    Evalúa el rendimiento de la calibración usando validación cruzada.
    
    Args:
        X: Características
        y: Etiquetas
        n_folds: Número de particiones para la validación cruzada
        
    Returns:
        Diccionario con resultados de validación cruzada
    """
    # Crear modelo base para calibrar
    base_model = IsotonicRegression(out_of_bounds='clip')
    
    # Realizar validación cruzada
    scores = cross_val_score(base_model, X, y, cv=n_folds, scoring='neg_brier_score')
    
    # Convertir a Brier score positivo (menor es mejor)
    brier_scores = -scores
    
    return {
        'mean_brier_score': float(brier_scores.mean()),
        'std_brier_score': float(brier_scores.std()),
        'individual_scores': [float(score) for score in brier_scores]
    }
