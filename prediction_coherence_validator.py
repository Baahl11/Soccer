"""
Prediction Coherence Validator

Este módulo implementa mecanismos para validar la coherencia entre diferentes aspectos
de las predicciones, especialmente entre ratings Elo y predicciones de goles.

Autor: Equipo de Desarrollo
Fecha: Mayo 25, 2025
"""

import logging
from typing import Dict, Any, Optional, Union, List
import math
import numpy as np

# Configuración de logging
logger = logging.getLogger(__name__)

class CoherenceValidator:
    """
    Clase para validar y ajustar la coherencia entre diferentes aspectos
    de las predicciones de partidos de fútbol.
    """
    
    def __init__(self):
        """Inicializa el validador de coherencia."""
        self.elo_goal_ratio_history = []  # Mantiene un historial de ratios para calibración
        self.max_history_size = 100  # Tamaño máximo del historial
    
    def is_prediction_coherent_with_elo(self, prediction: Dict[str, Any], elo_ratings: Dict[str, float]) -> bool:
        """
        Verifica si una predicción es coherente con los ratings Elo de los equipos.
        
        Args:
            prediction: Predicción a validar
            elo_ratings: Ratings Elo de ambos equipos
            
        Returns:
            True si la predicción es coherente con los ratings Elo, False en caso contrario
        """
        try:
            # Extrae valores necesarios
            home_elo = elo_ratings.get('home', 1500)
            away_elo = elo_ratings.get('away', 1500)
            
            # Buscar home_xg y away_xg en diferentes lugares posibles de la estructura
            home_xg = (
                prediction.get('home_xg') or
                prediction.get('predicted_home_goals') or
                prediction.get('xG', {}).get('home') or
                1.5  # Valor por defecto
            )
            
            away_xg = (
                prediction.get('away_xg') or
                prediction.get('predicted_away_goals') or
                prediction.get('xG', {}).get('away') or
                1.2  # Valor por defecto
            )
            
            # Calcular diferencia de Elo y diferencia de xG
            elo_diff = home_elo - away_elo
            xg_diff = home_xg - away_xg
            
            # Calcular la diferencia esperada de goles basada en Elo
            expected_xg_diff = self._calculate_expected_xg_diff(elo_diff)
            
            # Calcular discrepancia
            discrepancy = xg_diff - expected_xg_diff
            discrepancy_ratio = abs(discrepancy) / max(0.1, abs(expected_xg_diff))
            
            # La predicción es coherente si la discrepancia está por debajo del umbral
            return discrepancy_ratio <= 0.5
            
        except Exception as e:
            logger.error(f"Error al verificar coherencia con Elo: {e}")
            return False
    
    def validate_and_adjust_goal_predictions(self, 
                                           prediction: Dict[str, Any], 
                                           elo_ratings: Dict[str, float],
                                           validate_only: bool = False) -> Dict[str, Any]:
        """
        Valida y ajusta las predicciones de goles para que sean coherentes con los ratings Elo.
        
        Args:
            prediction: Predicción actual con valores de xG y probabilidades
            elo_ratings: Ratings Elo de ambos equipos
            validate_only: Si es True, solo valida sin ajustar
            
        Returns:
            Predicción ajustada o con información de validación adicional
        """
        try:
            # Extraer valores necesarios
            home_elo = elo_ratings.get('home', 1500)
            away_elo = elo_ratings.get('away', 1500)
            home_xg = prediction.get('home_xg', 1.5)
            away_xg = prediction.get('away_xg', 1.2)
            
            # Calcular diferencia de Elo y diferencia de xG
            elo_diff = home_elo - away_elo
            xg_diff = home_xg - away_xg
            
            # Calcular ratio esperado entre diferencia de Elo y diferencia de goles
            # Basado en estudios estadísticos:
            # - Cada 100 puntos Elo ~ 0.25 - 0.3 goles de diferencia
            expected_xg_diff = self._calculate_expected_xg_diff(elo_diff)
            
            # Calcular discrepancia
            discrepancy = xg_diff - expected_xg_diff
            discrepancy_ratio = abs(discrepancy) / max(0.1, abs(expected_xg_diff))
            
            # Añadir información de coherencia a la predicción
            prediction['coherence_analysis'] = {
                'elo_diff': elo_diff,
                'xg_diff': xg_diff,
                'expected_xg_diff': expected_xg_diff,
                'discrepancy': discrepancy,
                'discrepancy_ratio': discrepancy_ratio,
                'is_coherent': discrepancy_ratio <= 0.5,  # Umbral de discrepancia aceptable
                'adjustment_needed': discrepancy_ratio > 0.5
            }
            
            # Si solo validamos, devolvemos aquí
            if validate_only:
                return prediction
            
            # Ajustar predicciones si es necesario
            if discrepancy_ratio > 0.5:
                # Determinar cuánto ajustar (50% hacia el valor esperado)
                adjustment_factor = 0.5
                
                # Diferencia ajustada
                adjusted_diff = xg_diff - (discrepancy * adjustment_factor)
                
                # Distribuir el ajuste entre home_xg y away_xg
                adjustment = (adjusted_diff - xg_diff) / 2
                adjusted_home_xg = home_xg + adjustment
                adjusted_away_xg = away_xg - adjustment
                
                # Asegurarse de que los valores estén en rangos razonables
                adjusted_home_xg = max(0.3, min(5.0, adjusted_home_xg))
                adjusted_away_xg = max(0.3, min(5.0, adjusted_away_xg))
                
                # Actualizar predicción
                prediction['original_home_xg'] = home_xg
                prediction['original_away_xg'] = away_xg
                prediction['home_xg'] = adjusted_home_xg
                prediction['away_xg'] = adjusted_away_xg
                prediction['coherence_analysis']['adjusted'] = True
                prediction['coherence_analysis']['adjustment_magnitude'] = abs(adjustment * 2)
                
                # Recalcular probabilidades basadas en xG ajustados
                self._recalculate_probabilities(prediction)
            
            # Guardar ratio en historial para calibración futura
            self._update_history(elo_diff, xg_diff)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error validando coherencia de predicción: {str(e)}")
            if 'coherence_analysis' not in prediction:
                prediction['coherence_analysis'] = {
                    'error': str(e),
                    'is_coherent': False,
                    'adjustment_needed': False
                }
            return prediction
    
    def _calculate_expected_xg_diff(self, elo_diff: float) -> float:
        """
        Calcula la diferencia de goles esperada basada en la diferencia de Elo.
        
        Args:
            elo_diff: Diferencia de Elo entre equipos (local - visitante)
            
        Returns:
            Diferencia de xG esperada
        """
        # Fórmula basada en análisis estadístico de la relación entre Elo y diferencia de goles
        # Cada 100 puntos de diferencia ~ 0.3 goles aproximadamente con ventaja local
        home_advantage_goals = 0.3  # Ventaja por jugar en casa en términos de goles
        elo_to_goals_factor = 0.003  # Cada punto Elo ~ 0.003 goles aproximadamente
        
        expected_xg_diff = (elo_diff * elo_to_goals_factor) + home_advantage_goals
        
        # Aplicar transformación sigmoide para limitar valores extremos
        if abs(expected_xg_diff) > 2.5:
            sign = 1 if expected_xg_diff > 0 else -1
            expected_xg_diff = sign * (2.0 + math.tanh(abs(expected_xg_diff) - 2.0))
            
        return expected_xg_diff
    
    def _recalculate_probabilities(self, prediction: Dict[str, Any]) -> None:
        """
        Recalcula las probabilidades de resultado basadas en xG ajustados.
        
        Args:
            prediction: Diccionario con la predicción que se actualizará in-place
        """
        # Extraer xG ajustados
        home_xg = prediction.get('home_xg', 1.5)
        away_xg = prediction.get('away_xg', 1.2)
        
        # Calcular probabilidades usando distribución de Poisson
        max_goals = 7
        result_probs = {"home_win": 0, "draw": 0, "away_win": 0}
        
        # Inicializar matriz de probabilidades
        prob_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        # Calcular matriz de probabilidades conjuntas
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                # Probabilidad de que equipo local marque i goles
                p_home = np.exp(-home_xg) * (home_xg ** i) / math.factorial(i)
                # Probabilidad de que equipo visitante marque j goles
                p_away = np.exp(-away_xg) * (away_xg ** j) / math.factorial(j)
                # Probabilidad conjunta
                prob_matrix[i, j] = p_home * p_away
        
        # Calcular probabilidades de resultados
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                if i > j:
                    result_probs["home_win"] += prob_matrix[i, j]
                elif i < j:
                    result_probs["away_win"] += prob_matrix[i, j]
                else:
                    result_probs["draw"] += prob_matrix[i, j]
        
        # Normalizar para asegurar que suman 1
        total_prob = sum(result_probs.values())
        for key in result_probs:
            result_probs[key] /= total_prob
        
        # Actualizar predicción con nuevas probabilidades
        prediction['prob_home_win'] = result_probs["home_win"]
        prediction['prob_draw'] = result_probs["draw"]
        prediction['prob_away_win'] = result_probs["away_win"]
        
        # Actualizar probabilidad de goles
        prediction['prob_over_2_5'] = self._calculate_over_under_probability(home_xg, away_xg, 2.5, 'over')
        prediction['prob_under_2_5'] = 1 - prediction['prob_over_2_5']
    
    def _calculate_over_under_probability(self, 
                                        home_xg: float, 
                                        away_xg: float, 
                                        line: float, 
                                        direction: str) -> float:
        """
        Calcula la probabilidad de over/under para una línea dada.
        
        Args:
            home_xg: Valor xG del equipo local
            away_xg: Valor xG del equipo visitante
            line: Línea de goles (por ejemplo, 2.5)
            direction: 'over' o 'under'
            
        Returns:
            Probabilidad para el mercado especificado
        """
        total_xg = home_xg + away_xg
        
        # Usar distribución de Poisson para calcular la probabilidad
        max_goals = 15  # Límite práctico para el cálculo
        total_goals_prob = [np.exp(-total_xg) * (total_xg ** n) / math.factorial(n) for n in range(max_goals + 1)]
        
        if direction == 'over':
            # Probabilidad de que la suma de goles sea > línea
            return sum(total_goals_prob[int(line) + 1:])
        else:
            # Probabilidad de que la suma de goles sea <= línea
            return sum(total_goals_prob[:int(line) + 1])
    
    def _update_history(self, elo_diff: float, xg_diff: float) -> None:
        """
        Actualiza el historial de ratios entre diferencia de Elo y diferencia de goles.
        
        Args:
            elo_diff: Diferencia de Elo
            xg_diff: Diferencia de goles esperados
        """
        # Evitar división por cero
        if abs(elo_diff) < 10:
            return
            
        # Calcular ratio
        ratio = xg_diff / elo_diff
        
        # Añadir a historial
        self.elo_goal_ratio_history.append(ratio)
        
        # Limitar tamaño del historial
        if len(self.elo_goal_ratio_history) > self.max_history_size:
            self.elo_goal_ratio_history.pop(0)
    
    def get_calibration_stats(self) -> Dict[str, float]:
        """
        Obtiene estadísticas de calibración basadas en el historial.
        
        Returns:
            Diccionario con estadísticas de calibración
        """
        if not self.elo_goal_ratio_history:
            return {
                'mean_ratio': 0.003,
                'std_ratio': 0.001,
                'samples': 0
            }
            
        return {
            'mean_ratio': np.mean(self.elo_goal_ratio_history),
            'std_ratio': np.std(self.elo_goal_ratio_history),
            'samples': len(self.elo_goal_ratio_history)
        }
    
    def validate_prediction_consistency(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida la consistencia interna de una predicción (probabilidades, xG, etc.).
        
        Args:
            prediction: Predicción a validar
            
        Returns:
            Predicción con información de validación
        """
        issues = []
        
        # Verificar que las probabilidades suman aproximadamente 1
        prob_sum = prediction.get('prob_home_win', 0) + prediction.get('prob_draw', 0) + prediction.get('prob_away_win', 0)
        if abs(prob_sum - 1.0) > 0.02:
            issues.append(f"Las probabilidades no suman 1: {prob_sum:.2f}")
        
        # Verificar que over y under 2.5 suman aproximadamente 1
        ou_sum = prediction.get('prob_over_2_5', 0) + prediction.get('prob_under_2_5', 0)
        if abs(ou_sum - 1.0) > 0.02:
            issues.append(f"Las probabilidades de over/under no suman 1: {ou_sum:.2f}")
        
        # Verificar coherencia entre xG y probabilidad de victoria
        home_xg = prediction.get('home_xg', 1.5)
        away_xg = prediction.get('away_xg', 1.2)
        prob_home = prediction.get('prob_home_win', 0.4)
        
        if home_xg > away_xg * 1.5 and prob_home < 0.4:
            issues.append(f"El xG favorece mucho al local pero su probabilidad de victoria es baja: {prob_home:.2f}")
        elif away_xg > home_xg * 1.5 and prob_home > 0.3:
            issues.append(f"El xG favorece mucho al visitante pero la probabilidad de victoria local es alta: {prob_home:.2f}")
            
        # Añadir resultado de validación
        prediction['validation'] = {
            'consistent': len(issues) == 0,
            'issues': issues
        }
        
        return prediction


# Función de conveniencia para usar directamente
def validate_prediction_coherence(prediction: Dict[str, Any], 
                               elo_ratings: Dict[str, float],
                               validate_only: bool = False) -> Dict[str, Any]:
    """
    Valida y opcionalmente ajusta la coherencia de una predicción.
    
    Args:
        prediction: Predicción a validar/ajustar
        elo_ratings: Ratings Elo de ambos equipos
        validate_only: Si es True, solo valida sin ajustar
        
    Returns:
        Predicción validada/ajustada
    """
    validator = CoherenceValidator()
    return validator.validate_and_adjust_goal_predictions(prediction, elo_ratings, validate_only)
