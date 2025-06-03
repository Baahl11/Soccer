"""
Módulo para evaluar la confianza en las predicciones.

Este módulo implementa el cálculo dinámico de confianza para reemplazar
el valor estático de 0.65 que se usaba anteriormente.
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple, cast
from team_form import get_team_form, get_head_to_head_analysis
from calibration import PredictionConfidenceEvaluator

logger = logging.getLogger(__name__)

# Factores de confianza con sus pesos
CONFIDENCE_FACTORS = {
    'data_availability': 0.25,  # Disponibilidad de datos históricos
    'league_coverage': 0.20,    # Cobertura de la liga en nuestra base
    'team_stability': 0.15,     # Estabilidad de alineaciones y rendimiento
    'prediction_variance': 0.15, # Varianza entre diferentes modelos
    'historical_accuracy': 0.25  # Precisión histórica en esta liga/equipos
}

# Base de datos de cobertura de ligas (1.0 = cobertura completa)
LEAGUE_COVERAGE = {
    1: 0.95,   # Premier League
    2: 0.92,   # La Liga
    3: 0.90,   # Serie A
    4: 0.90,   # Bundesliga
    5: 0.85,   # Ligue 1
    6: 0.80,   # Primeira Liga
    7: 0.75,   # Eredivisie
    8: 0.85,   # Championship
    9: 0.75,   # Liga MX
    10: 0.70,  # MLS
    # Valores por defecto para otras ligas
}

def get_h2h_matches_count(team1_id: int, team2_id: int) -> int:
    """
    Obtener el número de partidos head-to-head entre dos equipos.
    
    Args:
        team1_id: ID del primer equipo
        team2_id: ID del segundo equipo
        
    Returns:
        Número de partidos entre ambos equipos
    """
    try:
        h2h_data = get_head_to_head_analysis(team1_id, team2_id)
        return h2h_data.get('total_matches', 0)
    except Exception as e:
        logger.error(f"Error obteniendo partidos h2h: {e}")
        return 0

def get_team_matches_count(team_id: int, league_id: Optional[int] = None) -> int:
    """
    Obtener el número de partidos de un equipo en una liga.
    
    Args:
        team_id: ID del equipo
        league_id: ID de la liga (opcional)
        
    Returns:
        Número de partidos del equipo
    """
    try:
        # Si league_id es None, usamos un valor predeterminado
        if league_id is None:
            default_league_id = 1  # Premier League como fallback
            team_form = get_team_form(team_id, league_id=default_league_id)
        else:
            team_form = get_team_form(team_id, league_id=league_id)
        return team_form.get('matches_played', 0)
    except Exception as e:
        logger.error(f"Error obteniendo partidos del equipo {team_id}: {e}")
        return 0

def get_league_coverage_score(league_id: int) -> float:
    """
    Obtener un score de cobertura para una liga específica.
    Algunas ligas tienen mejor cobertura que otras en nuestra base de datos.
    
    Args:
        league_id: ID de la liga
        
    Returns:
        Score de cobertura (0-1)
    """
    return LEAGUE_COVERAGE.get(league_id, 0.5)

def get_team_stability_score(team_id: int) -> float:
    """
    Evaluar la estabilidad de un equipo basado en cambios de alineación,
    transferencias recientes y consistencia de rendimiento.
    
    Args:
        team_id: ID del equipo
        
    Returns:
        Score de estabilidad (0-1)
    """
    try:
        # Como get_team_form requiere un league_id válido, usamos un valor predeterminado
        # cuando no tenemos una liga específica (1 = Premier League como fallback)
        default_league_id = 1  # Premier League como fallback
        team_form = get_team_form(team_id, league_id=default_league_id)
        
        if not team_form:
            return 0.5  # Valor neutral por defecto
        
        # Calculamos score basado en varias métricas:
        
        # 1. Consistencia en los resultados (desviación estándar baja = mayor consistencia)
        if 'form_values' in team_form and len(team_form['form_values']) >= 3:
            form_values = team_form['form_values']
            consistency = 1.0 - min(1.0, np.std(form_values) / 1.5)
        else:
            consistency = 0.5
        
        # 2. Estabilidad de alineación
        lineup_stability = team_form.get('lineup_stability', 0.5)
        
        # 3. Ausencia de lesiones importantes
        injury_factor = team_form.get('no_key_injuries', 0.5)
        
        # Combinamos los factores
        stability_score = (consistency * 0.4) + (lineup_stability * 0.4) + (injury_factor * 0.2)
        
        return min(1.0, max(0.0, stability_score))
        
    except Exception as e:
        logger.error(f"Error calculando estabilidad del equipo {team_id}: {e}")
        return 0.5  # Valor neutral por defecto

def get_historical_accuracy(league_id: int, team1_id: int, team2_id: int) -> float:
    """
    Evaluar la precisión histórica del modelo para esta liga y estos equipos.
    
    Args:
        league_id: ID de la liga
        team1_id: ID del primer equipo
        team2_id: ID del segundo equipo
        
    Returns:
        Score de precisión histórica (0-1)
    """
    try:
        # Intentar obtener métricas de precisión para la liga
        league_accuracy = _get_league_prediction_accuracy(league_id)
        
        # Intentar obtener métricas para los equipos específicos
        team1_accuracy = _get_team_prediction_accuracy(team1_id)
        team2_accuracy = _get_team_prediction_accuracy(team2_id)
        
        # Combinamos los resultados dando más peso a la precisión específica de los equipos
        weights = [0.4, 0.3, 0.3]  # Liga, Equipo1, Equipo2
        accuracy_values = [league_accuracy, team1_accuracy, team2_accuracy]
        
        # Calcular media ponderada de las precisiones disponibles
        available_values = [v for v in accuracy_values if v is not None]
        available_weights = weights[:len(available_values)]
        
        if not available_values:
            return 0.6  # Valor por defecto si no hay datos
            
        # Normalizar pesos
        norm_weights = [w/sum(available_weights) for w in available_weights]
        
        # Calcular media ponderada
        accuracy = sum(v * w for v, w in zip(available_values, norm_weights))
        
        return min(1.0, max(0.0, accuracy))
        
    except Exception as e:
        logger.error(f"Error calculando precisión histórica: {e}")
        return 0.6  # Valor moderado por defecto

def _get_league_prediction_accuracy(league_id: int) -> Optional[float]:
    """
    Obtener la precisión de predicción para una liga específica.
    """
    try:
        # En un sistema completo, esto vendría de una base de datos
        # de resultados históricos de predicciones
        league_accuracies = {
            1: 0.68,  # Premier League 
            2: 0.67,  # La Liga
            3: 0.65,  # Serie A
            4: 0.66,  # Bundesliga
            5: 0.64,  # Ligue 1
            # Más ligas...
        }
        return league_accuracies.get(league_id, 0.6)  # 0.6 como valor por defecto
    except Exception:
        return None

def _get_team_prediction_accuracy(team_id: int) -> Optional[float]:
    """
    Obtener la precisión de predicción para un equipo específico.
    """
    try:
        # En un sistema real, estos datos vendrían de una base de datos
        # con el histórico de predicciones para cada equipo
        # Aquí usamos valores ficticios para algunos equipos populares
        team_accuracies = {
            # IDs de equipos populares con sus precisiones
            1: 0.72,  # Example: Manchester City
            2: 0.70,  # Example: Liverpool
            3: 0.69,  # Example: Real Madrid
            4: 0.68,  # Example: Barcelona
            # Más equipos...
        }
        return team_accuracies.get(team_id)  # None si no existe
    except Exception:
        return None

def calculate_confidence_score(prediction_data: Dict[str, Any]) -> float:
    """
    Calcular un score de confianza dinámico basado en múltiples factores.
    
    Args:
        prediction_data: Datos de la predicción y el contexto
        
    Returns:
        Score de confianza entre 0.4 y 0.9
    """
    try:
        scores = {}
        
        # 1. Evaluar disponibilidad de datos
        team1_id = prediction_data.get('home_team_id')
        team2_id = prediction_data.get('away_team_id')
        
        if not team1_id or not team2_id:
            logger.warning("IDs de equipos no disponibles para calcular confianza")
            return 0.65  # Valor por defecto si faltan datos fundamentales
        
        h2h_matches = get_h2h_matches_count(team1_id, team2_id)
        team1_matches = get_team_matches_count(team1_id)
        team2_matches = get_team_matches_count(team2_id)
        
        # Escalar de 0 a 1
        # - Al menos 3 partidos h2h es ideal
        # - Al menos 10 partidos por equipo es ideal
        data_score = min(1.0, (h2h_matches/3 + team1_matches/10 + team2_matches/10) / 3)
        scores['data_availability'] = data_score
        
        # 2. Evaluar cobertura de liga
        league_id = prediction_data.get('league_id')
        league_score = get_league_coverage_score(league_id) if league_id else 0.5
        scores['league_coverage'] = league_score
        
        # 3. Evaluar estabilidad del equipo
        team1_stability = get_team_stability_score(team1_id)
        team2_stability = get_team_stability_score(team2_id)
        scores['team_stability'] = (team1_stability + team2_stability) / 2
        
        # 4. Evaluar varianza entre modelos de predicción
        if 'model_predictions' in prediction_data:
            predictions = prediction_data['model_predictions']
            if len(predictions) > 1:
                # Calcular coeficiente de variación en predicciones
                mean = np.mean(predictions)
                std = np.std(predictions)
                cv = std / mean if mean > 0 else 1.0
                variance_score = max(0, min(1, 1 - cv))
            else:
                variance_score = 0.5  # Valor neutral si solo hay un modelo
        else:
            variance_score = 0.5  # Valor neutral si no hay datos
        scores['prediction_variance'] = variance_score
          # 5. Evaluar precisión histórica
        if league_id is not None:
            accuracy_score = get_historical_accuracy(league_id, team1_id, team2_id)
        else:
            # Si no tenemos league_id, usamos valor neutral
            accuracy_score = 0.5
        scores['historical_accuracy'] = accuracy_score
        
        # Calcular puntuación final ponderada
        final_score = 0
        for factor, weight in CONFIDENCE_FACTORS.items():
            final_score += scores.get(factor, 0.5) * weight
            
        # Normalizar a un rango de 0.4 a 0.9
        # Nunca queremos 0 (sin confianza) ni 1 (confianza absoluta)
        normalized_score = 0.4 + (final_score * 0.5)
        
        logger.info(f"Confidence score calculated: {normalized_score:.2f}, factors: {scores}")
        return round(normalized_score, 2)
        
    except Exception as e:
        logger.error(f"Error calculando score de confianza: {e}")
        return 0.65  # Valor por defecto en caso de error
