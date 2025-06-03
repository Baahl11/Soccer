"""
Módulo para la integración de arquitectura Transformer para secuencias con el sistema existente.

Este módulo proporciona funciones para conectar el transformer de secuencias con los componentes
existentes del sistema de predicción, como el ensemble especializado y el flujo de predicción.
"""

import os
import sys
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import torch

# Importar módulos del sistema existente
from models import EnsemblePredictor
from psychological_features import PsychologicalFactorExtractor

# Importar módulo de transformers para secuencias
from sequence_transformer import (
    SequenceTransformerPredictor,
    predict_match_with_transformer,
    integrate_with_specialized_ensemble
)

# Configuración de logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class IntegratedPredictionSystem:
    """
    Sistema integrado de predicción que combina modelos tradicionales con Transformers.
    
    Esta clase coordina la generación de predicciones utilizando múltiples fuentes:
    1. Ensemble especializado de modelos tradicionales
    2. Transformer para modelado de secuencias temporales
    3. Características psicológicas
    """
    
    def __init__(
        self,
        transformer_model_path: str,
        feature_dim: int = 22,
        model_dim: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        sequence_length: int = 10,
        ensemble_models_config: Optional[Dict] = None,
        psychological_factors_enabled: bool = True,
        ensemble_weight: float = 0.6,
        transformer_weight: float = 0.3,
        psychological_weight: float = 0.1
    ):
        """
        Inicializa el sistema integrado de predicción.
        
        Args:
            transformer_model_path: Ruta al modelo transformer pre-entrenado
            feature_dim: Dimensión de features para el transformer
            model_dim: Dimensión interna del modelo transformer
            nhead: Número de cabezas de atención
            num_layers: Número de capas transformer
            sequence_length: Longitud de secuencia para el transformer
            ensemble_models_config: Configuración para los modelos de ensemble
            psychological_factors_enabled: Si se deben incluir factores psicológicos
            ensemble_weight: Peso del ensemble en la predicción final
            transformer_weight: Peso del transformer en la predicción final
            psychological_weight: Peso de factores psicológicos en la predicción final
        """
        self.sequence_length = sequence_length
        self.ensemble_weight = ensemble_weight
        self.transformer_weight = transformer_weight
        self.psychological_weight = psychological_weight
        
        # Inicializar predictor Transformer
        logger.info(f"Inicializando predictor Transformer desde {transformer_model_path}")
        if os.path.exists(transformer_model_path):
            self.transformer_predictor = SequenceTransformerPredictor(
                model_path=transformer_model_path,
                feature_dim=feature_dim,
                model_dim=model_dim,
                nhead=nhead,
                num_layers=num_layers,
                prediction_type='goals',
                sequence_length=sequence_length
            )
        else:
            logger.warning(f"No se encontró el modelo Transformer en {transformer_model_path}")
            self.transformer_predictor = None
        
        # Inicializar predictor de ensemble
        logger.info("Inicializando predictor de ensemble especializado")
        self.ensemble_predictor = EnsemblePredictor(
            models_config=ensemble_models_config
        )
        
        # Inicializar extractor de factores psicológicos si está habilitado
        self.psychological_factors_enabled = psychological_factors_enabled
        if psychological_factors_enabled:
            logger.info("Inicializando extractor de factores psicológicos")
            self.psychological_extractor = PsychologicalFactorExtractor()
        else:
            self.psychological_extractor = None
    
    def _prepare_ensemble_features(
        self,
        match_data: Dict[str, Any],
        league_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Convert match and league data into feature array for ensemble prediction.
        
        Args:
            match_data: Dictionary containing match information
            league_data: Optional league context data
            
        Returns:
            np.ndarray of features in the format expected by EnsemblePredictor
        """
        # Extract relevant features from match_data and league_data
        # This is a simplified version - in practice, you would extract more features
        features = []
        
        # Team IDs (encoded)
        home_team_id = int(match_data.get('home_team_id', 0))
        away_team_id = int(match_data.get('away_team_id', 0))
        features.extend([home_team_id, away_team_id])
        
        # Team rankings or ELO if available
        home_rank = float(match_data.get('home_team_rank', 0))
        away_rank = float(match_data.get('away_team_rank', 0))
        features.extend([home_rank, away_rank])
        
        # Recent form (if available)
        home_form = float(match_data.get('home_team_form', 0.5))
        away_form = float(match_data.get('away_team_form', 0.5))
        features.extend([home_form, away_form])
        
        # Home advantage factor
        features.append(1.0)  # Simple binary home advantage
        
        # League context features if available
        if league_data:
            league_id = int(league_data.get('league_id', 0))
            season = int(league_data.get('season', 0))
            features.extend([league_id, season])
        else:
            features.extend([0, 0])  # Default values
            
        return np.array(features).reshape(1, -1)  # Reshape for single prediction

    def predict_match(
        self,
        match_data: Dict[str, Any],
        previous_matches_home: List[Dict],
        previous_matches_away: List[Dict],
        league_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Realiza una predicción integrada para un partido.
        
        Args:
            match_data: Datos del partido a predecir
            previous_matches_home: Partidos previos del equipo local
            previous_matches_away: Partidos previos del equipo visitante
            league_data: Datos adicionales de la liga
            
        Returns:
            Predicción completa con contribuciones desglosadas
        """
        predictions = {}
        sources = []
        
        # 1. Predicción con ensemble especializado
        logger.info("Generando predicción con ensemble especializado")
        # Convert match and league data into features (X) format for EnsemblePredictor
        match_features = self._prepare_ensemble_features(match_data, league_data)
        ensemble_prediction = self.ensemble_predictor.predict(X=match_features)
        predictions['ensemble'] = ensemble_prediction
        sources.append('ensemble')
        
        # 2. Predicción con Transformer
        if self.transformer_predictor is not None:
            try:
                logger.info("Generando predicción con Transformer de secuencias")
                # Preparar secuencia para equipo local
                if len(previous_matches_home) > self.sequence_length:
                    previous_matches_home = previous_matches_home[-self.sequence_length:]
                
                # Predicción para equipo local
                transformer_home = self.transformer_predictor.predict_goals(
                    previous_matches_home
                )
                
                # Preparar secuencia para equipo visitante
                if len(previous_matches_away) > self.sequence_length:
                    previous_matches_away = previous_matches_away[-self.sequence_length:]
                
                # Predicción para equipo visitante
                transformer_away = self.transformer_predictor.predict_goals(
                    previous_matches_away
                )
                
                # Combinar predicciones
                transformer_prediction = {
                    'predicted_home_goals': transformer_home[0],  # Goles del equipo local
                    'predicted_away_goals': transformer_away[1]   # Goles del equipo visitante cuando es visitante
                }
                
                predictions['transformer'] = transformer_prediction
                sources.append('transformer')
            except Exception as e:
                logger.error(f"Error en predicción con Transformer: {e}")
        
        # 3. Factores psicológicos
        if self.psychological_factors_enabled and self.psychological_extractor is not None:
            try:
                logger.info("Analizando factores psicológicos")
                all_previous_matches = previous_matches_home + previous_matches_away
                psychological_factors = self.psychological_extractor.extract_features(
                    match_data=match_data,
                    historical_matches=all_previous_matches
                )
                predictions['psychological'] = psychological_factors
                sources.append('psychological')
            except Exception as e:
                logger.error(f"Error al extraer factores psicológicos: {e}")
                predictions['psychological'] = None
        
        # 4. Integrar predicciones
        integrated_prediction = self._integrate_predictions(predictions, sources)
        
        return integrated_prediction
    
    def _integrate_predictions(
        self,
        predictions: Dict[str, Dict],
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Integra predicciones de múltiples fuentes.
        
        Args:
            predictions: Diccionario con predicciones de diferentes fuentes
            sources: Lista de fuentes utilizadas
            
        Returns:
            Predicción integrada
        """
        # Inicializar predicción final
        final_prediction = {
            'predicted_home_goals': 0.0,
            'predicted_away_goals': 0.0,
            'source_contributions': {},
            'real_time_odds': {
                'home_win': 0.0,
                'draw': 0.0,
                'away_win': 0.0,
                'providers_data': {},
                'last_update': datetime.now().isoformat()
            }
        }

        # Procesar cada fuente de predicción
        for source in sources:
            weight = predictions[source].get('weight', 1.0)
            
            if source == 'odds':
                # Integrar datos de probabilidades en tiempo real
                odds_data = predictions[source].get('providers_data', {})
                final_prediction['real_time_odds']['providers_data'] = odds_data
                
                # Calcular probabilidades promedio ponderadas
                total_weight = 0
                for provider, data in odds_data.items():
                    provider_weight = data.get('reliability_score', 1.0)
                    total_weight += provider_weight
                    
                    final_prediction['real_time_odds']['home_win'] += data.get('home_win', 0.0) * provider_weight
                    final_prediction['real_time_odds']['draw'] += data.get('draw', 0.0) * provider_weight
                    final_prediction['real_time_odds']['away_win'] += data.get('away_win', 0.0) * provider_weight
                
                # Normalizar probabilidades
                if total_weight > 0:
                    final_prediction['real_time_odds']['home_win'] /= total_weight
                    final_prediction['real_time_odds']['draw'] /= total_weight
                    final_prediction['real_time_odds']['away_win'] /= total_weight
                
                # Contribución a la predicción de goles
                contribution_home = predictions[source].get('home_goals', 0) * weight
                contribution_away = predictions[source].get('away_goals', 0) * weight
                
                final_prediction['predicted_home_goals'] += contribution_home
                final_prediction['predicted_away_goals'] += contribution_away
                
                # Registrar contribución
                final_prediction['source_contributions'][source] = {
                    'weight': weight,
                    'home_goals_contribution': contribution_home,
                    'away_goals_contribution': contribution_away,
                    'odds_reliability': total_weight
                }
            elif source == 'ensemble' or source == 'transformer':
                # Añadir contribución ponderada a goles
                contribution_home = predictions[source]['predicted_home_goals'] * weight
                contribution_away = predictions[source]['predicted_away_goals'] * weight
                
                final_prediction['predicted_home_goals'] += contribution_home
                final_prediction['predicted_away_goals'] += contribution_away
                
                # Registrar contribución
                final_prediction['source_contributions'][source] = {
                    'weight': weight,
                    'home_goals_contribution': contribution_home,
                    'away_goals_contribution': contribution_away
                }
            
            elif source == 'psychological':
                # Añadir ajustes de factores psicológicos
                adjustment_home = predictions[source]['home_goals_adjustment'] * weight
                adjustment_away = predictions[source]['away_goals_adjustment'] * weight
                
                final_prediction['predicted_home_goals'] += adjustment_home
                final_prediction['predicted_away_goals'] += adjustment_away
                
                # Registrar contribución
                final_prediction['source_contributions'][source] = {
                    'weight': weight,
                    'home_goals_adjustment': adjustment_home,
                    'away_goals_adjustment': adjustment_away
                }
        
        # Añadir metadatos adicionales
        final_prediction['prediction_timestamp'] = datetime.now().isoformat()
        final_prediction['integration_method'] = 'weighted_average'
        
        # Asegurar valores no negativos
        final_prediction['predicted_home_goals'] = max(0, final_prediction['predicted_home_goals'])
        final_prediction['predicted_away_goals'] = max(0, final_prediction['predicted_away_goals'])
        
        logger.info(f"Predicción integrada: Local {final_prediction['predicted_home_goals']:.2f} - "
                  f"Visitante {final_prediction['predicted_away_goals']:.2f}")
        
        return final_prediction


def prepare_match_sequences(
    match_df: pd.DataFrame,
    team_id: str,
    last_n: int = 10,
    team_id_column: str = 'team_id'
) -> List[Dict]:
    """
    Prepara una secuencia de partidos para un equipo.
    
    Args:
        match_df: DataFrame con partidos históricos
        team_id: ID del equipo
        last_n: Número de partidos a considerar
        team_id_column: Nombre de la columna con el ID del equipo
        
    Returns:
        Lista de diccionarios con datos de partidos
    """
    # Filtrar partidos del equipo
    team_matches = match_df[match_df[team_id_column] == team_id].copy()
    
    # Ordenar por fecha
    if 'date' in team_matches.columns:
        team_matches = team_matches.sort_values(by='date')
    elif 'match_date' in team_matches.columns:
        team_matches = team_matches.sort_values(by='match_date')
    
    # Tomar los últimos N partidos
    last_matches = team_matches.tail(last_n)
    
    # Convertir a lista de diccionarios
    match_list = last_matches.to_dict('records')
    
    return match_list


def add_transformer_to_workflow(
    prediction_data: Dict[str, Any],
    transformer_model_path: str,
    previous_matches_home: List[Dict],
    previous_matches_away: List[Dict],
    sequence_length: int = 10
) -> Dict[str, Any]:
    """
    Añade predicciones de Transformer al flujo de trabajo existente.
    
    Args:
        prediction_data: Datos de predicción existentes
        transformer_model_path: Ruta al modelo transformer
        previous_matches_home: Partidos previos del equipo local
        previous_matches_away: Partidos previos del equipo visitante
        sequence_length: Longitud de secuencia
        
    Returns:
        Datos de predicción actualizados con componente Transformer
    """
    logger.info("Añadiendo predicción de Transformer al flujo de trabajo")
    
    try:
        # Limitar secuencias a longitud máxima
        if len(previous_matches_home) > sequence_length:
            previous_matches_home = previous_matches_home[-sequence_length:]
        
        if len(previous_matches_away) > sequence_length:
            previous_matches_away = previous_matches_away[-sequence_length:]
        
        # Generar predicción con Transformer para equipo local
        home_prediction = predict_match_with_transformer(
            model_path=transformer_model_path,
            previous_matches=previous_matches_home,
            prediction_type='goals'
        )
        
        # Generar predicción con Transformer para equipo visitante
        away_prediction = predict_match_with_transformer(
            model_path=transformer_model_path,
            previous_matches=previous_matches_away,
            prediction_type='goals'
        )
        
        # Combinar predicciones
        transformer_prediction = {
            'predicted_home_goals': home_prediction['predicted_home_goals'],
            'predicted_away_goals': away_prediction['predicted_away_goals']
        }
        
        # Añadir componente de predicción de Transformer
        prediction_data['transformer_component'] = {
            'home_goals': transformer_prediction['predicted_home_goals'],
            'away_goals': transformer_prediction['predicted_away_goals'],
            'sequence_length_used': min(len(previous_matches_home), len(previous_matches_away))
        }
        
        # Si existe una predicción raw, integrarla con la del Transformer
        if 'raw_predicted_home_goals' in prediction_data and 'raw_predicted_away_goals' in prediction_data:
            raw_prediction = {
                'predicted_home_goals': prediction_data['raw_predicted_home_goals'],
                'predicted_away_goals': prediction_data['raw_predicted_away_goals']
            }
            
            # Integrar predicciones (70% raw, 30% transformer)
            integrated = integrate_with_specialized_ensemble(
                transformer_prediction=transformer_prediction,
                ensemble_predictions=raw_prediction
            )
            
            # Actualizar predicción raw con la integrada
            prediction_data['raw_predicted_home_goals'] = integrated['predicted_home_goals']
            prediction_data['raw_predicted_away_goals'] = integrated['predicted_away_goals']
            prediction_data['integration_info'] = {
                'method': 'weighted_average',
                'components': ['raw_prediction', 'transformer_prediction'],
                'weights': {'raw_prediction': 0.7, 'transformer_prediction': 0.3}
            }
    
    except Exception as e:
        logger.error(f"Error al añadir predicción de Transformer: {e}")
        # Mantener la predicción original en caso de error
        prediction_data['transformer_error'] = str(e)
    
    return prediction_data
