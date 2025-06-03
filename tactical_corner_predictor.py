"""
Predictor táctico de corners que integra análisis de formaciones con el modelo ensemble.
"""

import logging
from typing import Dict, Any, Optional
from formation_analyzer import FormationAnalyzer
from voting_ensemble_corners import VotingEnsembleCornersModel
from corner_data_collector import FormationDataCollector

logger = logging.getLogger(__name__)

class TacticalCornerPredictor:
    """
    Integra el análisis táctico de formaciones con predicciones de corners
    basadas en el modelo de conjunto de votación.
    """
    
    def __init__(self):
        """Inicializa los componentes del predictor táctico"""
        self.formation_analyzer = FormationAnalyzer()
        self.voting_ensemble = VotingEnsembleCornersModel()
        self.formation_collector = FormationDataCollector()

    def analyze_tactical_matchup(self, 
                               home_formation: str, 
                               away_formation: str) -> Dict[str, Any]:
        """
        Analiza el emparejamiento táctico entre formaciones.
        
        Args:
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Dict con análisis táctico
        """
        try:
            # Obtener características tácticas de cada formación
            home_analysis = self.formation_analyzer.analyze_formation_matchup(home_formation, away_formation)
            
            # Obtener características de formación
            home_features = self.formation_collector.get_formation_features(home_formation)
            away_features = self.formation_collector.get_formation_features(away_formation)

            # Calcular índices tácticos
            home_tactical_indices = self._calculate_tactical_indices(home_formation)
            away_tactical_indices = self._calculate_tactical_indices(away_formation)
            
            # Calcular ventaja táctica
            formation_advantage = self.formation_collector.calculate_matchup_advantage(
                home_formation, away_formation
            )

            return {
                'home_tactical_stats': home_features,
                'away_tactical_stats': away_features,
                'home_tactical_indices': home_tactical_indices,
                'away_tactical_indices': away_tactical_indices,
                'formation_advantage': formation_advantage,
                'matchup_analysis': home_analysis,
                'tactical_recommendation': self._generate_tactical_recommendation(home_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis táctico: {str(e)}")
            return self._get_default_tactical_analysis()

    def predict_with_formations(self,
                              match_data: Dict[str, Any],
                              context_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Realiza predicción incorporando análisis de formaciones.
        
        Args:
            match_data: Datos del partido incluyendo formaciones
            context_factors: Factores contextuales adicionales
            
        Returns:
            Dict con predicción y análisis táctico
        """
        try:
            # Obtener formaciones
            home_formation = match_data.get('home_formation', '4-4-2')
            away_formation = match_data.get('away_formation', '4-4-2')
            
            # Analizar matchup táctico
            tactical_analysis = self.analyze_tactical_matchup(home_formation, away_formation)
            
            # Enriquecer datos con análisis táctico
            enriched_data = self.formation_collector.enrich_match_data(match_data)
            
            # Obtener predicción base
            base_prediction = self.voting_ensemble.predict(enriched_data)
            
            # Ajustar predicción con factores tácticos
            adjusted_prediction = self._adjust_prediction_with_tactics(
                base_prediction,
                tactical_analysis,
                context_factors
            )

            return {
                'prediction': adjusted_prediction,
                'base_prediction': base_prediction,
                'tactical_analysis': tactical_analysis,
                'confidence_score': self._calculate_confidence_score(
                    base_prediction,
                    tactical_analysis,
                    context_factors
                )
            }

        except Exception as e:
            logger.error(f"Error en predicción con formaciones: {str(e)}")
            # En caso de error, retornar predicción base sin ajustes tácticos
            try:
                base_prediction = self.voting_ensemble.predict(match_data)
                return {
                    'prediction': base_prediction,
                    'base_prediction': base_prediction,
                    'tactical_analysis': self._get_default_tactical_analysis(),
                    'confidence_score': 0.5  # Confianza media por defecto
                }
            except Exception as e2:
                logger.error(f"Error en predicción base: {str(e2)}")
                raise

    def _calculate_tactical_indices(self, formation: str) -> Dict[str, float]:
        """
        Calcula índices tácticos para una formación.
        
        Args:
            formation: Formación a analizar (ej: '4-3-3')
            
        Returns:
            Dict con índices tácticos
        """
        try:
            # Obtener datos base de la formación
            formation_data = self.formation_analyzer.formation_styles.get(formation, {})
            
            if not formation_data:
                return self._get_default_tactical_indices()
            
            # Calcular índices específicos
            wing_play_index = (
                formation_data.get('width', 0.6) * 
                formation_data.get('zones', {}).get('attack', 0.6)
            )
            
            pressing_index = (
                formation_data.get('pressing', 0.6) * 
                formation_data.get('zones', {}).get('midfield', 0.6)
            )
            
            possession_index = formation_data.get('possession', 0.6)
            
            return {
                'wing_play_index': wing_play_index,
                'pressing_index': pressing_index,
                'possession_index': possession_index,
                'formation_aggression': self._calculate_formation_aggression(formation),
                'corner_potential': self._calculate_corner_potential(
                    wing_play_index,
                    pressing_index,
                    possession_index
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculando índices tácticos: {str(e)}")
            return self._get_default_tactical_indices()
            
    def _get_default_tactical_indices(self) -> Dict[str, float]:
        """
        Retorna índices tácticos por defecto para casos de error.
        """
        return {
            'wing_play_index': 0.6,
            'pressing_index': 0.6,
            'possession_index': 0.6,
            'formation_aggression': 0.6,
            'corner_potential': 0.6
        }
        
    def _get_default_tactical_analysis(self) -> Dict[str, Any]:
        """
        Retorna análisis táctico por defecto para casos de error.
        """
        return {
            'home_tactical_stats': self._get_default_tactical_indices(),
            'away_tactical_stats': self._get_default_tactical_indices(),
            'formation_advantage': 0,
            'matchup_analysis': {
                'predicted_possession': 50,
                'tactical_advantage': 'neutral',
                'corner_tendency': 'average'
            },
            'tactical_recommendation': 'Mantener estilo de juego balanceado'
        }
        
    def _calculate_formation_aggression(self, formation: str) -> float:
        """
        Calcula índice de agresividad de una formación.
        """
        try:
            parts = formation.split('-')
            defenders = int(parts[0])
            
            # Más defensores = menos agresivo
            base_aggression = 1 - (defenders / 10)  # 3 def = 0.7, 5 def = 0.5
            
            # Ajustar por medio campo y ataque
            if len(parts) >= 3:
                midfielders = int(parts[1])
                attackers = int(parts[2])
                
                midfield_factor = midfielders / 10  # 5 mid = 0.5 bonus
                attack_factor = attackers / 10  # 3 atk = 0.3 bonus
                
                return min(1.0, base_aggression + midfield_factor + attack_factor)
            
            return base_aggression
            
        except Exception:
            return 0.6  # Valor por defecto
            
    def _calculate_corner_potential(self,
                                  wing_play: float,
                                  pressing: float,
                                  possession: float) -> float:
        """
        Calcula potencial de generar córners basado en índices tácticos.
        """
        # Pesos relativos de cada factor
        weights = {
            'wing_play': 0.5,    # Mayor impacto en córners
            'pressing': 0.3,     # Impacto medio
            'possession': 0.2    # Menor impacto
        }
        
        weighted_sum = (
            wing_play * weights['wing_play'] +
            pressing * weights['pressing'] +
            possession * weights['possession']
        )
        
        return min(1.0, weighted_sum)

    def _adjust_prediction_with_tactics(self,
                                base_prediction: Dict[str, float],
                                tactical_analysis: Dict[str, Any],
                                context_factors: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Ajusta la predicción base usando análisis táctico y factores contextuales.
        
        Args:
            base_prediction: Predicción base del modelo ensemble
            tactical_analysis: Análisis táctico del matchup
            context_factors: Factores contextuales adicionales
            
        Returns:
            Dict con predicción ajustada
        """
        adjusted = base_prediction.copy()
        
        try:
            # Obtener factores de ajuste táctico
            formation_advantage = tactical_analysis.get('formation_advantage', 0)
            home_style = tactical_analysis.get('home_tactical_stats', {}).get('style', 'balanced')
            away_style = tactical_analysis.get('away_tactical_stats', {}).get('style', 'balanced')
            
            # Calcular multiplicadores tácticos
            home_multiplier = self._calculate_style_multiplier(home_style)
            away_multiplier = self._calculate_style_multiplier(away_style)
            
            # Aplicar ajustes a predicciones base
            adjusted['home_corners'] = base_prediction['home_corners'] * (1 + formation_advantage * home_multiplier)
            adjusted['away_corners'] = base_prediction['away_corners'] * (1 + formation_advantage * away_multiplier)
            adjusted['total_corners'] = base_prediction['total_corners']
            
            # Ajustar por factores contextuales si existen
            if context_factors:
                weather_impact = context_factors.get('weather_impact', 0)
                pitch_condition = context_factors.get('pitch_condition', 1)
                
                adjusted['home_corners'] *= (1 + weather_impact) * pitch_condition
                adjusted['away_corners'] *= (1 + weather_impact) * pitch_condition
                adjusted['total_corners'] = (
                    adjusted['home_corners'] + adjusted['away_corners']
                )
            
        except Exception as e:
            logger.error(f"Error en ajuste táctico: {str(e)}")
        
        return adjusted

    def _calculate_style_multiplier(self, style: str) -> float:
        """
        Calcula multiplicador basado en estilo de juego.
        """
        style_multipliers = {
            'attacking': 1.2,
            'possession': 1.1,
            'wing_play': 1.15,
            'balanced': 1.0,
            'defensive': 0.9,
            'counter': 0.95
        }
        return style_multipliers.get(style.lower(), 1.0)

    def _generate_tactical_recommendation(self, matchup_analysis: Dict[str, Any]) -> str:
        """
        Genera recomendación táctica basada en análisis de matchup.
        
        Args:
            matchup_analysis: Análisis del emparejamiento táctico
            
        Returns:
            String con recomendación táctica
        """
        try:
            tactical_advantage = matchup_analysis.get('tactical_advantage', 'neutral')
            predicted_possession = matchup_analysis.get('predicted_possession', 50)
            corner_tendency = matchup_analysis.get('corner_tendency', 'average')
            
            if tactical_advantage == 'strong':
                if corner_tendency == 'high':
                    return 'Mantener presión alta y juego por bandas para maximizar córners'
                return 'Aprovechar ventaja táctica con juego ofensivo'
                
            elif tactical_advantage == 'slight':
                if predicted_possession > 55:
                    return 'Mantener posesión y buscar desbordes por bandas'
                return 'Equilibrar ataque y defensa, aprovechar contraataques'
                
            elif tactical_advantage == 'neutral':
                return 'Mantener estilo de juego balanceado'
                
            else:  # disadvantage
                if corner_tendency == 'low':
                    return 'Priorizar defensa sólida y contraataques rápidos'
                return 'Ajustar táctica para contrarrestar ventaja rival'
                
        except Exception as e:
            logger.error(f"Error generando recomendación: {str(e)}")
            return 'Mantener estilo de juego balanceado'
            
    def _calculate_confidence_score(self,
                                  base_prediction: Dict[str, float],
                                  tactical_analysis: Dict[str, Any],
                                  context_factors: Optional[Dict[str, Any]] = None) -> float:
        """
        Calcula score de confianza para la predicción.
        
        Args:
            base_prediction: Predicción base del modelo
            tactical_analysis: Análisis táctico
            context_factors: Factores contextuales
            
        Returns:
            Float entre 0 y 1 indicando confianza
        """
        try:
            # Peso base del modelo ensemble
            base_confidence = base_prediction.get('model_confidence', 0.6)
            
            # Ajuste por calidad del análisis táctico
            tactical_quality = self._assess_tactical_quality(tactical_analysis)
            
            # Ajuste por factores contextuales
            context_quality = 1.0
            if context_factors:
                # Reducir confianza si hay factores que aumentan incertidumbre
                if context_factors.get('weather_impact', 0) > 0.2:
                    context_quality *= 0.9
                if context_factors.get('pitch_condition', 1) < 0.8:
                    context_quality *= 0.9
            
            # Combinar factores (ponderado)
            confidence = (
                base_confidence * 0.6 +  # Mayor peso a predicción base
                tactical_quality * 0.3 +  # Peso medio a análisis táctico
                context_quality * 0.1     # Menor peso a factores contextuales
            )
            
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            logger.error(f"Error calculando confianza: {str(e)}")
            return 0.5  # Confianza media por defecto
            
    def _assess_tactical_quality(self, tactical_analysis: Dict[str, Any]) -> float:
        """
        Evalúa la calidad del análisis táctico.
        
        Args:
            tactical_analysis: Análisis táctico completo
            
        Returns:
            Float entre 0 y 1 indicando calidad
        """
        try:
            # Verificar completitud del análisis
            required_keys = [
                'home_tactical_stats',
                'away_tactical_stats',
                'formation_advantage',
                'matchup_analysis'
            ]
            
            completeness = sum(
                1 for key in required_keys
                if key in tactical_analysis
            ) / len(required_keys)
            
            # Verificar validez de índices tácticos
            home_stats = tactical_analysis.get('home_tactical_stats', {})
            away_stats = tactical_analysis.get('away_tactical_stats', {})
            
            indices_validity = all(
                0 <= value <= 1
                for stats in [home_stats, away_stats]
                for value in stats.values()
                if isinstance(value, (int, float))
            )
            
            base_quality = 0.7 if indices_validity else 0.5
            
            return base_quality * completeness
            
        except Exception:
            return 0.5  # Calidad media por defecto

    def _get_default_prediction(self) -> Dict[str, Any]:
        """Retorna una predicción por defecto cuando hay errores"""
        return {
            'home_corners_prediction': 5.0,
            'away_corners_prediction': 4.0,
            'confidence': 0.5,
            'message': 'Predicción por defecto debido a error'
        }
