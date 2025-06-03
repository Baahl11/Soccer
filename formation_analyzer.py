"""
Analizador táctico de formaciones de fútbol.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

class FormationAnalyzer:
    """Analiza aspectos tácticos de formaciones de fútbol"""
    
    def __init__(self):
        """Inicializa el analizador con datos tácticos base"""
        self.formation_styles = {
            '4-3-3': {
                'style': 'attacking',
                'width': 0.8,
                'pressing': 0.7,
                'possession': 0.7,
                'zones': {'attack': 0.8, 'midfield': 0.7, 'defense': 0.6}
            },
            '4-4-2': {
                'style': 'balanced',
                'width': 0.6,
                'pressing': 0.6,
                'possession': 0.5,
                'zones': {'attack': 0.6, 'midfield': 0.7, 'defense': 0.7}
            },
            '3-5-2': {
                'style': 'wing_play',
                'width': 0.9,
                'pressing': 0.6,
                'possession': 0.6,
                'zones': {'attack': 0.7, 'midfield': 0.8, 'defense': 0.5}
            },
            '5-3-2': {
                'style': 'defensive',
                'width': 0.5,
                'pressing': 0.4,
                'possession': 0.4,
                'zones': {'attack': 0.4, 'midfield': 0.6, 'defense': 0.9}
            },
            '4-2-3-1': {
                'style': 'possession',
                'width': 0.7,
                'pressing': 0.8,
                'possession': 0.8,
                'zones': {'attack': 0.7, 'midfield': 0.8, 'defense': 0.6}
            }
        }
    
    def analyze_formation(self, formation: str) -> Dict[str, Any]:
        """
        Analiza una formación específica.
        
        Args:
            formation: Formación en formato "X-X-X"
            
        Returns:
            Dict con análisis táctico
        """
        try:
            if formation not in self.formation_styles:
                formation = '4-4-2'  # formación por defecto
                
            style_data = self.formation_styles[formation]
            
            return {
                'formation': formation,
                'tactical_style': style_data['style'],
                'characteristics': {
                    'width_of_play': style_data['width'],
                    'pressing_intensity': style_data['pressing'],
                    'possession_orientation': style_data['possession']
                },
                'zone_control': style_data['zones']
            }
        except Exception as e:
            logger.error(f"Error analizando formación {formation}: {e}")
            return {}
    
    def compare_formations(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Compara dos formaciones y analiza sus ventajas tácticas.
        
        Args:
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Dict con análisis comparativo
        """
        try:
            home_analysis = self.analyze_formation(home_formation)
            away_analysis = self.analyze_formation(away_formation)
            
            if not home_analysis or not away_analysis:
                return {}
            
            # Calcular ventajas por zona
            zone_advantages = {}
            for zone in ['attack', 'midfield', 'defense']:
                home_control = home_analysis['zone_control'][zone]
                away_control = away_analysis['zone_control'][zone]
                zone_advantages[zone] = round(home_control - away_control, 2)
            
            # Calcular ventajas tácticas
            tactical_advantages = {
                'width_advantage': round(
                    home_analysis['characteristics']['width_of_play'] -
                    away_analysis['characteristics']['width_of_play'],
                    2
                ),
                'pressing_advantage': round(
                    home_analysis['characteristics']['pressing_intensity'] -
                    away_analysis['characteristics']['pressing_intensity'],
                    2
                ),
                'possession_advantage': round(
                    home_analysis['characteristics']['possession_orientation'] -
                    away_analysis['characteristics']['possession_orientation'],
                    2
                )
            }
            
            return {
                'zone_advantages': zone_advantages,
                'tactical_advantages': tactical_advantages,
                'home_style': home_analysis['tactical_style'],
                'away_style': away_analysis['tactical_style'],
                'expected_corner_impact': self.calculate_corner_impact(
                    zone_advantages,
                    tactical_advantages
                )
            }
            
        except Exception as e:
            logger.error(f"Error comparando formaciones: {e}")
            return {}
    
    def calculate_corner_impact(self,
                              zone_advantages: Dict[str, float],
                              tactical_advantages: Dict[str, float]) -> float:
        """
        Calcula el impacto esperado en la producción de corners.
        
        Args:
            zone_advantages: Ventajas por zona del campo
            tactical_advantages: Ventajas en aspectos tácticos
            
        Returns:
            Float indicando el impacto esperado (-1 a 1)
        """
        try:
            # Pesos de los factores
            weights = {
                'attack': 0.4,
                'midfield': 0.3,
                'defense': 0.3,
                'width': 0.4,
                'pressing': 0.3,
                'possession': 0.3
            }
            
            # Calcular impacto zonal
            zone_impact = (
                zone_advantages['attack'] * weights['attack'] +
                zone_advantages['midfield'] * weights['midfield'] +
                zone_advantages['defense'] * weights['defense']
            )
            
            # Calcular impacto táctico
            tactical_impact = (
                tactical_advantages['width_advantage'] * weights['width'] +
                tactical_advantages['pressing_advantage'] * weights['pressing'] +
                tactical_advantages['possession_advantage'] * weights['possession']
            )
            
            # Combinar impactos
            total_impact = (zone_impact + tactical_impact) / 2
            
            # Normalizar a rango -1 a 1
            return max(min(total_impact, 1), -1)
            
        except Exception as e:
            logger.error(f"Error calculando impacto en corners: {e}")
            return 0.0
        
    def analyze_formation_matchup(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza el matchup táctico entre dos formaciones.
        
        Args:
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Dict con análisis del matchup
        """
        try:
            # Obtener datos de formaciones
            home_data = self.formation_styles.get(home_formation, self.formation_styles['4-4-2'])
            away_data = self.formation_styles.get(away_formation, self.formation_styles['4-4-2'])
            
            # Calcular predicción de posesión
            possession_diff = home_data['possession'] - away_data['possession']
            predicted_possession = 50 + (possession_diff * 20)  # Convertir a escala 0-100
            
            # Analizar ventaja táctica
            home_strength = self._calculate_formation_strength(home_data)
            away_strength = self._calculate_formation_strength(away_data)
            
            strength_diff = home_strength - away_strength
            
            if strength_diff > 0.2:
                tactical_advantage = 'strong'
            elif strength_diff > 0.1:
                tactical_advantage = 'slight'
            elif strength_diff < -0.2:
                tactical_advantage = 'disadvantage'
            elif strength_diff < -0.1:
                tactical_advantage = 'slight disadvantage'
            else:
                tactical_advantage = 'neutral'
                
            # Analizar tendencia de córners
            wing_play_diff = home_data['width'] - away_data['width']
            pressing_diff = home_data['pressing'] - away_data['pressing']
            
            corner_tendency = self._analyze_corner_tendency(
                wing_play_diff,
                pressing_diff,
                possession_diff
            )
            
            return {
                'predicted_possession': predicted_possession,
                'tactical_advantage': tactical_advantage,
                'corner_tendency': corner_tendency,
                'key_matchups': self._analyze_zone_matchups(home_data, away_data)
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de matchup: {str(e)}")
            return {
                'predicted_possession': 50,
                'tactical_advantage': 'neutral',
                'corner_tendency': 'average',
                'key_matchups': {}
            }
            
    def _calculate_formation_strength(self, formation_data: Dict[str, Any]) -> float:
        """
        Calcula la fuerza general de una formación.
        """
        zones = formation_data.get('zones', {})
        return (
            zones.get('attack', 0.6) * 0.3 +
            zones.get('midfield', 0.6) * 0.4 +
            zones.get('defense', 0.6) * 0.3
        )
        
    def _analyze_corner_tendency(self,
                               wing_play_diff: float,
                               pressing_diff: float,
                               possession_diff: float) -> str:
        """
        Analiza la tendencia de córners basado en diferencias tácticas.
        """
        # Pesos para cada factor
        weights = [0.5, 0.3, 0.2]
        weighted_diff = (
            wing_play_diff * weights[0] +
            pressing_diff * weights[1] +
            possession_diff * weights[2]
        )
        
        if weighted_diff > 0.15:
            return 'high'
        elif weighted_diff > 0.05:
            return 'above_average'
        elif weighted_diff < -0.15:
            return 'low'
        elif weighted_diff < -0.05:
            return 'below_average'
        else:
            return 'average'
            
    def _analyze_zone_matchups(self,
                             home_data: Dict[str, Any],
                             away_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Analiza los matchups por zonas del campo.
        """
        home_zones = home_data.get('zones', {})
        away_zones = away_data.get('zones', {})
        
        results = {}
        for zone in ['attack', 'midfield', 'defense']:
            home_strength = home_zones.get(zone, 0.6)
            away_strength = away_zones.get(zone, 0.6)
            
            diff = home_strength - away_strength
            if diff > 0.15:
                results[zone] = 'strong advantage'
            elif diff > 0.05:
                results[zone] = 'slight advantage'
            elif diff < -0.15:
                results[zone] = 'strong disadvantage'
            elif diff < -0.05:
                results[zone] = 'slight disadvantage'
            else:
                results[zone] = 'neutral'
                
        return results
