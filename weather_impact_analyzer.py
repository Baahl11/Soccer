from typing import Dict, Any
import logging

class WeatherImpactAnalyzer:
    """
    Analyzes the impact of weather conditions on team performance and tactics
    """
    
    WEATHER_IMPACTS = {
        'rain': {
            'impact_level': 0.7,  # High impact
            'affected_aspects': ['passing_accuracy', 'ball_control', 'shooting'],
            'tactical_adjustments': ['more_direct_play', 'less_possession']
        },
        'wind': {
            'impact_level': 0.6,
            'affected_aspects': ['long_passes', 'crosses', 'set_pieces'],
            'tactical_adjustments': ['ground_passes', 'short_passing']
        },
        'heat': {
            'impact_level': 0.5,
            'affected_aspects': ['pressing_intensity', 'running_distance'],
            'tactical_adjustments': ['slower_tempo', 'possession_based']
        },
        'cold': {
            'impact_level': 0.3,
            'affected_aspects': ['muscle_injuries', 'ball_control'],
            'tactical_adjustments': ['high_intensity', 'constant_movement']
        },
        'snow': {
            'impact_level': 0.8,  # Very high impact
            'affected_aspects': ['ball_movement', 'player_movement', 'visibility'],
            'tactical_adjustments': ['direct_play', 'aerial_duels']
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_weather_impact(self, weather_conditions: Dict[str, Any], 
                             team_style: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how weather conditions will impact a team's play style
        
        Args:
            weather_conditions: Weather data including temperature, precipitation, wind
            team_style: Team's tactical profile and playing style
            
        Returns:
            Dict with weather impact analysis
        """
        try:
            impact_scores = {}
            tactical_adjustments = []
            
            # Analyze each weather condition's impact
            for condition, data in weather_conditions.items():
                if condition in self.WEATHER_IMPACTS:
                    condition_impact = self.WEATHER_IMPACTS[condition]
                    
                    # Calculate impact based on condition severity and base impact level
                    severity = data.get('severity', 1.0)
                    impact_score = condition_impact['impact_level'] * severity
                    
                    impact_scores[condition] = {
                        'score': round(impact_score, 2),
                        'affected_aspects': condition_impact['affected_aspects'],
                        'recommended_adjustments': condition_impact['tactical_adjustments']
                    }
                    
                    tactical_adjustments.extend(condition_impact['tactical_adjustments'])
            
            # Calculate overall impact score
            overall_impact = sum(impact['score'] for impact in impact_scores.values()) / len(impact_scores) if impact_scores else 0
            
            return {
                'overall_impact_score': round(overall_impact, 2),
                'condition_impacts': impact_scores,
                'recommended_tactical_adjustments': list(set(tactical_adjustments)),  # Remove duplicates
                'risk_level': self._calculate_risk_level(overall_impact)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing weather impact: {e}")
            return {
                'overall_impact_score': 0,
                'condition_impacts': {},
                'recommended_tactical_adjustments': [],
                'risk_level': 'low'
            }
    
    def _calculate_risk_level(self, impact_score: float) -> str:
        """Calculate risk level based on impact score"""
        if impact_score > 0.7:
            return 'high'
        elif impact_score > 0.4:
            return 'medium'
        else:
            return 'low'
