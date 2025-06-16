"""
Weather impact model for soccer predictions.
Based on academic research analyzing the effects of weather conditions on match outcomes.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_weather_adjustment_factors(weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate adjustment factors for different aspects of the game based on weather conditions.
    Research-based adjustments from analysis of historical matches.
    
    Args:
        weather_data: Dictionary containing weather information including:
            - condition: str (clear, rain, snow, etc.)
            - intensity: str (light, moderate, heavy)
            - temperature: float (in Celsius)
            - humidity: float (percentage)
            - wind_speed: float (km/h)
            
    Returns:
        Dictionary of adjustment factors for:
        - goals (total goals impact)
        - xg_home (home team xG impact)
        - xg_away (away team xG impact)
        - home_advantage (impact on home advantage)
    """
    if not weather_data:
        return {
            'goals': 1.0,
            'xg_home': 1.0,
            'xg_away': 1.0,
            'home_advantage': 1.0
        }
        
    try:
        factors = {
            'goals': 1.0,
            'xg_home': 1.0,
            'xg_away': 1.0,
            'home_advantage': 1.0
        }
        
        condition = weather_data.get('condition', '').lower()
        intensity = weather_data.get('intensity', '').lower()
        temperature = weather_data.get('temperature', 20)
        wind_speed = weather_data.get('wind_speed', 0)
        humidity = weather_data.get('humidity', 50)
        
        # Rain effects (research shows 8-15% reduction in scoring)
        if condition == 'rain':
            if intensity == 'heavy':
                factors['goals'] *= 0.85
                factors['xg_home'] *= 0.85
                factors['xg_away'] *= 0.85
                factors['home_advantage'] *= 1.05  # Home teams adapt better
            elif intensity == 'moderate':
                factors['goals'] *= 0.90
                factors['xg_home'] *= 0.90
                factors['xg_away'] *= 0.90
                factors['home_advantage'] *= 1.03
            else:  # light rain
                factors['goals'] *= 0.95
                factors['xg_home'] *= 0.95
                factors['xg_away'] *= 0.95
                factors['home_advantage'] *= 1.02
                
        # Snow effects (significant impact on scoring)
        elif condition == 'snow':
            factors['goals'] *= 0.80
            factors['xg_home'] *= 0.80
            factors['xg_away'] *= 0.80
            factors['home_advantage'] *= 1.10  # Home teams much more familiar with conditions
            
        # Wind effects (stronger impact on away teams)
        if wind_speed > 30:  # Strong wind
            factors['goals'] *= 0.88
            factors['xg_home'] *= 0.90
            factors['xg_away'] *= 0.85  # Away teams struggle more
            factors['home_advantage'] *= 1.08
        elif wind_speed > 20:  # Moderate wind
            factors['goals'] *= 0.93
            factors['xg_home'] *= 0.95
            factors['xg_away'] *= 0.90
            factors['home_advantage'] *= 1.05
            
        # Temperature effects
        if temperature < 5:  # Cold weather
            factors['goals'] *= 0.95
            factors['xg_home'] *= 0.95
            factors['xg_away'] *= 0.95
        elif temperature > 30:  # Hot weather
            factors['goals'] *= 0.93  # Slower pace
            factors['xg_home'] *= 0.93
            factors['xg_away'] *= 0.93
            
        # Humidity effects (high humidity reduces scoring)
        if humidity > 80:
            factors['goals'] *= 0.95
            factors['xg_home'] *= 0.95
            factors['xg_away'] *= 0.95
            
        return factors
        
    except Exception as e:
        logger.error(f"Error calculating weather adjustments: {e}")
        return {
            'goals': 1.0,
            'xg_home': 1.0,
            'xg_away': 1.0,
            'home_advantage': 1.0
        }
