"""Contextual feature analyzer for corner prediction model.

This module implements contextual features like:
- Stadium and attendance effects
- Head-to-head history
- Weather conditions
- Manager tactics and historical performance
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from weather_api import WeatherConditions

logger = logging.getLogger(__name__)

class ContextualFeatures:
    """Analyzes contextual match features for corner prediction."""
    
    def __init__(self):
        self.weather_impact = {
            'rain': -0.1,     # Slightly reduces corners
            'snow': -0.2,     # More significant reduction
            'wind': -0.15,    # Moderate reduction
            'clear': 0.0,     # Neutral
            'cloudy': 0.0     # Neutral
        }
        
    def analyze_context(self, match_data: Dict[str, Any],
                       h2h_history: List[Dict[str, Any]],
                       weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Analyzes contextual factors affecting corner probability.
        
        Args:
            match_data: Current match info including stadium, attendance etc
            h2h_history: Historical head-to-head matches
            weather_data: Weather conditions if available
            
        Returns:
            Dict containing calculated contextual features
        """
        try:
            # Initialize with default values
            context_features = self._get_default_features()
            
            if not match_data:
                return context_features
                
            # Calculate stadium impact
            context_features.update(
                self._analyze_stadium_effect(match_data)
            )
            
            # Calculate head-to-head patterns
            if h2h_history:
                context_features.update(
                    self._analyze_h2h_patterns(h2h_history)
                )
                
            # Add weather effects
            if weather_data:
                context_features.update(
                    self._analyze_weather_impact(weather_data)
                )
                
            # Add managerial impact
            context_features.update(
                self._analyze_manager_impact(match_data, h2h_history)
            )
            
            return context_features
            
        except Exception as e:
            logger.error(f"Error analyzing contextual features: {e}")
            return self._get_default_features()
            
    def _analyze_stadium_effect(self, match_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyzes impact of stadium and attendance."""
        try:
            # Get stadium data
            capacity = match_data.get('stadium_capacity', 30000)
            attendance = match_data.get('attendance', 0)
            pitch_size = match_data.get('pitch_dimensions', {'length': 105, 'width': 68})
            
            # Calculate attendance percentage
            attendance_pct = attendance / max(1, capacity)
            
            # Calculate pitch size impact (larger pitch = more corners)
            pitch_area = pitch_size.get('length', 105) * pitch_size.get('width', 68)
            standard_area = 105 * 68  # Standard pitch size
            pitch_factor = pitch_area / standard_area
            
            # Calculate stadium familiarity
            is_home = match_data.get('is_home', False)
            stadium_games = match_data.get('games_at_stadium', 10)
            familiarity = min(1.0, stadium_games / 20)  # Caps at 20 games
            
            return {
                'crowd_intensity': min(1.0, attendance_pct * 1.2),  # Boost small stadiums
                'pitch_size_effect': min(1.5, pitch_factor),
                'stadium_familiarity': familiarity if is_home else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing stadium effect: {e}")
            return {
                'crowd_intensity': 0.5,
                'pitch_size_effect': 1.0,
                'stadium_familiarity': 0.0
            }
            
    def _analyze_h2h_patterns(self, h2h_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyzes historical head-to-head patterns."""
        try:
            if not h2h_history:
                return {'h2h_corner_trend': 0.0, 'h2h_intensity': 0.5}
                
            # Get corners from recent h2h games
            recent_corners = [m.get('total_corners', 10) for m in h2h_history[-5:]]
            older_corners = [m.get('total_corners', 10) for m in h2h_history[:-5]]
            
            # Calculate corner trend
            recent_avg = np.mean(recent_corners) if recent_corners else 10
            historic_avg = np.mean(older_corners) if older_corners else 10
            
            corner_trend = (recent_avg - historic_avg) / max(1, historic_avg)
            
            # Calculate match intensity based on cards and fouls
            intensity_metrics = []
            for match in h2h_history[-5:]:
                cards = match.get('yellow_cards', 0) + match.get('red_cards', 0) * 2
                fouls = match.get('fouls', 0)
                intensity = (cards / 5 + fouls / 20) / 2  # Normalize to 0-1
                intensity_metrics.append(intensity)
                
            avg_intensity = np.mean(intensity_metrics) if intensity_metrics else 0.5
            
            return {
                'h2h_corner_trend': max(-0.5, min(0.5, corner_trend)),
                'h2h_intensity': min(1.0, avg_intensity)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing h2h patterns: {e}")
            return {'h2h_corner_trend': 0.0, 'h2h_intensity': 0.5}
            
    def _analyze_weather_impact(self, weather_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyzes impact of weather conditions."""
        try:
            conditions = weather_data.get('conditions', 'clear').lower()
            wind_speed = weather_data.get('wind_speed', 0)  # km/h
            precipitation = weather_data.get('precipitation', 0)  # mm
            temperature = weather_data.get('temperature', 15)  # Celsius
            
            # Get base weather impact
            weather_effect = self.weather_impact.get(conditions, 0.0)
            
            # Adjust for wind speed
            if wind_speed > 30:  # Strong wind
                weather_effect -= 0.2
            elif wind_speed > 20:  # Moderate wind
                weather_effect -= 0.1
                
            # Adjust for heavy rain/snow
            if precipitation > 10:  # Heavy precipitation
                weather_effect -= 0.1
                
            # Adjust for extreme temperatures
            temp_effect = 0.0
            if temperature < 5:
                temp_effect = -0.1
            elif temperature > 30:
                temp_effect = -0.05
                
            return {
                'weather_impact': max(-0.5, min(0.0, weather_effect)),
                'temperature_impact': max(-0.2, min(0.0, temp_effect))
            }
            
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}")
            return {'weather_impact': 0.0, 'temperature_impact': 0.0}
            
    def _analyze_manager_impact(self, match_data: Dict[str, Any],
                              h2h_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyzes impact of manager tactics and style."""
        try:
            home_manager = match_data.get('home_manager', {})
            away_manager = match_data.get('away_manager', {})
            
            # Get manager styles
            home_attacking = home_manager.get('attacking_preference', 0.5)  # 0-1 scale
            away_attacking = away_manager.get('attacking_preference', 0.5)
            
            # Calculate tactical mismatch
            tactical_mismatch = abs(home_attacking - away_attacking)
            
            # Get manager experience in h2h matches
            home_h2h_games = sum(1 for m in h2h_history 
                               if m.get('home_manager') == home_manager.get('name'))
            away_h2h_games = sum(1 for m in h2h_history
                               if m.get('away_manager') == away_manager.get('name'))
                               
            h2h_experience = (home_h2h_games + away_h2h_games) / max(2, len(h2h_history))
            
            return {
                'tactical_mismatch': min(1.0, tactical_mismatch * 2),
                'manager_h2h_experience': min(1.0, h2h_experience)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing manager impact: {e}")
            return {'tactical_mismatch': 0.0, 'manager_h2h_experience': 0.0}
            
    def _get_default_features(self) -> Dict[str, float]:
        """Returns default feature values when data is missing."""
        return {
            'crowd_intensity': 0.5,
            'pitch_size_effect': 1.0,
            'stadium_familiarity': 0.0,
            'h2h_corner_trend': 0.0,
            'h2h_intensity': 0.5,
            'weather_impact': 0.0,
            'temperature_impact': 0.0,
            'tactical_mismatch': 0.0,
            'manager_h2h_experience': 0.0
        }
