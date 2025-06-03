"""Match statistics analyzer for corner prediction model.

This module implements match statistics features, analyzing team performance metrics like:
- Shots and shots on target
- Possession and passing
- Pressing and defensive actions
- Team playstyles and tactical indicators
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MatchStatisticsAnalyzer:
    """Analyzes match statistics for corner prediction."""
    
    def __init__(self):
        self.shot_impact_weight = 0.4  # Weight for shot-based features
        self.possession_weight = 0.3   # Weight for possession-based features
        self.pressure_weight = 0.3     # Weight for pressing/defensive features
        
    def analyze_match_stats(self, match_data: Dict[str, Any], 
                          team_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyzes match statistics to generate predictive features.
        
        Args:
            match_data: Current match statistics and context
            team_history: Previous matches data for the team
            
        Returns:
            Dict containing calculated features
        """
        try:
            # Initialize empty stats if not provided
            if not match_data or not team_history:
                return self._get_default_stats()
            
            # Calculate shot-based features
            shot_metrics = self._analyze_shot_patterns(match_data, team_history)
            
            # Calculate possession-based features  
            possession_metrics = self._analyze_possession(match_data, team_history)
            
            # Calculate pressing/defensive features
            pressing_metrics = self._analyze_pressing(match_data, team_history)
            
            # Combine all features with weights
            stats = {
                'shot_intensity_score': shot_metrics['shot_intensity'] * self.shot_impact_weight,
                'shot_accuracy_trend': shot_metrics['accuracy_trend'] * self.shot_impact_weight,
                'possession_dominance': possession_metrics['dominance'] * self.possession_weight,
                'passing_efficiency': possession_metrics['passing_score'] * self.possession_weight,
                'pressing_intensity': pressing_metrics['intensity'] * self.pressure_weight,
                'defensive_line_height': pressing_metrics['line_height'] * self.pressure_weight,
                'counter_attack_threat': self._calculate_counter_threat(shot_metrics, pressing_metrics),
                'set_piece_threat': self._calculate_set_piece_threat(match_data, team_history)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing match statistics: {e}")
            return self._get_default_stats()
            
    def _analyze_shot_patterns(self, match_data: Dict[str, Any], 
                             team_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyzes shot patterns and their effectiveness."""
        try:
            # Get current match shots
            shots = match_data.get('shots', 0)
            shots_on_target = match_data.get('shots_on_target', 0)
            
            # Calculate historical shot conversion
            hist_shots = [m.get('shots', 0) for m in team_history[-5:]]  # Last 5 matches
            hist_conversion = np.mean([m.get('goals', 0) / max(1, m.get('shots', 1)) 
                                    for m in team_history[-5:]])
            
            # Calculate metrics
            shot_intensity = shots / max(1, np.mean(hist_shots))
            accuracy_trend = shots_on_target / max(1, shots) - hist_conversion
            
            return {
                'shot_intensity': min(2.0, shot_intensity),  # Cap at 2.0
                'accuracy_trend': max(-1.0, min(1.0, accuracy_trend))  # Bound between -1 and 1
            }
            
        except Exception as e:
            logger.error(f"Error analyzing shot patterns: {e}")
            return {'shot_intensity': 1.0, 'accuracy_trend': 0.0}
            
    def _analyze_possession(self, match_data: Dict[str, Any],
                          team_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyzes possession patterns and passing effectiveness."""
        try:
            # Get possession metrics
            possession_pct = match_data.get('possession', 50) / 100
            passes_completed = match_data.get('passes_completed', 0)
            passes_attempted = match_data.get('passes_attempted', 1)
            
            # Calculate historical possession
            hist_possession = np.mean([m.get('possession', 50)/100 for m in team_history[-5:]])
            
            # Calculate metrics
            dominance = possession_pct / hist_possession
            passing_score = (passes_completed / max(1, passes_attempted) * 
                           (1 + 0.5 * (possession_pct - 0.5)))  # Bonus for high possession
            
            return {
                'dominance': min(2.0, dominance),
                'passing_score': min(1.0, passing_score)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing possession: {e}")
            return {'dominance': 1.0, 'passing_score': 0.5}
            
    def _analyze_pressing(self, match_data: Dict[str, Any],
                       team_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyzes pressing and defensive positioning."""
        try:
            # Get pressing metrics
            pressures = match_data.get('pressures', 0)
            pressure_regains = match_data.get('pressure_regains', 0)
            defensive_line = match_data.get('defensive_line_height', 50)  # 0-100 scale
            
            # Calculate historical pressing
            hist_pressures = np.mean([m.get('pressures', 0) for m in team_history[-5:]])
            
            # Calculate metrics
            intensity = pressures / max(1, hist_pressures)
            efficiency = pressure_regains / max(1, pressures)
            line_height = defensive_line / 50  # Normalize to 0-2 scale
            
            return {
                'intensity': min(2.0, intensity),
                'line_height': min(2.0, line_height)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pressing: {e}")
            return {'intensity': 1.0, 'line_height': 1.0}
            
    def _calculate_counter_threat(self, shot_metrics: Dict[str, float],
                                pressing_metrics: Dict[str, float]) -> float:
        """Calculates counter-attacking threat based on shot and pressing metrics."""
        try:
            # Combine shot intensity and pressing to estimate counter threat
            counter_score = (shot_metrics['shot_intensity'] * 0.6 +
                           pressing_metrics['intensity'] * 0.4)
                           
            return min(1.0, counter_score)
            
        except Exception as e:
            logger.error(f"Error calculating counter threat: {e}")
            return 0.5
            
    def _calculate_set_piece_threat(self, match_data: Dict[str, Any],
                                  team_history: List[Dict[str, Any]]) -> float:
        """Calculates set piece threat level based on historical conversion."""
        try:
            # Get set piece stats
            corners = match_data.get('corners', 0)
            free_kicks = match_data.get('free_kicks', 0)
            
            # Calculate historical conversion from set pieces
            hist_conversion = np.mean([
                (m.get('set_piece_goals', 0) / 
                 max(1, m.get('corners', 0) + m.get('free_kicks', 0)))
                for m in team_history[-10:]  # Use last 10 matches
            ])
            
            # Calculate current threat level
            current_threat = (corners + free_kicks) * hist_conversion
            
            return min(1.0, current_threat)
            
        except Exception as e:
            logger.error(f"Error calculating set piece threat: {e}")
            return 0.5
            
    def _get_default_stats(self) -> Dict[str, float]:
        """Returns default feature values when data is missing."""
        return {
            'shot_intensity_score': 1.0,
            'shot_accuracy_trend': 0.0,
            'possession_dominance': 1.0,
            'passing_efficiency': 0.5,
            'pressing_intensity': 1.0,
            'defensive_line_height': 1.0,
            'counter_attack_threat': 0.5,
            'set_piece_threat': 0.5
        }
