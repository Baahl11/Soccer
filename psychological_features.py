"""
Módulo para análisis de factores psicológicos en fútbol.

Este módulo implementa análisis de factores psicológicos cuantificables en fútbol 
basados en investigaciones recientes de la revista Psychology of Sport and Exercise (2025) 
que identificaron correlaciones significativas entre factores psicológicos y 
desviaciones en resultados esperados.

Funcionalidades principales:
- Índices de respuesta a la presión (rendimiento en partidos decisivos)
- Factores de momento (racha actual ajustada por oponentes)
- Métricas de "rebote" después de derrotas importantes
"""

from typing import Dict, List, Optional, Any, Union, TypedDict
from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)

def normalize_value(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Normaliza un valor al rango dado."""
    return float(min(max_val, max(min_val, value)))

def safe_float_division(num: float, denom: float, default: float = 0.0) -> float:
    """Divide dos números de forma segura con un valor predeterminado para el denominador cero."""
    try:
        if abs(denom) < 1e-10:  # Evitar división por números muy pequeños
            return default
        return float(num / denom)
    except (ZeroDivisionError, TypeError):
        return default

class RivalryFactors(TypedDict):
    home_team_rivalry_factor: float
    away_team_rivalry_factor: float
    rivalry_intensity: float

class MatchResult(TypedDict):
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    date: str

class PsychologicalFeatures(TypedDict):
    home_pressure_response: float
    away_pressure_response: float
    home_psychological_momentum: float
    away_psychological_momentum: float
    home_bounce_back_factor: float
    away_bounce_back_factor: float
    home_rivalry_factor: float
    away_rivalry_factor: float
    rivalry_intensity: float
    home_pressure_sensitivity: float
    away_pressure_sensitivity: float
    psychological_advantage: float
    pressure_differential: float

class PsychologicalFactors:
    """Analysis of quantifiable psychological factors in soccer."""
    
    def __init__(self):
        """Initialize psychological factors analyzer."""
        self._rivalry_data: Dict[str, RivalryFactors] = {}

    def _identify_high_pressure_matches(
        self,
        matches: List[Dict[str, Any]],
        team_id: int
    ) -> List[Dict[str, Any]]:
        """
        Identify matches with high psychological pressure based on:
        - Recent matches (last 5 games)
        - Important opponent strength
        - Goal difference significance
        """
        if not matches:
            return []

        sorted_matches = sorted(
            matches,
            key=lambda x: datetime.strptime(x.get('date', '1900-01-01'), "%Y-%m-%d")
        )
        
        high_pressure_matches = []
        recent_matches = sorted_matches[-5:]  # Focus on last 5 matches
        
        for match in recent_matches:
            opponent_id = match.get('home_team_id') if match.get('away_team_id') == team_id else match.get('away_team_id')
            if opponent_id is None:
                continue
                
            opponent_strength = self._calculate_opponent_strength(match, int(opponent_id))
            home_goals = int(match.get('home_goals', 0))
            away_goals = int(match.get('away_goals', 0))
            goal_diff = abs(home_goals - away_goals)
            
            # Consider a match high-pressure if:
            # 1. Strong opponent (strength > 0.7) OR
            # 2. Close game (goal difference <= 1) OR
            # 3. Recent match (last 3 games)
            if (opponent_strength > 0.7 or 
                goal_diff <= 1 or 
                match in sorted_matches[-3:]):
                high_pressure_matches.append(match)
        
        return high_pressure_matches

    def _calculate_opponent_strength(self, match: Dict[str, Any], opponent_id: int) -> float:
        """Calculate the relative strength of an opponent."""
        try:
            # Implementation depends on available data
            # For now, return default strength
            return 1.0
        except Exception as e:
            logger.error(f"Error calculating opponent strength: {e}")
            return 1.0
        
    def _get_match_result(self, match: Dict[str, Any], team_id: int) -> float:
        """Get normalized match result (-1 to 1 scale)."""
        try:
            is_home = match.get('home_team_id') == team_id
            home_goals = float(match.get('home_goals', 0))
            away_goals = float(match.get('away_goals', 0))
            
            if is_home:
                return safe_float_division(home_goals - away_goals, max(1.0, max(home_goals, away_goals)))
            return safe_float_division(away_goals - home_goals, max(1.0, max(home_goals, away_goals)))
        except Exception as e:
            logger.error(f"Error calculating match result: {e}")
            return 0.0

    def calculate_pressure_response_index(
        self,
        matches: List[Dict[str, Any]],
        team_id: int
    ) -> float:
        if not matches:
            return 0.5
        
        pressure_matches = self._identify_high_pressure_matches(matches, team_id)
        if not pressure_matches:
            return 0.5
            
        total_response = 0.0
        for match in pressure_matches:
            opponent_id = match.get('home_team_id') if match.get('away_team_id') == team_id else match.get('away_team_id')
            if opponent_id is None:
                continue
                
            opponent_strength = self._calculate_opponent_strength(match, int(opponent_id))
            match_result = self._get_match_result(match, team_id)
            
            total_response += match_result * opponent_strength
            
        pressure_index = safe_float_division(total_response, len(pressure_matches), 0.0)
        return normalize_value((pressure_index + 1) / 2)  # Normalize to [0,1]

    def calculate_psychological_momentum(
        self,
        matches: List[Dict[str, Any]],
        team_id: int,
        current_date: Optional[str] = None
    ) -> float:
        """Calculate psychological momentum (-1 to 1 scale)."""
        if not matches:
            return 0.0
            
        sorted_matches = sorted(
            matches,
            key=lambda x: datetime.strptime(x.get('date', '1900-01-01'), "%Y-%m-%d")
        )
        
        recent_matches = sorted_matches[-5:]  # Last 5 matches
        momentum = 0.0
        weight = 1.0
        
        for match in reversed(recent_matches):
            result = self._get_match_result(match, team_id)
            opponent_id = match.get('home_team_id') if match.get('away_team_id') == team_id else match.get('away_team_id')
            
            if opponent_id is not None:
                opponent_strength = self._calculate_opponent_strength(match, int(opponent_id))
                momentum += result * weight * opponent_strength
                
            weight *= 0.8  # Decay weight for older matches
            
        return normalize_value(momentum / 3.0, -1.0, 1.0)  # Normalize to [-1,1]

    def calculate_bounce_back_factor(
        self,
        matches: List[Dict[str, Any]],
        team_id: int,
        current_date: Optional[str] = None
    ) -> float:
        if not matches:
            return 0.5
            
        sorted_matches = sorted(
            matches,
            key=lambda x: datetime.strptime(x.get('date', '1900-01-01'), "%Y-%m-%d")
        )
        
        recent_matches = sorted_matches[-min(10, len(sorted_matches)):]
        bounce_back_strength = 0.0
        pattern_count = 0
        
        for i in range(1, len(recent_matches)):
            current_match = recent_matches[i]
            previous_match = recent_matches[i-1]
            
            prev_result = self._get_match_result(previous_match, team_id)
            
            if prev_result < 0:
                current_result = self._get_match_result(current_match, team_id)
                
                if current_result > prev_result:
                    bounce_strength = (current_result - prev_result) / 2
                    bounce_back_strength += bounce_strength
                    pattern_count += 1
        
        if pattern_count == 0:
            return 0.5
            
        avg_bounce = safe_float_division(bounce_back_strength, pattern_count, 0.0)
        normalized_bounce = (avg_bounce + 1) / 2
        
        last_match = recent_matches[-1]
        last_result = self._get_match_result(last_match, team_id)
        
        if last_result < -0.3:
            bounce_factor = 0.3 + (0.7 * normalized_bounce) - (last_result * 0.2)
            return normalize_value(bounce_factor)
        
        return 0.5

    def calculate_rivalry_impact(
        self,
        home_team_id: int,
        away_team_id: int,
        historical_matches: Optional[List[Dict[str, Any]]] = None
    ) -> RivalryFactors:
        rivalry_key = f"{min(home_team_id, away_team_id)}_{max(home_team_id, away_team_id)}"
        
        if rivalry_key in self._rivalry_data:
            return self._rivalry_data[rivalry_key]
        
        factors: RivalryFactors = {
            'home_team_rivalry_factor': 0.5,
            'away_team_rivalry_factor': 0.5,
            'rivalry_intensity': 0.3
        }
        
        if historical_matches:
            # Calculate rivalry metrics based on historical matches
            # This is a placeholder - actual implementation would analyze historical data
            pass
            
        self._rivalry_data[rivalry_key] = factors
        return factors

    def calculate_pressure_sensitivity(
        self,
        team_id: int,
        matches: List[Dict[str, Any]],
        high_pressure_matches: Optional[List[Dict[str, Any]]] = None
    ) -> float:
        if not matches:
            return 0.5
            
        if high_pressure_matches is None:
            high_pressure_matches = self._identify_high_pressure_matches(matches, team_id)
        
        if not high_pressure_matches:
            return 0.5
            
        high_pressure_perf = 0.0
        normal_pressure_perf = 0.0
        
        for match in matches:
            result = self._get_match_result(match, team_id)
            if match in high_pressure_matches:
                high_pressure_perf += result
            else:
                normal_pressure_perf += result
                
        hp_matches = len(high_pressure_matches)
        normal_matches = len(matches) - hp_matches
        
        hp_avg = safe_float_division(high_pressure_perf, hp_matches, 0.0)
        np_avg = safe_float_division(normal_pressure_perf, normal_matches, 0.0) if normal_matches > 0 else 0.0
        
        sensitivity = abs(hp_avg - np_avg)
        normalized_sensitivity = float(sensitivity / 2.0)  # Scale to [0,1]
        
        return normalize_value(normalized_sensitivity)

    def get_psychological_features(
        self,
        home_team_id: int,
        away_team_id: int,
        home_matches: List[Dict[str, Any]],
        away_matches: List[Dict[str, Any]],
        h2h_matches: Optional[List[Dict[str, Any]]] = None,
        match_date: Optional[str] = None
    ) -> PsychologicalFeatures:
        if match_date is None:
            match_date = datetime.now().strftime("%Y-%m-%d")
            
        # Calculate all psychological factors
        home_pressure_response = self.calculate_pressure_response_index(home_matches, home_team_id)
        away_pressure_response = self.calculate_pressure_response_index(away_matches, away_team_id)
        
        home_momentum = self.calculate_psychological_momentum(home_matches, home_team_id, match_date)
        away_momentum = self.calculate_psychological_momentum(away_matches, away_team_id, match_date)
        
        home_bounce = self.calculate_bounce_back_factor(home_matches, home_team_id, match_date)
        away_bounce = self.calculate_bounce_back_factor(away_matches, away_team_id, match_date)
        
        rivalry = self.calculate_rivalry_impact(home_team_id, away_team_id, h2h_matches)
        
        home_pressure_sensitivity = self.calculate_pressure_sensitivity(home_team_id, home_matches)
        away_pressure_sensitivity = self.calculate_pressure_sensitivity(away_team_id, away_matches)
        
        # Consolidate all features
        return {
            'home_pressure_response': home_pressure_response,
            'away_pressure_response': away_pressure_response,
            'home_psychological_momentum': home_momentum,
            'away_psychological_momentum': away_momentum,
            'home_bounce_back_factor': home_bounce,
            'away_bounce_back_factor': away_bounce,
            'home_rivalry_factor': rivalry['home_team_rivalry_factor'],
            'away_rivalry_factor': rivalry['away_team_rivalry_factor'],
            'rivalry_intensity': rivalry['rivalry_intensity'],
            'home_pressure_sensitivity': home_pressure_sensitivity,
            'away_pressure_sensitivity': away_pressure_sensitivity,
            'psychological_advantage': home_pressure_response - away_pressure_response + 
                                    (home_momentum - away_momentum) / 2 +
                                    (home_bounce - away_bounce) / 2,
            'pressure_differential': away_pressure_sensitivity - home_pressure_sensitivity
        }

class PsychologicalFactorExtractor:
    """
    Class to extract and manage psychological factors for the transformer integration.
    Acts as a high-level interface wrapping PsychologicalFactors functionality.
    """
    
    def __init__(self):
        """Initialize the psychological factor extractor with a factors analyzer."""
        self._factors_analyzer = PsychologicalFactors()
        
    def extract_features(
        self,
        match_data: Dict[str, Any],
        historical_matches: Optional[List[Dict[str, Any]]] = None,
        current_date: Optional[str] = None
    ) -> PsychologicalFeatures:
        """
        Extract psychological features for a match using available data.
        
        Args:
            match_data: Dictionary containing current match information
            historical_matches: Optional list of historical matches
            current_date: Optional current date string (YYYY-MM-DD)
            
        Returns:
            PsychologicalFeatures dictionary with all psychological factors
        """
        home_team_id = int(match_data.get('home_team_id', 0))
        away_team_id = int(match_data.get('away_team_id', 0))
        
        if not historical_matches:
            historical_matches = []
            
        # Calculate pressure response
        home_pressure = self._factors_analyzer.calculate_pressure_response_index(
            historical_matches, home_team_id
        )
        away_pressure = self._factors_analyzer.calculate_pressure_response_index(
            historical_matches, away_team_id
        )
        
        # Calculate psychological momentum
        home_momentum = self._factors_analyzer.calculate_psychological_momentum(
            historical_matches, home_team_id, current_date
        )
        away_momentum = self._factors_analyzer.calculate_psychological_momentum(
            historical_matches, away_team_id, current_date
        )
        
        # Calculate bounce back factors
        home_bounce = self._factors_analyzer.calculate_bounce_back_factor(
            historical_matches, home_team_id, current_date
        )
        away_bounce = self._factors_analyzer.calculate_bounce_back_factor(
            historical_matches, away_team_id, current_date
        )
        
        # Calculate rivalry impact
        rivalry_factors = self._factors_analyzer.calculate_rivalry_impact(
            home_team_id, away_team_id, historical_matches
        )
        
        # Calculate pressure sensitivity (differential between normal and high-pressure performance)
        pressure_diff = home_pressure - away_pressure
        psychological_adv = (home_momentum - away_momentum + home_bounce - away_bounce) / 3
        
        return {
            'home_pressure_response': home_pressure,
            'away_pressure_response': away_pressure,
            'home_psychological_momentum': home_momentum,
            'away_psychological_momentum': away_momentum,
            'home_bounce_back_factor': home_bounce,
            'away_bounce_back_factor': away_bounce,
            'home_rivalry_factor': rivalry_factors.get('home_team_rivalry_factor', 0.5),
            'away_rivalry_factor': rivalry_factors.get('away_team_rivalry_factor', 0.5),
            'rivalry_intensity': rivalry_factors.get('rivalry_intensity', 0.0),
            'home_pressure_sensitivity': home_pressure,
            'away_pressure_sensitivity': away_pressure,
            'pressure_differential': pressure_diff,
            'psychological_advantage': psychological_adv
        }
        
    def get_raw_factors(self) -> PsychologicalFactors:
        """Get access to the underlying factors analyzer for advanced usage."""
        return self._factors_analyzer
