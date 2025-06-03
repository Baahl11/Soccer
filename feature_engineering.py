"""Feature engineering for corner prediction model.

This module integrates multiple feature analyzers to generate a comprehensive
feature set for corner prediction, including:
- Schedule congestion analysis
- Seasonal effects 
- Match statistics
- Contextual factors
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from scipy.stats import poisson
from weather_api import WeatherConditions
from match_statistics import MatchStatisticsAnalyzer
from contextual_features import ContextualFeatures

logger = logging.getLogger(__name__)

class ScheduleCongestion:
    """Analyzes schedule congestion and team fatigue"""
    
    def __init__(self):
        self.recovery_threshold = 3  # minimum ideal recovery days
        self.congestion_window = 30  # window for analyzing congestion
        
    def analyze_schedule_load(self, matches: List[Dict[str, Any]], current_date: str) -> Dict[str, float]:
        """
        Analyzes schedule load for a team
        
        Args:
            matches: List of previous matches
            current_date: Date of analysis
            
        Returns:
            Dict with congestion metrics
        """
        if not matches:
            return self._get_default_load_metrics()
            
        try:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            
            # Analyze matches in congestion window
            recent_matches = [
                m for m in matches 
                if (current_dt - datetime.strptime(m.get('date', current_date), "%Y-%m-%d")).days <= self.congestion_window
            ]
            
            if not recent_matches:
                return self._get_default_load_metrics()
            
            # Calculate congestion metrics
            match_dates = [datetime.strptime(m['date'], "%Y-%m-%d") for m in recent_matches]
            rest_days: List[int] = []
            for i in range(1, len(match_dates)):
                rest = (match_dates[i] - match_dates[i-1]).days
                rest_days.append(rest)
                
            # Calculate metrics
            avg_rest = float(np.mean(rest_days) if rest_days else 7)
            min_rest = int(min(rest_days)) if rest_days else 7
            games_in_window = len(recent_matches)
            insufficient_rest = sum(1 for r in rest_days if r < self.recovery_threshold)
            
            # Calculate indices
            congestion_index = games_in_window / (self.congestion_window/7)  # Normalized by weeks
            fatigue_risk = float((insufficient_rest / max(1, len(rest_days))) * (1 + (1 / max(1, float(avg_rest)))))
            recovery_quality = float(min(1.0, float(avg_rest) / self.recovery_threshold))
            
            return {
                'congestion_index': float(min(2.0, congestion_index)),
                'fatigue_risk': float(min(1.0, fatigue_risk)),
                'recovery_quality': recovery_quality,
                'games_in_window': games_in_window,
                'avg_rest_days': float(avg_rest),
                'min_rest_days': float(min_rest),
                'high_risk_games': insufficient_rest
            }
            
        except Exception as e:
            logger.error(f"Error analyzing schedule load: {e}")
            return self._get_default_load_metrics()
    
    def _get_default_load_metrics(self) -> Dict[str, float]:
        """Returns default metrics when insufficient data"""
        return {
            'congestion_index': 1.0,
            'fatigue_risk': 0.5,
            'recovery_quality': 1.0,
            'games_in_window': 4,
            'avg_rest_days': 7.0,
            'min_rest_days': 5.0,
            'high_risk_games': 0
        }
        
class SeasonalAnalyzer:
    """Analyzes seasonal patterns and calendar effects"""
    
    def __init__(self):
        self.season_segments = {
            'early': (0.0, 0.25),
            'mid': (0.25, 0.75),
            'late': (0.75, 1.0)
        }
        self.high_intensity_months = [8, 12, 4, 5]  # August, December, April, May
        
    def analyze_seasonal_effects(self, matches: List[Dict[str, Any]], 
                               current_date: str,
                               season: int) -> Tuple[Dict[str, float], str]:
        """
        Analyzes seasonal patterns and effects
        
        Args:
            matches: Historical matches
            current_date: Current match date
            season: Season year/number
            
        Returns:
            Tuple of (seasonal metrics dict, season segment)
        """
        try:
            current_dt = datetime.strptime(current_date, "%Y-%m-%d")
            month = current_dt.month
            
            # Calculate season progress
            progress = self.get_season_progress(current_dt)
            segment = self.get_season_segment(progress)
            
            # Calculate monthly importance
            month_importance = self.calculate_month_importance(month)
            
            # Analyze historical pattern for this period
            segment_matches = [m for m in matches 
                             if self.get_season_segment(
                                 self.get_season_progress(
                                     datetime.strptime(m['date'], "%Y-%m-%d")
                                 )
                             ) == segment]
                             
            avg_corners = np.mean([m.get('total_corners', 10) for m in segment_matches]) if segment_matches else 10
            
            return {
                'season_progress': float(progress),
                'month_importance': float(month_importance),
                'segment_corners_avg': float(avg_corners)
            }, segment
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal effects: {e}")
            return {
                'season_progress': 0.5,
                'month_importance': 0.5,
                'segment_corners_avg': 10.0
            }, 'mid'
            
    def get_season_progress(self, date: datetime) -> float:
        """Calculates progress through the season (0-1)"""
        # Assuming season starts in August (8) and ends in May (5)
        month = date.month
        if month >= 8:
            progress = (month - 8) / 10
        else:
            progress = ((month + 4) / 10)
        return float(progress)
        
    def get_season_segment(self, progress: float) -> str:
        """Determines the season segment based on progress"""
        for segment, (start, end) in self.season_segments.items():
            if start <= progress < end:
                return segment
        return 'late'
        
    def calculate_month_importance(self, month: int) -> float:
        """Calculates the relative importance of the month"""
        if month in self.high_intensity_months:
            return 1.0
        elif month in [9, 1, 2]:  # Early/mid season
            return 0.7
        else:
            return 0.5

class AdvancedFeatureEngineering:
    """Main feature engineering coordinator"""
    
    def __init__(self):
        self.schedule_analyzer = ScheduleCongestion()
        self.seasonal_analyzer = SeasonalAnalyzer()
        self.stats_analyzer = MatchStatisticsAnalyzer()
        self.context_analyzer = ContextualFeatures()
        
    def extract_features(self, match_data: Dict[str, Any],
                        team_history: List[Dict[str, Any]],
                        current_date: str,
                        season: Optional[int] = None,
                        weather_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Extracts all features for corner prediction.
        
        Args:
            match_data: Current match information
            team_history: Historical match data
            current_date: Current match date
            season: Season number/year
            weather_data: Optional weather data
            
        Returns:
            Dict containing all calculated features
        """
        try:
            features = {}
            
            # Get schedule congestion features
            schedule_features = self.schedule_analyzer.analyze_schedule_load(
                team_history, current_date
            )
            features.update(schedule_features)
            
            # Get seasonal features
            if season:
                seasonal_features, _ = self.seasonal_analyzer.analyze_seasonal_effects(
                    team_history, current_date, season
                )
                features.update(seasonal_features)
                
            # Get match statistics features
            stats_features = self.stats_analyzer.analyze_match_stats(
                match_data, team_history
            )
            features.update(stats_features)
            
            # Get contextual features
            context_features = self.context_analyzer.analyze_context(
                match_data, team_history, weather_data
            )
            features.update(context_features)
            
            logger.info(f"Generated {len(features)} features for corner prediction")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {}