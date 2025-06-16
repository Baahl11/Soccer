"""
League-specific calibration system for goal predictions.
Handles different league characteristics and adjusts predictions accordingly.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import logging
from datetime import datetime
import json
import os
from scipy.stats import poisson

logger = logging.getLogger(__name__)

class LeagueCalibrator:
    def __init__(self, league_data_path: str = 'data/league_characteristics.json'):
        """
        Initialize league calibration system.
        
        Args:
            league_data_path: Path to league characteristics data file
        """
        self.league_data_path = league_data_path
        self.league_characteristics = self._load_league_data()
        self.calibration_factors = {}
        
    def _load_league_data(self) -> Dict[str, Any]:
        """Load league characteristics data"""
        try:
            if os.path.exists(self.league_data_path):
                with open(self.league_data_path, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"League data file not found at {self.league_data_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading league data: {e}")
            return {}
            
    def calibrate_prediction(
        self,
        prediction: Dict[str, Any],
        league_id: int,
        match_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate a prediction based on league characteristics.
        
        Args:
            prediction: Original prediction
            league_id: League ID
            match_context: Additional context about the match
            
        Returns:
            Calibrated prediction
        """
        try:
            league_info = self.league_characteristics.get(str(league_id), {})
            if not league_info:
                logger.warning(f"No league info found for league {league_id}")
                return prediction
                
            # Get league characteristics
            avg_goals = league_info.get('avg_goals_per_game', 2.6)
            home_advantage = league_info.get('home_advantage_factor', 1.0)
            high_scoring = league_info.get('high_scoring_league', False)
            defensive_league = league_info.get('defensive_league', False)
            
            # Get historical stats
            historical = league_info.get('historical_stats', {})
            avg_home_goals = historical.get('avg_home_goals', avg_goals * 0.6)
            avg_away_goals = historical.get('avg_away_goals', avg_goals * 0.4)
            btts_pct = historical.get('btts_percentage', 50) / 100
            over_2_5_pct = historical.get('over_2_5_percentage', 50) / 100
            
            # Get season characteristics
            season = league_info.get('season_characteristics', {})
            season_phase_goals = self._get_season_phase_goals(match_context, season)
            
            # Apply calibrations
            calibrated = prediction.copy()
            
            # Adjust goals based on seasonal factors and historical data
            base_adjustment = season_phase_goals / avg_goals
            home_goals_factor = avg_home_goals / (avg_goals * 0.6)
            away_goals_factor = avg_away_goals / (avg_goals * 0.4)
            
            # Special league adjustments
            if high_scoring:
                base_adjustment *= 1.1
            if defensive_league:
                base_adjustment *= 0.9
                
            # Apply calibrated adjustments
            calibrated['home_xg'] = prediction['home_xg'] * base_adjustment * home_goals_factor * home_advantage
            calibrated['away_xg'] = prediction['away_xg'] * base_adjustment * away_goals_factor
            calibrated['total_xg'] = calibrated['home_xg'] + calibrated['away_xg']
            
            # Calculate additional predictions
            calibrated['btts_prob'] = self._calculate_btts_probability(
                calibrated['home_xg'],
                calibrated['away_xg'],
                btts_pct
            )
            
            calibrated['over_under'] = self._calculate_over_under_probabilities(
                calibrated['home_xg'],
                calibrated['away_xg'],
                over_2_5_pct
            )
            
            calibrated['exact_score_probs'] = self._calculate_exact_score_probabilities(
                calibrated['home_xg'],
                calibrated['away_xg']
            )
            
            # Add calibration metadata
            calibrated['calibration_info'] = {
                'league_id': league_id,
                'league_name': league_info.get('name', ''),
                'league_avg_goals': avg_goals,
                'home_advantage': home_advantage,
                'season_phase_goals': season_phase_goals,
                'base_adjustment': base_adjustment,
                'home_goals_factor': home_goals_factor,
                'away_goals_factor': away_goals_factor,
                'timestamp': datetime.now().isoformat()
            }
            
            return calibrated
            
        except Exception as e:
            logger.error(f"Error in league calibration: {e}")
            return prediction
            
    def _get_season_phase_goals(
        self,
        match_context: Optional[Dict[str, Any]],
        season_characteristics: Dict[str, Any]
    ) -> float:
        """Get average goals for current season phase"""
        if not match_context:
            return season_characteristics.get('mid_season_goals', 2.6)
            
        phase = match_context.get('season_phase', 'mid_season')
        if phase == 'early_season':
            return season_characteristics.get('early_season_goals', 2.6)
        elif phase == 'end_season':
            return season_characteristics.get('end_season_goals', 2.6)
        else:
            return season_characteristics.get('mid_season_goals', 2.6)
            
    def _calculate_btts_probability(
        self,
        home_xg: float,
        away_xg: float,
        league_btts_rate: float
    ) -> float:
        """Calculate both teams to score probability"""
        try:
            # Calculate raw probability
            prob_home_scores = 1 - poisson.pmf(0, home_xg)
            prob_away_scores = 1 - poisson.pmf(0, away_xg)
            raw_btts_prob = prob_home_scores * prob_away_scores
            
            # Blend with league average
            return round(0.7 * raw_btts_prob + 0.3 * league_btts_rate, 3)
            
        except Exception as e:
            logger.error(f"Error calculating BTTS probability: {e}")
            return 0.5
            
    def _calculate_over_under_probabilities(
        self,
        home_xg: float,
        away_xg: float,
        league_over_rate: float
    ) -> Dict[str, float]:
        """Calculate over/under probabilities for various thresholds"""
        try:
            results = {}
            thresholds = [0.5, 1.5, 2.5, 3.5, 4.5]
            
            for threshold in thresholds:
                # Calculate raw probability from Poisson
                raw_prob = 0
                for h in range(10):
                    for a in range(10):
                        if h + a > threshold:
                            raw_prob += (poisson.pmf(h, home_xg) * 
                                    poisson.pmf(a, away_xg))
                
                # For 2.5 threshold, blend with league average
                if threshold == 2.5:
                    prob = 0.7 * raw_prob + 0.3 * league_over_rate
                else:
                    prob = raw_prob
                    
                results[f'over_{threshold:.1f}'] = round(float(prob), 3)
                results[f'under_{threshold:.1f}'] = round(float(1 - prob), 3)
                
            return results
            
        except Exception as e:
            logger.error(f"Error calculating over/under probabilities: {e}")
            return {}
            
    def _calculate_exact_score_probabilities(
        self,
        home_xg: float,
        away_xg: float
    ) -> Dict[str, float]:
        """Calculate probabilities for exact scores"""
        try:
            results = {}
            max_goals = 5
            
            for h in range(max_goals + 1):
                for a in range(max_goals + 1):
                    prob = poisson.pmf(h, home_xg) * poisson.pmf(a, away_xg)
                    results[f'{h}-{a}'] = round(float(prob), 4)
                    
            # Add catchall for higher scores
            other_prob = 1 - sum(results.values())
            results['other'] = round(float(other_prob), 4)
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating exact scores: {e}")
            return {}
