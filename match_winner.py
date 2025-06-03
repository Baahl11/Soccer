"""
Soccer Match Winner Prediction Module

This module provides functions to predict match winners (home win, draw, away win) with confidence
percentages for soccer matches. It uses a combination of expected goals (xG), team form data,
head-to-head history, and contextual factors to generate predictions with associated confidence levels.

The implementation is based on academic research in soccer analytics, primarily:
1. Dixon, M.J., Coles, S.G. (1997) "Modelling Association Football Scores and Inefficiencies in the Football Betting Market"
2. Baio, G., Blangiardo, M. (2010) "Bayesian hierarchical model for the prediction of football results"
3. Koopman, S.J., Lit, R. (2019) "Forecasting Football Match Results in National League Competitions Using Score-Driven Models"
"""

import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from scipy.stats import poisson, nbinom
from enum import Enum

logger = logging.getLogger(__name__)

class MatchOutcome(Enum):
    """Enum representing possible match outcomes"""
    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"
    
class WinnerPredictor:
    """
    Class for predicting match winners with confidence ratings.
    
    This class combines expected goals (xG) predictions with team form, head-to-head data,
    and contextual factors to produce match winner predictions with associated confidence levels.
    """
    
    def __init__(self):
        """Initialize the match winner predictor"""
        # Correlation parameter for low-scoring matches (Dixon-Coles adjustment)
        self.rho = 0.13
        
        # Factors affecting confidence calculation
        self.consistency_weight = 0.3
        self.data_quality_weight = 0.25
        self.context_weight = 0.2
        self.league_weight = 0.25
        
        # Major leagues with higher data quality
        self.major_leagues = [39, 78, 140, 135, 61]  # Premier League, Bundesliga, La Liga, Serie A, Ligue 1
        
    def predict_winner(
        self,
        home_xg: float,
        away_xg: float,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h: Dict[str, Any],
        league_id: int,
        context_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Predict the match winner with confidence percentages.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            home_form: Form data for home team
            away_form: Form data for away team
            h2h: Head-to-head data between teams
            league_id: League ID 
            context_factors: Optional contextual factors (weather, injuries, etc.)
            
        Returns:
            Dictionary with prediction results
        """
        # 1. Calculate basic 1X2 probabilities using Dixon-Coles adjusted Poisson
        home_win_prob, draw_prob, away_win_prob = self._calculate_1x2_probabilities(home_xg, away_xg)
        
        # 2. Refine probability calculations using team form and head-to-head data
        refined_probs = self._refine_probabilities(
            home_win_prob, draw_prob, away_win_prob, 
            home_form, away_form, h2h
        )
        
        # 3. Apply context adjustments if provided
        if context_factors:
            refined_probs = self._apply_context_adjustments(refined_probs, context_factors)
        
        # 4. Determine most likely outcome
        most_likely_outcome = self._get_most_likely_outcome(refined_probs)
          # 5. Calculate confidence level
        confidence_score, confidence_factors = self._calculate_confidence(
            refined_probs, 
            home_form, 
            away_form, 
            league_id,
            context_factors
        )
        
        # 6. Format final result
        return self._format_result(
            refined_probs, 
            most_likely_outcome, 
            confidence_score,
            confidence_factors
        )
    
    def _calculate_1x2_probabilities(
        self, 
        home_xg: float, 
        away_xg: float
    ) -> Tuple[float, float, float]:
        """
        Calculate 1X2 probabilities using Dixon-Coles adjusted Poisson.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            
        Returns:
            Tuple of (home_win_prob, draw_prob, away_win_prob)
        """
        # Maximum goals to consider in probability calculation
        max_goals = 10
        
        # Initialize probability counters
        home_win_prob = 0.0
        draw_prob = 0.0
        away_win_prob = 0.0
        
        # Calculate probabilities for all possible score combinations
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                # Get basic Poisson probabilities
                p_h = poisson.pmf(h, home_xg)
                p_a = poisson.pmf(a, away_xg)
                
                # Apply Dixon-Coles adjustment for low-scoring games
                if h <= 1 and a <= 1:
                    if h == 0 and a == 0:
                        p = p_h * p_a * (1 + self.rho)
                    elif h == 0 and a == 1:
                        p = p_h * p_a * (1 - self.rho)
                    elif h == 1 and a == 0:
                        p = p_h * p_a * (1 - self.rho)
                    elif h == 1 and a == 1:
                        p = p_h * p_a * (1 + self.rho)
                    else:
                        p = p_h * p_a
                else:
                    p = p_h * p_a
                
                # Assign probability to appropriate outcome
                if h > a:
                    home_win_prob += p
                elif h == a:
                    draw_prob += p
                else:
                    away_win_prob += p
          # Normalize probabilities to ensure they sum to 1
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
              # Apply home advantage factor if not already reflected in xG values
        # This ensures home team gets significant advantage in close scenarios
        home_advantage = 0.10  # 10% home advantage boost
        
        # For equal xG, adjust draw probability upward to meet expectations
        if abs(home_xg - away_xg) < 0.1:
            # For very even matches, ensure draw probability is at least 27%
            if draw_prob < 0.27:
                shortage = 0.27 - draw_prob
                home_win_prob -= shortage * 0.6
                away_win_prob -= shortage * 0.4
                draw_prob = 0.27
                
        # Then add home advantage        
        if abs(home_xg - away_xg) < 0.5:
            # Scale advantage based on how close the xG values are
            adjustment_factor = 1.0 - (abs(home_xg - away_xg) / 0.5)
            adjustment = home_advantage * adjustment_factor
            
            # Take probability mostly from away_win, preserve draw
            home_win_prob += adjustment
            away_win_prob -= adjustment
            
            # Make sure no probability goes negative
            if away_win_prob < 0:
                draw_prob += away_win_prob
                away_win_prob = 0
            if draw_prob < 0:
                home_win_prob += draw_prob
                draw_prob = 0
                
        # Ensure probabilities sum to exactly 1.0
        total_prob = home_win_prob + draw_prob + away_win_prob
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
            
        return float(home_win_prob), float(draw_prob), float(away_win_prob)
    
    def _refine_probabilities(
        self,
        home_win_prob: float,
        draw_prob: float,
        away_win_prob: float,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Refine probability calculations using team form and head-to-head data.
        
        Args:
            home_win_prob: Initial home win probability
            draw_prob: Initial draw probability
            away_win_prob: Initial away win probability
            home_form: Form data for home team
            away_form: Form data for away team
            h2h: Head-to-head data
            
        Returns:
            Dictionary with refined probabilities
        """
        # Convert probabilities to odds ratios for easier manipulation
        home_odds = (1 - home_win_prob) / home_win_prob if home_win_prob > 0 else float('inf')
        draw_odds = (1 - draw_prob) / draw_prob if draw_prob > 0 else float('inf')
        away_odds = (1 - away_win_prob) / away_win_prob if away_win_prob > 0 else float('inf')
        
        # Form-based adjustments
        home_form_factor = min(1.3, max(0.7, 1.0 + (home_form.get('form_trend', 0.0) * 0.15)))
        away_form_factor = min(1.3, max(0.7, 1.0 + (away_form.get('form_trend', 0.0) * 0.15)))
        
        # Apply form factors to odds (lower odds = higher probability)
        home_odds /= home_form_factor
        away_odds /= away_form_factor
        
        # Head-to-head adjustment
        if h2h and h2h.get('matches_played', 0) > 3:  # Minimum matches for reliable h2h data
            h2h_home_win_pct = h2h.get('home_win_pct', 0.5)
            h2h_draw_pct = h2h.get('draw_pct', 0.25)
            h2h_away_win_pct = h2h.get('away_win_pct', 0.25)
            
            # Adjust based on historical head-to-head performance (with a 20% weight)
            h2h_weight = 0.2
            home_odds = home_odds * (1 - h2h_weight) + (1 - h2h_home_win_pct) / max(0.01, h2h_home_win_pct) * h2h_weight
            draw_odds = draw_odds * (1 - h2h_weight) + (1 - h2h_draw_pct) / max(0.01, h2h_draw_pct) * h2h_weight
            away_odds = away_odds * (1 - h2h_weight) + (1 - h2h_away_win_pct) / max(0.01, h2h_away_win_pct) * h2h_weight
        
        # Convert odds back to probabilities
        home_win_prob_refined = 1 / (1 + home_odds) if home_odds != float('inf') else 0
        draw_prob_refined = 1 / (1 + draw_odds) if draw_odds != float('inf') else 0
        away_win_prob_refined = 1 / (1 + away_odds) if away_odds != float('inf') else 0
          # Normalize to ensure probabilities sum to 1
        total_prob = home_win_prob_refined + draw_prob_refined + away_win_prob_refined
        if total_prob > 0:
            home_win_prob_refined /= total_prob
            draw_prob_refined /= total_prob
            away_win_prob_refined /= total_prob
            
        # Convert to float values with exact 5 decimal places without rounding
        home_win = float(home_win_prob_refined)
        draw = float(draw_prob_refined)
        away_win = float(away_win_prob_refined)
          # Make final adjustment to ensure sum is EXACTLY 1.0
        total = home_win + draw + away_win
        
        # Force exact equality to 1.0 regardless of floating-point precision
        if abs(total - 1.0) > 1e-14:
            # Distribute the tiny difference to the largest probability
            if home_win >= draw and home_win >= away_win:
                home_win = home_win + (1.0 - total)
            elif draw >= home_win and draw >= away_win:
                draw = draw + (1.0 - total)
            else:
                away_win = away_win + (1.0 - total)
                
        # Double-check that we have exactly 1.0 now
        result = {
            MatchOutcome.HOME_WIN.value: round(home_win, 4),
            MatchOutcome.DRAW.value: round(draw, 4),
            MatchOutcome.AWAY_WIN.value: round(away_win, 4)
        }
          # Verify and adjust rounded values to ensure they sum to 1.0 exactly
        rounded_sum = sum(result.values())
        if abs(rounded_sum - 1.0) > 1e-10:
            # Find the maximum value and adjust it
            max_key = max(result.keys(), key=lambda k: result[k])
            result[max_key] = round(result[max_key] + (1.0 - rounded_sum), 4)
            
        return result
    
    def _apply_context_adjustments(
        self,
        probabilities: Dict[str, float],
        context_factors: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Apply context adjustments to probabilities.
        
        Args:
            probabilities: Dictionary with probabilities for each outcome
            context_factors: Dictionary with contextual factors
            
        Returns:
            Dictionary with adjusted probabilities
        """
        home_win_prob = probabilities[MatchOutcome.HOME_WIN.value]
        draw_prob = probabilities[MatchOutcome.DRAW.value]
        away_win_prob = probabilities[MatchOutcome.AWAY_WIN.value]
        
        # Weather adjustments
        if context_factors.get('is_rainy', False) or context_factors.get('is_snowy', False):
            # Bad weather tends to favor lower scoring, which increases draw probability
            draw_boost = 0.05
            home_factor = (1.0 - draw_boost) * home_win_prob / (home_win_prob + away_win_prob)
            away_factor = (1.0 - draw_boost) * away_win_prob / (home_win_prob + away_win_prob)
            
            draw_prob += draw_boost
            home_win_prob = home_win_prob * (1.0 - draw_boost) * home_factor
            away_win_prob = away_win_prob * (1.0 - draw_boost) * away_factor
            
        # Injury adjustments
        if context_factors.get('home_injuries_impact', 0) > 0:
            # Significant home team injuries reduce home win probability
            impact = min(0.15, context_factors.get('home_injuries_impact', 0) * 0.05)
            home_win_prob -= impact
            # Distribute the probability reduction to draw and away win
            draw_prob += impact * 0.4
            away_win_prob += impact * 0.6
            
        if context_factors.get('away_injuries_impact', 0) > 0:
            # Significant away team injuries reduce away win probability
            impact = min(0.15, context_factors.get('away_injuries_impact', 0) * 0.05)
            away_win_prob -= impact
            # Distribute the probability reduction to draw and home win
            draw_prob += impact * 0.4
            home_win_prob += impact * 0.6
            
        # Match importance adjustment
        if context_factors.get('high_stakes', False):
            # High-stakes matches tend to be more conservative
            if home_win_prob > away_win_prob:
                # Favorite team plays more cautiously
                draw_boost = min(0.08, home_win_prob * 0.15)
                home_win_prob -= draw_boost
                draw_prob += draw_boost
            elif away_win_prob > home_win_prob:
                # Favorite team plays more cautiously
                draw_boost = min(0.08, away_win_prob * 0.15)
                away_win_prob -= draw_boost
                draw_prob += draw_boost
        
        # Normalize to ensure probabilities sum to 1
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
        
        return {
            MatchOutcome.HOME_WIN.value: round(float(home_win_prob), 4),
            MatchOutcome.DRAW.value: round(float(draw_prob), 4),
            MatchOutcome.AWAY_WIN.value: round(float(away_win_prob), 4)
        }
    
    def _get_most_likely_outcome(self, probabilities: Dict[str, float]) -> str:
        """
        Get the most likely match outcome.
        
        Args:
            probabilities: Dictionary with probabilities for each outcome
            
        Returns:
            Most likely outcome as string
        """
        return max(probabilities, key=lambda k: probabilities[k])
    
    def _calculate_confidence(
        self,
        probabilities: Dict[str, float],
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        league_id: int,
        context_factors: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, List[str]]:
        """
        Calculate confidence score and factors.
        
        Args:
            probabilities: Dictionary with probabilities for each outcome
            home_form: Form data for home team
            away_form: Form data for away team
            league_id: League ID
            context_factors: Optional contextual factors
            
        Returns:
            Tuple of (confidence_score, confidence_factors)
        """
        # Base confidence derived from highest probability
        max_prob = max(probabilities.values())
        base_confidence = 0.5 + (max_prob - 0.33) * 1.5  # Scale from 0.5 to 0.95
        
        # Data completeness and consistency factors
        consistency_score = 0.0
        if home_form.get('matches_played', 0) >= 5:
            consistency_score += 0.1
        if away_form.get('matches_played', 0) >= 5:
            consistency_score += 0.1
            
        # Team consistency factor
        home_consistency = home_form.get('consistency', 0.5)
        away_consistency = away_form.get('consistency', 0.5)
        avg_consistency = (home_consistency + away_consistency) / 2
        consistency_score += avg_consistency * 0.1
        
        # Data quality factor
        data_quality = 0.0
        if home_form and away_form:
            data_quality += 0.15
            
        # Context completeness factor
        context_score = 0.0
        if context_factors:
            if context_factors.get('weather_data_available', False):
                context_score += 0.05
            if context_factors.get('lineup_data_available', False):
                context_score += 0.1
            if context_factors.get('recent_injuries_data_available', False):
                context_score += 0.05
                
        # League data quality factor
        league_quality = 0.15
        if league_id in self.major_leagues:
            league_quality = 0.25
            
        # Combine factors with weights
        confidence_score = (
            (base_confidence * 0.5) +
            (consistency_score * self.consistency_weight) +
            (data_quality * self.data_quality_weight) +
            (context_score * self.context_weight) +
            (league_quality * self.league_weight)
        )
        
        # Cap the confidence score within reasonable bounds
        confidence_score = min(0.95, max(0.4, confidence_score))
        
        # Generate confidence factors explanation
        confidence_factors = []
        
        # Add factors based on probability distribution
        probability_gap = sorted(probabilities.values(), reverse=True)
        if len(probability_gap) >= 2:
            gap = probability_gap[0] - probability_gap[1]
            if gap > 0.2:
                confidence_factors.append("Strong probability difference between outcomes")
            elif gap < 0.05:
                confidence_factors.append("Close probability distribution among outcomes")
        
        # Add data quality factors
        if league_id in self.major_leagues:
            confidence_factors.append("High-quality league data available")
        
        if home_form.get('matches_played', 0) < 3 or away_form.get('matches_played', 0) < 3:
            confidence_factors.append("Limited match history for one or both teams")
        
        # Context factors
        if context_factors:
            if context_factors.get('high_stakes', False):
                confidence_factors.append("High-stakes match may lead to unexpected outcomes")
            if context_factors.get('is_derby', False):
                confidence_factors.append("Derby match with historical significance")
            if context_factors.get('significant_injuries', False):
                confidence_factors.append("Significant player injuries affecting team strength")
        
        return round(float(confidence_score), 2), confidence_factors
    
    def _format_result(
        self,
        probabilities: Dict[str, float],
        most_likely_outcome: str,
        confidence_score: float,
        confidence_factors: List[str]
    ) -> Dict[str, Any]:
        """
        Format the final prediction result.
        
        Args:
            probabilities: Dictionary with probabilities for each outcome
            most_likely_outcome: The most likely outcome
            confidence_score: Confidence score (0-1)
            confidence_factors: List of confidence factors
            
        Returns:
            Dictionary with formatted prediction
        """
        # Convert confidence score to percentage
        confidence_percentage = round(confidence_score * 100, 1)
        
        # Determine confidence level text
        confidence_level = "low"
        if confidence_score >= 0.75:
            confidence_level = "high"
        elif confidence_score >= 0.6:
            confidence_level = "medium"
            
        return {
            "most_likely_outcome": most_likely_outcome,
            "probabilities": {
                "home_win": round(float(probabilities[MatchOutcome.HOME_WIN.value]) * 100, 1),
                "draw": round(float(probabilities[MatchOutcome.DRAW.value]) * 100, 1),
                "away_win": round(float(probabilities[MatchOutcome.AWAY_WIN.value]) * 100, 1)
            },
            "confidence": {
                "score": confidence_score,
                "percentage": confidence_percentage,
                "level": confidence_level,
                "factors": confidence_factors
            }
        }


def predict_match_winner(
    home_team_id: int,
    away_team_id: int,
    home_xg: float,
    away_xg: float,
    home_form: Dict[str, Any],
    away_form: Dict[str, Any],
    h2h: Dict[str, Any],
    league_id: int,
    context_factors: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Predict match winner with confidence percentages.
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID  
        home_xg: Expected goals for home team
        away_xg: Expected goals for away team
        home_form: Form data for home team
        away_form: Form data for away team
        h2h: Head-to-head data
        league_id: League ID
        context_factors: Optional contextual factors
        
    Returns:
        Dictionary with prediction results
    """
    predictor = WinnerPredictor()
    result = predictor.predict_winner(
        home_xg, away_xg, home_form, away_form, h2h, league_id, context_factors
    )
    
    # Add team IDs to result
    result["home_team_id"] = home_team_id
    result["away_team_id"] = away_team_id
    result["league_id"] = league_id
    
    return result
