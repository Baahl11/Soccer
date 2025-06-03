"""
Enhanced Expected Goals (xG) model for more accurate scoring predictions
based on academic research from soccer analytics.

This model improves on basic xG by incorporating:
1. Shot quality metrics
2. Team attacking/defending styles
3. Context-specific adjustments
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
from team_form import get_team_form

logger = logging.getLogger(__name__)

class EnhancedXGModel:
    """
    Enhanced Expected Goals model that provides more accurate predictions
    by incorporating contextual factors beyond basic shot metrics.
    """
    
    def __init__(self, model_path: str = 'models/enhanced_xg_model.pkl'):
        """Initialize the enhanced xG model"""
        self.model_path = model_path
        self.model = self._load_model()
        
        # Default model coefficients if no model is available
        self.team_attack_importance = 0.6
        self.team_defense_importance = 0.7
        self.home_advantage = 0.2
        self.form_weight = 0.3
        self.base_conversion = 0.11  # Base conversion rate
        
    def _load_model(self) -> Optional[GradientBoostingRegressor]:
        """Load the trained model if available"""
        try:
            if os.path.exists(self.model_path):
                return joblib.load(self.model_path)
            else:
                logger.warning(f"xG model not found at {self.model_path}, using statistical estimation")
                return None
        except Exception as e:
            logger.error(f"Error loading xG model: {e}")
            return None
            
    def predict_match_xg(
        self, 
        home_team_id: int, 
        away_team_id: int,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h: Dict[str, Any],
        league_id: int,
        elo_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """
        Predict expected goals for both teams in a match using enhanced xG model
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            home_form: Form data for home team
            away_form: Form data for away team
            h2h: Head-to-head data between teams
            league_id: League ID
            elo_info: Optional ELO rating information for teams
            
        Returns:
            Tuple of (home_xg, away_xg)
        """
        try:
            # If we have a trained model, use it
            if self.model is not None:
                features = self._extract_prediction_features(home_team_id, away_team_id, home_form, away_form, h2h, league_id, elo_info)
                predictions = self.model.predict(features.reshape(1, -1))
                return float(predictions[0][0]), float(predictions[0][1])
            
            # Otherwise use statistical estimation
            return self._statistical_xg_estimate(home_team_id, away_team_id, home_form, away_form, h2h, elo_info)
            
        except Exception as e:
            logger.error(f"Error predicting match xG: {e}")
            # Fallback to reasonable defaults
            return 1.4, 1.1
            
    def _extract_prediction_features(
        self, 
        home_team_id: int, 
        away_team_id: int,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h: Dict[str, Any],
        league_id: int,
        elo_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        """Extract features for xG prediction"""
        features = np.zeros(14)  # 14 key features
        
        # Team attacking and defensive strength
        features[0] = home_form.get('avg_goals_scored', 1.4)
        features[1] = home_form.get('avg_goals_conceded', 1.2)
        features[2] = away_form.get('avg_goals_scored', 1.2)
        features[3] = away_form.get('avg_goals_conceded', 1.4)
        
        # Form metrics
        features[4] = home_form.get('form_score', 0.5)
        features[5] = away_form.get('form_score', 0.5)
        
        # Shots and conversion rates
        features[6] = home_form.get('avg_shots', 12.0)
        features[7] = away_form.get('avg_shots', 10.0)
        features[8] = home_form.get('avg_shots_on_target', 4.0)
        features[9] = away_form.get('avg_shots_on_target', 3.5)
        
        # Head-to-head data
        features[10] = h2h.get('average_goals', 2.5)
        features[11] = h2h.get('h2h_home_dominance', 0.0)  # -1 to 1 value
        
        # Team IDs as proxy for team quality (normalized)
        features[12] = home_team_id / 1000
        features[13] = away_team_id / 1000
        
        return features
      
    def _statistical_xg_estimate(
        self, 
        home_team_id: int, 
        away_team_id: int,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h: Dict[str, Any],
        elo_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, float]:
        """
        Estimate expected goals using statistical approach when model isn't available
        
        This implements the academic approach of combining:
        1. Team attacking strength
        2. Opponent defensive weakness
        3. Home advantage
        4. Recent form
        """
        # Base values from form
        home_attack = home_form.get('avg_goals_scored', 1.3)
        home_defense = 1/max(0.5, home_form.get('avg_goals_conceded', 1.0))
        away_attack = away_form.get('avg_goals_scored', 1.1)
        away_defense = 1/max(0.5, away_form.get('avg_goals_conceded', 1.0))
        
        # Form adjustment
        home_form_factor = 0.8 + (home_form.get('form_score', 0.5) * 0.4)
        away_form_factor = 0.8 + (away_form.get('form_score', 0.5) * 0.4)
        
        # Calculate xG using team attack vs opponent defense
        home_xg = home_attack * (1/away_defense) * home_form_factor * (1 + self.home_advantage)
        away_xg = away_attack * (1/home_defense) * away_form_factor
        
        # H2H adjustment
        if h2h and h2h.get('matches_played', 0) > 0:
            h2h_factor = h2h.get('h2h_home_dominance', 0) * 0.15  # Scale factor
            home_xg *= (1 + h2h_factor)
            away_xg *= (1 - h2h_factor)
        
        # Ensure reasonable ranges
        home_xg = min(4.0, max(0.3, home_xg))
        away_xg = min(3.5, max(0.2, away_xg))
        
        return home_xg, away_xg
    
    def calculate_over_under_probabilities(
        self, 
        home_xg: float, 
        away_xg: float, 
        thresholds: List[float] = [2.5, 3.5]
    ) -> Dict[str, float]:
        """
        Calculate over/under probabilities based on xG values
        using Poisson distribution with correlation adjustment
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            thresholds: List of goals thresholds
            
        Returns:
            Dictionary of over probabilities
        """
        from scipy.stats import poisson
        import numpy as np
        
        # Dixon-Coles correlation parameter (academic approach)
        # Accounts for correlation between teams' scoring
        rho = 0.1
        
        results = {}
        total_xg = home_xg + away_xg
        
        # Calculate probability for each threshold
        for threshold in thresholds:
            threshold_key = f"over_{threshold:.1f}"
            
            # Academic approach: Calculate probability via simulation including correlation
            # This gives more accurate results than simple Poisson
            total_prob = 0
            max_goals = 10  # Reasonable upper limit
            
            for h in range(max_goals):
                for a in range(max_goals):
                    # Basic Poisson probabilities
                    p_h = poisson.pmf(h, home_xg)
                    p_a = poisson.pmf(a, away_xg)
                    
                    # Dixon-Coles adjustment for low scores
                    if h == 0 and a == 0:
                        p = p_h * p_a * (1 + rho)
                    elif h == 0 and a == 1:
                        p = p_h * p_a * (1 - rho)
                    elif h == 1 and a == 0:
                        p = p_h * p_a * (1 - rho)
                    elif h == 1 and a == 1:
                        p = p_h * p_a * (1 + rho)
                    else:
                        p = p_h * p_a
                    
                    # Add to over probability if total exceeds threshold
                    if h + a > threshold:
                        total_prob += p
            
            results[threshold_key] = float(round(float(total_prob), 4))
            
        return results

    def calculate_over_under_probabilities_advanced(
        self, 
        home_xg: float, 
        away_xg: float, 
        thresholds: List[float] = [0.5, 1.5, 2.5, 3.5, 4.5],
        use_negative_binomial: bool = True,
        context_factors: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """
        Calculate over/under probabilities based on xG values using advanced methods
        including negative binomial distribution for better modeling of variance
        and contextual adjustments.
        
        Args:
            home_xg: Expected goals for home team
            away_xg: Expected goals for away team
            thresholds: List of goals thresholds
            use_negative_binomial: Whether to use negative binomial (True) or Poisson (False)
            context_factors: Additional contextual factors that may affect goal distribution
                
        Returns:
            Dictionary of over probabilities for each threshold
        """
        from scipy.stats import poisson, nbinom
        import numpy as np
        
        # Dixon-Coles correlation parameter (academic research-based)
        rho = 0.13
        
        # Apply contextual adjustments if provided
        total_xg = home_xg + away_xg
        if context_factors:
            # Weather conditions affect total goals (research shows ~10% reduction in rain)
            if context_factors.get('is_rainy', False):
                home_xg *= 0.92
                away_xg *= 0.92
            
            # High stakes matches tend to have fewer goals
            if context_factors.get('is_high_stakes', False):
                home_xg *= 0.95
                away_xg *= 0.95
        
        # Dispersion parameter for negative binomial (smaller = more overdispersion)
        # Research shows values between 3-5 model soccer goals well
        dispersion = 4.2
        
        results = {}
        
        # Calculate probability for each threshold
        for threshold in thresholds:
            threshold_key = f"over_{threshold:.1f}"
            
            if use_negative_binomial:
                # Academic approach: Use negative binomial for better modeling
                # Convert mean to negative binomial parameters
                p_home = dispersion / (dispersion + home_xg)
                p_away = dispersion / (dispersion + away_xg)
                
                # Calculate over probability through simulation including correlation
                total_prob = 0
                max_goals = 15  # Reasonable upper limit for accurate tail probabilities
                
                for h in range(max_goals):
                    for a in range(max_goals):
                        # Negative binomial probabilities
                        p_h = nbinom.pmf(h, dispersion, p_home)
                        p_a = nbinom.pmf(a, dispersion, p_away)
                        
                        # Dixon-Coles adjustment for low scores
                        if h <= 1 and a <= 1:
                            if h == 0 and a == 0:
                                p = p_h * p_a * (1 + rho)
                            elif h == 0 and a == 1:
                                p = p_h * p_a * (1 - rho)
                            elif h == 1 and a == 0:
                                p = p_h * p_a * (1 - rho)
                            elif h == 1 and a == 1:
                                p = p_h * p_a * (1 + rho)
                            else:
                                p = p_h * p_a
                        else:
                            p = p_h * p_a
                        
                        # Add to over probability if total exceeds threshold
                        if h + a > threshold:
                            total_prob += p
            else:
                # Poisson approach: Calculate probability via simulation including correlation
                total_prob = 0
                max_goals = 12  # Reasonable upper limit
                
                for h in range(max_goals):
                    for a in range(max_goals):
                        # Basic Poisson probabilities
                        p_h = poisson.pmf(h, home_xg)
                        p_a = poisson.pmf(a, away_xg)
                        
                        # Dixon-Coles adjustment for low scores
                        if h <= 1 and a <= 1:
                            if h == 0 and a == 0:
                                p = p_h * p_a * (1 + rho)
                            elif h == 0 and a == 1:
                                p = p_h * p_a * (1 - rho)
                            elif h == 1 and a == 0:
                                p = p_h * p_a * (1 - rho)
                            elif h == 1 and a == 1:
                                p = p_h * p_a * (1 + rho)
                            else:
                                p = p_h * p_a
                        else:
                            p = p_h * p_a
                        
                        # Add to over probability if total exceeds threshold
                        if h + a > threshold:
                            total_prob += p
            
            # Dynamic adjustment for extreme thresholds
            if threshold >= 4.5 and total_xg < 2.5:
                # Research shows markets slightly overestimate high threshold overs in low xG games
                total_prob *= 0.92
            elif threshold <= 1.5 and total_xg > 3.5:
                # Markets slightly underestimate low threshold overs in high xG games
                total_prob *= 1.05
            
            results[threshold_key] = round(float(total_prob), 4)
        
        # Calculate exact goal probabilities for 0-5+ goals
        # This is valuable additional information for specific markets
        exact_probs = {}
        max_exact = 5
        
        for total in range(max_exact + 1):
            if total < max_exact:
                exact_probs[f"exactly_{total}"] = _calculate_exact_goals_probability(
                    home_xg, away_xg, total, rho, use_negative_binomial, dispersion)
            else:
                # 5+ goals combines all possibilities of 5 or more
                exact_probs[f"{total}+"] = 1.0 - sum(
                    exact_probs.get(f"exactly_{i}", 0.0) for i in range(total))
        
        results.update(exact_probs)
        return results

def _calculate_exact_goals_probability(
    home_xg: float,
    away_xg: float,
    total_goals: int,
    rho: float = 0.13,
    use_negative_binomial: bool = True,
    dispersion: float = 4.2
) -> float:
    """
    Calculate probability of exactly N total goals in a match.
    
    Args:
        home_xg: Expected goals for home team
        away_xg: Expected goals for away team
        total_goals: The exact number of total goals to calculate
        rho: Dixon-Coles correlation parameter
        use_negative_binomial: Whether to use negative binomial distribution
        dispersion: Dispersion parameter for negative binomial
        
    Returns:
        Probability of exactly total_goals being scored
    """
    from scipy.stats import poisson, nbinom
    
    total_prob = 0.0
    
    # All combinations that sum to total_goals
    for h in range(total_goals + 1):
        a = total_goals - h
        
        if use_negative_binomial:
            # Convert mean to negative binomial parameters
            p_home = dispersion / (dispersion + home_xg)
            p_away = dispersion / (dispersion + away_xg)
            
            # Negative binomial probabilities
            p_h = nbinom.pmf(h, dispersion, p_home)
            p_a = nbinom.pmf(a, dispersion, p_away)
        else:
            # Poisson probabilities
            p_h = poisson.pmf(h, home_xg)
            p_a = poisson.pmf(a, away_xg)
        
        # Apply Dixon-Coles correction for low scores
        if h <= 1 and a <= 1:
            if h == 0 and a == 0:
                p = p_h * p_a * (1 + rho)
            elif h == 0 and a == 1:
                p = p_h * p_a * (1 - rho)
            elif h == 1 and a == 0:
                p = p_h * p_a * (1 - rho)
            elif h == 1 and a == 1:
                p = p_h * p_a * (1 + rho)
            else:
                p = p_h * p_a
        else:
            p = p_h * p_a
        
        total_prob += p
    
    return round(float(total_prob), 4)

def get_enhanced_goal_predictions(
    home_team_id: int,
    away_team_id: int,
    home_form: Dict[str, Any],
    away_form: Dict[str, Any],
    h2h: Dict[str, Any],
    league_id: int,
    elo_ratings: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get enhanced goal predictions using the xG model
    
    Args:
        home_team_id: ID of home team
        away_team_id: ID of away team
        home_form: Form data for home team
        away_form: Form data for away team
        h2h: Head-to-head data
        league_id: League ID
        elo_ratings: Optional ELO ratings data for improved prediction
        
    Returns:
        Dictionary with predictions
    """
    xg_model = EnhancedXGModel()
    
    # Inicializar variables para ajustes de ELO
    elo_adjustment_factor = 0.0
    elo_info = {}
    
    # Aplicar ajustes de ELO si están disponibles
    if elo_ratings:
        elo_info = {
            'home_elo': elo_ratings.get('home_elo', 1500),
            'away_elo': elo_ratings.get('away_elo', 1500),
            'elo_diff': elo_ratings.get('elo_diff', 0),
            'elo_win_probability': elo_ratings.get('elo_win_probability', 0.5)
        }
    
    home_xg, away_xg = xg_model.predict_match_xg(
        home_team_id,
        away_team_id,
        home_form,
        away_form,
        h2h,
        league_id,
        elo_info=elo_info if elo_ratings else None
    )
    
    # Get over/under probabilities
    over_under_probs = xg_model.calculate_over_under_probabilities(
        home_xg, 
        away_xg, 
        thresholds=[0.5, 1.5, 2.5, 3.5, 4.5]
    )
    
    # Calculate BTTS (both teams to score) probability
    # Academic approach: P(BTTS) = 1 - P(home=0) - P(away=0) + P(home=0 & away=0)
    btts_prob = 1 - np.exp(-home_xg) - np.exp(-away_xg) + np.exp(-(home_xg + away_xg))
    
    return {
        'home_xg': float(round(float(home_xg), 2)),
        'away_xg': float(round(float(away_xg), 2)),
        'total_xg': float(round(float(home_xg + away_xg), 2)),
        'over_under': over_under_probs,
        'btts_prob': float(round(float(btts_prob), 4))
    }
