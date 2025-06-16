"""
Enhanced prediction integration with advanced weather modeling.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from .weather_model import calculate_weather_adjustment_factors
from .xg_model import EnhancedXGModel, get_enhanced_goal_predictions
from .specialized_ensemble import predict_goals_with_ensemble
from .bayesian_goals_model_new import BayesianGoalsModel

logger = logging.getLogger(__name__)

def generate_enhanced_goals_prediction(
    home_team_id: int,
    away_team_id: int,
    league_id: int,
    home_form: Dict[str, Any],
    away_form: Dict[str, Any],
    h2h: Dict[str, Any],
    weather_data: Optional[Dict[str, Any]] = None,
    elo_ratings: Optional[Dict[str, Any]] = None,
    context_factors: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate enhanced goals prediction using multiple models and weather adjustments.
    
    Args:
        home_team_id: ID of home team
        away_team_id: ID of away team
        league_id: League ID
        home_form: Form data for home team
        away_form: Form data for away team
        h2h: Head-to-head statistics
        weather_data: Optional weather information
        elo_ratings: Optional ELO ratings
        context_factors: Additional contextual factors
        
    Returns:
        Dictionary containing predictions and metadata
    """
    try:
        # Get weather adjustment factors
        weather_factors = calculate_weather_adjustment_factors(weather_data)
        
        # Step 1: Get XG-based prediction
        xg_predictions = get_enhanced_goal_predictions(
            home_team_id,
            away_team_id,
            home_form,
            away_form,
            h2h,
            league_id,
            elo_ratings
        )
        
        # Apply weather adjustments to XG values
        xg_predictions['home_xg'] *= weather_factors['xg_home']
        xg_predictions['away_xg'] *= weather_factors['xg_away']
        xg_predictions['total_xg'] = xg_predictions['home_xg'] + xg_predictions['away_xg']
        
        # Step 2: Get ensemble prediction
        ensemble_data = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'league_id': league_id,
            'home_form': home_form,
            'away_form': away_form,
            'weather_factors': weather_factors,
            'elo_ratings': elo_ratings
        }
        
        ensemble_pred = predict_goals_with_ensemble(
            match_data=ensemble_data,
            historical_matches=None,  # Would be loaded from database
            use_bayesian=True,
            context_aware=True
        )
        
        # Step 3: Calculate final prediction combining models
        # Use weighted average, giving more weight to XG in good weather
        # and more weight to ensemble in adverse conditions
        if weather_factors['goals'] < 0.9:  # Bad weather
            xg_weight = 0.4
            ensemble_weight = 0.6
        else:  # Good weather
            xg_weight = 0.6
            ensemble_weight = 0.4
            
        final_home_xg = (xg_predictions['home_xg'] * xg_weight + 
                        ensemble_pred['predicted_home_goals'] * ensemble_weight)
        final_away_xg = (xg_predictions['away_xg'] * xg_weight + 
                        ensemble_pred['predicted_away_goals'] * ensemble_weight)
        
        # Calculate advanced probabilities using the combined prediction
        model = EnhancedXGModel()
        over_under_probs = model.calculate_over_under_probabilities_advanced(
            final_home_xg,
            final_away_xg,
            thresholds=[0.5, 1.5, 2.5, 3.5, 4.5],
            use_negative_binomial=True,
            context_factors=context_factors
        )
        
        # Calculate BTTS probability
        import numpy as np
        btts_prob = 1 - np.exp(-final_home_xg) - np.exp(-final_away_xg) + np.exp(-(final_home_xg + final_away_xg))
        
        # Add confidence score based on data quality and weather
        confidence = 0.8  # Base confidence
        if weather_factors['goals'] < 0.9:
            confidence *= 0.9  # Reduce confidence in bad weather
        if len(home_form) < 5 or len(away_form) < 5:
            confidence *= 0.9  # Reduce confidence with limited data
            
        result = {
            'home_xg': round(float(final_home_xg), 2),
            'away_xg': round(float(final_away_xg), 2),
            'total_xg': round(float(final_home_xg + final_away_xg), 2),
            'over_under': over_under_probs,
            'btts_prob': round(float(btts_prob), 4),
            'confidence': round(float(confidence), 2),
            'method': 'enhanced_ensemble_weather',
            'weather_impact': {
                'total_factor': round(float(weather_factors['goals']), 3),
                'home_factor': round(float(weather_factors['xg_home']), 3),
                'away_factor': round(float(weather_factors['xg_away']), 3),
                'home_advantage_factor': round(float(weather_factors['home_advantage']), 3)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error generating enhanced goals prediction: {e}")
        return {
            'home_xg': 1.3,
            'away_xg': 1.1,
            'total_xg': 2.4,
            'over_under': {'over_2.5': 0.5},
            'btts_prob': 0.55,
            'confidence': 0.4,
            'method': 'fallback',
            'error': str(e)
        }
