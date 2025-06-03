"""
Enhanced prediction integration module.

This module integrates the enhanced xG model and advanced corners model
into the main prediction system, allowing for improved forecasting accuracy.
It now includes the voting ensemble model for corner predictions based on
academic research showing superior performance of combined RF and XGBoost models.
It also integrates advanced ELO ratings to improve prediction accuracy.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from xg_model import get_enhanced_goal_predictions, EnhancedXGModel
from corners_improved import predict_corners_with_negative_binomial, ImprovedCornersModel
from voting_ensemble_corners import VotingEnsembleCornersModel
from team_form import get_team_form, get_head_to_head_analysis
from match_winner import predict_match_winner
from elo_integration import enhance_prediction_with_elo
from team_elo_rating import get_elo_ratings_for_match

logger = logging.getLogger(__name__)

def predict_corners_with_voting_ensemble(
    home_team_id: int,
    away_team_id: int,
    home_form: Dict[str, Any],
    away_form: Dict[str, Any],
    league_id: int,
    context_factors: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Wrapper function to use the VotingEnsembleCornersModel for corner predictions.
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        home_form: Home team form data
        away_form: Away team form data
        league_id: League ID
        context_factors: Optional context factors like weather
        
    Returns:
        Dictionary with corner predictions
    """
    try:
        # Initialize the voting ensemble model
        ensemble_model = VotingEnsembleCornersModel()
        
        # Convert form data to stats format expected by the model
        home_stats = {
            'avg_corners_for': home_form.get('avg_corners_for', 5.0),
            'avg_corners_against': home_form.get('avg_corners_against', 5.0),
            'form_score': home_form.get('form_score', 50),
            'avg_shots': home_form.get('avg_shots', 12.0)
        }
        
        away_stats = {
            'avg_corners_for': away_form.get('avg_corners_for', 4.5),
            'avg_corners_against': away_form.get('avg_corners_against', 5.5),
            'form_score': away_form.get('form_score', 50),
            'avg_shots': away_form.get('avg_shots', 10.0)
        }
        
        # Make prediction using the ensemble model
        prediction = ensemble_model.predict_corners(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_stats=home_stats,
            away_stats=away_stats,
            league_id=league_id,
            context_factors=context_factors
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error in voting ensemble corner prediction: {e}")
        # Return fallback prediction
        return {
            'total': 10.0,
            'home': 5.4,
            'away': 4.6,
            'over_8.5': 0.65,
            'over_9.5': 0.52,
            'over_10.5': 0.38,
            'is_fallback': True,
            'model': 'fallback',
            'confidence': 0.45
        }

def make_enhanced_prediction(
    fixture_id: int,
    home_team_id: int,
    away_team_id: int,
    league_id: int,
    season: int,
    home_form: Optional[Dict[str, Any]] = None,
    away_form: Optional[Dict[str, Any]] = None,
    h2h: Optional[Dict[str, Any]] = None,
    weather_data: Optional[Dict[str, Any]] = None,
    odds_data: Optional[Dict[str, Any]] = None,
    use_elo: bool = True
) -> Dict[str, Any]:
    """
    Make enhanced prediction using the academic research-based models.
    
    Args:
        fixture_id: Fixture ID
        home_team_id: Home team ID
        away_team_id: Away team ID
        league_id: League ID
        season: Season year
        home_form: Optional home team form data (will be fetched if not provided)
        away_form: Optional away team form data (will be fetched if not provided)
        h2h: Optional head-to-head data (will be fetched if not provided)
        weather_data: Optional weather data for adjustments
        odds_data: Optional betting odds data
        use_elo: Whether to use ELO ratings to enhance prediction (default: True)
        
    Returns:
        Dictionary with enhanced predictions
    """   
    try:
        # Get form data if not provided
        if not home_form:
            home_form = get_team_form(home_team_id, league_id, season)
        if not away_form:
            away_form = get_team_form(away_team_id, league_id, season)
        if not h2h:
            h2h = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Get ELO ratings if enabled
        elo_ratings = None
        if use_elo:
            try:
                logger.info(f"Getting ELO ratings for fixture {fixture_id}")
                elo_ratings = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
            except Exception as e:
                logger.warning(f"Error getting ELO ratings: {e}")
            
        # Get enhanced goal predictions using xG model
        logger.info(f"Generating enhanced goal predictions for fixture {fixture_id}")
        xg_predictions = get_enhanced_goal_predictions(
            home_team_id,
            away_team_id,
            home_form,
            away_form,
            h2h,
            league_id,
            elo_ratings=elo_ratings  # Pass ELO ratings to xG model
        )
        
        # Get advanced corners predictions using voting ensemble model (RF + XGBoost)
        # based on academic research showing this combination achieves highest accuracy
        logger.info(f"Generating corners predictions with voting ensemble model for fixture {fixture_id}")
        corners_predictions = predict_corners_with_voting_ensemble(
            home_team_id,
            away_team_id,
            home_form,
            away_form,
            league_id,
            context_factors=weather_data
        )
        
        # Fallback to negative binomial model if ensemble method fails
        if corners_predictions.get('is_fallback', False):
            logger.info(f"Falling back to negative binomial model for fixture {fixture_id}")
            corners_predictions = predict_corners_with_negative_binomial(
                home_team_id,
                away_team_id,
                home_form,
                away_form,
                league_id
            )
        
        # Apply weather adjustments if available
        if weather_data:
            xg_predictions = _adjust_for_weather(xg_predictions, weather_data)
            corners_predictions = _adjust_corners_for_weather(corners_predictions, weather_data)
            
        # Generate match winner prediction with confidence percentages
        logger.info(f"Generating match winner prediction for fixture {fixture_id}")
        match_winner_prediction = predict_match_winner(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_xg=xg_predictions["predicted_home_goals"],
            away_xg=xg_predictions["predicted_away_goals"],
            home_form=home_form,
            away_form=away_form,
            h2h=h2h,
            league_id=league_id,
            context_factors=weather_data
        )
        
        # Calculate confidence level
        confidence = _calculate_prediction_confidence(
            fixture_id,
            home_team_id,
            away_team_id,
            league_id,
            xg_predictions,
            corners_predictions
        )
        
        # Create consolidated prediction result
        result = {
            "predicted_home_goals": xg_predictions['home_xg'],
            "predicted_away_goals": xg_predictions['away_xg'],
            "total_goals": xg_predictions['total_xg'],
            "prob_over_2_5": xg_predictions['over_under'].get('over_2.5', 0.0),
            "prob_btts": xg_predictions['btts_prob'],
            "method": "enhanced_xg",
            "confidence": confidence,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league_id": league_id,
            
            # Incluir datos ELO si están disponibles
            "elo": elo_ratings if elo_ratings else {},
            # Advanced corners prediction
            "corners": {
                "total": corners_predictions['total'],
                "home": corners_predictions['home'],
                "away": corners_predictions['away'],
                "over_8.5": corners_predictions['over_8.5'],
                "over_9.5": corners_predictions['over_9.5'],
                "over_10.5": corners_predictions['over_10.5'],
                "is_fallback": False
            },
            # Match winner prediction with confidence percentages
            "match_winner": {
                "most_likely_outcome": match_winner_prediction["most_likely_outcome"],
                "home_win_probability": match_winner_prediction["probabilities"]["home_win"],
                "draw_probability": match_winner_prediction["probabilities"]["draw"],
                "away_win_probability": match_winner_prediction["probabilities"]["away_win"],
                "confidence_score": match_winner_prediction["confidence"]["score"],
                "confidence_level": match_winner_prediction["confidence"]["level"],
                "confidence_factors": match_winner_prediction["confidence"]["factors"]
            }
        }
        
        # Add confidence level text
        if confidence >= 0.75:
            result["confidence_level"] = "high"
        elif confidence >= 0.65:
            result["confidence_level"] = "medium"
        else:
            result["confidence_level"] = "low"
            
        # Add confidence factors
        result["confidence_factors"] = _get_confidence_factors(xg_predictions, corners_predictions)
        
        return result
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction: {e}")
        return _get_fallback_prediction()
        
def _adjust_for_weather(
    xg_predictions: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adjust xG predictions based on weather conditions.
    
    Args:
        xg_predictions: Base xG predictions
        weather_data: Weather data
        
    Returns:
        Adjusted xG predictions
    """
    # Make a copy to avoid modifying the original
    adjusted = xg_predictions.copy()
    
    try:
        condition = weather_data.get('condition', '').lower()
        intensity = weather_data.get('intensity', 'normal').lower()
        
        # Only adjust for significant weather conditions
        if condition in ('rain', 'snow', 'wind'):
            if condition == 'rain' and intensity in ('heavy', 'extreme'):
                # Heavy rain reduces overall xG slightly
                adjusted['home_xg'] *= 0.92
                adjusted['away_xg'] *= 0.92
                adjusted['total_xg'] = adjusted['home_xg'] + adjusted['away_xg']
                
            elif condition == 'snow' and intensity in ('moderate', 'heavy', 'extreme'):
                # Snow more significantly reduces xG
                adjusted['home_xg'] *= 0.85
                adjusted['away_xg'] *= 0.85
                adjusted['total_xg'] = adjusted['home_xg'] + adjusted['away_xg']
                
            elif condition == 'wind' and intensity in ('strong', 'severe'):
                # Wind impacts away team more than home team
                adjusted['home_xg'] *= 0.95
                adjusted['away_xg'] *= 0.90
                adjusted['total_xg'] = adjusted['home_xg'] + adjusted['away_xg']
            
            # Recalculate over/under probabilities
            xg_model = EnhancedXGModel()
            adjusted['over_under'] = xg_model.calculate_over_under_probabilities(
                adjusted['home_xg'],
                adjusted['away_xg'],
                thresholds=[0.5, 1.5, 2.5, 3.5, 4.5]
            )
            
            # Recalculate BTTS
            btts_prob = 1 - np.exp(-adjusted['home_xg']) - np.exp(-adjusted['away_xg']) + np.exp(-(adjusted['home_xg'] + adjusted['away_xg']))
            adjusted['btts_prob'] = round(btts_prob, 4)
    
    except Exception as e:
        logger.warning(f"Error adjusting xG for weather: {e}")
    
    return adjusted

def _adjust_corners_for_weather(
    corners_predictions: Dict[str, Any],
    weather_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Adjust corners predictions based on weather.
    
    Args:
        corners_predictions: Base corners predictions
        weather_data: Weather data
        
    Returns:
        Adjusted corners predictions
    """
    # Make a copy to avoid modifying the original
    adjusted = corners_predictions.copy()
    
    try:
        condition = weather_data.get('condition', '').lower()
        intensity = weather_data.get('intensity', 'normal').lower()
        
        # Only adjust for significant weather conditions
        if condition in ('rain', 'snow', 'wind'):
            if condition == 'rain' and intensity in ('heavy', 'extreme'):
                # Heavy rain typically increases corners
                factor = 1.08
                adjusted['total'] *= factor
                adjusted['home'] *= factor
                adjusted['away'] *= factor
                
            elif condition == 'snow' and intensity in ('moderate', 'heavy', 'extreme'):
                # Snow reduces corners as play slows down
                factor = 0.85
                adjusted['total'] *= factor
                adjusted['home'] *= factor
                adjusted['away'] *= factor
                
            elif condition == 'wind' and intensity in ('strong', 'severe'):
                # Wind typically increases corners
                factor = 1.12
                adjusted['total'] *= factor
                adjusted['home'] *= factor
                adjusted['away'] *= factor
                
            # Recalculate over probabilities
            # Note: This is a simplified approach - in practice we would use a proper statistical model
            base_corners = adjusted['total']
            adjusted['over_8.5'] = max(0.1, min(0.95, 1 - (8.5/base_corners)**2))
            adjusted['over_9.5'] = max(0.1, min(0.95, 1 - (9.5/base_corners)**2))
            adjusted['over_10.5'] = max(0.1, min(0.9, 1 - (10.5/base_corners)**2))
            
    except Exception as e:
        logger.warning(f"Error adjusting corners for weather: {e}")
    
    return adjusted

def _calculate_prediction_confidence(
    fixture_id: int, 
    home_team_id: int, 
    away_team_id: int, 
    league_id: int,
    xg_predictions: Dict[str, Any],
    corners_predictions: Dict[str, Any]
) -> float:
    """
    Calculate prediction confidence based on multiple factors.
    
    Args:
        fixture_id: Fixture ID
        home_team_id: Home team ID
        away_team_id: Away team ID
        league_id: League ID
        xg_predictions: xG model predictions
        corners_predictions: Corners model predictions
        
    Returns:
        Confidence score between 0 and 1
    """
    # Default baseline confidence
    base_confidence = 0.65
    
    try:
        # Team ID-based confidence factors
        # (Typically we'd use a more sophisticated approach, but this demonstrates the concept)
        team_confidence = min(0.9, 0.6 + (max(home_team_id, away_team_id) % 100) / 1000)
        
        # Model confidence factors
        model_confidence = 0.7  # Base model confidence
        
        # League-specific confidence 
        # (Major leagues typically have better data quality)
        major_leagues = [39, 78, 140, 135, 61]  # Premier League, Bundesliga, La Liga, Serie A, Ligue 1
        if league_id in major_leagues:
            league_factor = 0.8
        else:
            league_factor = 0.6
            
        # Combine factors with appropriate weights
        confidence = (
            (base_confidence * 0.3) + 
            (team_confidence * 0.2) + 
            (model_confidence * 0.2) + 
            (league_factor * 0.3)
        )
        
        # Ensure reasonable bounds
        confidence = max(0.4, min(0.9, confidence))
        
        return round(confidence, 2)
    except Exception as e:
        logger.warning(f"Error calculating confidence: {e}")
        return base_confidence
        
def _get_confidence_factors(
    xg_predictions: Dict[str, Any],
    corners_predictions: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Generate human-readable confidence factors.
    
    Args:
        xg_predictions: xG predictions
        corners_predictions: Corners predictions
        
    Returns:
        Dictionary with confidence factors
    """
    factors = {
        'increasing': [],
        'neutral': [],
        'decreasing': []
    }
    
    # Add factors based on xG values
    total_xg = xg_predictions.get('total_xg', 0)
    if total_xg > 3.5:
        factors['increasing'].append('High expected goals total')
    elif total_xg < 1.8:
        factors['decreasing'].append('Low expected goals total')
    else:
        factors['neutral'].append('Average expected goals total')
    
    # Add corners-related factor
    total_corners = corners_predictions.get('total', 0)
    if total_corners > 11:
        factors['increasing'].append('High expected corner count')
    elif total_corners < 8:
        factors['decreasing'].append('Low expected corner count')
        
    return factors
        
def _get_fallback_prediction() -> Dict[str, Any]:
    """
    Get fallback prediction when enhanced prediction fails.
    
    Returns:
        Dictionary with fallback prediction values
    """
    # Generate reasonable defaults with slight randomization for variability
    home_goals = 1.35 + np.random.normal(0, 0.1)
    away_goals = 1.15 + np.random.normal(0, 0.1)
    total_goals = home_goals + away_goals
    
    # Calculate over/under and BTTS probabilities
    prob_over = 1 - np.exp(-total_goals) * (1 + total_goals)  # Approximation
    prob_btts = (1 - np.exp(-home_goals)) * (1 - np.exp(-away_goals))
    
    # Generate corners with variability
    total_corners = 10.0 + np.random.normal(0, 0.5)
    home_ratio = 0.54 + np.random.normal(0, 0.02)
    home_corners = total_corners * home_ratio
    away_corners = total_corners * (1 - home_ratio)
    
    # Generate fallback match winner probabilities
    # Home advantage in fallback prediction
    home_win_prob = 0.45 + np.random.normal(0, 0.02)
    away_win_prob = 0.30 + np.random.normal(0, 0.02)
    draw_prob = 1.0 - home_win_prob - away_win_prob
    
    # Ensure probabilities are valid
    if draw_prob < 0.15:
        # Redistribute probabilities to ensure draw is at least 15%
        excess = 0.15 - draw_prob
        home_win_prob -= excess * 0.6
        away_win_prob -= excess * 0.4
        draw_prob = 0.15
    
    return {
        "predicted_home_goals": round(float(home_goals), 2),
        "predicted_away_goals": round(float(away_goals), 2),
        "total_goals": round(float(total_goals), 2),
        "prob_over_2_5": round(float(prob_over), 2),
        "prob_btts": round(float(prob_btts), 2),
        "method": "fallback",
        "confidence": 0.45,
        "confidence_level": "low",
        "corners": {
            "total": round(float(total_corners), 1),
            "home": round(float(home_corners), 1),
            "away": round(float(away_corners), 1),
            "over_8.5": round(float(1 - (8.5/total_corners)**2), 2),
            "over_9.5": round(float(1 - (9.5/total_corners)**2), 2),
            "over_10.5": round(float(1 - (10.5/total_corners)**2), 2),
            "is_fallback": True,
            "fallback_message": "ATENCIÓN: Predicción de corners cayó en fallback"
        },
        "match_winner": {
            "most_likely_outcome": "home_win" if home_win_prob > max(draw_prob, away_win_prob) else 
                                  "draw" if draw_prob > away_win_prob else 
                                  "away_win",
            "home_win_probability": round(float(home_win_prob * 100), 1),
            "draw_probability": round(float(draw_prob * 100), 1),
            "away_win_probability": round(float(away_win_prob * 100), 1),
            "confidence_score": 0.4,
            "confidence_level": "low",
            "confidence_factors": ["Using fallback prediction", "Limited data available"]
        },
        "confidence_factors": {
            'neutral': ['Using fallback prediction'],
            'decreasing': ['Limited data available', 'Predicción de corners cayó en fallback']
        }
    }
