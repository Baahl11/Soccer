"""
ELO Integration Module

This module provides integration between the ELO rating system and the main prediction pipeline.
It enhances prediction outputs with ELO-derived data.
"""

import logging
from typing import Dict, Any, List, Optional
from team_elo_rating import get_elo_ratings_for_match

logger = logging.getLogger(__name__)

def enhance_prediction_with_elo(prediction_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhance a prediction with additional ELO-based insights
    
    Args:
        prediction_data: Original prediction data dictionary
        
    Returns:
        Enhanced prediction with ELO insights
    """
    enhanced = prediction_data.copy()
    
    # Extract team and league IDs
    home_team_id = prediction_data.get('home_team_id')
    away_team_id = prediction_data.get('away_team_id')
    league_id = prediction_data.get('league_id')
    
    if not home_team_id or not away_team_id:
        logger.warning("Missing team IDs, cannot enhance with ELO data")
        return prediction_data
    
    # Get ELO ratings and derived data
    try:
        elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
        
        # Add ELO-based insights
        enhanced['elo_insights'] = {
            'home_elo': elo_data['home_elo'],
            'away_elo': elo_data['away_elo'],
            'elo_diff': elo_data['elo_diff'],
            'win_probability': elo_data['elo_win_probability'],
            'draw_probability': elo_data['elo_draw_probability'],
            'loss_probability': elo_data['elo_loss_probability'],
            'expected_goal_diff': elo_data['elo_expected_goal_diff']
        }
        
        # Add strength comparison text
        elo_diff = elo_data['elo_diff']
        if abs(elo_diff) < 25:
            strength_comparison = "Teams are very evenly matched"
        elif abs(elo_diff) < 75:
            stronger_team = "Home" if elo_diff > 0 else "Away"
            strength_comparison = f"{stronger_team} team has a slight advantage"
        elif abs(elo_diff) < 150:
            stronger_team = "Home" if elo_diff > 0 else "Away"
            strength_comparison = f"{stronger_team} team has a clear advantage"
        else:
            stronger_team = "Home" if elo_diff > 0 else "Away"
            strength_comparison = f"{stronger_team} team is significantly stronger"
        
        # Add draw likelihood based on ELO
        draw_probability = elo_data.get('elo_draw_probability', 0)
        if draw_probability > 0.3:
            draw_assessment = "High likelihood of a draw"
        elif draw_probability > 0.25:
            draw_assessment = "Moderate likelihood of a draw"
        else:
            draw_assessment = "Low likelihood of a draw"
        
        # Add expected goals insight
        exp_goal_diff = elo_data.get('elo_expected_goal_diff', 0)
        if abs(exp_goal_diff) < 0.3:
            goals_insight = "Expect a close, low-scoring match"
        elif abs(exp_goal_diff) < 0.7:
            stronger_team = "Home" if exp_goal_diff > 0 else "Away"
            goals_insight = f"Expect {stronger_team} team to score slightly more"
        else:
            stronger_team = "Home" if exp_goal_diff > 0 else "Away"
            goals_insight = f"Expect {stronger_team} team to dominate scoring"
        
        # Add interpretation to ELO insights
        enhanced['elo_insights'].update({
            "strength_comparison": strength_comparison,
            "draw_assessment": draw_assessment,
            "goals_insight": goals_insight
        })
        
        # Adjust confidence based on ELO difference
        base_confidence = prediction_data.get("confidence", 0.7)
        elo_confidence_factor = min(0.2, abs(elo_diff) / 400)
        enhanced_confidence = min(0.95, base_confidence + elo_confidence_factor)
        
        enhanced['enhanced_confidence'] = {
            "score": round(enhanced_confidence, 2),
            "factors": {
                "base": round(base_confidence, 2),
                "elo_adjustment": round(elo_confidence_factor, 2)
            }
        }
        
    except Exception as e:
        logger.error(f"Error enhancing prediction with ELO data: {str(e)}")
    
    return enhanced

def enhance_predictions_batch(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enhance a batch of predictions with ELO-based insights
    
    Args:
        predictions: List of prediction dictionaries
        
    Returns:
        Enhanced predictions with ELO insights
    """
    return [enhance_prediction_with_elo(pred) for pred in predictions]

def generate_elo_enhanced_prediction(
    home_team_id: int, 
    away_team_id: int, 
    home_team_name: str,
    away_team_name: str,
    league_id: Optional[int] = None,
    base_prediction: Optional[str] = None,
    base_confidence: Optional[float] = None
) -> Dict[str, Any]:
    """
    Generate a new prediction enhanced with ELO data
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        home_team_name: Home team name
        away_team_name: Away team name
        league_id: League ID (optional)
        base_prediction: Base prediction (Home/Draw/Away) if available
        base_confidence: Base confidence score if available
        
    Returns:
        New prediction with ELO-based insights
    """
    # Create base prediction object
    prediction = {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "league_id": league_id
    }
    
    # Add base prediction if available
    if base_prediction:
        prediction["prediction"] = base_prediction
    
    if base_confidence:
        prediction["confidence"] = base_confidence
    
    # Enhance with ELO data
    return enhance_prediction_with_elo(prediction)
