"""
ELO Enhanced Demo

This script demonstrates how to enhance soccer prediction results with ELO ratings.
It focuses on adding valuable ELO-derived data to the prediction outputs.
"""

import json
import logging
from typing import Dict, Any
import os
from team_elo_rating import get_elo_ratings_for_match

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
            
        # Add all insights to prediction data
        elo_insights = {
            **elo_data,
            "strength_comparison": strength_comparison,
            "draw_assessment": draw_assessment,
            "goals_insight": goals_insight
        }
        
        enhanced = prediction_data.copy()
        enhanced['elo_insights'] = elo_insights
        
        # Adjust confidence based on ELO
        base_confidence = prediction_data.get("confidence", 0.7)
        elo_confidence_factor = min(0.2, abs(elo_diff) / 400)
        enhanced_confidence = min(0.95, base_confidence + elo_confidence_factor)
        
        enhanced['enhanced_confidence'] = {
            "confidence_score": round(enhanced_confidence, 2),
            "confidence_factors": {
                "base_model_confidence": round(base_confidence, 2),
                "elo_adjustment": round(elo_confidence_factor, 2)
            }
        }
        
        return enhanced
    
    except Exception as e:
        logger.error(f"Error enhancing prediction with ELO: {str(e)}")
        return prediction_data

def run_demo():
    """Run the demonstration"""
    # Sample predictions
    predictions = [
        {
            "match_id": 12345,
            "home_team_id": 529,  # Barcelona
            "away_team_id": 541,  # Real Madrid
            "home_team_name": "Barcelona",
            "away_team_name": "Real Madrid",
            "league_id": 140,  # La Liga
            "prediction": "Home",
            "confidence": 0.72
        },
        {
            "match_id": 67890,
            "home_team_id": 50,  # Manchester City
            "away_team_id": 40,  # Liverpool
            "home_team_name": "Manchester City",
            "away_team_name": "Liverpool",
            "league_id": 39,  # Premier League
            "prediction": "Home",
            "confidence": 0.68
        },
        {
            "match_id": 11122,
            "home_team_id": 489,  # Juventus
            "away_team_id": 496,  # AC Milan
            "home_team_name": "Juventus",
            "away_team_name": "AC Milan",
            "league_id": 135,  # Serie A
            "prediction": "Draw",
            "confidence": 0.58
        }
    ]
    
    logger.info("=== ENHANCED PREDICTION ELO DEMO ===\n")
    
    # Enhance each prediction with ELO data
    for i, prediction in enumerate(predictions):
        logger.info(f"Match {i+1}: {prediction['home_team_name']} vs {prediction['away_team_name']}")
        logger.info(f"Base prediction: {prediction['prediction']}, confidence: {prediction['confidence']}")
        
        enhanced = enhance_prediction_with_elo(prediction)
        
        logger.info("ELO-enhanced data:")
        if 'elo_insights' in enhanced:
            elo = enhanced['elo_insights']
            logger.info(f"  Home ELO: {elo.get('home_elo')}, Away ELO: {elo.get('away_elo')}")
            logger.info(f"  ELO difference: {elo.get('elo_diff')}")
            logger.info(f"  Strength comparison: {elo.get('strength_comparison')}")
            logger.info(f"  Draw assessment: {elo.get('draw_assessment')}")
            logger.info(f"  Goals insight: {elo.get('goals_insight')}")
        
        if 'enhanced_confidence' in enhanced:
            ec = enhanced['enhanced_confidence']
            logger.info(f"  Enhanced confidence: {ec.get('confidence_score')} " +
                      f"(base: {ec.get('confidence_factors', {}).get('base_model_confidence')}, " +
                      f"ELO adjustment: {ec.get('confidence_factors', {}).get('elo_adjustment')})")
        
        logger.info("\n" + "-" * 50)
    
    logger.info("\nDemo completed successfully!")

if __name__ == "__main__":
    run_demo()
