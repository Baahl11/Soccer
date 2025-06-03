"""
Test script for ELO ratings integration with prediction system.

This simplified script focuses on testing how ELO ratings can enhance
soccer predictions with additional insights about team strengths.
"""

import json
import logging
from typing import Dict, Any

# Import the function from team_elo_rating for direct ELO analysis
from team_elo_rating import get_elo_ratings_for_match
from elo_enhanced_demo import enhance_prediction_with_elo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_sample_prediction() -> Dict[str, Any]:
    """Create a sample prediction for testing"""
    return {
        "match_id": 12345,
        "home_team_id": 529,  # Barcelona
        "away_team_id": 541,  # Real Madrid
        "home_team_name": "Barcelona",
        "away_team_name": "Real Madrid",
        "league_id": 140,  # La Liga
        "prediction": "Home",
        "confidence": 0.72,
        "home_win_probability": 0.65,
        "draw_probability": 0.20,
        "away_win_probability": 0.15
    }

def test_elo_ratings_integration():
    """Test ELO ratings integration directly"""
    home_team_id = 529  # Barcelona
    away_team_id = 541  # Real Madrid
    league_id = 140  # La Liga
    
    elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
    
    logger.info("ELO Ratings for Barcelona vs Real Madrid:")
    logger.info(json.dumps(elo_data, indent=2))
    
    return elo_data

def test_single_prediction_enhancement():
    """Test enhancing a single prediction"""
    # Create a sample prediction
    prediction = create_sample_prediction()
    
    # Enhance the prediction with ELO data
    enhanced = enhance_prediction_with_elo(prediction)
    
    logger.info("Original prediction:")
    logger.info(json.dumps(prediction, indent=2))
    
    logger.info("\nEnhanced prediction with ELO insights:")
    
    # Print only the ELO fields for clarity
    elo_fields = {
        "elo_insights": enhanced.get("elo_insights", {}),
        "elo_enhanced_confidence": enhanced.get("elo_enhanced_confidence")
    }
    logger.info(json.dumps(elo_fields, indent=2))
    
    return enhanced

def test_batch_predictions():
    """Test enhancing multiple predictions"""
    # Create multiple sample predictions
    predictions = [
        create_sample_prediction(),
        {
            "match_id": 67890,
            "home_team_id": 50,   # Manchester City
            "away_team_id": 40,   # Liverpool
            "home_team_name": "Manchester City",
            "away_team_name": "Liverpool",
            "league_id": 39,      # Premier League
            "prediction": "Home",
            "confidence": 0.68,
            "home_win_probability": 0.60,
            "draw_probability": 0.25,
            "away_win_probability": 0.15
        },
        {
            "match_id": 78901,
            "home_team_id": 489,   # Juventus
            "away_team_id": 496,   # AC Milan
            "home_team_name": "Juventus",
            "away_team_name": "AC Milan",
            "league_id": 135,      # Serie A
            "prediction": "Draw",
            "confidence": 0.58,
            "home_win_probability": 0.35,
            "draw_probability": 0.40,
            "away_win_probability": 0.25
        }
    ]
    
    # Process each prediction
    enhanced_predictions = []
    for prediction in predictions:
        enhanced = enhance_prediction_with_elo(prediction)
        enhanced_predictions.append(enhanced)
    
    # Show summary of enhanced predictions
    logger.info(f"Enhanced {len(enhanced_predictions)} predictions")
    
    for i, enhanced in enumerate(enhanced_predictions):
        match_details = f"{enhanced['home_team_name']} vs {enhanced['away_team_name']}"
        elo_diff = enhanced.get('elo_insights', {}).get('elo_diff', 0)
        confidence = enhanced.get('elo_enhanced_confidence', enhanced.get('confidence'))
        
        logger.info(f"\nMatch {i+1}: {match_details}")
        logger.info(f"ELO Difference: {elo_diff}")
        logger.info(f"Enhanced confidence: {confidence}")
        logger.info(f"Strength: {enhanced.get('elo_insights', {}).get('strength_comparison', 'N/A')}")
    
    return enhanced_predictions

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING ELO RATINGS INTEGRATION")
    print("=" * 60)
    test_elo_ratings_integration()
    
    print("\n" + "=" * 60)
    print("TESTING SINGLE PREDICTION ENHANCEMENT")
    print("=" * 60)
    test_single_prediction_enhancement()
    
    print("\n" + "=" * 60)
    print("TESTING BATCH PREDICTIONS")
    print("=" * 60)
    test_batch_predictions()

if __name__ == "__main__":
    print("Script starting...")
    try:
        main()
        print("Script completed successfully!")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
