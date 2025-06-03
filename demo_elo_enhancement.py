"""
Demo script for ELO Enhanced Predictions

This script demonstrates how ELO-enhanced predictions work by generating a sample
prediction for a match and showing the additional insights provided by the ELO system.
"""

import logging
import json
from pprint import pprint
from typing import Dict, Any
from prediction_integration import enrich_prediction_with_contextual_data
from team_elo_rating import get_elo_ratings_for_match

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sample_prediction() -> Dict[str, Any]:
    """Create a sample base prediction"""
    return {
        'predicted_home_goals': 1.7,
        'predicted_away_goals': 1.2,
        'total_goals': 2.9,
        'prob_over_2_5': 0.63,
        'prob_btts': 0.68,
        'prob_1': 0.48,
        'prob_X': 0.26,
        'prob_2': 0.26,
        'confidence': 0.65,
        'prediction': 'Home',
        'method': 'statistical'
    }

def demonstrate_elo_enhancement():
    """Demonstrate the ELO enhancement process with example teams"""
    # Use Premier League teams for demonstration
    home_team_id = 39  # Manchester City
    away_team_id = 40  # Liverpool
    league_id = 39     # Premier League
    
    # Create base prediction
    base_prediction = create_sample_prediction()
    print("BASIC PREDICTION (WITHOUT ELO):")
    pprint(base_prediction)
    print("\n" + "-"*80 + "\n")
    
    # Get raw ELO data
    print("RAW ELO DATA:")
    elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
    pprint(elo_data)
    print("\n" + "-"*80 + "\n")
    
    # Enrich with contextual data including ELO
    print("PREDICTION WITH ELO ENHANCEMENT:")
    enriched = enrich_prediction_with_contextual_data(
        base_prediction,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league_id=league_id
    )
    
    # Format nicely for presentation
    print(json.dumps(enriched, indent=2, sort_keys=False))
    print("\n" + "-"*80 + "\n")
    
    # Show key ELO insights
    print("KEY ELO INSIGHTS:")
    if 'elo_insights' in enriched:
        for key, value in enriched['elo_insights'].items():
            print(f"- {key}: {value}")
    print("\n")
    
    if 'elo_enhanced_metrics' in enriched:
        print("ENHANCED ELO METRICS:")
        for key, value in enriched['elo_enhanced_metrics'].items():
            if isinstance(value, dict):
                print(f"- {key}:")
                for sub_key, sub_value in value.items():
                    print(f"  * {sub_key}: {sub_value}")
            else:
                print(f"- {key}: {value}")
    print("\n")
    
    if 'blended_probabilities' in enriched:
        print("BLENDED PROBABILITIES (ELO + STATISTICAL MODEL):")
        for key, value in enriched['blended_probabilities'].items():
            if isinstance(value, dict):
                print(f"- {key}:")
                for sub_key, sub_value in value.items():
                    print(f"  * {sub_key}: {sub_value}")
            else:
                print(f"- {key}: {value}")
    
if __name__ == "__main__":
    demonstrate_elo_enhancement()
