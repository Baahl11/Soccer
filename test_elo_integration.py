"""
Test script to validate the integration of Elo ratings in the prediction system.
This creates a mock prediction and enriches it with Elo data.
"""

import logging
from pprint import pprint
from prediction_integration import enrich_prediction_with_contextual_data
from team_elo_rating import get_elo_ratings_for_match

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_elo_integration():
    """Test the Elo rating integration in predictions"""
    
    # Sample prediction (similar to what would be returned by calculate_statistical_prediction)
    base_prediction = {
        'predicted_home_goals': 1.8,
        'predicted_away_goals': 1.2,
        'total_goals': 3.0,
        'prob_over_2_5': 0.65,
        'prob_btts': 0.7
    }
    
    # Use real team IDs for Premier League teams
    home_team_id = 39    # Manchester City
    away_team_id = 40    # Liverpool
    league_id = 39       # Premier League
    
    # Print Elo data directly for reference
    print("Elo ratings for the match:")
    elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
    pprint(elo_data)
    print("\n")
    
    # Enrich prediction with Elo data
    enriched = enrich_prediction_with_contextual_data(
        base_prediction,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league_id=league_id
    )
    
    # Print the enriched prediction
    print("Enriched prediction with Elo data:")
    pprint(enriched)
    
    # Validate Elo data was added
    if 'elo' in enriched.get('adjustments_applied', []):
        print("\nSuccess: Elo data was successfully integrated into the prediction!")
        
        # Check if all expected Elo fields are present
        expected_fields = [
            ('elo_ratings', dict),
            ('elo_probabilities', dict),
            ('elo_expected_goal_diff', float)
        ]
        
        all_fields_present = True
        for field, field_type in expected_fields:
            if field not in enriched:
                print(f"Missing field: {field}")
                all_fields_present = False
            elif not isinstance(enriched[field], field_type):
                print(f"Field {field} has wrong type: {type(enriched[field])} (expected {field_type})")
                all_fields_present = False
        
        if all_fields_present:
            print("All Elo fields are present with the correct types.")
            
            # Compare data from direct Elo call with integrated data
            elo_win_prob = elo_data['elo_win_probability']
            integrated_win_prob = enriched['elo_probabilities']['win']
            
            if elo_win_prob == integrated_win_prob:
                print(f"Elo win probability matches: {elo_win_prob}")
            else:
                print(f"Elo win probability mismatch: direct={elo_win_prob}, integrated={integrated_win_prob}")
    else:
        print("\nFailure: Elo data was not added to the prediction.")

if __name__ == "__main__":
    test_elo_integration()
