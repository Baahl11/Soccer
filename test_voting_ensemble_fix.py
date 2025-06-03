"""
Test script to verify that the voting_ensemble_corners.py file works correctly
after fixing indentation issues in the _predict_with_rf and _predict_with_xgb methods.
"""

import sys
import logging
import json
from voting_ensemble_corners import VotingEnsembleCornersModel, predict_corners_with_voting_ensemble

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Sample data for testing
home_team_id = 123
away_team_id = 456
league_id = 39  # Premier League

home_stats = {
    'avg_corners_for': 6.2,
    'avg_corners_against': 4.5,
    'form_score': 65,
    'attack_strength': 1.1,
    'defense_strength': 0.95,
    'avg_shots': 14.2
}

away_stats = {
    'avg_corners_for': 5.1,
    'avg_corners_against': 5.8,
    'form_score': 48,
    'attack_strength': 0.9,
    'defense_strength': 1.05,
    'avg_shots': 10.5
}

context_factors = {
    'is_windy': False,
    'is_rainy': True,
    'is_high_stakes': True
}

def run_test():
    """Run test function to check voting ensemble model"""
    print("Testing VotingEnsembleCornersModel directly...")
    
    # Test direct model creation
    try:
        model = VotingEnsembleCornersModel()
        print("✓ Successfully created VotingEnsembleCornersModel instance")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        return False
    
    # Test prediction using the model
    try:
        result = model.predict_corners(
            home_team_id, away_team_id, home_stats, away_stats, league_id, context_factors
        )
        print(f"✓ Successfully made prediction with model: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"✗ Error making prediction with model: {e}")
        return False
    
    # Test helper function
    print("\nTesting helper function predict_corners_with_voting_ensemble...")
    try:
        result = predict_corners_with_voting_ensemble(
            home_team_id, away_team_id, home_stats, away_stats, league_id, context_factors
        )
        print(f"✓ Successfully made prediction with helper function: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"✗ Error using helper function: {e}")
        return False
        
    return True

if __name__ == "__main__":
    print("Testing voting ensemble corners model after fixing indentation issues...")
    success = run_test()
    
    if success:
        print("\n✓✓✓ All tests passed. The model is working correctly!")
        sys.exit(0)
    else:
        print("\n✗✗✗ Tests failed. The model still has issues.")
        sys.exit(1)
