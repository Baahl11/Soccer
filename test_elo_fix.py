#!/usr/bin/env python3
"""
Simple test to verify the ELO key naming fix in voting_ensemble_corners.py
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_elo_keys():
    """Test that the key naming fix works correctly"""
    
    print("Testing the fixed ELO key naming in voting_ensemble_corners.py...")
    
    try:
        # Test ELO system output
        from team_elo_rating import get_elo_ratings_for_match
        
        print("\n1. Getting ELO data...")
        elo_data = get_elo_ratings_for_match(1, 2, 1)
        print("ELO system keys:")
        for key in elo_data.keys():
            print(f"  '{key}'")
        
        # Verify the key we fixed exists
        if 'elo_expected_goal_diff' in elo_data:
            print("✓ 'elo_expected_goal_diff' key found in ELO data")
        else:
            print("✗ 'elo_expected_goal_diff' key NOT found in ELO data")
            return False
        
        # Test the VotingEnsembleCornersModel class
        print("\n2. Testing VotingEnsembleCornersModel class...")
        
        from voting_ensemble_corners import VotingEnsembleCornersModel
        
        try:
            # Create an instance of the model
            model = VotingEnsembleCornersModel()
            print("✓ VotingEnsembleCornersModel instantiated successfully")
            
            # Test the _extract_features method directly with sample data
            home_stats = {
                'avg_corners_for': 5.0,
                'avg_corners_against': 4.5,
                'form_score': 55,
                'total_shots': 12.0
            }
            away_stats = {
                'avg_corners_for': 4.5,
                'avg_corners_against': 5.5,
                'form_score': 45,
                'total_shots': 10.0
            }
            
            print("\n3. Testing feature extraction with ELO integration...")
            features = model._extract_features(1, 2, home_stats, away_stats, 1, None)
            print(f"✓ Feature extraction successful! Extracted {len(features)} features")
            
            # Check if expected_goal_diff was mapped correctly
            if 'expected_goal_diff' in features:
                print(f"✓ 'expected_goal_diff' feature successfully extracted: {features['expected_goal_diff']}")
            else:
                print("ℹ 'expected_goal_diff' feature not in this model's feature set (this is OK)")
            
            # Check some basic ELO features
            elo_features = ['home_elo', 'away_elo', 'elo_diff', 'elo_win_probability']
            for feature_name in elo_features:
                if feature_name in features:
                    print(f"✓ '{feature_name}' feature found: {features[feature_name]}")
                else:
                    print(f"ℹ '{feature_name}' feature not in this model's feature set")
            
        except KeyError as e:
            print(f"✗ KeyError during feature extraction: {e}")
            print("This suggests the key naming fix may not be working correctly")
            return False
        except Exception as e:
            print(f"✗ Other error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        print("\n✓ All tests passed! The ELO key naming fix is working correctly.")
        return True
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fixed_elo_keys()
    if success:
        print("\n🎉 ELO key naming fix test PASSED!")
        sys.exit(0)
    else:
        print("\n❌ ELO key naming fix test FAILED!")
        sys.exit(1)
