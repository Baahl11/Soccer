"""
Test script to verify that all integration fixes are working correctly.
"""

import sys
import traceback

def test_imports():
    """Test that all modules can be imported without errors"""
    print("Testing imports...")
    
    try:
        # Test enhanced_predictions import
        from enhanced_predictions import make_enhanced_prediction, predict_corners_with_voting_ensemble
        print("✓ enhanced_predictions imported successfully")
        
        # Test tactical_corner_predictor import
        from tactical_corner_predictor import TacticalCornerPredictor
        print("✓ tactical_corner_predictor imported successfully")
        
        # Test voting_ensemble_corners import
        from voting_ensemble_corners import VotingEnsembleCornersModel
        print("✓ voting_ensemble_corners imported successfully")
        
        # Test other key modules
        from team_form import get_team_form, get_head_to_head_analysis
        print("✓ team_form imported successfully")
        
        from match_winner import predict_match_winner
        print("✓ match_winner imported successfully")
        
        from elo_integration import enhance_prediction_with_elo
        print("✓ elo_integration imported successfully")
        
        from team_elo_rating import get_elo_ratings_for_match
        print("✓ team_elo_rating imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test VotingEnsembleCornersModel initialization
        from voting_ensemble_corners import VotingEnsembleCornersModel
        ensemble_model = VotingEnsembleCornersModel()
        print("✓ VotingEnsembleCornersModel initialized successfully")
        
        # Test TacticalCornerPredictor initialization
        from tactical_corner_predictor import TacticalCornerPredictor
        tactical_predictor = TacticalCornerPredictor()
        print("✓ TacticalCornerPredictor initialized successfully")
        
        # Test predict_corners_with_voting_ensemble function
        from enhanced_predictions import predict_corners_with_voting_ensemble
        
        # Create sample data
        home_form = {
            'avg_corners_for': 5.2,
            'avg_corners_against': 4.8,
            'form_score': 65,
            'avg_shots': 14.0
        }
        
        away_form = {
            'avg_corners_for': 4.5,
            'avg_corners_against': 5.5,
            'form_score': 45,
            'avg_shots': 11.0
        }
        
        # Test the function
        result = predict_corners_with_voting_ensemble(
            home_team_id=1,
            away_team_id=2,
            home_form=home_form,
            away_form=away_form,
            league_id=39
        )
        
        print("✓ predict_corners_with_voting_ensemble executed successfully")
        print(f"  Result keys: {list(result.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test error: {e}")
        traceback.print_exc()
        return False

def test_enhanced_prediction():
    """Test the main enhanced prediction function"""
    print("\nTesting enhanced prediction...")
    
    try:
        from enhanced_predictions import make_enhanced_prediction
        
        # Test with minimal parameters (should use fallback for missing data)
        result = make_enhanced_prediction(
            fixture_id=12345,
            home_team_id=1,
            away_team_id=2,
            league_id=39,
            season=2024,
            use_elo=False  # Disable ELO to avoid API calls
        )
        
        print("✓ make_enhanced_prediction executed successfully")
        print(f"  Result keys: {list(result.keys())}")
        
        # Check that required keys are present
        required_keys = ['predicted_home_goals', 'predicted_away_goals', 'corners', 'match_winner']
        missing_keys = [key for key in required_keys if key not in result]
        
        if missing_keys:
            print(f"✗ Missing required keys: {missing_keys}")
            return False
        else:
            print("✓ All required keys present in result")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced prediction test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("INTEGRATION FIXES VERIFICATION")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_enhanced_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All integration fixes are working correctly!")
        return True
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
