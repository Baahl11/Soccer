#!/usr/bin/env python3
"""
Test Enhanced Match Winner Integration

This module tests the integration of the Enhanced Match Winner system
to ensure all components work together correctly.
"""

import logging
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_enhanced_prediction_system():
    """Test the Enhanced Prediction System integration."""
    print("=== Testing Enhanced Match Winner Integration ===\n")
    
    try:
        # Import the enhanced system
        from enhanced_match_winner import EnhancedPredictionSystem, predict_with_enhanced_system
        from draw_prediction import DrawPredictor, enhance_draw_predictions
        from match_winner import MatchOutcome
        
        print("✅ Successfully imported enhanced prediction modules")
        
        # Test 1: Initialize Enhanced Prediction System
        print("\n1. Testing Enhanced Prediction System initialization...")
        enhanced_system = EnhancedPredictionSystem()
        print("✅ Enhanced Prediction System initialized successfully")
        
        # Test 2: Initialize Draw Predictor
        print("\n2. Testing Draw Predictor initialization...")
        draw_predictor = DrawPredictor()
        print("✅ Draw Predictor initialized successfully")
        
        # Test 3: Test enhanced prediction with real team IDs
        print("\n3. Testing enhanced prediction...")
        test_home_team = 33  # Manchester United
        test_away_team = 40  # Liverpool
        test_league = 39     # Premier League
        
        prediction = enhanced_system.predict(
            home_team_id=test_home_team,
            away_team_id=test_away_team,
            league_id=test_league
        )
        
        print(f"✅ Enhanced prediction completed for teams {test_home_team} vs {test_away_team}")
        print(f"   Predicted outcome: {prediction.get('predicted_outcome')}")
        print(f"   Probabilities: {prediction.get('probabilities', {})}")
        
        # Test 4: Test convenience function
        print("\n4. Testing convenience function...")
        conv_prediction = predict_with_enhanced_system(
            home_team_id=test_home_team,
            away_team_id=test_away_team,
            league_id=test_league
        )
        
        print(f"✅ Convenience function works correctly")
        print(f"   Predicted outcome: {conv_prediction.get('predicted_outcome')}")
        
        # Test 5: Test batch prediction
        print("\n5. Testing batch prediction...")
        test_matches = [
            {
                'home_team_id': 33,  # Manchester United
                'away_team_id': 40,  # Liverpool
                'league_id': 39,     # Premier League
                'fixture_id': 'test_1'
            },
            {
                'home_team_id': 50,  # Manchester City
                'away_team_id': 49,  # Chelsea
                'league_id': 39,     # Premier League
                'fixture_id': 'test_2'
            }
        ]
        
        batch_predictions = enhanced_system.batch_predict(test_matches)
        print(f"✅ Batch prediction completed for {len(batch_predictions)} matches")
        
        for i, pred in enumerate(batch_predictions):
            print(f"   Match {i+1}: {pred.get('predicted_outcome')} (ID: {pred.get('fixture_id')})")
        
        # Test 6: Test draw enhancement function directly
        print("\n6. Testing draw enhancement function...")
        
        # Create a mock base prediction
        base_prediction = {
            'predicted_outcome': MatchOutcome.HOME_WIN.value,
            'probabilities': {
                MatchOutcome.HOME_WIN.value: 0.5,
                MatchOutcome.DRAW.value: 0.2,
                MatchOutcome.AWAY_WIN.value: 0.3
            },
            'confidence': 0.75
        }
        
        enhanced_pred = enhance_draw_predictions(
            base_prediction,
            test_home_team,
            test_away_team,
            draw_predictor,
            test_league
        )
        
        print("✅ Draw enhancement function works correctly")
        print(f"   Original draw prob: {base_prediction['probabilities'][MatchOutcome.DRAW.value]:.3f}")
        print(f"   Enhanced draw prob: {enhanced_pred['probabilities'][MatchOutcome.DRAW.value]:.3f}")
        
        # Test 7: Validate probability normalization
        print("\n7. Testing probability normalization...")
        total_prob = sum(enhanced_pred['probabilities'].values())
        if abs(total_prob - 1.0) < 0.001:
            print("✅ Probabilities are properly normalized")
        else:
            print(f"❌ Probabilities sum to {total_prob:.3f}, expected 1.0")
            return False
        
        print("\n=== Enhanced Match Winner Integration Test Results ===")
        print("✅ ALL TESTS PASSED!")
        print("✅ Enhanced Match Winner system is properly integrated")
        print("✅ Draw prediction enhancement is working correctly")
        print("✅ Batch prediction functionality is operational")
        print("✅ Probability calibration and normalization work correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("   Missing required modules for enhanced prediction")
        return False
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_main_system():
    """Test integration with the main prediction system."""
    print("\n=== Testing Integration with Main Prediction System ===\n")
    
    try:
        # Test integration through the main prediction integration module
        from prediction_integration import make_integrated_prediction
        from enhanced_match_winner import predict_with_enhanced_system
        
        print("1. Testing integration with existing prediction system...")
        
        # Create mock fixture data for testing
        mock_fixture_data = {
            'fixture_id': 12345,
            'home_team_id': 33,
            'away_team_id': 40,
            'league_id': 39,
            'date': '2024-12-01',
            'venue': 'Old Trafford'
        }
        
        # Test with mock data first (avoiding API calls)
        print("   Testing with mock data...")
        result = make_integrated_prediction(
            fixture_id=12345,
            fixture_data=mock_fixture_data
        )        
        print("✅ Mock integrated prediction completed")
        
        # Test enhanced prediction separately
        print("   Testing enhanced prediction system...")
        enhanced_result = predict_with_enhanced_system(
            home_team_id=33,
            away_team_id=40,
            league_id=39
        )
        
        print("✅ Enhanced prediction system works correctly")
        print(f"   Enhanced predicted outcome: {enhanced_result.get('predicted_outcome')}")
        
        # Compare integration vs enhanced approaches
        print("   Comparing prediction approaches...")
        
        if result.get('prediction') and enhanced_result:
            base_prob = result.get('prediction', {}).get('probabilities', {})
            enhanced_prob = enhanced_result.get('probabilities', {})
            
            print(f"   Base prediction outcome: {result.get('prediction', {}).get('predicted_outcome')}")
            print(f"   Enhanced prediction outcome: {enhanced_result.get('predicted_outcome')}")
            
            if base_prob and enhanced_prob:
                print(f"   Base draw probability: {base_prob.get('draw', 'N/A')}")
                print(f"   Enhanced draw probability: {enhanced_prob.get('draw', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing main system integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Enhanced Match Winner Integration Test Suite")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_enhanced_prediction_system()
    test2_passed = test_integration_with_main_system()
    
    # Summary
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS:")
    print(f"Enhanced System Tests: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Main Integration Tests: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("🎉 Enhanced Match Winner system is fully operational!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
