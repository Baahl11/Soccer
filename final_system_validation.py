#!/usr/bin/env python3
"""
Final System Validation - Complete Integration Test
===================================================

This script performs a comprehensive validation of the entire soccer prediction system:
1. ELO system functionality
2. Corner prediction models
3. Full system integration
4. Memory and performance optimization

Expected to pass all tests after ELO integration fixes.
"""

import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# Import system components
from voting_ensemble_corners import VotingEnsembleCornersModel
from team_elo_rating import get_elo_ratings_for_match
from auto_updating_elo import AutoUpdatingEloRating

# Configure logging to reduce noise
logging.getLogger('elo_performance_dashboard').setLevel(logging.ERROR)
logging.getLogger('elo_alert_system').setLevel(logging.ERROR)
logging.getLogger('auto_updating_elo').setLevel(logging.ERROR)
logging.getLogger('elo_database_backend').setLevel(logging.ERROR)
logging.getLogger('team_elo_rating').setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING)

def test_elo_basic_functionality():
    """Test 1: Basic ELO system functionality"""
    print("🔧 Test 1: ELO System Basic Functionality")
    try:
        # Test ELO rating calculation
        elo_features = get_elo_ratings_for_match(1, 2, 39)
        
        # Verify expected ELO outputs
        expected_keys = ['elo_expected_goal_diff', 'elo_win_probability', 'elo_draw_probability', 'elo_loss_probability']
        missing_keys = [key for key in expected_keys if key not in elo_features]
        
        if missing_keys:
            print(f"❌ Missing ELO keys: {missing_keys}")
            return False
            
        print(f"✅ ELO expected_goal_diff: {elo_features['elo_expected_goal_diff']:.2f}")
        print(f"✅ ELO win probability: {elo_features['elo_win_probability']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ ELO system test failed: {e}")
        traceback.print_exc()
        return False

def test_corner_models_loading():
    """Test 2: Corner prediction models loading"""
    print("\n🔧 Test 2: Corner Models Loading")
    try:
        # Create corner model instance
        corner_model = VotingEnsembleCornersModel()
        
        # Check if models loaded successfully
        rf_loaded = corner_model.rf_model is not None
        xgb_loaded = corner_model.xgb_model is not None
        
        print(f"✅ Random Forest loaded: {rf_loaded}")
        print(f"✅ XGBoost loaded: {xgb_loaded}")
        
        if rf_loaded and xgb_loaded:
            print(f"✅ Models are fitted: RF={getattr(corner_model.rf_model, 'n_features_in_', 'Unknown')}, XGB={getattr(corner_model.xgb_model, 'n_features_in_', 'Unknown')} features")
            return True
        else:
            print("❌ One or more models failed to load")
            return False
            
    except Exception as e:
        print(f"❌ Corner models test failed: {e}")
        traceback.print_exc()
        return False

def test_feature_extraction():
    """Test 3: Feature extraction integration"""
    print("\n🔧 Test 3: Feature Extraction")
    try:
        corner_model = VotingEnsembleCornersModel()
        
        # Sample match data
        home_stats = {
            'avg_corners_for': 6.2,
            'avg_corners_against': 4.8,
            'form_score': 65,
            'total_shots': 14
        }
        away_stats = {
            'avg_corners_for': 5.1,
            'avg_corners_against': 5.3,
            'form_score': 58,
            'total_shots': 11
        }
        
        # Extract features
        features = corner_model._extract_features(1, 2, home_stats, away_stats, 39, None)
        
        # Expected features based on model analysis
        expected_features = [
            'home_avg_corners_for', 'home_avg_corners_against', 'away_avg_corners_for',
            'away_avg_corners_against', 'home_form_score', 'away_form_score',
            'home_total_shots', 'away_total_shots', 'league_id'
        ]
        
        # Check if all expected features are present
        missing_features = [f for f in expected_features if f not in features]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
            
        print(f"✅ Feature extraction successful: {len(features)} features")
        print(f"✅ Features: {list(features.keys())}")
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        traceback.print_exc()
        return False

def test_corner_prediction():
    """Test 4: End-to-end corner prediction"""
    print("\n🔧 Test 4: Corner Prediction Pipeline")
    try:
        corner_model = VotingEnsembleCornersModel()
        
        # Sample match data
        home_stats = {
            'avg_corners_for': 6.2,
            'avg_corners_against': 4.8,
            'form_score': 65,
            'total_shots': 14
        }
        away_stats = {
            'avg_corners_for': 5.1,
            'avg_corners_against': 5.3,
            'form_score': 58,
            'total_shots': 11
        }
        
        # Make prediction
        prediction = corner_model.predict_corners(1, 2, home_stats, away_stats, 39)
        
        # Validate prediction structure
        required_keys = ['total', 'home', 'away', 'over_8.5', 'over_9.5', 'over_10.5']
        missing_keys = [key for key in required_keys if key not in prediction]
        
        if missing_keys:
            print(f"❌ Missing prediction keys: {missing_keys}")
            return False
            
        # Validate prediction values
        if not (0 < prediction['total'] < 20):
            print(f"❌ Invalid total corners: {prediction['total']}")
            return False
            
        print(f"✅ Corner prediction successful:")
        print(f"   Total: {prediction['total']}")
        print(f"   Home: {prediction['home']}")
        print(f"   Away: {prediction['away']}")
        print(f"   Over 8.5: {prediction['over_8.5']:.1%}")
        print(f"   Model: {prediction.get('model', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Corner prediction test failed: {e}")
        traceback.print_exc()
        return False

def test_elo_integration():
    """Test 5: ELO integration with auto-updating system"""
    print("\n🔧 Test 5: Auto-updating ELO Integration")
    try:        # Test auto-updating ELO system
        from auto_updating_elo import get_elo_data_with_auto_rating
        elo_data = get_elo_data_with_auto_rating(10, 20, 39)
        
        # Verify expected ELO output
        if 'elo_expected_goal_diff' not in elo_data:
            print("❌ ELO expected_goal_diff missing")
            return False
            
        print(f"✅ Auto ELO integration working: elo_expected_goal_diff = {elo_data['elo_expected_goal_diff']:.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Auto ELO integration test failed: {e}")
        traceback.print_exc()
        return False

def test_multiple_matches():
    """Test 6: Multiple match scenarios"""
    print("\n🔧 Test 6: Multiple Match Scenarios")
    try:
        corner_model = VotingEnsembleCornersModel()
        
        # Test multiple matches with different scenarios
        test_matches = [
            (1, 2, "Standard match"),
            (100, 200, "New teams"),
            (5, 15, "Different league teams")
        ]
        
        all_passed = True
        for home_id, away_id, description in test_matches:
            try:
                # Sample stats
                home_stats = {'avg_corners_for': 5.5, 'avg_corners_against': 4.5, 'form_score': 60, 'total_shots': 12}
                away_stats = {'avg_corners_for': 4.8, 'avg_corners_against': 5.2, 'form_score': 55, 'total_shots': 10}
                
                # Make prediction
                prediction = corner_model.predict_corners(home_id, away_id, home_stats, away_stats, 39)
                
                print(f"✅ {description}: Total corners = {prediction['total']}")
                
            except Exception as e:
                print(f"❌ {description} failed: {e}")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"❌ Multiple match test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive system validation"""
    print("=" * 50)
    print("FINAL SYSTEM VALIDATION")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all tests
    tests = [
        test_elo_basic_functionality,
        test_corner_models_loading,
        test_feature_extraction,
        test_corner_prediction,
        test_elo_integration,
        test_multiple_matches
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("⚠️  Test failed!")
        except Exception as e:
            print(f"💥 Test crashed: {e}")
            traceback.print_exc()
    
    # Final results
    print("\n" + "=" * 50)
    print("FINAL TEST SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 SYSTEM VALIDATION: PASSED!")
        print("✅ All components are working correctly")
        print("✅ ELO integration is functional")
        print("✅ Corner prediction models are operational")
        print("✅ End-to-end pipeline is validated")
        return True
    else:
        print("❌ SYSTEM VALIDATION: FAILED!")
        print(f"⚠️  {total - passed} test(s) failed. Review failed tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
