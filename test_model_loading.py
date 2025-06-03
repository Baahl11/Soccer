#!/usr/bin/env python3
"""
Test script to verify corner prediction models are loading correctly.
"""

import os
import sys
import logging
from pathlib import Path

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from voting_ensemble_corners import VotingEnsembleCornersModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the corner prediction models load correctly"""
    
    print("=" * 60)
    print("CORNER PREDICTION MODEL LOADING TEST")
    print("=" * 60)
    
    # Check if model files exist
    model_files = [
        'models/random_forest_corners.pkl',
        'models/xgboost_corners.pkl',
        'models/random_forest_corners.joblib',
        'models/xgboost_corners.json'
    ]
    
    print("\n1. Checking for model files:")
    for file_path in model_files:
        exists = os.path.exists(file_path)
        status = "✓ EXISTS" if exists else "✗ NOT FOUND"
        print(f"   {file_path}: {status}")
    
    print("\n2. Initializing VotingEnsembleCornersModel...")
    try:
        model = VotingEnsembleCornersModel()
        print("   ✓ Model initialization successful")
        
        # Check if models were loaded
        print("\n3. Checking model loading status:")
        rf_loaded = model.rf_model is not None
        xgb_loaded = model.xgb_model is not None
        
        print(f"   Random Forest model: {'✓ LOADED' if rf_loaded else '✗ NOT LOADED'}")
        print(f"   XGBoost model: {'✓ LOADED' if xgb_loaded else '✗ NOT LOADED'}")
        print(f"   Is fitted: {'✓ YES' if model.is_fitted else '✗ NO'}")
        
        if rf_loaded:
            print(f"   RF model type: {type(model.rf_model)}")
        if xgb_loaded:
            print(f"   XGB model type: {type(model.xgb_model)}")
            
        # Test basic prediction if models are loaded
        if rf_loaded and xgb_loaded:
            print("\n4. Testing corner prediction...")
            
            # Sample data for testing
            home_stats = {
                'avg_corners_for': 5.2,
                'avg_corners_against': 4.8,
                'form_score': 0.6,
                'total_shots': 15,
                'elo_rating': 1500
            }
            
            away_stats = {
                'avg_corners_for': 4.9,
                'avg_corners_against': 5.1,
                'form_score': 0.4,
                'total_shots': 12,
                'elo_rating': 1450
            }
            
            try:
                result = model.predict_corners(
                    home_team_id=1,
                    away_team_id=2,
                    home_stats=home_stats,
                    away_stats=away_stats,
                    league_id=39
                )
                
                print("   ✓ Prediction successful!")
                print(f"   Expected total corners: {result.get('expected_total_corners', 'N/A')}")
                print(f"   Confidence score: {result.get('confidence_score', 'N/A')}")
                
            except Exception as e:
                print(f"   ✗ Prediction failed: {e}")
        else:
            print("\n4. Skipping prediction test (models not loaded)")
            
    except Exception as e:
        print(f"   ✗ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    test_model_loading()
