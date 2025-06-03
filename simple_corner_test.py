#!/usr/bin/env python3
"""
Simple test to verify corner prediction models are working.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from voting_ensemble_corners import VotingEnsembleCornersModel
    
    print("Testing corner prediction model loading...")
    
    model = VotingEnsembleCornersModel()
    
    print(f"RF Model loaded: {model.rf_model is not None}")
    print(f"XGB Model loaded: {model.xgb_model is not None}")
    print(f"Is fitted: {model.is_fitted}")
    
    if model.is_fitted:
        print("✅ SUCCESS: Corner prediction models are now loading correctly!")
    else:
        print("❌ ISSUE: Models still not loading properly")
        
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
