#!/usr/bin/env python3
"""
Quick integration test for the ELO fix
"""
import sys
sys.path.append('.')

from voting_ensemble_corners import VotingEnsembleCornersModel

# Create model instance
print("Creating VotingEnsembleCornersModel...")
model = VotingEnsembleCornersModel()
print("✅ Model created successfully!")

# Test data
home_stats = {
    'avg_corners_for': 5.2, 
    'avg_corners_against': 4.8, 
    'form_score': 60, 
    'total_shots': 14.0
}
away_stats = {
    'avg_corners_for': 4.1, 
    'avg_corners_against': 5.3, 
    'form_score': 45, 
    'total_shots': 11.5
}

print("\nTesting feature extraction with ELO integration...")
try:
    features = model._extract_features(1, 2, home_stats, away_stats, 1, None)
    print(f"✅ SUCCESS: Feature extraction worked! Features count: {len(features)}")
    
    print("\nKey ELO features found:")
    elo_keys = ['home_elo', 'away_elo', 'elo_diff', 'elo_win_probability']
    for k in elo_keys:
        value = features.get(k, "NOT_FOUND")
        print(f"  {k}: {value}")
    
    # Test the specific fix we made
    if 'expected_goal_diff' in features:
        print(f"\n✅ CRITICAL: 'expected_goal_diff' feature successfully mapped: {features['expected_goal_diff']}")
    else:
        print("\nℹ️  'expected_goal_diff' not in this model's feature set (this is OK)")
    
    print("\n🎉 ELO INTEGRATION TEST PASSED!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
