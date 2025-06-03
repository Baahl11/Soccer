#!/usr/bin/env python3
"""
Debug script to trace why all predictions return identical probabilities
"""

import logging
import json
from typing import Dict, Any
from enhanced_match_winner import EnhancedPredictionSystem
from match_winner import predict_match_winner, WinnerPredictor

# Configure logging to see all debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def debug_prediction_pipeline(home_team_id: int, away_team_id: int, league_id: int):
    """
    Debug the prediction pipeline to see where identical values come from
    """
    print(f"\n🔍 DEBUGGING PREDICTION PIPELINE")
    print(f"Teams: {home_team_id} vs {away_team_id} (League: {league_id})")
    print("=" * 60)
    
    # Step 1: Test base prediction system directly
    print("\n1️⃣ Testing base WinnerPredictor directly...")
    try:
        predictor = WinnerPredictor()
        
        # Use default/minimal form data to see what happens
        default_form = {
            'goals_scored_pg': 1.3,
            'goals_conceded_pg': 1.1,
            'wins': 5,
            'losses': 3,
            'draws': 2,
            'total_matches': 10,
            'win_percentage': 50.0
        }
        
        default_h2h = {
            'total_matches': 3,
            'team1_wins': 1,
            'team2_wins': 1,
            'draws': 1
        }
        
        base_result = predictor.predict_winner(
            home_xg=1.3,
            away_xg=1.1,
            home_form=default_form,
            away_form=default_form,
            h2h=default_h2h,
            league_id=league_id
        )
        
        print(f"Base probabilities: {base_result.get('probabilities', {})}")
        
    except Exception as e:
        print(f"❌ Error in base predictor: {e}")
    
    # Step 2: Test enhanced system
    print("\n2️⃣ Testing EnhancedPredictionSystem...")
    try:
        enhanced_system = EnhancedPredictionSystem()
        enhanced_result = enhanced_system.predict(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id
        )
        
        print(f"Enhanced probabilities: {enhanced_result.get('probabilities', {})}")
        
    except Exception as e:
        print(f"❌ Error in enhanced system: {e}")
    
    # Step 3: Test with different xG values to see if they affect output
    print("\n3️⃣ Testing with different xG values...")
    test_scenarios = [
        (0.8, 2.5),  # Away team heavily favored
        (2.5, 0.8),  # Home team heavily favored
        (1.5, 1.5),  # Even match
        (3.0, 0.5),  # Very uneven
    ]
    
    for home_xg, away_xg in test_scenarios:
        try:
            result = predictor.predict_winner(
                home_xg=home_xg,
                away_xg=away_xg,
                home_form=default_form,
                away_form=default_form,
                h2h=default_h2h,
                league_id=league_id
            )
            probs = result.get('probabilities', {})
            print(f"xG {home_xg}:{away_xg} -> Home:{probs.get('home_win',0):.1f}% Draw:{probs.get('draw',0):.1f}% Away:{probs.get('away_win',0):.1f}%")
        except Exception as e:
            print(f"❌ Error with xG {home_xg}:{away_xg}: {e}")

def test_multiple_team_combinations():
    """
    Test multiple team combinations to confirm they all return the same values
    """
    print(f"\n🎲 TESTING MULTIPLE TEAM COMBINATIONS")
    print("=" * 60)
    
    team_combinations = [
        (33, 40, 39),   # Man United vs Liverpool (Premier League)
        (529, 530, 140), # Barcelona vs Real Madrid (La Liga)
        (157, 165, 78),  # Bayern vs Dortmund (Bundesliga)
        (496, 489, 135), # Juventus vs AC Milan (Serie A)
        (85, 81, 61),    # PSG vs Marseille (Ligue 1)
    ]
    
    enhanced_system = EnhancedPredictionSystem()
    
    for home_id, away_id, league_id in team_combinations:
        try:
            result = enhanced_system.predict(
                home_team_id=home_id,
                away_team_id=away_id,
                league_id=league_id
            )
            
            probs = result.get('probabilities', {})
            print(f"Teams {home_id} vs {away_id}: Home:{probs.get('home_win',0):.1f}% Draw:{probs.get('draw',0):.1f}% Away:{probs.get('away_win',0):.1f}%")
            
        except Exception as e:
            print(f"❌ Error with teams {home_id} vs {away_id}: {e}")

def check_for_hardcoded_values():
    """
    Search for hardcoded values that match our problematic output
    """
    print(f"\n🔎 CHECKING FOR HARDCODED VALUES")
    print("=" * 60)
    
    # The exact values we're seeing: 42.1%, 35.7%, 22.2%
    # In decimal: 0.421, 0.357, 0.222
    
    problematic_values = [42.1, 35.7, 22.2, 0.421, 0.357, 0.222]
    
    print("Looking for these problematic values in the prediction pipeline:")
    for val in problematic_values:
        print(f"  - {val}")
    
    print("\nThis suggests there's a fallback/default prediction being used")
    print("instead of calculating team-specific probabilities.")

if __name__ == "__main__":
    print("🚨 DEBUGGING IDENTICAL PROBABILITY ISSUE")
    print("This script will help identify why all predictions return the same values")
    
    # Test with the same team combination from your earlier tests
    debug_prediction_pipeline(33, 40, 39)  # Man United vs Liverpool
    
    test_multiple_team_combinations()
    
    check_for_hardcoded_values()
    
    print(f"\n📝 NEXT STEPS:")
    print("1. Check if the base prediction system is receiving proper form data")
    print("2. Verify that xG calculations are working correctly")
    print("3. Look for fallback mechanisms that might be triggered")
    print("4. Check if team-specific data is being retrieved properly")
