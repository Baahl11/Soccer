#!/usr/bin/env python3
"""
Test script to verify the enhanced dynamic confidence calculation
"""

import sys
import os

def test_enhanced_confidence():
    """Test the enhanced confidence calculation with various scenarios"""
    
    print("=== ENHANCED DYNAMIC CONFIDENCE TEST ===")
    
    try:
        # Add the current directory to Python path
        sys.path.insert(0, os.getcwd())
        
        # Import the enhanced function from app.py
        from app import calculate_dynamic_confidence
        print("✅ Successfully imported enhanced calculate_dynamic_confidence")
        
        # Test Case 1: High Confidence Scenario (Clear favorite with good data)
        print("\n--- Test Case 1: High Confidence Scenario ---")
        high_confidence_match = {
            "fixture_id": 12345,
            "home_team_id": 50,    # Manchester City
            "away_team_id": 42,    # Arsenal (but weaker form)
            "league_id": 39,       # Premier League (high quality)
            "predicted_home_goals": 2.8,
            "predicted_away_goals": 1.1,
            "home_win_probability": 0.75,  # Clear favorite
            "draw_probability": 0.15,
            "away_win_probability": 0.10,
            "home_team_form": {
                "win_percentage": 85,
                "goals_scored": 25,
                "goals_conceded": 8
            },
            "away_team_form": {
                "win_percentage": 55,  # Much lower
                "goals_scored": 15,
                "goals_conceded": 18
            },
            "head_to_head": {
                "total_matches": 8,
                "home_wins": 6,     # Home team dominates historically
                "away_wins": 1,
                "draws": 1
            },
            "home_team_injuries": {
                "key_players_injured": 0,
                "severity_score": 0.1
            },
            "away_team_injuries": {
                "key_players_injured": 2,  # Away team has injuries
                "severity_score": 0.6
            },
            "weather": {
                "temperature": 18,
                "wind_speed": 8,
                "precipitation": 0
            }
        }
        
        high_conf = calculate_dynamic_confidence(high_confidence_match)
        print(f"High confidence scenario result: {high_conf}")
        
        # Test Case 2: Low Confidence Scenario (Very close match)
        print("\n--- Test Case 2: Low Confidence Scenario ---")
        low_confidence_match = {
            "fixture_id": 67890,
            "home_team_id": 45,
            "away_team_id": 51,
            "league_id": 200,      # Lower league
            "predicted_home_goals": 1.6,
            "predicted_away_goals": 1.4,
            "home_win_probability": 0.38,  # Very close probabilities
            "draw_probability": 0.32,
            "away_win_probability": 0.30,
            "home_team_form": {
                "win_percentage": 52,
                "goals_scored": 12,
                "goals_conceded": 11
            },
            "away_team_form": {
                "win_percentage": 48,  # Very similar form
                "goals_scored": 11,
                "goals_conceded": 12
            },
            "head_to_head": {
                "total_matches": 3,    # Limited history
                "home_wins": 1,
                "away_wins": 1,
                "draws": 1
            },
            "home_team_injuries": {
                "key_players_injured": 1,
                "severity_score": 0.3
            },
            "away_team_injuries": {
                "key_players_injured": 1,  # Similar injury situation
                "severity_score": 0.3
            },
            "weather": {
                "temperature": 2,      # Extreme cold
                "wind_speed": 30,      # Strong wind
                "precipitation": 15    # Heavy rain
            }
        }
        
        low_conf = calculate_dynamic_confidence(low_confidence_match)
        print(f"Low confidence scenario result: {low_conf}")
        
        # Test Case 3: Medium Confidence Scenario (Moderate favorite)
        print("\n--- Test Case 3: Medium Confidence Scenario ---")
        medium_confidence_match = {
            "fixture_id": 11111,
            "home_team_id": 40,
            "away_team_id": 49,
            "league_id": 78,       # Bundesliga (high quality)
            "predicted_home_goals": 2.1,
            "predicted_away_goals": 1.5,
            "home_win_probability": 0.58,  # Moderate favorite
            "draw_probability": 0.25,
            "away_win_probability": 0.17,
            "home_team_form": {
                "win_percentage": 70,
                "goals_scored": 18,
                "goals_conceded": 10
            },
            "away_team_form": {
                "win_percentage": 60,  # Good but not as strong
                "goals_scored": 16,
                "goals_conceded": 12
            },
            "head_to_head": {
                "total_matches": 6,
                "home_wins": 3,     # Balanced history
                "away_wins": 2,
                "draws": 1
            },
            "weather": {
                "temperature": 22,
                "wind_speed": 10,
                "precipitation": 0
            }
        }
        
        medium_conf = calculate_dynamic_confidence(medium_confidence_match)
        print(f"Medium confidence scenario result: {medium_conf}")
        
        # Test Case 4: Missing Data Scenario (Fallback test)
        print("\n--- Test Case 4: Missing Data Scenario ---")
        missing_data_match = {
            "fixture_id": 22222,
            "home_team_id": 30,
            "away_team_id": 35,
            "league_id": 39,
            "predicted_home_goals": 1.8,
            "predicted_away_goals": 1.3,
            "home_win_probability": 0.50,
            "draw_probability": 0.30,
            "away_win_probability": 0.20
            # Missing form, h2h, injuries, weather data
        }
        
        missing_conf = calculate_dynamic_confidence(missing_data_match)
        print(f"Missing data scenario result: {missing_conf}")
        
        # Analysis
        print("\n=== CONFIDENCE ANALYSIS ===")
        confidences = [high_conf, low_conf, medium_conf, missing_conf]
        print(f"All confidence values: {confidences}")
        
        # Check variation
        conf_range = max(confidences) - min(confidences)
        print(f"Confidence range: {conf_range:.2f}")
        
        if conf_range > 0.25:
            print("✅ EXCELLENT: Wide confidence variation achieved")
        elif conf_range > 0.15:
            print("✅ GOOD: Good confidence variation")
        elif conf_range > 0.08:
            print("⚠️ ACCEPTABLE: Some confidence variation")
        else:
            print("❌ POOR: Limited confidence variation")
        
        # Check expected ordering
        if high_conf > medium_conf > low_conf:
            print("✅ EXCELLENT: Confidence ordering matches scenario expectations")
        else:
            print("⚠️ Note: Confidence ordering doesn't fully match expectations")
            
        # Check bounds
        if all(0.35 <= c <= 0.95 for c in confidences):
            print("✅ All confidence values within expected bounds (0.35-0.95)")
        else:
            print("❌ Some confidence values out of expected bounds")
            
        print("\n=== EXPECTED CONFIDENCE LEVELS ===")
        print("• Very High (0.80+): Clear favorites with strong supporting data")
        print("• High (0.70-0.80): Good favorites with solid data")
        print("• Medium (0.55-0.70): Moderate favorites or balanced matches")
        print("• Low (0.40-0.55): Close matches with uncertainty factors")
        print("• Very Low (0.35-0.40): Highly unpredictable scenarios")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the Soccer directory")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing Enhanced Dynamic Confidence System...")
    success = test_enhanced_confidence()
    
    if success:
        print("\n🎉 ENHANCED CONFIDENCE SYSTEM TEST COMPLETED SUCCESSFULLY")
    else:
        print("\n❌ TEST FAILED")
