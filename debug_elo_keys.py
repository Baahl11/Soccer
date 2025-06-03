#!/usr/bin/env python3
"""
Quick debug script to check what keys the ELO system actually returns.
"""

from team_elo_rating import get_elo_ratings_for_match

# Test with sample team IDs
home_team_id = 1  # Real Madrid
away_team_id = 2  # Barcelona  
league_id = 1     # La Liga

try:
    elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
    print("ELO data returned:")
    for key, value in elo_data.items():
        print(f"  '{key}': {value}")
    
    print("\nLooking for specific keys:")
    print(f"  'expected_goal_diff': {'expected_goal_diff' in elo_data}")
    print(f"  'elo_expected_goal_diff': {'elo_expected_goal_diff' in elo_data}")
    print(f"  'win_probability': {'win_probability' in elo_data}")
    print(f"  'elo_win_probability': {'elo_win_probability' in elo_data}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
