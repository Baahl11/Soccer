#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Test the auto-updating ELO system directly."""

print("Starting ELO test...")

import os
import json

# Print current teams in ratings file
ratings_file = 'data/team_elo_ratings.json'
if os.path.exists(ratings_file):
    with open(ratings_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print(f"Current teams in database: {len(data.get('ratings', {}))}")
        
        # Check if Bosnian teams exist
        bosnian_ids = [3361, 3382, 3380, 3363]
        for team_id in bosnian_ids:
            team_id_str = str(team_id)
            if team_id_str in data.get('ratings', {}):
                print(f"Team {team_id} already exists with rating: {data['ratings'][team_id_str]}")
            else:
                print(f"Team {team_id} not found in database")
else:
    print(f"Ratings file not found: {ratings_file}")

# Import our auto-updating system
from auto_updating_elo import get_elo_data_with_auto_rating

# Test with Bosnian teams (Tuzla City vs Zvijezda Gradačac)
print("\nTesting Tuzla City vs Zvijezda Gradačac:")
elo_data = get_elo_data_with_auto_rating(3361, 3382, 392)
print(f"Home ELO: {elo_data['home_elo']}")
print(f"Away ELO: {elo_data['away_elo']}")
print(f"ELO difference: {elo_data['elo_diff']}")
print(f"Win probability: {elo_data['elo_win_probability']}")
print(f"Draw probability: {elo_data['elo_draw_probability']}")
print(f"Loss probability: {elo_data['elo_loss_probability']}")
print(f"Expected goal difference: {elo_data['elo_expected_goal_diff']}")

print("\nTest completed successfully!")
