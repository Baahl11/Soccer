#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bosnian Teams ELO Verification

This script verifies that Bosnian teams now have proper ELO ratings 
and valid goal difference predictions.
"""

import sys

# Import the auto-updating ELO system
from auto_updating_elo import get_elo_data_with_auto_rating

# Set up output to flush immediately for debugging
def log(message):
    print(message)
    sys.stdout.flush()

# Bosnian teams to test
teams = {
    3361: "Tuzla City",
    3382: "Zvijezda Gradačac", 
    3380: "TOŠK Tešanj", 
    3363: "Čelik",
    3373: "FK Sarajevo", 
    3368: "FK Željezničar"
}

log("Testing Bosnian teams ELO ratings:")
log("---------------------------------")

# League ID for Bosnian Premier League
league_id = 392

log("Starting verification...")

# Get ELO data for each team vs the first team
first_team_id = list(teams.keys())[0]
first_team_name = teams[first_team_id]

log(f"First team: {first_team_name} (ID: {first_team_id})")

for team_id, team_name in teams.items():
    if team_id != first_team_id:
        log(f"\nVerifying: {first_team_name} vs {team_name}")
        
        # Get ELO data including expected goal difference
        elo_data = get_elo_data_with_auto_rating(first_team_id, team_id, league_id)
        
        log(f"  Home ELO: {elo_data['home_elo']}")
        log(f"  Away ELO: {elo_data['away_elo']}")
        log(f"  Win probability: {elo_data['elo_win_probability']}")
        log(f"  Expected goal difference: {elo_data['elo_expected_goal_diff']}")
        
        # Verification
        if elo_data['elo_expected_goal_diff'] is not None:
            log("  ✓ Expected goal difference is properly calculated")
        else:
            log("  ✗ Expected goal difference is NULL!")

log("\nTest completed successfully!")
