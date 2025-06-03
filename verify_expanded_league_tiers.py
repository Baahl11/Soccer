#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
League Tier System Verification

This script verifies that the expanded league tier system is working correctly
by checking teams from different leagues across the tiers.
"""

import logging
import sys
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("league_tier_test")

# Import the auto-updating ELO system
from auto_updating_elo import AutoUpdatingEloRating, get_elo_data_with_auto_rating

def log(message):
    """Log message and flush stdout"""
    print(message)
    sys.stdout.flush()

# Initialize the auto-updating ELO system
auto_elo = AutoUpdatingEloRating()

# Test data: League ID -> [(team_id, team_name), ...]
test_leagues = {
    # Top tier leagues
    39: [(33, "Manchester United"), (50, "Manchester City")],  # Premier League
    140: [(529, "Barcelona"), (541, "Real Madrid")],  # La Liga
    
    # Second tier leagues
    88: [(194, "Ajax"), (197, "PSV")],  # Eredivisie
    203: [(3205, "Galatasaray"), (3215, "Fenerbahçe")],  # Super Lig
    
    # Third tier leagues
    103: [(2601, "Red Bull Salzburg"), (2605, "Rapid Wien")],  # Austrian Bundesliga
    179: [(247, "Celtic"), (249, "Rangers")],  # Scottish Premiership
    
    # Fourth tier leagues
    392: [(3361, "Tuzla City"), (3368, "FK Željezničar")],  # Premier League Bosnia
    265: [(3504, "WIT Georgia"), (14861, "Gori")],  # Erovnuli Liga (Georgia)
    
    # Other leagues
    253: [(7177, "Los Angeles FC"), (7173, "Toronto FC")],  # MLS
    322: [(8341, "Vissel Kobe"), (8342, "Kashima Antlers")]  # J1 League
}

def test_league_tiers():
    """Test the league tier system with teams from different leagues"""
    log("\nLeague Tier System Verification")
    log("=============================\n")

    # Test classification of leagues
    log("League Tier Classification:")
    log("-------------------------")
      # Dictionary mapping league IDs to their names
    league_names = {
        39: "Premier League (England)",
        140: "La Liga (Spain)",
        135: "Serie A (Italy)", 
        78: "Bundesliga (Germany)",
        61: "Ligue 1 (France)",
        88: "Eredivisie (Netherlands)",
        94: "Primeira Liga (Portugal)",
        203: "Super Lig (Turkey)",
        119: "Danish Superliga (Denmark)",
        106: "Ekstraklasa (Poland)",
        392: "Premier League Bosnia (Bosnia)",
        265: "Erovnuli Liga (Georgia)",
        271: "Israeli Premier League (Israel)",
        290: "South African National First Division",
        329: "Meistriliiga (Estonia)",
        103: "Austrian Bundesliga (Austria)",
        179: "Scottish Premiership (Scotland)",
        253: "MLS (USA & Canada)",
        322: "J1 League (Japan)"
    }

    for league_id in sorted(test_leagues.keys()):
        tier = auto_elo.get_league_tier(league_id)
        league_str = league_names.get(league_id, f"League {league_id}")
        
        log(f"League ID {league_id} ({league_str}): {tier} tier - Default ELO: {auto_elo.league_tiers.get(league_id)}")

    # Test default team ratings for each league
    log("\nDefault Team Rating by League:")
    log("----------------------------")
    
    # Dictionary to collect average default ratings by league tier
    tier_ratings = {"Top": [], "Second": [], "Third": [], "Fourth": [], "Unknown": []}
    
    for league_id, teams in test_leagues.items():
        league_tier = auto_elo.get_league_tier(league_id)
        log(f"\nLeague ID {league_id} ({league_tier} tier):")
        
        # Generate a test team ID that doesn't exist
        test_team_id = 99000 + league_id
        
        # Get rating for this test team in this league context
        default_rating = auto_elo._calculate_smart_default_rating(test_team_id, league_id)
        tier_ratings[league_tier].append(default_rating)
        
        log(f"  Default rating for new team: {default_rating:.1f}")
        log(f"  League base rating: {auto_elo.league_tiers.get(league_id, auto_elo.default_league_elo)}")
        
        # Test a real match from this league
        if len(teams) >= 2:
            home_team, away_team = teams[0], teams[1]
            log(f"  Test match: {home_team[1]} vs {away_team[1]}")
            
            # Get match data
            match_data = get_elo_data_with_auto_rating(home_team[0], away_team[0], league_id)
            log(f"  Home ELO: {match_data['home_elo']}")
            log(f"  Away ELO: {match_data['away_elo']}")
            log(f"  Expected goal diff: {match_data['elo_expected_goal_diff']}")
            
    # Summarize tier average ratings
    log("\nAverage Default Ratings by Tier:")
    log("-----------------------------")
    for tier, ratings in tier_ratings.items():
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            log(f"{tier} tier average: {avg_rating:.1f}")

if __name__ == "__main__":
    test_league_tiers()
