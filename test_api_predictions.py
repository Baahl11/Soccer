#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bosnian Teams ELO Verification

This script verifies that Bosnian teams now have proper ELO ratings 
and valid goal difference predictions.
"""

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_test")

def test_prediction_with_bosnian_teams():
    """Simulate API prediction with Bosnian teams"""
    # Import prediction function used by API
    from predictions import make_global_prediction
    
    # Function takes fixture ID, but internally extracts team IDs
    # For testing, we'll mock that part by directly using our auto-updating ELO
    
    from auto_updating_elo import get_elo_data_with_auto_rating
    
    # Bosnian teams: Tuzla City vs Zvijezda Gradačac
    home_team_id = 3361  # Tuzla City
    away_team_id = 3382  # Zvijezda Gradačac
    league_id = 392      # Bosnian Premier League
    
    # Get ELO data including expected goal difference
    elo_data = get_elo_data_with_auto_rating(home_team_id, away_team_id, league_id)
    
    logger.info(f"ELO data for Tuzla City vs Zvijezda Gradačac:")
    logger.info(f"Home ELO: {elo_data['home_elo']}")
    logger.info(f"Away ELO: {elo_data['away_elo']}")
    logger.info(f"ELO difference: {elo_data['elo_diff']}")
    logger.info(f"Expected goal difference: {elo_data['elo_expected_goal_diff']}")
    
    # Verify expected goal difference is not null
    if elo_data['elo_expected_goal_diff'] is not None:
        logger.info("✓ Expected goal difference is properly calculated")
    else:
        logger.error("✗ Expected goal difference is NULL!")
    
    # Test with other Bosnian teams
    logger.info("\nTesting additional Bosnian team combinations:")
    
    team_pairs = [
        (3380, 3363),  # TOŠK Tešanj vs Čelik
        (3373, 3368),  # FK Sarajevo vs FK Željezničar
        (3372, 3366)   # HSK Zrinjski vs FK Borac Banja Luka
    ]
    
    for home_id, away_id in team_pairs:
        # Get ELO data
        pair_data = get_elo_data_with_auto_rating(home_id, away_id, league_id)
        
        logger.info(f"Match {home_id} vs {away_id}:")
        logger.info(f"  Home ELO: {pair_data['home_elo']}")
        logger.info(f"  Away ELO: {pair_data['away_elo']}")
        logger.info(f"  Expected goal diff: {pair_data['elo_expected_goal_diff']}")
        
        # Verify expected goal difference is not null
        if pair_data['elo_expected_goal_diff'] is not None:
            logger.info("  ✓ Expected goal difference is properly calculated")
        else:
            logger.error("  ✗ Expected goal difference is NULL!")

if __name__ == "__main__":
    logger.info("Starting API prediction test with Bosnian teams")
    test_prediction_with_bosnian_teams()
    logger.info("Test completed")
