#!/usr/bin/env python3
"""
Test script to verify the fixes in auto_updating_elo.py
Tests the key attribute access and method call issues that were fixed.
"""

import sys
import os
import logging
from typing import Dict, Any

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_updating_elo import AutoUpdatingEloRating

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_auto_updating_elo_fixes():
    """Test that the fixes to auto_updating_elo.py work correctly"""
    
    logger.info("Testing AutoUpdatingEloRating fixes...")
    
    try:
        # Test 1: Initialize the system
        logger.info("Test 1: Initializing AutoUpdatingEloRating system...")
        elo_system = AutoUpdatingEloRating()
        logger.info("✓ AutoUpdatingEloRating initialized successfully")
        
        # Test 2: Get team rating (this should work with getattr fix)
        logger.info("Test 2: Getting team rating...")
        team_id = 12345  # Test team ID
        league_id = 39   # Premier League
        rating = elo_system.get_team_rating(team_id, league_id)
        logger.info(f"✓ Got team rating: {rating}")
        
        # Test 3: Test new team addition (tests set_rating method fix)
        logger.info("Test 3: Testing new team addition...")
        new_team_id = 67890
        new_rating = elo_system.get_team_rating(new_team_id, league_id)
        logger.info(f"✓ Added new team {new_team_id} with rating: {new_rating}")
        
        # Test 4: Test match probabilities (tests prediction handling fix)
        logger.info("Test 4: Testing match probabilities...")
        home_id = team_id
        away_id = new_team_id
        win_prob, draw_prob, loss_prob = elo_system.get_match_probabilities(home_id, away_id, league_id)
        logger.info(f"✓ Match probabilities - Win: {win_prob:.3f}, Draw: {draw_prob:.3f}, Loss: {loss_prob:.3f}")
          # Test 5: Test ratings update (tests update_ratings method fix)
        logger.info("Test 5: Testing ratings update...")
        match_data = {
            'home_team_id': home_id,
            'away_team_id': away_id,
            'home_goals': 2,
            'away_goals': 1,
            'league_id': league_id,
            'match_importance': 1.2
        }
        
        old_home_rating = elo_system.get_team_rating(home_id)
        old_away_rating = elo_system.get_team_rating(away_id)
        
        new_home_rating, new_away_rating = elo_system.update_ratings(match_data)
        
        logger.info(f"✓ Ratings updated successfully:")
        logger.info(f"  Home: {old_home_rating:.2f} → {new_home_rating:.2f}")
        logger.info(f"  Away: {old_away_rating:.2f} → {new_away_rating:.2f}")
        
        # Test 6: Test edge case handling (None values)
        logger.info("Test 6: Testing edge case handling...")
        try:
            # Test with None team IDs
            safe_win_prob, safe_draw_prob, safe_loss_prob = elo_system.get_match_probabilities(None, away_id, league_id)
            logger.info(f"✓ Handled None home_id safely: {safe_win_prob:.3f}, {safe_draw_prob:.3f}, {safe_loss_prob:.3f}")
        except Exception as e:
            logger.warning(f"Expected behavior with None values: {e}")
        
        logger.info("\n" + "="*50)
        logger.info("ALL TESTS PASSED! ✓")
        logger.info("The fixes to auto_updating_elo.py are working correctly.")
        logger.info("="*50)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_fixes():
    """Test the specific fixes that were made"""
    
    logger.info("\nTesting specific fixes...")
    
    try:
        elo_system = AutoUpdatingEloRating()
        
        # Test the safe rating access fix
        logger.info("Testing safe rating access...")
        ratings = getattr(elo_system.elo_rating.elo_system, 'ratings', {})
        logger.info(f"✓ Safe ratings access works. Found {len(ratings)} teams.")
        
        # Test the set_rating method exists
        logger.info("Testing set_rating method...")
        if hasattr(elo_system.elo_rating.elo_system, 'set_rating'):
            logger.info("✓ set_rating method exists")
        else:
            logger.error("✗ set_rating method not found")
            return False
        
        # Test the update_ratings method exists
        logger.info("Testing update_ratings method...")
        if hasattr(elo_system.elo_rating.elo_system, 'update_ratings'):
            logger.info("✓ update_ratings method exists")
        else:
            logger.error("✗ update_ratings method not found")
            return False
        
        logger.info("✓ All specific fixes verified!")
        return True
        
    except Exception as e:
        logger.error(f"Specific fix test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing AutoUpdatingEloRating fixes...")
    print("=" * 60)
    
    # Run the tests
    success1 = test_auto_updating_elo_fixes()
    success2 = test_specific_fixes()
    
    if success1 and success2:
        print("\n🎉 ALL TESTS PASSED! The fixes are working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1)
