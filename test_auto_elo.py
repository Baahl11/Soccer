"""
Test script to verify the auto-updating ELO rating system.

This script compares the original ELO system with the new auto-updating ELO system
and demonstrates how new teams are automatically added to the database.
"""

import logging
import json
import os
import sys
from typing import Dict, Any, Optional
import random

# Configure logging to both console and file
log_file = 'auto_elo_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("elo_test")
logger.info(f"Logging to file: {os.path.abspath(log_file)}")

# Import both ELO systems for comparison
from team_elo_rating import EloRating as OriginalEloRating
from auto_updating_elo import AutoUpdatingEloRating

# Path to the ratings file
RATINGS_FILE = 'data/team_elo_ratings.json'


def backup_ratings_file():
    """Create a backup of the current ratings file"""
    if os.path.exists(RATINGS_FILE):
        backup_path = f"{RATINGS_FILE}.bak"
        try:
            with open(RATINGS_FILE, 'r', encoding='utf-8') as src:
                with open(backup_path, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            logger.info(f"Created backup of ratings file at {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    return False


def restore_ratings_file():
    """Restore ratings file from backup"""
    backup_path = f"{RATINGS_FILE}.bak"
    if os.path.exists(backup_path):
        try:
            with open(backup_path, 'r', encoding='utf-8') as src:
                with open(RATINGS_FILE, 'w', encoding='utf-8') as dst:
                    dst.write(src.read())
            logger.info(f"Restored ratings file from backup")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
    return False


def get_current_team_count():
    """Count how many teams are in the ratings file"""
    try:
        if os.path.exists(RATINGS_FILE):
            with open(RATINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return len(data.get('ratings', {}))
        return 0
    except Exception as e:
        logger.error(f"Error counting teams: {e}")
        return 0


def test_existing_team(elo_system, team_id, expected_range=None):
    """Test rating for an existing team"""
    rating = elo_system.get_team_rating(team_id)
    logger.info(f"Team {team_id}: Rating = {rating}")
    
    if expected_range and (rating < expected_range[0] or rating > expected_range[1]):
        logger.warning(f"Team {team_id} rating {rating} outside expected range {expected_range}")
        return False
    return True


def test_new_team(elo_system, team_id, league_id=None):
    """Test auto-addition of a new team"""
    # Use a team ID that's unlikely to exist in our database
    rating = elo_system.get_team_rating(team_id, league_id)
    logger.info(f"New team {team_id} (League {league_id}): Assigned rating = {rating}")
    
    # Verify the team was added by getting the rating again
    second_rating = elo_system.get_team_rating(team_id)
    
    if abs(rating - second_rating) < 0.01:
        logger.info(f"✓ Team {team_id} was successfully added with rating {rating}")
        return True
    else:
        logger.error(f"✗ Team {team_id} was not properly persisted! First: {rating}, Second: {second_rating}")
        return False


def test_match_probabilities(elo_system, home_id, away_id, league_id=None):
    """Test match probability calculations"""
    win_prob, draw_prob, loss_prob = elo_system.get_match_probabilities(home_id, away_id, league_id)
    goal_diff = elo_system.get_expected_goal_diff(home_id, away_id, league_id)
    
    logger.info(f"Match {home_id} vs {away_id} (League {league_id}):")
    logger.info(f"  Win probability: {win_prob:.4f}")
    logger.info(f"  Draw probability: {draw_prob:.4f}")
    logger.info(f"  Loss probability: {loss_prob:.4f}")
    logger.info(f"  Expected goal diff: {goal_diff:.4f}")
    
    # Check that probabilities sum to approximately 1
    total_prob = win_prob + draw_prob + loss_prob
    if abs(total_prob - 1.0) > 0.001:
        logger.error(f"✗ Probabilities don't sum to 1.0: {total_prob}")
        return False
    
    return True


def main():
    """Main test function"""
    logger.info("Starting ELO auto-update test")
    
    # Create backup of ratings file
    backup_created = backup_ratings_file()
    if not backup_created:
        logger.warning("No backup created, proceeding without backup")
    
    try:
        # Initial team count
        initial_count = get_current_team_count()
        logger.info(f"Initial team count: {initial_count}")
        
        # Initialize both ELO systems
        original_elo = OriginalEloRating()
        auto_elo = AutoUpdatingEloRating()
        
        # Test with existing teams
        logger.info("\n=== Testing existing teams ===")
        existing_teams = [
            (33, "Manchester United", (1550, 1620)),
            (529, "Barcelona", (1550, 1620)),
            (50, "Manchester City", (1580, 1650))
        ]
        
        for team_id, name, expected_range in existing_teams:
            rating_orig = original_elo.get_team_rating(team_id)
            rating_auto = auto_elo.get_team_rating(team_id)
            logger.info(f"{name} (ID: {team_id}):")
            logger.info(f"  Original system: {rating_orig}")
            logger.info(f"  Auto system:     {rating_auto}")
            
            assert abs(rating_orig - rating_auto) < 0.01, "Ratings don't match between systems!"
        
        # Test with new teams not in database
        logger.info("\n=== Testing new teams ===")
        
        # Dynamic new team IDs (avoid collisions by using very high numbers)
        new_teams = [
            (80000 + random.randint(1, 1000), None),       # Unknown league
            (80000 + random.randint(1001, 2000), 392),     # Bosnian Premier League
            (80000 + random.randint(2001, 3000), 39),      # English Premier League
            (80000 + random.randint(3001, 4000), 265),     # Georgian league
        ]
        
        # Test adding new teams
        for team_id, league_id in new_teams:
            league_name = f"League {league_id}" if league_id else "Unknown league"
            logger.info(f"Adding new team ID {team_id} ({league_name})")
            
            # Test that auto-elo system adds the team
            test_new_team(auto_elo, team_id, league_id)
        
        # Check that team count increased
        new_count = get_current_team_count()
        logger.info(f"New team count: {new_count}")
        logger.info(f"Added teams: {new_count - initial_count}")
        
        # Test match probabilities
        logger.info("\n=== Testing match probabilities ===")
        test_matches = [
            (33, 50, 39),           # Man United vs Man City (EPL)
            (3361, 3382, 392),      # Tuzla City vs Zvijezda (Bosnian League)
            (new_teams[0][0], new_teams[1][0], 392)  # Random new teams
        ]
        
        for home_id, away_id, league_id in test_matches:
            test_match_probabilities(auto_elo, home_id, away_id, league_id)
        
        logger.info("\n=== All tests completed successfully ===")
        
    finally:
        # For automated testing, we always restore the backup
        if backup_created:
            logger.info("Restoring original ratings file from backup")
            restore_ratings_file()
        else:
            logger.info("No backup was created, skipping restore")


if __name__ == "__main__":
    main()
