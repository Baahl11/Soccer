"""
Script to add missing Bosnian teams to the ELO ratings file.
This will fix the issue where Bosnian teams have default ELO ratings of 1500.
"""
import json
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to ELO ratings file
ELO_RATINGS_FILE = 'data/team_elo_ratings.json'

def load_ratings():
    """Load current ELO ratings"""
    try:
        if os.path.exists(ELO_RATINGS_FILE):
            with open(ELO_RATINGS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            logger.error(f"Ratings file not found: {ELO_RATINGS_FILE}")
            return {"ratings": {}, "last_updated": datetime.now().strftime("%Y-%m-%d")}
    except Exception as e:
        logger.error(f"Error loading ratings: {e}")
        return {"ratings": {}, "last_updated": datetime.now().strftime("%Y-%m-%d")}

def save_ratings(ratings_data):
    """Save ratings to file"""
    try:
        os.makedirs(os.path.dirname(ELO_RATINGS_FILE), exist_ok=True)
        with open(ELO_RATINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(ratings_data, f, indent=2)
        logger.info(f"Ratings saved successfully to {ELO_RATINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving ratings: {e}")
        return False

def add_bosnian_teams():
    """Add Bosnian teams to ELO ratings"""
    try:
        # Team IDs from the prediction data
        bosnian_teams = {
            3361: {"name": "Tuzla City", "rating": 1550.0},
            3382: {"name": "Zvijezda Gradačac", "rating": 1465.0},
            3380: {"name": "TOŠK Tešanj", "rating": 1490.0},
            3363: {"name": "Čelik", "rating": 1525.0},
            # Add more Bosnian teams if needed
            3373: {"name": "FK Sarajevo", "rating": 1540.0},
            3368: {"name": "FK Željezničar", "rating": 1535.0},
            3372: {"name": "HSK Zrinjski", "rating": 1545.0},
            3366: {"name": "FK Borac Banja Luka", "rating": 1530.0},
            3379: {"name": "Sloboda", "rating": 1485.0},
            3369: {"name": "Velež", "rating": 1495.0},
        }
        
        # Load current ratings
        ratings_data = load_ratings()
        
        # Add new teams or update existing ones
        for team_id, team_info in bosnian_teams.items():
            team_id_str = str(team_id)
            if team_id_str not in ratings_data["ratings"]:
                ratings_data["ratings"][team_id_str] = team_info["rating"]
                logger.info(f"Added team: {team_info['name']} (ID: {team_id}) with rating {team_info['rating']}")
            else:
                # If team exists but has default rating of 1500, update it
                current_rating = ratings_data["ratings"][team_id_str]
                if current_rating == 1500.0:
                    ratings_data["ratings"][team_id_str] = team_info["rating"]
                    logger.info(f"Updated team: {team_info['name']} (ID: {team_id}) from 1500.0 to {team_info['rating']}")
        
        # Update last_updated timestamp
        ratings_data["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Save updated ratings
        save_ratings(ratings_data)
        
        logger.info("Bosnian teams added/updated successfully")
        return True
    except Exception as e:
        logger.error(f"Error adding Bosnian teams: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting Bosnian teams ELO ratings update...")
    success = add_bosnian_teams()
    if success:
        logger.info("Bosnian teams update completed successfully")
    else:
        logger.error("Failed to update Bosnian teams")
