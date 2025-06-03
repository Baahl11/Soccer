"""
Test script to verify that the advanced features implementation is working correctly
"""
import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
import logging

from advanced_features import MomentumFeatures, TeamStrengthIndex

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Create sample match data
    logger.info("Generating sample match data...")
    matches = generate_sample_matches()
    
    # Initialize the momentum feature calculator
    momentum_calculator = MomentumFeatures()
    
    # Test momentum features
    team_id = 1  # Use team 1 for our test
    match_date = '2025-04-01'
    
    logger.info(f"Calculating momentum features for team {team_id}...")
    momentum_features = momentum_calculator.calculate_momentum_features(
        matches, team_id, match_date, is_home=True
    )
    
    # Display results
    logger.info("Momentum Features:")
    for feature, value in sorted(momentum_features.items()):
        logger.info(f"  {feature}: {value:.4f}")
    
    # Test team strength index
    strength_calculator = TeamStrengthIndex()
    logger.info(f"Calculating strength indices for team {team_id}...")
    
    # Create sample team data
    team_data = {
        'team_id': team_id,
        'league_position': 4,
        'league_size': 20,
        'trophy_score': 5,
        'league_prestige': 0.8,
        'shots_on_target_ratio': 0.45,
        'shot_conversion': 0.12,
        'defensive_actions': 42,
        'tactical_data': {
            'pressing_success': 0.7,
            'possession_control': 0.65,
            'buildup_success': 0.6,
            'tactical_flexibility': 0.55
        }
    }
    
    strength_indices = strength_calculator.calculate_team_strength(team_data, matches)
    
    # Display results
    logger.info("Team Strength Indices:")
    for index, value in sorted(strength_indices.items()):
        logger.info(f"  {index}: {value:.4f}")
    
    logger.info("Test completed successfully!")

def generate_sample_matches(num_matches=20):
    """Generate sample match data"""
    matches = []
    team_ids = list(range(1, 6))  # 5 teams
    
    start_date = datetime(2025, 1, 1)
    
    for i in range(num_matches):
        # Select random teams
        home_team = np.random.choice(team_ids)
        away_team = np.random.choice([t for t in team_ids if t != home_team])
        
        # Generate random scores with some bias for home team
        home_goals = np.random.poisson(1.5)
        away_goals = np.random.poisson(1.2)
        
        # Generate date
        match_date = (start_date + timedelta(days=i*3)).strftime('%Y-%m-%d')
        
        # Create match data
        match = {
            'match_id': i+1,
            'date': match_date,
            'home_team_id': int(home_team),
            'away_team_id': int(away_team),
            'home_goals': int(home_goals),
            'away_goals': int(away_goals),
            'home_xg': float(np.random.normal(1.4, 0.3)),
            'away_xg': float(np.random.normal(1.2, 0.3))
        }
        matches.append(match)
    
    return matches

if __name__ == "__main__":
    main()
