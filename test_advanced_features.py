import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Any

from advanced_features import MomentumFeatures, TeamStrengthIndex, add_advanced_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_sample_matches(num_teams=10, matches_per_team=20):
    """Generate sample match data for demonstration"""
    teams = list(range(1, num_teams + 1))
    start_date = datetime(2024, 8, 15)  # Season start
    
    matches = []
    match_id = 1
    
    for team1 in teams:
        for team2 in teams:
            if team1 == team2:
                continue
                
            # Generate home and away matches
            for match_num in range(matches_per_team // (num_teams - 1)):
                match_date = start_date + timedelta(days=(match_id % 7))
                
                # Team strength influences score probability
                team1_strength = 0.4 + (team1 / num_teams) * 0.6
                team2_strength = 0.4 + (team2 / num_teams) * 0.6
                
                # Generate scores with some randomness but influenced by team strength
                # Higher index teams are stronger on average
                home_goals_prob = np.random.poisson(2 * team1_strength)
                away_goals_prob = np.random.poisson(1.5 * team2_strength)
                
                # Generate xG values (expected goals)
                home_xg = max(0.3, min(3.5, home_goals_prob * 0.9 + np.random.normal(0, 0.3)))
                away_xg = max(0.3, min(3.5, away_goals_prob * 0.9 + np.random.normal(0, 0.3)))
                
                # Sometimes actual goals differ from xG
                home_goals = np.random.poisson(home_xg)
                away_goals = np.random.poisson(away_xg)
                
                # Generate other stats based on strengths
                home_possession = 40 + int((team1_strength - 0.4) * 40)
                home_shots = max(5, int(home_xg * 3 + np.random.normal(0, 2)))
                away_shots = max(5, int(away_xg * 3 + np.random.normal(0, 2)))
                home_shots_on_target = max(1, int(home_shots * (0.3 + team1_strength * 0.2)))
                away_shots_on_target = max(1, int(away_shots * (0.3 + team2_strength * 0.2)))
                
                match = {
                    'match_id': match_id,
                    'date': (match_date + timedelta(days=match_num * 7)).strftime('%Y-%m-%d'),
                    'league_id': 1,
                    'season': 2024,
                    'home_team_id': team1,
                    'away_team_id': team2,
                    'home_team': f'Team {team1}',
                    'away_team': f'Team {team2}',
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'home_xg': home_xg,
                    'away_xg': away_xg,
                    'home_possession': home_possession,
                    'away_possession': 100 - home_possession,
                    'home_shots': home_shots,
                    'away_shots': away_shots,
                    'home_shots_on_target': home_shots_on_target,
                    'away_shots_on_target': away_shots_on_target,
                    'home_corners': int(home_shots / 3) + np.random.randint(0, 3),
                    'away_corners': int(away_shots / 3) + np.random.randint(0, 3),
                }
                
                # Add result field
                if home_goals > away_goals:
                    match['result'] = 'H'  # Home win
                elif home_goals < away_goals:
                    match['result'] = 'A'  # Away win
                else:
                    match['result'] = 'D'  # Draw
                
                matches.append(match)
                match_id += 1
                
    return pd.DataFrame(matches)

def generate_team_data(df, num_teams=10):
    """Generate team metadata"""
    team_data = {}
    
    for team_id in range(1, num_teams + 1):
        # Basic team strength (higher team_id = stronger team)
        base_strength = 0.4 + (team_id / num_teams) * 0.6
        
        # Extract team matches
        team_matches = df[(df['home_team_id'] == team_id) | (df['away_team_id'] == team_id)]
        
        # Calculate some basic stats
        total_matches = len(team_matches)
        
        wins = len(team_matches[(team_matches['home_team_id'] == team_id) & 
                              (team_matches['result'] == 'H')]) + \
               len(team_matches[(team_matches['away_team_id'] == team_id) & 
                              (team_matches['result'] == 'A')])
                              
        draws = len(team_matches[team_matches['result'] == 'D'])
        
        # Points and position roughly based on team_id
        points = wins * 3 + draws
        position = num_teams + 1 - team_id
        
        team_data[team_id] = {
            'team_id': team_id,
            'team_name': f'Team {team_id}',
            'league_position': position,
            'league_size': num_teams,
            'points': points,
            'win_rate': wins / total_matches if total_matches > 0 else 0.5,
            'shot_conversion': base_strength * 0.2,
            'shots_on_target_ratio': 0.3 + base_strength * 0.2,
            'defensive_actions': 25 + int(base_strength * 10),
            'trophy_score': max(0, team_id - 5),
            'league_prestige': 0.7,
            'tactical_data': {
                'pressing_success': base_strength,
                'possession_control': base_strength,
                'buildup_success': base_strength,
                'tactical_flexibility': 0.4 + np.random.random() * 0.3
            }
        }
        
    return team_data

def test_momentum_features():
    """Test the MomentumFeatures class"""
    logger.info("Testing MomentumFeatures")
    
    # Generate sample data
    matches_df = generate_sample_matches()
    
    # Save sample data for later inspection
    data_dir = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_dir, exist_ok=True)
    matches_df.to_csv(os.path.join(data_dir, 'sample_matches.csv'), index=False)
    
    # Initialize MomentumFeatures
    momentum_calculator = MomentumFeatures()
    
    # Pick a team and date for analysis
    team_id = 8  # Strong team
    match_date = '2025-02-01'
    
    # Convert DataFrame rows to dictionaries
    matches_list = matches_df.to_dict('records')
    
    # Calculate momentum features
    home_momentum = momentum_calculator.calculate_momentum_features(
        matches_list, team_id, match_date, is_home=True
    )
    
    logger.info(f"Momentum features for Team {team_id} (home):")
    for feature, value in sorted(home_momentum.items()):
        logger.info(f"  {feature}: {value:.4f}")
    
    # Visualize momentum across different windows
    windows = momentum_calculator.windows
    win_ratios = [home_momentum[f'home_win_ratio_{w}'] for w in windows]
    form_trends = [home_momentum[f'home_form_trend_{w}'] for w in windows]
    
    plt.figure(figsize=(10, 6))
    plt.bar(windows, win_ratios, width=0.4, label='Win Ratio', alpha=0.7)
    plt.plot(windows, form_trends, 'r-o', label='Form Trend')
    plt.xlabel('Window Size')
    plt.ylabel('Value')
    plt.title(f'Momentum Metrics for Team {team_id} Across Different Windows')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.7)
    plt.savefig('momentum_windows.png')
    
    return matches_df

def test_team_strength_indices(matches_df):
    """Test the TeamStrengthIndex class"""
    logger.info("Testing TeamStrengthIndex")
    
    # Generate team data
    team_data = generate_team_data(matches_df)
    
    # Initialize TeamStrengthIndex
    strength_calculator = TeamStrengthIndex()
    
    # Calculate strength for each team
    team_strengths = {}
    
    for team_id in range(1, 11):
        # Convert DataFrame rows to dictionaries
        team_matches = matches_df[
            (matches_df['home_team_id'] == team_id) | 
            (matches_df['away_team_id'] == team_id)
        ].to_dict('records')
        
        # Add team_id to each match dictionary for easier processing
        for match in team_matches:
            match['team_id'] = team_id
        
        # Calculate strength indices
        strength_indices = strength_calculator.calculate_team_strength(
            team_data[team_id], team_matches
        )
        
        team_strengths[team_id] = strength_indices
        
        logger.info(f"Strength indices for Team {team_id}:")
        for feature, value in sorted(strength_indices.items()):
            logger.info(f"  {feature}: {value:.4f}")
    
    # Create visualization of strength indices
    teams = list(range(1, 11))
    overall_strengths = [team_strengths[t]['overall_strength'] for t in teams]
    offensive_strengths = [team_strengths[t]['offensive_strength'] for t in teams]
    defensive_strengths = [team_strengths[t]['defensive_strength'] for t in teams]
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(teams))
    width = 0.25
    
    plt.bar(x - width, overall_strengths, width, label='Overall Strength', color='blue', alpha=0.7)
    plt.bar(x, offensive_strengths, width, label='Offensive Strength', color='red', alpha=0.7)
    plt.bar(x + width, defensive_strengths, width, label='Defensive Strength', color='green', alpha=0.7)
    
    plt.xlabel('Team ID')
    plt.ylabel('Strength Index')
    plt.title('Team Strength Indices Comparison')
    plt.xticks(x, [f'Team {t}' for t in teams])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('team_strength_indices.png')
    
    return team_data, team_strengths

def test_integrated_features(matches_df, team_data):
    """Test the integration of all advanced features"""
    logger.info("Testing integrated advanced features")
    
    # Add all advanced features
    league_data = {1: {'league_avg_goals': 2.75, 'league_competitive_index': 0.8}}
    
    # Use the helper function to add all features
    enhanced_df = add_advanced_features(matches_df, team_data, league_data)
    
    # Check the new features
    momentum_cols = [col for col in enhanced_df.columns if 'momentum' in col]
    strength_cols = [col for col in enhanced_df.columns if 'strength' in col]
    
    logger.info(f"Added {len(momentum_cols)} momentum-related features")
    logger.info(f"Added {len(strength_cols)} strength-related features")
    
    # Sample correlation analysis
    corr_cols = ['home_overall_strength', 'away_overall_strength', 
               'home_momentum_index', 'away_momentum_index',
               'home_goals', 'away_goals']
    
    corr = enhanced_df[corr_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Advanced Features and Goals')
    plt.tight_layout()
    plt.savefig('feature_correlation.png')
    
    # Save the enhanced dataset
    enhanced_df.to_csv('data/enhanced_matches.csv', index=False)
    
    logger.info(f"Enhanced dataset has {enhanced_df.shape[1]} features")
    
    return enhanced_df

def run_all_tests():
    """Run all feature tests"""
    matches_df = test_momentum_features()
    team_data, team_strengths = test_team_strength_indices(matches_df)
    enhanced_df = test_integrated_features(matches_df, team_data)
    
    logger.info("All tests completed successfully!")
    
    return {
        'matches': matches_df,
        'team_data': team_data,
        'team_strengths': team_strengths,
        'enhanced_data': enhanced_df
    }

if __name__ == "__main__":
    results = run_all_tests()
