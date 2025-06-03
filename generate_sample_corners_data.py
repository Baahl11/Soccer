"""
Generate sample corner data for testing the corner prediction model.
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from pathlib import Path

# Ensure data directory exists
DATA_DIR = "data"
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)

def generate_sample_corners_data(num_samples=500):
    """Generate synthetic corner kick data for model training and testing."""
    
    # Team IDs (fictional)
    team_ids = list(range(1001, 1041))  # 40 teams
    league_ids = [39, 140, 135, 78, 61]  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    
    # Generate fixture IDs
    fixture_ids = list(range(10001, 10001 + num_samples))
    
    # Base date
    base_date = datetime.now() - timedelta(days=180)
    
    # Generate data
    data = []
    
    for i in range(num_samples):
        # Select random teams
        home_team_id = np.random.choice(team_ids)
        away_team_id = np.random.choice([t for t in team_ids if t != home_team_id])
        
        # Select random league
        league_id = np.random.choice(league_ids)
        
        # Generate date
        match_date = base_date + timedelta(days=np.random.randint(0, 180))
        
        # Team form scores and corner stats (realistic distributions)
        home_form_score = np.random.normal(60, 15)  # Form score centered around 60
        away_form_score = np.random.normal(50, 15)  # Away teams slightly lower form
        
        # Corner averages
        home_avg_corners_for = np.random.normal(5.8, 1.2)
        home_avg_corners_against = np.random.normal(4.2, 1.0)
        away_avg_corners_for = np.random.normal(4.5, 1.1)
        away_avg_corners_against = np.random.normal(5.5, 1.2)
        
        # Create statistics for the match
        home_stats = {
            'ball_possession': np.random.normal(55, 8),
            'total_shots': np.random.normal(14, 4),
            'shots_on_target': np.random.normal(5, 2),
            'shots_off_target': np.random.normal(9, 3),
        }
        
        away_stats = {
            'ball_possession': 100 - home_stats['ball_possession'],
            'total_shots': np.random.normal(11, 4),
            'shots_on_target': np.random.normal(4, 2),
            'shots_off_target': np.random.normal(7, 3),
        }
        
        # Generate actual corner counts (with correlation to form and averages)
        base_home_corners = 0.7 * home_avg_corners_for + 0.3 * away_avg_corners_against
        base_away_corners = 0.7 * away_avg_corners_for + 0.3 * home_avg_corners_against
        
        # Add form effect
        form_effect = (home_form_score - away_form_score) / 100
        
        # Add possession and shots effect
        stats_effect_home = (home_stats['total_shots'] - away_stats['total_shots']) / 20
        
        # Add randomness
        home_corners = max(0, int(np.random.normal(base_home_corners + form_effect + stats_effect_home, 2)))
        away_corners = max(0, int(np.random.normal(base_away_corners - form_effect - stats_effect_home, 2)))
        
        # Generate match scores
        home_goals = max(0, int(np.random.normal(1.5 + form_effect * 0.5, 1.2)))
        away_goals = max(0, int(np.random.normal(1.2 - form_effect * 0.5, 1.1)))
        
        # Create data record
        data_record = {
            'fixture_id': fixture_ids[i],
            'league_id': league_id,
            'season': '2024',
            'date': match_date.strftime('%Y-%m-%d'),
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team_name': f'Team-{home_team_id}',
            'away_team_name': f'Team-{away_team_id}',
            'home_score': home_goals,
            'away_score': away_goals,
            'total_corners': home_corners + away_corners,
            'home_corners': home_corners,
            'away_corners': away_corners,
            'home_form_score': home_form_score,
            'away_form_score': away_form_score,
            'home_avg_corners_for': home_avg_corners_for,
            'home_avg_corners_against': home_avg_corners_against,
            'away_avg_corners_for': away_avg_corners_for,
            'away_avg_corners_against': away_avg_corners_against,
            'home_ball_possession': home_stats['ball_possession'],
            'away_ball_possession': away_stats['ball_possession'],
            'home_total_shots': home_stats['total_shots'],
            'away_total_shots': away_stats['total_shots'],
            'home_shots_on_target': home_stats['shots_on_target'],
            'away_shots_on_target': away_stats['shots_on_target'],
            'home_shots_off_target': home_stats['shots_off_target'],
            'away_shots_off_target': away_stats['shots_off_target'],
        }
        
        data.append(data_record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some additional features
    df['is_derby'] = np.random.choice([0, 1], size=len(df), p=[0.9, 0.1])
    df['weather_condition'] = np.random.choice(['Clear', 'Rain', 'Snow'], size=len(df), p=[0.7, 0.25, 0.05])
    
    return df

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate sample corner kick data for model training')
    parser.add_argument('--samples', type=int, default=1000, help='Number of sample matches to generate')
    args = parser.parse_args()
    
    # Generate sample data
    print(f"Generating {args.samples} sample corner kick records...")
    df = generate_sample_corners_data(num_samples=args.samples)
    
    # Save to CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(DATA_DIR, f'corners_training_data_SAMPLE_{timestamp}.csv')
    df.to_csv(output_path, index=False)
    
    print(f"Sample data saved to {output_path}")
    print(f"Generated {len(df)} records")
    print(f"Average corners per match: {df['total_corners'].mean():.2f}")
    print(f"Home team average corners: {df['home_corners'].mean():.2f}")
    print(f"Away team average corners: {df['away_corners'].mean():.2f}")
