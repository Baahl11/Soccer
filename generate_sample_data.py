"""
Generate sample soccer match data for testing time series cross-validation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Create the data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Number of matches to generate
n_matches = 300

# Start date for the matches
start_date = datetime(2023, 1, 1)

# Generate dates
dates = [start_date + timedelta(days=i) for i in range(n_matches)]
dates_str = [d.strftime('%Y-%m-%d') for d in dates]

# Teams
teams = ["Manchester United", "Manchester City", "Liverpool", "Chelsea", 
         "Arsenal", "Tottenham", "Leicester", "West Ham", "Everton", "Newcastle",
         "Barcelona", "Real Madrid", "Atletico Madrid", "Sevilla", "Valencia",
         "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "PSG", "Juventus"]

# Generate match data
data = []
for i in range(n_matches):
    # Select random home and away teams
    all_teams = teams.copy()
    home_team = np.random.choice(all_teams)
    all_teams.remove(home_team)
    away_team = np.random.choice(all_teams)
    
    # Generate team stats
    home_possession = np.random.uniform(30, 70)
    away_possession = 100 - home_possession
    
    home_shots = np.random.randint(5, 25)
    away_shots = np.random.randint(5, 25)
    
    home_shots_on_target = np.random.randint(1, home_shots+1)
    away_shots_on_target = np.random.randint(1, away_shots+1)
    
    home_corners = np.random.randint(1, 15)
    away_corners = np.random.randint(1, 15)
    
    home_fouls = np.random.randint(5, 20)
    away_fouls = np.random.randint(5, 20)
    
    home_yellow_cards = np.random.randint(0, 5)
    away_yellow_cards = np.random.randint(0, 5)
    
    home_red_cards = np.random.randint(0, 2)
    away_red_cards = np.random.randint(0, 2)
    
    # Generate goals based on shots on target
    home_goals = np.random.randint(0, min(home_shots_on_target+1, 6))
    away_goals = np.random.randint(0, min(away_shots_on_target+1, 6))
    
    # Result (0 = away win, 1 = draw, 2 = home win)
    if home_goals > away_goals:
        result = 2
    elif home_goals < away_goals:
        result = 0
    else:
        result = 1
    
    # Add match to data
    match = {
        'date': dates_str[i],
        'league_id': np.random.choice([39, 140, 135, 78, 61]),
        'season': 2023,
        'home_team': home_team,
        'away_team': away_team,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'result': result,
        'home_possession': home_possession,
        'away_possession': away_possession,
        'home_shots': home_shots,
        'away_shots': away_shots,
        'home_shots_on_target': home_shots_on_target,
        'away_shots_on_target': away_shots_on_target,
        'home_corners': home_corners,
        'away_corners': away_corners,
        'home_fouls': home_fouls,
        'away_fouls': away_fouls,
        'home_yellow_cards': home_yellow_cards,
        'away_yellow_cards': away_yellow_cards,
        'home_red_cards': home_red_cards,
        'away_red_cards': away_red_cards,
        'home_form': np.random.uniform(0, 1),
        'away_form': np.random.uniform(0, 1),
        'home_ranking': np.random.randint(1, 21),
        'away_ranking': np.random.randint(1, 21),
    }
    data.append(match)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("data/sample_matches.csv", index=False)
print(f"Generated {n_matches} sample matches and saved to data/sample_matches.csv")
