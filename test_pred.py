from voting_ensemble_corners import predict_corners_with_voting_ensemble
import json

# Test data
home_team_id = 123
away_team_id = 456
league_id = 39  # Premier League

home_stats = {
    'avg_corners_for': 6.2,
    'avg_corners_against': 4.5,
    'form_score': 65,
    'attack_strength': 1.1,
    'defense_strength': 0.95,
    'avg_shots': 14.2
}

away_stats = {
    'avg_corners_for': 5.1,
    'avg_corners_against': 5.8,
    'form_score': 48,
    'attack_strength': 0.9,
    'defense_strength': 1.05,
    'avg_shots': 10.5
}

# Make prediction
result = predict_corners_with_voting_ensemble(
    home_team_id, 
    away_team_id, 
    home_stats, 
    away_stats, 
    league_id
)

# Print result as JSON
print(json.dumps(result, indent=2))
