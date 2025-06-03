import requests
import json

# Test with different team combinations
test_cases = [
    {'home_team_id': 33, 'away_team_id': 34},  
    {'home_team_id': 40, 'away_team_id': 49},   
    {'home_team_id': 47, 'away_team_id': 35},  
]

for i, team_combo in enumerate(test_cases, 1):
    print(f'Test Case {i}: Team {team_combo["home_team_id"]} vs Team {team_combo["away_team_id"]}')
    response = requests.post('http://127.0.0.1:5000/api/predict', json=team_combo)
    
    if response.status_code == 200:
        result = response.json()
        probs = result.get('probabilities', {})
        pred = result.get('prediction', {})
        
        print(f'  Prediction: {pred.get("outcome", "Unknown")} ({pred.get("confidence", 0)}% confidence)')
        print(f'  Probabilities: Home {probs.get("home_win", 0)}%, Draw {probs.get("draw", 0)}%, Away {probs.get("away_win", 0)}%')
        
        # Verify probabilities sum to ~100%
        total = probs.get('home_win', 0) + probs.get('draw', 0) + probs.get('away_win', 0)
        print(f'  Total probability: {total}%')
        print()
    else:
        print(f'  Error: {response.status_code} - {response.text}')
