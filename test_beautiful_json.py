import requests
import json

# Test the formatted endpoint
response = requests.post('http://127.0.0.1:5000/api/predict/formatted', 
                        json={'home_team_id': 33, 'away_team_id': 34})

if response.status_code == 200:
    result = response.json()
    print("🏆 BEAUTIFUL JSON PREDICTION RESULT:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print(f"Error: {response.status_code} - {response.text}")
