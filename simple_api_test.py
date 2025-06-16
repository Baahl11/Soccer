import requests
import json

def test_api_response():
    """Test the API response to see current team names."""
    try:
        print("🧪 Testing API Response for Team Names")
        print("=" * 50)
        
        # Make API request
        url = "http://127.0.0.1:5000/api/upcoming_predictions"
        params = {
            "limit": 3,
            "include_additional_data": "true"
        }
        
        print(f"📡 Making request to: {url}")
        print(f"📋 Parameters: {params}")
        
        response = requests.get(url, params=params, timeout=60)
        print(f"✅ Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            print(f"🏆 Found {len(matches)} matches")
            print()
            
            for i, match in enumerate(matches, 1):
                fixture = match.get('fixture', {})
                home_team = fixture.get('home_team', 'N/A')
                away_team = fixture.get('away_team', 'N/A')
                fixture_id = fixture.get('id', 'N/A')
                date = fixture.get('date', 'N/A')
                
                print(f"Match {i}:")
                print(f"  🆔 ID: {fixture_id}")
                print(f"  🏠 Home: {home_team}")
                print(f"  🚪 Away: {away_team}")
                print(f"  📅 Date: {date}")
                
                # Check for placeholder names
                if "Team A" in str(home_team) or "Team B" in str(away_team):
                    print(f"  ⚠️  Still using placeholder names!")
                else:
                    print(f"  ✅ Real team names detected!")
                print()
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    test_api_response()
