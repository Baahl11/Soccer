#!/usr/bin/env python3

import requests
import json

def test_api_endpoint():
    """Test the Flask API endpoint for odds-based fixture discovery"""
    base_url = "http://localhost:5000"
    
    # Test with limit=5
    print("Testing /api/odds-based-fixtures with limit=5...")
    try:
        response = requests.get(f"{base_url}/api/odds-based-fixtures?limit=5")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Count: {data['count']}")
            print(f"Discovery Method: {data['discovery_method']}")
            print(f"Timestamp: {data['timestamp']}")
            
            print("\nMatches found:")
            for i, match in enumerate(data['matches'], 1):
                print(f"  {i}. {match['home_team']} vs {match['away_team']}")
                print(f"     League: {match['league_name']}")
                print(f"     Date: {match['date']}")
                print(f"     Fixture ID: {match['fixture_id']}")
                print(f"     Has Odds: {match['has_odds']}")
                print()
            
            # Verify team names are not placeholder format
            placeholder_names = [match for match in data['matches'] 
                               if 'Team A' in match['home_team'] or 'Team A' in match['away_team']
                               or 'Team B' in match['home_team'] or 'Team B' in match['away_team']]
            
            if placeholder_names:
                print(f"⚠️  WARNING: Found {len(placeholder_names)} matches with placeholder team names!")
                for match in placeholder_names:
                    print(f"   - {match['home_team']} vs {match['away_team']}")
            else:
                print("✅ SUCCESS: All matches have proper team names (no placeholders found)")
                
        else:
            print(f"❌ ERROR: API returned status {response.status_code}")
            print(f"Response: {response.text}")
            
    except Exception as e:
        print(f"❌ ERROR: Failed to connect to API: {e}")

if __name__ == "__main__":
    test_api_endpoint()
