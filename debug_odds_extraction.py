#!/usr/bin/env python3
"""
Debug the actual team IDs being extracted from odds data
"""
import logging
import sys
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odds_based_fixture_discovery import OddsBasedFixtureDiscovery
from data import get_api_instance

def debug_odds_data():
    """Debug the actual odds data being retrieved"""
    
    print("🔍 Debugging odds data extraction...")
    
    try:
        discovery = OddsBasedFixtureDiscovery()
        
        # Get raw odds data from API
        import requests
        
        # Test the odds endpoint directly
        headers = discovery.headers
        endpoint = f"{discovery.api_base_url}/odds"
        
        # Get today's odds
        from datetime import datetime, timedelta
        today = datetime.now().strftime('%Y-%m-%d')
        
        params = {
            'date': today,
            'timezone': 'UTC'
        }
        
        print(f"📡 Making direct API call to: {endpoint}")
        print(f"   Params: {params}")
        
        response = requests.get(endpoint, headers=headers, params=params, timeout=15)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"   Results: {data.get('results', 0)}")
            print(f"   Response items: {len(data.get('response', []))}")
            
            # Analyze first few items
            for i, odds_data in enumerate(data.get('response', [])[:3]):
                print(f"\n🎯 Odds Data {i+1}:")
                
                fixture_info = odds_data.get('fixture', {})
                teams_info = odds_data.get('teams', {})
                league_info = odds_data.get('league', {})
                
                print(f"   Fixture ID: {fixture_info.get('id')}")
                print(f"   Date: {fixture_info.get('date')}")
                print(f"   League: {league_info.get('name')} (ID: {league_info.get('id')})")
                
                if teams_info:
                    home_info = teams_info.get('home', {})
                    away_info = teams_info.get('away', {})
                    
                    print(f"   Home Team: {home_info.get('name', 'N/A')} (ID: {home_info.get('id', 'N/A')})")
                    print(f"   Away Team: {away_info.get('name', 'N/A')} (ID: {away_info.get('id', 'N/A')})")
                    
                    # Test team name lookup for these actual IDs
                    home_id = home_info.get('id')
                    away_id = away_info.get('id')
                    
                    if home_id:
                        print(f"   🧪 Testing team lookup for home team ID {home_id}:")
                        try:
                            api = get_api_instance()
                            team_data = api.get_team_info(home_id)
                            if team_data and team_data.get('response'):
                                team_name = team_data['response'][0].get('team', {}).get('name', 'No name')
                                print(f"      API Result: {team_name}")
                            else:
                                print(f"      API Result: Empty response - {team_data.get('results', 0)} results")
                        except Exception as e:
                            print(f"      API Error: {e}")
                    
                    if away_id and away_id != home_id:  # Don't test same ID twice
                        print(f"   🧪 Testing team lookup for away team ID {away_id}:")
                        try:
                            api = get_api_instance()
                            team_data = api.get_team_info(away_id)
                            if team_data and team_data.get('response'):
                                team_name = team_data['response'][0].get('team', {}).get('name', 'No name')
                                print(f"      API Result: {team_name}")
                            else:
                                print(f"      API Result: Empty response - {team_data.get('results', 0)} results")
                        except Exception as e:
                            print(f"      API Error: {e}")
                else:
                    print("   Teams info: Not available")
                    
                print(f"   Bookmakers: {len(odds_data.get('bookmakers', []))}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text[:500]}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_odds_data()
