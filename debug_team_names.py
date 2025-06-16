#!/usr/bin/env python3
"""
Debug team name resolution in odds-based discovery
"""
import logging
import sys
import os

# Set up logging to see debug messages
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from odds_based_fixture_discovery import OddsBasedFixtureDiscovery
from data import FootballAPI, get_api_instance

def test_team_name_resolution():
    """Test team name resolution specifically"""
    
    print("🔍 Testing team name resolution...")
    
    # Test the FootballAPI directly
    try:
        api = get_api_instance()
        print(f"✅ API instance created: {type(api)}")
        
        # Test getting team info for a known team ID
        test_team_ids = [1218558, 1218553, 1218556]  # IDs from the log
        
        print(f"\n🧪 Testing get_multiple_teams_info with IDs: {test_team_ids}")
        
        team_names = api.get_multiple_teams_info(test_team_ids)
        print(f"Result: {team_names}")
        
        # Test individual team info
        for team_id in test_team_ids[:2]:  # Test first 2 only
            print(f"\n🧪 Testing get_team_info for ID: {team_id}")
            team_info = api.get_team_info(team_id)
            print(f"Result: {team_info}")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test the odds discovery with debugging
    print("\n" + "="*50)
    print("🔍 Testing odds discovery with debug logging...")
    
    try:
        discovery = OddsBasedFixtureDiscovery()
        
        # Get one match for detailed analysis
        matches = discovery.get_matches_with_odds_next_24h(1)
        
        if matches:
            match = matches[0]
            print(f"\n📊 First match details:")
            print(f"  Fixture ID: {match.get('fixture_id')}")
            print(f"  Home Team: {match.get('home_team')}")
            print(f"  Away Team: {match.get('away_team')}")
            print(f"  Home Team ID: {match.get('home_team_id')}")
            print(f"  Away Team ID: {match.get('away_team_id')}")
            print(f"  Discovery Method: {match.get('discovery_method')}")
            
            # If team names are unknown, try to resolve them manually
            home_team_id = match.get('home_team_id')
            away_team_id = match.get('away_team_id')
            
            if home_team_id or away_team_id:
                print(f"\n🔧 Manual team name resolution test:")
                
                if home_team_id:
                    print(f"  Testing home team ID: {home_team_id}")
                    try:
                        home_info = api.get_team_info(home_team_id)
                        print(f"  Home team info: {home_info}")
                    except Exception as e:
                        print(f"  Home team lookup failed: {e}")
                
                if away_team_id:
                    print(f"  Testing away team ID: {away_team_id}")
                    try:
                        away_info = api.get_team_info(away_team_id)
                        print(f"  Away team info: {away_info}")
                    except Exception as e:
                        print(f"  Away team lookup failed: {e}")
            
        else:
            print("❌ No matches found")
            
    except Exception as e:
        print(f"❌ Discovery test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_team_name_resolution()
