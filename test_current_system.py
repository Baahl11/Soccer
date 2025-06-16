#!/usr/bin/env python3
"""
Quick test to check the current odds-based discovery system
"""
import requests
import json
import sys
import os

def test_odds_based_discovery():
    """Test the odds-based discovery endpoint"""
    try:
        print("Testing odds-based discovery endpoint...")
        response = requests.get('http://localhost:5000/api/odds-based-fixtures', timeout=30)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response structure: {type(data)}")
            
            if isinstance(data, dict):
                if 'matches' in data:
                    matches = data['matches']
                    print(f"Number of matches found: {len(matches)}")
                    
                    # Analyze first few matches
                    for i, match in enumerate(matches[:3]):
                        print(f"\nMatch {i+1}:")
                        print(f"  Home Team: {match.get('home_team', 'N/A')}")
                        print(f"  Away Team: {match.get('away_team', 'N/A')}")
                        print(f"  Date: {match.get('date', 'N/A')}")
                        print(f"  League: {match.get('league_name', 'N/A')}")
                        print(f"  Has Odds: {match.get('has_odds', 'N/A')}")
                        
                        # Check if team names are placeholder-style
                        home_team = match.get('home_team', '')
                        away_team = match.get('away_team', '')
                        
                        if 'Team A' in home_team or 'Team B' in away_team or 'Unknown' in home_team or 'Unknown' in away_team:
                            print(f"  ⚠️  ISSUE: Placeholder team names detected!")
                        else:
                            print(f"  ✅ Team names look good")
                
                elif 'error' in data:
                    print(f"API returned error: {data['error']}")
                else:
                    print(f"Unexpected response structure: {data}")
            else:
                print(f"Response is not a dict: {data}")
        else:
            print(f"Failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - is the Flask server running on port 5000?")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_direct_import():
    """Test by importing the module directly"""
    try:
        print("\n" + "="*50)
        print("Testing direct module import...")
        
        # Add the current directory to Python path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from odds_based_fixture_discovery import OddsBasedFixtureDiscovery
        from data import FootballAPI
        
        print("✅ Modules imported successfully")
          # Create instances
        discovery = OddsBasedFixtureDiscovery()
        
        print("✅ Instances created successfully")
        
        # Test discovery
        print("Running discovery...")
        matches = discovery.get_matches_with_odds_next_24h(limit=3)
        
        print(f"Found {len(matches)} matches")
        
        for i, match in enumerate(matches):
            print(f"\nMatch {i+1}:")
            print(f"  Home Team: {match.get('home_team', 'N/A')}")
            print(f"  Away Team: {match.get('away_team', 'N/A')}")
            print(f"  Date: {match.get('date', 'N/A')}")
            print(f"  League: {match.get('league_name', 'N/A')}")
            
            # Check team names
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            if 'Team A' in home_team or 'Team B' in away_team or 'Unknown' in home_team or 'Unknown' in away_team:
                print(f"  ⚠️  ISSUE: Placeholder team names detected!")
            else:
                print(f"  ✅ Team names look good")
        
    except Exception as e:
        print(f"❌ Direct import test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧪 Testing Current Odds-Based Discovery System")
    print("=" * 50)
    
    # Test via API endpoint
    test_odds_based_discovery()
    
    # Test via direct import
    test_direct_import()
