#!/usr/bin/env python3
"""
Test with debug logging enabled
"""
import logging
import sys
import os

# Set up debug logging
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_with_debug():
    """Test odds discovery with full debug logging"""
    
    print("🔍 Testing with DEBUG logging enabled...")
    
    try:
        from odds_based_fixture_discovery import OddsBasedFixtureDiscovery
        
        discovery = OddsBasedFixtureDiscovery()
        
        # Get just 1 match to see all the debug output
        print("\n" + "="*60)
        print("Starting discovery with 1 match limit...")
        matches = discovery.get_matches_with_odds_next_24h(1)
        
        print(f"\nResult: Found {len(matches)} matches")
        
        if matches:
            match = matches[0]
            print(f"\nMatch details:")
            print(f"  Home Team: '{match.get('home_team')}'")
            print(f"  Away Team: '{match.get('away_team')}'")
            print(f"  Home Team ID: {match.get('home_team_id')}")
            print(f"  Away Team ID: {match.get('away_team_id')}")
            print(f"  Date: {match.get('date')}")
            print(f"  League: {match.get('league_name')}")
            print(f"  Discovery Method: {match.get('discovery_method')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_with_debug()
