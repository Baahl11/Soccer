#!/usr/bin/env python3

"""
Test script to verify team name fetching in odds-based discovery system.
"""

import logging
from odds_based_fixture_discovery import OddsBasedFixtureDiscovery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_team_name_fetching():
    """Test the team name fetching functionality."""
    print("🧪 Testing Team Name Fetching in Odds-Based Discovery")
    print("=" * 60)
    
    try:
        discovery = OddsBasedFixtureDiscovery()
        
        # Get matches with odds (limit to 3 for testing)
        print("1️⃣ Getting matches with odds...")
        matches = discovery.get_matches_with_odds_next_24h(limit=3)
        
        print(f"✅ Found {len(matches)} matches")
        print()
        
        for i, match in enumerate(matches, 1):
            print(f"🏆 Match {i}:")
            print(f"   🆔 Fixture ID: {match.get('fixture_id')}")
            print(f"   🏠 Home Team: {match.get('home_team')} (ID: {match.get('home_team_id')})")
            print(f"   🚪 Away Team: {match.get('away_team')} (ID: {match.get('away_team_id')})")
            print(f"   📅 Date: {match.get('date')}")
            print(f"   🏟️ League: {match.get('league_name')} (ID: {match.get('league_id')})")
            print(f"   🎯 Discovery Method: {match.get('discovery_method')}")
            print(f"   💰 Has Odds: {match.get('has_odds')}")
            print()
            
            # Check if team names look like placeholders
            home_team = match.get('home_team', '')
            away_team = match.get('away_team', '')
            
            if 'Team A' in home_team or 'Team B' in away_team:
                print(f"   ⚠️  WARNING: Still using placeholder names!")
            else:
                print(f"   ✅ Real team names detected!")
            print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_team_name_fetching()
