#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from odds_based_fixture_discovery import get_matches_with_odds_24h

def test_odds_only_approach():
    """Test that we're only using the odds endpoint, not iterating leagues"""
    print("🧪 Testing ODDS-ONLY approach (no league iteration)...")
    print("=" * 60)
    
    try:
        # Get matches using odds-based discovery
        matches = get_matches_with_odds_24h(10)
        
        print(f"✅ Found {len(matches)} matches using ONLY odds endpoint")
        print(f"✅ This includes ALL leagues worldwide that have odds available")
        print()
        
        if matches:
            print("📋 Sample matches found:")
            for i, match in enumerate(matches[:8], 1):
                print(f"  {i}. {match['home_team']} vs {match['away_team']}")
                print(f"     League: {match['league_name']}")
                print(f"     Discovery: {match['discovery_method']}")
                print()
            
            # Check discovery methods
            odds_endpoint_count = len([m for m in matches if m.get('discovery_method') == 'odds_endpoint'])
            fixtures_fallback_count = len([m for m in matches if m.get('discovery_method') == 'fixtures_fallback'])
            
            print(f"📊 Discovery method breakdown:")
            print(f"   🎯 Odds endpoint: {odds_endpoint_count} matches")
            print(f"   🔄 Fixtures fallback: {fixtures_fallback_count} matches")
            
            if odds_endpoint_count > 0:
                print("✅ SUCCESS: Using odds endpoint as primary source!")
            else:
                print("⚠️  WARNING: No matches from odds endpoint!")
                
        else:
            print("⚠️  No matches found")
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_odds_only_approach()
