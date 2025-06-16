#!/usr/bin/env python3
"""
Quick verification test for the enhanced odds-based discovery system.
"""

import requests
import json
from datetime import datetime

def test_odds_discovery():
    """Test the enhanced odds-based discovery endpoint."""
    print(f"🧪 Testing Enhanced Odds-Based Discovery System")
    print(f"⏰ Test time: {datetime.now()}")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test 1: Explicit odds discovery
    print("\n1️⃣ Testing explicit odds-based discovery...")
    try:
        response = requests.get(f"{base_url}/api/upcoming_predictions", 
                               params={
                                   'use_odds_discovery': 'true',
                                   'limit': 2,
                                   'include_additional_data': 'true'
                               }, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {len(data.get('match_predictions', []))} matches")
            
            if data.get('match_predictions'):
                match = data['match_predictions'][0]
                print(f"   📍 Sample match: {match.get('home_team')} vs {match.get('away_team')}")
                print(f"   📅 Date: {match.get('date')}")
                print(f"   🎯 Predictions: Home {match.get('home_win_prob'):.3f}, Away {match.get('away_win_prob'):.3f}")
                
                # Check for odds analysis
                if 'odds_analysis' in match:
                    odds = match['odds_analysis']
                    print(f"   📊 Odds Analysis: {len(odds.get('bookmakers', []))} bookmakers")
                    if 'best_odds' in odds:
                        best = odds['best_odds']
                        print(f"   💰 Best Odds: Home {best.get('home', 'N/A')}, Away {best.get('away', 'N/A')}")
                else:
                    print(f"   ⚠️  No odds analysis found in response")
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    # Test 2: Auto discovery (no league_id)
    print("\n2️⃣ Testing automatic odds discovery (no league_id)...")
    try:
        response = requests.get(f"{base_url}/api/upcoming_predictions", 
                               params={
                                   'limit': 2,
                                   'include_additional_data': 'true'
                               }, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Found {len(data.get('match_predictions', []))} matches")
        else:
            print(f"⚠️  Got {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"❌ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Verification completed!")

if __name__ == "__main__":
    test_odds_discovery()
