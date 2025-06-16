#!/usr/bin/env python3
"""
Test script for the enhanced /api/upcoming_predictions endpoint with odds-based discovery
"""

import requests
import json
import sys
from datetime import datetime

def test_odds_based_discovery():
    """Test the new odds-based discovery functionality"""
    base_url = "http://127.0.0.1:5000"
    
    print("Testing odds-based fixture discovery endpoint...")
    print("=" * 60)
    
    # Test 1: Odds-based discovery without league_id/season
    print("\n1. Testing odds-based discovery (no league_id/season):")
    print("-" * 50)
    
    endpoint = f"{base_url}/api/upcoming_predictions"
    params = {
        "limit": 5,
        "include_additional_data": "true"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            match_predictions = data.get("match_predictions", [])
            print(f"Number of matches found: {len(match_predictions)}")
            
            if match_predictions:
                print("\nFirst match details:")
                first_match = match_predictions[0]
                print(f"  Fixture ID: {first_match.get('fixture_id')}")
                print(f"  Home Team: {first_match.get('home_team')}")
                print(f"  Away Team: {first_match.get('away_team')}")
                print(f"  Date: {first_match.get('date')}")
                print(f"  League ID: {first_match.get('league_id')}")
                print(f"  Has Odds Analysis: {'odds_analysis' in first_match}")
                print(f"  Confidence: {first_match.get('confidence')}")
                
                # Check if odds data is available
                if 'odds_analysis' in first_match:
                    odds = first_match['odds_analysis']
                    print(f"  Odds Available: {bool(odds.get('bookmaker_odds'))}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing odds-based discovery: {e}")
    
    # Test 2: Explicit odds-based discovery flag
    print("\n2. Testing explicit odds-based discovery flag:")
    print("-" * 50)
    
    params = {
        "use_odds_discovery": "true",
        "limit": 3,
        "include_additional_data": "true"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            match_predictions = data.get("match_predictions", [])
            print(f"Number of matches found: {len(match_predictions)}")
            
            if match_predictions:
                print("\nMatches with odds discovery:")
                for i, match in enumerate(match_predictions, 1):
                    print(f"  {i}. {match.get('home_team')} vs {match.get('away_team')}")
                    print(f"     Fixture ID: {match.get('fixture_id')}")
                    print(f"     Date: {match.get('date')}")
                    print(f"     Has odds: {'odds_analysis' in match}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing explicit odds discovery: {e}")
    
    # Test 3: Traditional approach (with league_id and season)
    print("\n3. Testing traditional approach (league_id + season):")
    print("-" * 50)
    
    params = {
        "league_id": 71,  # Premier League
        "season": 2024,
        "limit": 3,
        "include_additional_data": "true"
    }
    
    try:
        response = requests.get(endpoint, params=params, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            match_predictions = data.get("match_predictions", [])
            print(f"Number of matches found: {len(match_predictions)}")
            
            if match_predictions:
                print("\nTraditional approach matches:")
                for i, match in enumerate(match_predictions, 1):
                    print(f"  {i}. {match.get('home_team')} vs {match.get('away_team')}")
                    print(f"     League ID: {match.get('league_id')}")
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error testing traditional approach: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")

def test_odds_discovery_module():
    """Test the odds discovery module directly"""
    print("\nTesting odds discovery module directly...")
    print("-" * 50)
    
    try:
        from odds_based_fixture_discovery import get_matches_with_odds_24h, get_matches_with_verified_odds
        
        print("Testing get_matches_with_odds_24h()...")
        matches = get_matches_with_odds_24h(limit=3)
        print(f"Found {len(matches)} matches with odds in next 24h")
        
        if matches:
            print("Sample match:")
            match = matches[0]
            print(f"  Fixture ID: {match.get('fixture_id')}")
            print(f"  Home: {match.get('home_team')}")
            print(f"  Away: {match.get('away_team')}")
            print(f"  Date: {match.get('date')}")
            
        print("\nTesting get_matches_with_verified_odds()...")
        verified_matches = get_matches_with_verified_odds(limit=3)
        print(f"Found {len(verified_matches)} matches with verified odds")
        
    except ImportError as e:
        print(f"Could not import odds discovery module: {e}")
    except Exception as e:
        print(f"Error testing odds discovery module: {e}")

if __name__ == "__main__":
    print(f"Starting tests at {datetime.now()}")
    
    # Test the module directly first
    test_odds_discovery_module()
    
    # Test the endpoint
    test_odds_based_discovery()
    
    print(f"\nAll tests completed at {datetime.now()}")
