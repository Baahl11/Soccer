#!/usr/bin/env python3
"""
Debug script to check the actual structure of odds API responses
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import FootballAPI
import json
from datetime import datetime, timedelta

def debug_odds_api_structure():
    """Debug the actual structure of odds API responses"""
    
    print("Debug: Checking odds API response structure...")
    print("=" * 60)
    
    api = FootballAPI()
    
    # Test live odds endpoint
    print("\n1. Live Odds Endpoint Structure:")
    print("-" * 40)
    
    try:
        live_odds = api.get_live_odds()
        if live_odds and 'response' in live_odds:
            matches = live_odds['response']
            print(f"Total matches in live odds: {len(matches)}")
            
            if matches:
                print("\nFirst match structure:")
                first_match = matches[0]
                print(json.dumps(first_match, indent=2, ensure_ascii=False))
                
                # Check teams structure
                teams = first_match.get('teams', {})
                print(f"\nTeams structure: {teams}")
                home_team = teams.get('home', {})
                away_team = teams.get('away', {})
                print(f"Home team: {home_team}")
                print(f"Away team: {away_team}")
        else:
            print("No live odds found or invalid response")
            
    except Exception as e:
        print(f"Error getting live odds: {e}")
    
    # Test regular odds endpoint for today
    print("\n2. Regular Odds Endpoint Structure:")
    print("-" * 40)
    
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        odds_today = api.get_odds(date=today)
        
        if odds_today and 'response' in odds_today:
            matches = odds_today['response']
            print(f"Total matches for today: {len(matches)}")
            
            if matches:
                print(f"\nFirst match structure for {today}:")
                first_match = matches[0]
                print(json.dumps(first_match, indent=2, ensure_ascii=False))
        else:
            print("No odds found for today or invalid response")
            
    except Exception as e:
        print(f"Error getting odds for today: {e}")

if __name__ == "__main__":
    debug_odds_api_structure()
