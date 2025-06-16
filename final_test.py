#!/usr/bin/env python3

import requests

def comprehensive_test():
    print("=== COMPREHENSIVE API TEST ===")
    
    # Test the API
    response = requests.get('http://localhost:5000/api/odds-based-fixtures?limit=15')
    data = response.json()
    
    print(f'Status: {response.status_code}')
    print(f'Count returned: {data["count"]}')
    print(f'Discovery method: {data["discovery_method"]}')
    print()
    
    print('Team names (checking for placeholders):')
    for i, match in enumerate(data['matches'], 1):
        print(f'{i:2d}. {match["home_team"]} vs {match["away_team"]}')
    
    # Check for placeholder names
    placeholder_patterns = ['Team A', 'Team B', '(', ')']
    has_placeholder = any(
        any(pattern in match['home_team'] or pattern in match['away_team'] 
            for pattern in placeholder_patterns)
        for match in data['matches']
    )
    
    print()
    print('RESULT: ' + ('❌ FOUND PLACEHOLDERS' if has_placeholder else '✅ ALL REAL TEAM NAMES'))
    
    # Additional checks
    print(f"\nAdditional verification:")
    print(f"- Requested limit: 15")
    print(f"- Actual returned: {data['count']}")
    print(f"- Match limit working: {'✅ YES' if data['count'] <= 15 else '❌ NO'}")
    
    # Check if all matches have required fields
    required_fields = ['home_team', 'away_team', 'fixture_id', 'has_odds', 'league_name', 'date']
    all_have_fields = all(
        all(field in match for field in required_fields)
        for match in data['matches']
    )
    print(f"- All matches have required fields: {'✅ YES' if all_have_fields else '❌ NO'}")
    
    # Check if all matches have odds
    all_have_odds = all(match['has_odds'] for match in data['matches'])
    print(f"- All matches have odds: {'✅ YES' if all_have_odds else '❌ NO'}")

if __name__ == "__main__":
    comprehensive_test()
