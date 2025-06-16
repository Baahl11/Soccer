#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test Team Name Fix

This script tests whether the team name fetching enhancement is working properly
in the odds-based discovery system.
"""

import logging
import requests
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("team_name_test")

def test_team_name_resolution():
    """Test if the API is returning real team names instead of placeholders"""
    
    logger.info("Testing team name resolution in odds-based discovery")
    logger.info("=" * 60)
    
    try:
        # Make API request to get upcoming matches
        url = "http://127.0.0.1:5000/api/upcoming_predictions"
        params = {
            "limit": 5,
            "include_additional_data": "true"
        }
        
        logger.info(f"Making request to: {url}")
        logger.info(f"Parameters: {params}")
        
        response = requests.get(url, params=params, timeout=60)
        logger.info(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            matches = data.get('matches', [])
            logger.info(f"Found {len(matches)} matches")
            
            if not matches:
                logger.warning("No matches returned - cannot test team names")
                return
            
            # Analyze team names
            team_name_stats = {
                'total_matches': len(matches),
                'placeholder_count': 0,
                'real_names_count': 0,
                'unknown_count': 0,
                'matches_analyzed': []
            }
            
            for i, match in enumerate(matches, 1):
                fixture = match.get('fixture', {})
                home_team = fixture.get('home_team', 'N/A')
                away_team = fixture.get('away_team', 'N/A')
                fixture_id = fixture.get('fixture_id', 'N/A')
                discovery_method = fixture.get('discovery_method', 'Unknown')
                
                # Analyze team names
                is_placeholder = False
                is_unknown = False
                
                # Check for placeholder patterns
                if ("Team A" in str(home_team) or "Team B" in str(away_team) or 
                    "(" in str(home_team) and ")" in str(home_team) or
                    "(" in str(away_team) and ")" in str(away_team)):
                    is_placeholder = True
                    team_name_stats['placeholder_count'] += 1
                elif home_team == "Unknown" or away_team == "Unknown":
                    is_unknown = True
                    team_name_stats['unknown_count'] += 1
                else:
                    team_name_stats['real_names_count'] += 1
                
                match_analysis = {
                    'match_num': i,
                    'fixture_id': fixture_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'discovery_method': discovery_method,
                    'is_placeholder': is_placeholder,
                    'is_unknown': is_unknown,
                    'has_real_names': not (is_placeholder or is_unknown)
                }
                
                team_name_stats['matches_analyzed'].append(match_analysis)
                
                # Log match details
                logger.info(f"\nMatch {i}:")
                logger.info(f"  Fixture ID: {fixture_id}")
                logger.info(f"  Home Team: {home_team}")
                logger.info(f"  Away Team: {away_team}")
                logger.info(f"  Discovery Method: {discovery_method}")
                
                if is_placeholder:
                    logger.warning(f"  ⚠️  PLACEHOLDER NAMES DETECTED")
                elif is_unknown:
                    logger.warning(f"  ⚠️  UNKNOWN TEAM NAMES")
                else:
                    logger.info(f"  ✅ Real team names!")
            
            # Summary
            logger.info("\n" + "=" * 60)
            logger.info("TEAM NAME ANALYSIS SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total matches analyzed: {team_name_stats['total_matches']}")
            logger.info(f"Matches with real names: {team_name_stats['real_names_count']}")
            logger.info(f"Matches with placeholders: {team_name_stats['placeholder_count']}")
            logger.info(f"Matches with unknown names: {team_name_stats['unknown_count']}")
            
            success_rate = (team_name_stats['real_names_count'] / team_name_stats['total_matches']) * 100
            logger.info(f"Real name success rate: {success_rate:.1f}%")
            
            if success_rate >= 80:
                logger.info("✅ EXCELLENT: Team name fetching is working well!")
            elif success_rate >= 60:
                logger.info("🔶 GOOD: Team name fetching is mostly working")
            elif success_rate >= 40:
                logger.info("⚠️  FAIR: Team name fetching needs improvement")
            else:
                logger.warning("❌ POOR: Team name fetching is not working properly")
            
            # Save detailed results
            with open('team_name_test_results.json', 'w') as f:
                json.dump(team_name_stats, f, indent=2, default=str)
            logger.info(f"\nDetailed results saved to: team_name_test_results.json")
            
        else:
            logger.error(f"API request failed with status {response.status_code}")
            logger.error(f"Response: {response.text[:500]}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)

def test_direct_odds_discovery():
    """Test the odds-based discovery directly"""
    
    logger.info("\nTesting direct odds-based discovery...")
    logger.info("-" * 40)
    
    try:
        from odds_based_fixture_discovery import get_matches_with_odds_24h
        
        # Get matches directly from the discovery system
        matches = get_matches_with_odds_24h(limit=3)
        
        logger.info(f"Direct discovery found {len(matches)} matches")
        
        for i, match in enumerate(matches, 1):
            logger.info(f"\nDirect Match {i}:")
            logger.info(f"  Home: {match.get('home_team', 'N/A')}")
            logger.info(f"  Away: {match.get('away_team', 'N/A')}")
            logger.info(f"  Method: {match.get('discovery_method', 'N/A')}")
            logger.info(f"  Has Odds: {match.get('has_odds', False)}")
            
            # Check for placeholder names
            home = str(match.get('home_team', ''))
            away = str(match.get('away_team', ''))
            
            if "Team A" in home or "Team B" in away or "(" in home or ")" in away:
                logger.warning(f"  ⚠️  Placeholder names detected in direct discovery")
            else:
                logger.info(f"  ✅ Real names in direct discovery")
        
    except Exception as e:
        logger.error(f"Error in direct discovery test: {e}")

if __name__ == "__main__":
    logger.info("Starting comprehensive team name resolution test")
    logger.info(f"Test started at: {datetime.now()}")
    
    # Test API endpoint
    test_team_name_resolution()
    
    # Test direct discovery
    test_direct_odds_discovery()
    
    logger.info(f"\nTest completed at: {datetime.now()}")
