"""
Odds-Based Fixture Discovery Module

This module implements an improved approach for obtaining matches with odds available
in the next 72 hours using the odds endpoints directly, instead of searching league by league.
"""

import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from data import get_api_instance
from config import API_BASE_URL, API_FOOTBALL_KEY, ODDS_ENDPOINTS
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class OddsBasedFixtureDiscovery:
    """
    Class for discovering fixtures with available odds using the odds endpoints directly.
    This provides a more efficient alternative to league-by-league searching.
    """
    
    def __init__(self):
        self.api_base_url = API_BASE_URL
        self.api_key = API_FOOTBALL_KEY
        self.headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': self.api_key
        }
        self.cache_dir = Path("cache/odds_discovery")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_matches_with_odds_next_24h(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get matches with available odds in the next 72 hours using odds endpoints.
        
        Args:
            limit: Maximum number of matches to return
              Returns:
            List of match data with available odds
        """
        try:
            logger.info("🔍 Discovering upcoming matches with pre-match odds available from ALL leagues...")
            
            # PRIMARY APPROACH: Use pre-match odds endpoint directly - this gets ALL matches with odds from ALL leagues
            # No need for league iteration since the odds endpoint already covers everything
            matches = self._get_matches_from_prematch_odds()
            
            logger.info(f"Found {len(matches)} total matches from odds endpoint (all leagues included)")
            
            # Remove duplicates and sort by date
            unique_matches = self._deduplicate_and_sort_matches(matches)
            
            # Filter to next 72 hours only
            filtered_matches = self._filter_matches_next_72h(unique_matches)
            
            # Limit results
            final_matches = filtered_matches[:limit] if filtered_matches else []
            
            logger.info(f"✅ Found {len(final_matches)} matches with odds in next 72 hours from ALL leagues worldwide")
            
            return final_matches
            
        except Exception as e:
            logger.error(f"Error in odds-based fixture discovery: {e}")
            return []
    
    def _get_matches_from_prematch_odds(self) -> List[Dict[str, Any]]:
        """Get matches from pre-match odds endpoint with date filtering for next 3 days."""
        try:
            logger.info("🔄 Checking pre-match odds endpoint for next 3 days...")
            
            matches = []
            
            # Check today, tomorrow, and day after tomorrow
            for days_ahead in range(0, 3):
                try:
                    target_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    
                    endpoint = f"{self.api_base_url}/odds"
                    params = {
                        'date': target_date,
                        'timezone': 'UTC'
                    }
                    
                    response = requests.get(endpoint, headers=self.headers, params=params, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        day_matches = []
                        
                        for odds_data in data.get('response', []):
                            fixture_info = odds_data.get('fixture', {})
                            if fixture_info and fixture_info.get('id'):
                                match_data = self._extract_match_from_odds_data(odds_data)
                                if match_data:
                                    day_matches.append(match_data)
                        
                        logger.info(f"Found {len(day_matches)} matches from pre-match odds for {target_date}")
                        matches.extend(day_matches)
                    else:
                        logger.warning(f"Pre-match odds endpoint returned {response.status_code} for {target_date}")
                        
                except Exception as e:
                    logger.warning(f"Error getting pre-match odds for {target_date}: {e}")
                    continue
            
            logger.info(f"Total matches from pre-match odds: {len(matches)}")
            return matches
                
        except Exception as e:
            logger.warning(f"Error getting matches from pre-match odds: {e}")
            return []
    
    def _extract_match_from_odds_data(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:        """Extract standardized match data from odds API response."""
        try:
            fixture_info = odds_data.get('fixture', {})
            teams_info = odds_data.get('teams', {})
            league_info = odds_data.get('league', {})
              if not fixture_info.get('id'):
                return None
            
            # Extract team names with better error handling
            home_team_name = "Unknown"
            away_team_name = "Unknown"
            home_team_id = None
            away_team_id = None
            
            # Debug logging to see what we're getting
            logger.debug(f"Processing odds data for fixture {fixture_info.get('id')}")
            logger.debug(f"Teams info structure: {teams_info}")
            
            if teams_info:
                home_info = teams_info.get('home', {})
                away_info = teams_info.get('away', {})
                
                logger.debug(f"Home info: {home_info}")
                logger.debug(f"Away info: {away_info}")
                
                if home_info and isinstance(home_info, dict):
                    home_team_name = home_info.get('name', 'Unknown')
                    home_team_id = home_info.get('id')
                    logger.debug(f"Extracted home team: {home_team_name} (ID: {home_team_id})")
                if away_info and isinstance(away_info, dict):
                    away_team_name = away_info.get('name', 'Unknown')
                    away_team_id = away_info.get('id')
                    logger.debug(f"Extracted away team: {away_team_name} (ID: {away_team_id})")
            else:
                logger.debug("No teams_info found in odds data - fetching fixture details separately")
                # Fetch fixture details from fixtures endpoint to get team information
                try:
                    from data import get_fixture_data
                    fixture_data = get_fixture_data(fixture_info.get('id'))
                    
                    if fixture_data and 'response' in fixture_data:
                        fixture_response = fixture_data.get('response', [])
                        if fixture_response and len(fixture_response) > 0:
                            fixture_details = fixture_response[0]
                            teams_details = fixture_details.get('teams', {})
                            
                            home_team_info = teams_details.get('home', {})
                            away_team_info = teams_details.get('away', {})
                            
                            if home_team_info:
                                home_team_name = home_team_info.get('name', 'Unknown')
                                home_team_id = home_team_info.get('id')
                                logger.debug(f"Fetched home team from fixture endpoint: {home_team_name} (ID: {home_team_id})")
                                
                            if away_team_info:
                                away_team_name = away_team_info.get('name', 'Unknown')
                                away_team_id = away_team_info.get('id')
                                logger.debug(f"Fetched away team from fixture endpoint: {away_team_name} (ID: {away_team_id})")
                        else:
                            logger.debug("No fixture details found in response")
                    else:
                        logger.debug("No fixture data response received")
                          except Exception as e:
                    logger.debug(f"Failed to fetch fixture details: {e}")
            
            # If team names are still unknown after both methods, use fallback with team IDs
            if (home_team_name == "Unknown" or away_team_name == "Unknown"):
                fixture_id = fixture_info.get('id')
                logger.debug(f"Using fallback team names for fixture {fixture_id}")
                
                if home_team_name == "Unknown":
                    if home_team_id:
                        home_team_name = f"Team A ({home_team_id})"
                    else:
                        home_team_name = f"Team A ({fixture_id})"
                
                if away_team_name == "Unknown":
                    if away_team_id:
                        away_team_name = f"Team B ({away_team_id})"
                    else:
                        away_team_name = f"Team B ({fixture_id})"
                        
                logger.debug(f"Final fallback team names: {home_team_name} vs {away_team_name}")
            
            return {
                'fixture_id': fixture_info.get('id'),
                'date': fixture_info.get('date'),
                'status': fixture_info.get('status', {}).get('short', 'NS'),
                'home_team': home_team_name,
                'away_team': away_team_name,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'league_id': league_info.get('id'),
                'league_name': league_info.get('name'),
                'has_odds': True,
                'odds_available': len(odds_data.get('bookmakers', [])) > 0,
                'discovery_method': 'odds_endpoint'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting match from odds data: {e}")
            return None
    
    def _extract_match_from_fixture_data(self, fixture_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract standardized match data from fixtures API response."""
        try:
            fixture_info = fixture_data.get('fixture', {})
            teams_info = fixture_data.get('teams', {})
            league_info = fixture_data.get('league', {})
            
            if not fixture_info.get('id'):
                return None
            
            return {
                'fixture_id': fixture_info.get('id'),
                'date': fixture_info.get('date'),
                'status': fixture_info.get('status', {}).get('short', 'NS'),
                'home_team': teams_info.get('home', {}).get('name', 'Unknown'),
                'away_team': teams_info.get('away', {}).get('name', 'Unknown'),
                'home_team_id': teams_info.get('home', {}).get('id'),
                'away_team_id': teams_info.get('away', {}).get('id'),
                'league_id': league_info.get('id'),
                'league_name': league_info.get('name'),
                'has_odds': False,  # We assume odds might be available but haven't confirmed
                'odds_available': False,  # Will be checked later if needed
                'discovery_method': 'fixtures_fallback'
            }
            
        except Exception as e:
            logger.debug(f"Error extracting match from fixture data: {e}")
            return None
    
    def _deduplicate_and_sort_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate matches and sort by date."""
        try:
            # Use fixture_id as unique identifier
            seen_fixtures = set()
            unique_matches = []
            
            for match in matches:
                fixture_id = match.get('fixture_id')
                if fixture_id and fixture_id not in seen_fixtures:
                    seen_fixtures.add(fixture_id)
                    unique_matches.append(match)
            
            # Sort by date (handle None values)
            unique_matches.sort(key=lambda x: x.get('date') or '9999-12-31T23:59:59')
            
            return unique_matches
            
        except Exception as e:
            logger.warning(f"Error deduplicating matches: {e}")
            return matches
    
    def _filter_matches_next_72h(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter matches to only include those in the next 72 hours (3 days)."""
        try:
            now = datetime.now()
            cutoff_time = now + timedelta(hours=72)  # Extended to 3 days
            
            filtered_matches = []
            
            for match in matches:
                match_date_str = match.get('date')
                if not match_date_str:
                    continue
                
                try:
                    # Parse the date string (assuming ISO format)
                    match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                    
                    # Convert to local time if needed (removing timezone info for comparison)
                    match_date = match_date.replace(tzinfo=None)
                    
                    # Check if match is in the future (within next 72 hours)
                    if now <= match_date <= cutoff_time:
                        filtered_matches.append(match)
                        
                except Exception as e:
                    logger.debug(f"Error parsing date {match_date_str}: {e}")
                    # Include match if we can't parse the date (better safe than sorry)
                    filtered_matches.append(match)
            
            logger.info(f"Filtered {len(filtered_matches)} matches within next 72 hours from {len(matches)} total matches")
            return filtered_matches
            
        except Exception as e:
            logger.warning(f"Error filtering matches by time: {e}")
            return matches
    
    def verify_odds_availability(self, fixture_id: int) -> bool:
        """
        Verify if odds are actually available for a specific fixture.
        
        Args:
            fixture_id: ID of the fixture to check
            
        Returns:
            True if odds are available, False otherwise
        """
        try:
            from optimize_odds_integration import get_fixture_odds
            
            odds_data = get_fixture_odds(fixture_id, use_cache=True)
            
            # Check if we got real odds (not simulated)
            if odds_data and not odds_data.get('simulated', True):
                return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Error verifying odds for fixture {fixture_id}: {e}")
            return False


def get_matches_with_odds_24h(limit: int = 20) -> List[Dict[str, Any]]:
    """
    Convenience function to get matches with odds in the next 72 hours.
    
    Args:
        limit: Maximum number of matches to return
        
    Returns:
        List of match data with available odds
    """
    discovery = OddsBasedFixtureDiscovery()
    return discovery.get_matches_with_odds_next_24h(limit)


def get_matches_with_verified_odds(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Get matches with verified odds availability (slower but more accurate).
    
    Args:
        limit: Maximum number of matches to return
        
    Returns:
        List of match data with verified odds availability
    """
    discovery = OddsBasedFixtureDiscovery()
    
    # Get potential matches
    potential_matches = discovery.get_matches_with_odds_next_24h(limit * 2)  # Get more to account for filtering
    
    # Verify odds availability for each match
    verified_matches = []
    
    for match in potential_matches:
        if len(verified_matches) >= limit:
            break
            
        fixture_id = match.get('fixture_id')
        if fixture_id and discovery.verify_odds_availability(fixture_id):
            match['odds_verified'] = True
            verified_matches.append(match)
        elif match.get('has_odds', False):
            # Include matches that we know have odds from the odds endpoint
            match['odds_verified'] = False
            verified_matches.append(match)
    
    return verified_matches


if __name__ == "__main__":
    # Test the odds-based discovery
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Testing Odds-Based Fixture Discovery...")
    
    # Test basic discovery
    matches = get_matches_with_odds_24h(10)
    print(f"\nFound {len(matches)} matches with odds in next 72 hours:")
    
    for i, match in enumerate(matches, 1):
        print(f"{i}. {match['home_team']} vs {match['away_team']}")
        print(f"   League: {match.get('league_name', 'Unknown')}")
        print(f"   Date: {match['date']}")
        print(f"   Method: {match['discovery_method']}")
        print(f"   Has Odds: {match['has_odds']}")
        print()
    
    # Test verified odds discovery
    print("\nTesting verified odds discovery...")
    verified_matches = get_matches_with_verified_odds(5)
    print(f"\nFound {len(verified_matches)} matches with verified odds:")
    
    for i, match in enumerate(verified_matches, 1):
        print(f"{i}. {match['home_team']} vs {match['away_team']}")
        print(f"   Odds Verified: {match.get('odds_verified', False)}")
        print()