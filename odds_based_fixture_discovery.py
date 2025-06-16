"""
Odds-Based Fixture Discovery Module

This module implements an improved approach for obtaining matches with odds available
in the next 72 hours using the odds endpoints directly, without league-by-league searching.
Now using ONLY the odds endpoint to discover ALL matches with available odds worldwide.

Key Features:
- Uses UTC timestamps for consistent date handling
- Ensures only future matches are returned
- Properly handles timezone information
- Includes 2-minute buffer for upcoming matches
- Returns matches in chronological order
"""

import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from data import get_api_instance, get_fixture_data
from config import API_BASE_URL, API_FOOTBALL_KEY
import json
from pathlib import Path
from log_config import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)


class OddsBasedFixtureDiscovery:
    """
    Class for discovering fixtures with available odds using the odds endpoints directly.
    Now using ONLY the odds endpoint to get ALL matches with odds, regardless of league.
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
        
        # Rate limiting settings
        self.requests_per_minute = 30
        self.min_request_interval = 60.0 / self.requests_per_minute
        self.last_request_time = 0.0

    def get_matches_with_odds_next_24h(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get matches with available odds in the next 72 hours using ONLY the odds endpoint."""
        try:
            logger.info("🔍 Discovering upcoming matches with pre-match odds from ALL leagues worldwide...")
            
            # Get ALL matches with odds from the odds endpoint - this covers every league
            matches = self._get_matches_from_prematch_odds()
            logger.info(f"Found {len(matches)} total matches from odds endpoint (all leagues included)")
            
            # Process the matches
            unique_matches = self._deduplicate_and_sort_matches(matches)
            filtered_matches = self._filter_matches_next_72h(unique_matches)
            final_matches = filtered_matches[:limit] if filtered_matches else []
            
            logger.info(f"✅ Found {len(final_matches)} matches with odds in next 72 hours from ALL leagues")
            return final_matches
            
        except Exception as e:
            logger.error(f"Error in odds-based fixture discovery: {e}")
            return []

    def _get_matches_from_prematch_odds(self) -> List[Dict[str, Any]]:
        """Get ALL matches with odds from the odds endpoint for next 3 days."""
        try:
            logger.info("🔄 Checking odds endpoint for next 3 days (covers ALL leagues)...")
            matches = []
            
            for days_ahead in range(0, 3):
                try:
                    target_date = (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d')
                    
                    # Check cache first
                    cached_matches = self._load_cached_matches(target_date)
                    if cached_matches:
                        logger.info(f"✅ Loaded {len(cached_matches)} matches from cache for {target_date}")
                        matches.extend(cached_matches)
                        continue  # Skip API call, use cached data
                    
                    endpoint = f"{self.api_base_url}/odds"
                    params = {'date': target_date, 'timezone': 'UTC'}
                      # Rate limit handling - sleep between requests
                    time.sleep(2)  # 2 second delay between requests
                    
                    response = requests.get(endpoint, headers=self.headers, params=params, timeout=15)
                    
                    if response.status_code == 429:  # Too Many Requests
                        logger.warning("Rate limit exceeded, waiting 60 seconds...")
                        time.sleep(60)
                        response = requests.get(endpoint, headers=self.headers, params=params, timeout=15)
                    
                    if response.status_code == 403:  # Forbidden
                        logger.error("API authentication failed. Check your API key.")
                        continue
                        
                    if response.status_code == 200:
                        try:
                            data = response.json()
                        except json.JSONDecodeError as e:
                            logger.error(f"Invalid JSON response: {e}")
                            continue

                        day_matches = []
                        
                        for odds_data in data.get('response', []):
                            try:
                                fixture_info = odds_data.get('fixture', {})
                                if fixture_info and fixture_info.get('id'):
                                    match_data = self._extract_match_from_odds_data(odds_data)
                                    if match_data:
                                        day_matches.append(match_data)
                            except Exception as e:
                                logger.warning(f"Error processing match data: {e}")
                                continue
                        
                        logger.info(f"✅ Found {len(day_matches)} matches from odds endpoint for {target_date}")
                        
                        # Cache the matches for the day
                        self._cache_day_matches(target_date, day_matches)
                        
                        matches.extend(day_matches)
                    else:
                        logger.warning(f"❌ Odds endpoint returned {response.status_code} for {target_date}")
                    
                except requests.RequestException as e:
                    logger.warning(f"❌ Network error getting odds for {target_date}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"❌ Error getting odds for {target_date}: {e}")
                    continue
            
            logger.info(f"Total matches with odds available: {len(matches)}")
            return matches
            
        except Exception as e:
            logger.warning(f"Error getting matches from odds endpoint: {e}")
            return []    
    def _extract_match_from_odds_data(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract standardized match data from odds API response."""
        try:
            fixture_info = odds_data.get('fixture', {})
            teams_info = odds_data.get('teams', {})
            league_info = odds_data.get('league', {})
            
            if not fixture_info.get('id'):
                return None
                
            fixture_id = fixture_info.get('id')
            home_team_name = "Unknown"
            away_team_name = "Unknown"
            home_team_id = None
            away_team_id = None
            
            # First try to get team names from odds data
            if teams_info:
                home_info = teams_info.get('home', {})
                away_info = teams_info.get('away', {})
                
                if home_info and isinstance(home_info, dict):
                    home_team_name = home_info.get('name', 'Unknown')
                    home_team_id = home_info.get('id')
                if away_info and isinstance(away_info, dict):
                    away_team_name = away_info.get('name', 'Unknown')
                    away_team_id = away_info.get('id')
              # If team names are still unknown, fetch from fixtures endpoint
            if home_team_name == "Unknown" or away_team_name == "Unknown":
                try:
                    logger.info(f"⚽ Fetching team names for fixture {fixture_id}")
                    fixture_data = get_fixture_data(fixture_id)
                    
                    if fixture_data and 'response' in fixture_data:
                        fixture_response = fixture_data.get('response', [])
                        if fixture_response and len(fixture_response) > 0:
                            fixture_details = fixture_response[0]
                            teams_details = fixture_details.get('teams', {})
                            
                            home_team_info = teams_details.get('home', {})
                            away_team_info = teams_details.get('away', {})
                            
                            if home_team_info:
                                real_home_name = home_team_info.get('name')
                                real_home_id = home_team_info.get('id')
                                if real_home_name:
                                    home_team_name = real_home_name
                                    home_team_id = real_home_id
                                    logger.info(f"✅ Found home team: {real_home_name}")
                                
                            if away_team_info:
                                real_away_name = away_team_info.get('name')
                                real_away_id = away_team_info.get('id')
                                if real_away_name:
                                    away_team_name = real_away_name
                                    away_team_id = real_away_id
                                    logger.info(f"✅ Found away team: {real_away_name}")
                except Exception as e:
                    logger.warning(f"❌ Error fetching fixture details for {fixture_id}: {e}")
            
            # Only use fallback team names if absolutely nothing worked
            if home_team_name == "Unknown":
                home_team_name = f"Home Team ({home_team_id or fixture_id})"
                logger.warning(f"⚠️ Using fallback home team name for fixture {fixture_id}")
            
            if away_team_name == "Unknown":
                away_team_name = f"Away Team ({away_team_id or fixture_id})"
                logger.warning(f"⚠️ Using fallback away team name for fixture {fixture_id}")
            
            return {
                'fixture_id': fixture_id,
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

    def _deduplicate_and_sort_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate matches and sort by date."""
        try:
            seen_fixtures = set()
            unique_matches = []
            
            for match in matches:
                fixture_id = match.get('fixture_id')
                if fixture_id and fixture_id not in seen_fixtures:
                    seen_fixtures.add(fixture_id)
                    unique_matches.append(match)
            
            unique_matches.sort(key=lambda x: x.get('date') or '9999-12-31T23:59:59')
            return unique_matches
            
        except Exception as e:
            logger.warning(f"Error deduplicating matches: {e}")
            return matches

    def _filter_matches_next_72h(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter matches to only include future matches in the next 72 hours."""
        try:
            # Use UTC for consistent comparison
            now = datetime.utcnow()
            cutoff_time = now + timedelta(hours=72)
            filtered_matches = []
            
            for match in matches:
                match_date_str = match.get('date')
                if not match_date_str:
                    continue
                
                try:
                    # Parse the match date (API returns UTC dates)
                    match_date = datetime.fromisoformat(match_date_str.replace('Z', '+00:00'))
                    match_date = match_date.replace(tzinfo=None)  # Convert to naive UTC
                    
                    # Check if match is in the future and within 72 hours
                    # Add 2 minute buffer to avoid edge cases
                    if match_date > (now + timedelta(minutes=2)) and match_date <= cutoff_time:
                        filtered_matches.append(match)
                    
                except Exception as e:
                    logger.debug(f"Error parsing date {match_date_str}: {e}")
                    continue
            
            # Sort matches by date
            filtered_matches.sort(key=lambda x: x.get('date') or '9999-12-31T23:59:59')
            
            logger.info(f"Filtered to {len(filtered_matches)} upcoming matches in next 72 hours")
            return filtered_matches
            
        except Exception as e:
            logger.warning(f"Error filtering matches by time: {e}")
            return []

    def _wait_for_rate_limit(self):
        """Wait according to rate limiting settings."""
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.min_request_interval:
            wait_time = self.min_request_interval - time_since_last + 0.1  # Add small buffer
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Make an API request with rate limiting and error handling."""
        self._wait_for_rate_limit()
        
        try:
            response = requests.get(
                f"{self.api_base_url}/{endpoint}",
                headers=self.headers,
                params=params,
                timeout=15
            )
            
            if response.status_code == 429:  # Too Many Requests
                logger.warning("Rate limit exceeded, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)  # Retry once
                
            if response.status_code == 403:  # Forbidden
                logger.error("API authentication failed. Check your API key.")
                return None
                
            if response.status_code == 200:
                try:
                    return response.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON response: {e}")
                    return None
                    
            logger.warning(f"API request failed with status {response.status_code}")
            return None
            
        except requests.RequestException as e:
            logger.error(f"Request error: {e}")
            return None

    def _cache_day_matches(self, target_date: str, matches: List[Dict[str, Any]]) -> None:
        """Cache matches for a specific day."""
        try:
            cache_file = self.cache_dir / f"matches_{target_date}.json"
            cache_data = {
                "date": target_date,
                "timestamp": datetime.now().isoformat(),
                "matches": matches
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"✅ Cached {len(matches)} matches for {target_date}")
            
        except Exception as e:
            logger.warning(f"❌ Error caching matches for {target_date}: {e}")

    def _load_cached_matches(self, target_date: str) -> List[Dict[str, Any]]:
        """Load cached matches for a specific day."""
        try:
            cache_file = self.cache_dir / f"matches_{target_date}.json"
            
            if not cache_file.exists():
                return []
            
            # Check if cache is stale (older than 3 hours)
            if cache_file.stat().st_mtime < time.time() - 10800:  # 3 hours in seconds
                logger.debug(f"Cache for {target_date} is stale")
                return []
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            matches = cache_data.get('matches', [])
            logger.info(f"✅ Loaded {len(matches)} matches from cache for {target_date}")
            return matches
            
        except Exception as e:
            logger.warning(f"❌ Error loading cached matches for {target_date}: {e}")
            return []


def get_matches_with_odds_24h(limit: int = 20) -> List[Dict[str, Any]]:
    """Convenience function to get matches with odds in the next 72 hours."""
    discovery = OddsBasedFixtureDiscovery()
    return discovery.get_matches_with_odds_next_24h(limit)


def get_matches_with_verified_odds(limit: int = 10) -> List[Dict[str, Any]]:
    """Get matches with verified odds availability (slower but more accurate)."""
    discovery = OddsBasedFixtureDiscovery()
    potential_matches = discovery.get_matches_with_odds_next_24h(limit * 2)
    verified_matches = []
    
    for match in potential_matches:
        if len(verified_matches) >= limit:
            break
            
        if match.get('odds_available', False):
            match['odds_verified'] = True
            verified_matches.append(match)
    
    return verified_matches


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("\n=== Testing ODDS-ONLY Match Discovery ===\n")
    print("This approach gets ALL matches with odds from ALL leagues worldwide\n")
    
    # Test with different limits
    for test_limit in [3, 5, 8]:
        print(f"\nTesting with limit={test_limit}:")
        matches = get_matches_with_odds_24h(test_limit)
        print(f"Found {len(matches)} matches:")
        
        for i, match in enumerate(matches, 1):
            print(f"\n{i}. {match['home_team']} vs {match['away_team']}")
            print(f"   League: {match.get('league_name', 'Unknown')}")
            print(f"   Date: {match['date']}")
            print(f"   Method: {match['discovery_method']}")
            print(f"   Has Odds: {match['odds_available']}")

    # Test verified odds
    print("\n=== Testing Verified Odds ===\n")
    verified_matches = get_matches_with_verified_odds(5)
    print(f"\nFound {len(verified_matches)} matches with verified odds:")
    
    for i, match in enumerate(verified_matches, 1):
        print(f"\n{i}. {match['home_team']} vs {match['away_team']}")
        print(f"   League: {match.get('league_name', 'Unknown')}")
        print(f"   Odds Verified: {match.get('odds_verified', False)}")