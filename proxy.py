# proxy.py - will handle API calls through the new FootballAPI class
from football_api import FootballAPI
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

def get_upcoming_matches(days_ahead: int = 3, league_id: int = None) -> List[Dict[str, Any]]:
    """Get upcoming matches for next N days."""
    try:
        api = FootballAPI()
        
        # Calculate date range
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Format dates for API request
        from_date = today.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Build params
        params = {
            'from': from_date,
            'to': to_date,
            'status': 'NS'  # Not Started
        }
        
        if league_id:
            params['league'] = league_id
            
        # Get fixtures
        fixtures = api.get_fixtures(params)
        
        if not fixtures:
            logger.warning(f"No fixtures found for next {days_ahead} days")
            return []
            
        # Format response
        formatted_fixtures = []
        for fixture in fixtures:
            try:
                fixture_data = fixture.get('fixture', {})
                teams = fixture.get('teams', {})
                
                formatted_fixture = {
                    'fixture_id': fixture_data.get('id'),
                    'date': fixture_data.get('date'),
                    'home_team': teams.get('home', {}).get('name'),
                    'away_team': teams.get('away', {}).get('name'),
                    'home_team_id': teams.get('home', {}).get('id'),
                    'away_team_id': teams.get('away', {}).get('id'),
                    'league_id': fixture.get('league', {}).get('id'),
                    'league_name': fixture.get('league', {}).get('name')
                }
                
                # Only add complete fixtures
                if all(formatted_fixture.values()):
                    formatted_fixtures.append(formatted_fixture)
                    
            except Exception as e:
                logger.warning(f"Error formatting fixture: {e}")
                continue
                
        return formatted_fixtures
        
    except Exception as e:
        logger.error(f"Error getting upcoming fixtures: {e}")
        return []
