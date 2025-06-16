"""
League historical data analysis and processing.
Handles historical data aggregation and analysis for league calibration.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LeagueHistoricalStats:
    """Historical statistics for a league"""
    league_id: int
    name: str
    matches_analyzed: int
    date_range: Tuple[str, str]
    avg_goals_per_game: float
    home_goals_avg: float
    away_goals_avg: float
    btts_percentage: float
    over_2_5_percentage: float
    clean_sheets_home_pct: float
    clean_sheets_away_pct: float
    season_patterns: Dict[str, Any]

class LeagueHistoricalAnalyzer:
    def __init__(self, data_path: str = 'data/historical_matches'):
        """
        Initialize league historical analyzer.
        
        Args:
            data_path: Path to historical match data directory
        """
        self.data_path = data_path
        self.cache = {}
        
    def analyze_league(
        self,
        league_id: int,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_matches: int = 100
    ) -> Optional[LeagueHistoricalStats]:
        """
        Analyze historical data for a specific league.
        
        Args:
            league_id: League ID to analyze
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
            min_matches: Minimum matches required for analysis
            
        Returns:
            LeagueHistoricalStats if successful, None otherwise
        """
        try:
            # Check cache first
            cache_key = f"{league_id}_{start_date}_{end_date}"
            if cache_key in self.cache:
                return self.cache[cache_key]

            # Load matches data
            matches = self._load_matches_data(league_id, start_date, end_date)
            if len(matches) < min_matches:
                logger.warning(
                    f"Insufficient matches ({len(matches)}) for league {league_id}. "
                    f"Minimum required: {min_matches}"
                )
                return None
                
            # Calculate basic stats
            total_matches = len(matches)
            total_goals = sum(m['home_goals'] + m['away_goals'] for m in matches)
            home_goals = sum(m['home_goals'] for m in matches)
            away_goals = sum(m['away_goals'] for m in matches)
            
            btts_matches = sum(1 for m in matches 
                             if m['home_goals'] > 0 and m['away_goals'] > 0)
            over_matches = sum(1 for m in matches 
                             if m['home_goals'] + m['away_goals'] > 2.5)
            clean_sheets_home = sum(1 for m in matches if m['away_goals'] == 0)
            clean_sheets_away = sum(1 for m in matches if m['home_goals'] == 0)
            
            # Analyze seasonal patterns
            season_patterns = self._analyze_seasonal_patterns(matches)
            
            # Create stats object
            stats = LeagueHistoricalStats(
                league_id=league_id,
                name=matches[0].get('league_name', ''),
                matches_analyzed=total_matches,
                date_range=(matches[0]['date'], matches[-1]['date']),
                avg_goals_per_game=round(total_goals / total_matches, 3),
                home_goals_avg=round(home_goals / total_matches, 3),
                away_goals_avg=round(away_goals / total_matches, 3),
                btts_percentage=round(btts_matches * 100 / total_matches, 2),
                over_2_5_percentage=round(over_matches * 100 / total_matches, 2),
                clean_sheets_home_pct=round(clean_sheets_home * 100 / total_matches, 2),
                clean_sheets_away_pct=round(clean_sheets_away * 100 / total_matches, 2),
                season_patterns=season_patterns
            )
            
            # Cache results
            self.cache[cache_key] = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error analyzing league {league_id}: {e}")
            return None
            
    def _load_matches_data(
        self,
        league_id: int,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Load and filter historical matches data"""
        try:
            # Attempt to load from JSON file
            league_file = os.path.join(self.data_path, f"league_{league_id}.json")
            if not os.path.exists(league_file):
                logger.error(f"No historical data file found for league {league_id}")
                return []
                
            with open(league_file, 'r') as f:
                matches = json.load(f)
                
            # Filter by date if specified
            if start_date:
                matches = [m for m in matches if m['date'] >= start_date]
            if end_date:
                matches = [m for m in matches if m['date'] <= end_date]
                
            # Sort by date
            matches.sort(key=lambda x: x['date'])
            return matches
            
        except Exception as e:
            logger.error(f"Error loading matches data for league {league_id}: {e}")
            return []
            
    def _analyze_seasonal_patterns(
        self,
        matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze seasonal patterns in match data"""
        try:
            # Group matches by season month (1-10, assuming August-May season)
            monthly_stats = {}
            for match in matches:
                date = datetime.strptime(match['date'], '%Y-%m-%d')
                month = date.month
                season_month = (month - 8) % 12 + 1
                
                if season_month not in monthly_stats:
                    monthly_stats[season_month] = {
                        'matches': 0,
                        'total_goals': 0,
                        'home_goals': 0,
                        'away_goals': 0,
                        'btts': 0,
                        'over_2_5': 0
                    }
                    
                stats = monthly_stats[season_month]
                stats['matches'] += 1
                stats['total_goals'] += match['home_goals'] + match['away_goals']
                stats['home_goals'] += match['home_goals']
                stats['away_goals'] += match['away_goals']
                stats['btts'] += 1 if match['home_goals'] > 0 and match['away_goals'] > 0 else 0
                stats['over_2_5'] += 1 if match['home_goals'] + match['away_goals'] > 2.5 else 0
                
            # Calculate averages for each phase
            early_season = {'goals': 0, 'matches': 0}
            mid_season = {'goals': 0, 'matches': 0}
            end_season = {'goals': 0, 'matches': 0}
            
            for month, stats in monthly_stats.items():
                avg_goals = stats['total_goals'] / stats['matches']
                if month <= 3:  # Early season (Aug-Oct)
                    early_season['goals'] += stats['total_goals']
                    early_season['matches'] += stats['matches']
                elif month <= 7:  # Mid season (Nov-Mar)
                    mid_season['goals'] += stats['total_goals']
                    mid_season['matches'] += stats['matches']
                else:  # End season (Apr-May)
                    end_season['goals'] += stats['total_goals']
                    end_season['matches'] += stats['matches']
                    
            return {
                'early_season_goals': round(early_season['goals'] / early_season['matches'], 3),
                'mid_season_goals': round(mid_season['goals'] / mid_season['matches'], 3),
                'end_season_goals': round(end_season['goals'] / end_season['matches'], 3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing seasonal patterns: {e}")
            return {
                'early_season_goals': 2.6,
                'mid_season_goals': 2.6,
                'end_season_goals': 2.6
            }
