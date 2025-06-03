"""
Improved ELO Rating System with automatic team addition

This module enhances the existing ELO rating system to automatically add new teams
when they are first encountered, rather than requiring manual addition of teams.
"""

import logging
import os
import json
import random
import statistics
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union
from collections import defaultdict
from dataclasses import dataclass, field, asdict
import time

from edge_case_handler import EdgeCaseHandler, EdgeCaseConfig

# Import advanced monitoring systems
try:
    from elo_performance_dashboard import PerformanceDashboard, SystemHealthMetrics, EloPerformanceMetrics
    from elo_alert_system import AlertManager, AlertSeverity, AlertType, NotificationConfig
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    # Note: logger will be defined later, this warning will be logged during initialization

# Import database integration adapter
try:
    from elo_database_integration import DatabaseIntegrationAdapter, DatabaseConfig, DatabaseType
    from elo_database_backend import TeamRating, MatchResult, LeagueInfo
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False

# Performance metrics data structure
@dataclass
class PerformanceMetrics:
    total_operations: int = 0
    new_teams_added: int = 0
    rating_adjustments: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    operation_times: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list)) 
    rating_distributions: Dict[str, Dict[str, float]] = field(default_factory=lambda: defaultdict(dict))
    cache_hits: int = 0
    cache_misses: int = 0
    error_count: int = 0  # Added for enhanced monitoring
    matches_processed: int = 0  # Added for enhanced monitoring

# Configure enhanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add file handler for detailed logging
if not os.path.exists('logs'):
    os.makedirs('logs')
fh = logging.FileHandler('logs/elo_performance.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Team ELO information cache to store team metadata
TEAM_INFO_CACHE: Dict[int, Dict[str, Any]] = {}

# Global performance metrics
METRICS = PerformanceMetrics()

class AutoUpdatingEloRating:
    """
    ELO rating system that automatically adds new teams and intelligently
    assigns initial ratings based on league context.
    
    Phase 2 enhancements:
    - Advanced performance monitoring and dashboard
    - Real-time alerting system
    - Enhanced metrics collection
    - System health monitoring
    """
    def __init__(self, ratings_file: str = 'data/team_elo_ratings.json', 
                 edge_case_config: Optional[EdgeCaseConfig] = None,
                 enable_monitoring: bool = True,
                 monitoring_interval: int = 300,  # 5 minutes default
                 use_database: bool = True,
                 database_config: Optional[DatabaseConfig] = None):
        """Initialize the auto-updating ELO rating system with advanced monitoring and database backend"""
        self.start_time = time.time()
        
        # Import here to avoid circular imports
        from team_elo_rating import EloRating
        from elo_database_integration import DatabaseIntegrationAdapter
        
        # Initialize database integration adapter
        self.use_database: bool = use_database and DATABASE_AVAILABLE
        self.database_adapter: Optional[DatabaseIntegrationAdapter] = None
        
        if self.use_database:
            try:
                self.database_adapter = DatabaseIntegrationAdapter(
                    use_database=True,
                    json_file_path=ratings_file,
                    database_config=database_config
                )
                logger.info("Database integration adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database adapter: {e}")
                self.use_database = False
          # Use the existing ELO system as a base
        self.elo_rating = EloRating(ratings_file=ratings_file)
        self.ratings_file = ratings_file
        
        # Load ratings from database if available
        if self.use_database and self.database_adapter:
            self._load_ratings_from_database()

        # Initialize edge case handler
        self.edge_case_handler = EdgeCaseHandler(config=edge_case_config)
        
        # Initialize advanced monitoring systems
        self.monitoring_enabled = enable_monitoring and MONITORING_AVAILABLE
        self.performance_dashboard = None
        self.alert_manager = None
        
        if self.monitoring_enabled:
            try:
                # Initialize performance dashboard
                self.performance_dashboard = PerformanceDashboard()
                
                # Initialize alert manager with default configuration
                notification_config = NotificationConfig(
                    email_enabled=False  # Can be configured later
                )
                self.alert_manager = AlertManager(notification_config=notification_config)
                
                # Start monitoring in background
                self.performance_dashboard.start_monitoring(self, monitoring_interval)
                
                logger.info("Advanced monitoring systems initialized and started")
                
            except Exception as e:
                logger.error(f"Failed to initialize monitoring systems: {e}")
                self.monitoring_enabled = False
        
        # League rating tiers for better initial ratings
        # Maps league_id to approximate average ELO based on UEFA coefficients, FIFA rankings, and historical performance
        self.league_tiers = {
            # === TIER 1: ELITE LEAGUES (1540-1580) ===
            # Top 5 European leagues + Champions League
            39: 1580,   # Premier League (England) - UEFA Coef: 1st
            140: 1565,  # La Liga (Spain) - UEFA Coef: 2nd  
            135: 1555,  # Serie A (Italy) - UEFA Coef: 3rd
            78: 1550,   # Bundesliga (Germany) - UEFA Coef: 4th
            61: 1540,   # Ligue 1 (France) - UEFA Coef: 5th
            2: 1590,    # UEFA Champions League - Elite competition
            
            # === TIER 2: STRONG EUROPEAN LEAGUES (1500-1530) ===
            88: 1520,   # Eredivisie (Netherlands) - UEFA Coef: 6th
            94: 1515,   # Primeira Liga (Portugal) - UEFA Coef: 7th
            203: 1510,  # Süper Lig (Turkey) - UEFA Coef: 8th
            144: 1505,  # Belgian Pro League - UEFA Coef: 9th
            218: 1500,  # Austrian Bundesliga - UEFA Coef: 10th
            3: 1525,    # UEFA Europa League - Strong competition
            
            # === TIER 3: COMPETITIVE EUROPEAN LEAGUES (1470-1495) ===
            207: 1495,  # Swiss Super League - UEFA Coef: 11th
            179: 1490,  # Scottish Premiership - UEFA Coef: 12th
            235: 1485,  # Russian Premier League - UEFA Coef: 13th (suspended)
            210: 1485,  # Croatian HNL - UEFA Coef: 14th
            119: 1480,  # Danish Superliga - UEFA Coef: 15th
            106: 1480,  # Ekstraklasa (Poland) - UEFA Coef: 16th
            197: 1475,  # Super League (Greece) - UEFA Coef: 17th
            283: 1475,  # Liga I (Romania) - UEFA Coef: 18th
            345: 1470,  # Czech First League - UEFA Coef: 19th
            848: 1485,  # UEFA Conference League - Third tier European competition
            
            # === TIER 4: SOLID EUROPEAN LEAGUES (1440-1465) ===
            113: 1465,  # Allsvenskan (Sweden) - UEFA Coef: 20th
            103: 1460,  # Eliteserien (Norway) - UEFA Coef: 21st
            333: 1460,  # Ukrainian Premier League - UEFA Coef: 22nd (war-affected)
            372: 1455,  # Serbian SuperLiga - UEFA Coef: 23rd
            271: 1455,  # Hungarian NB I - UEFA Coef: 24th
            330: 1450,  # Slovenian PrvaLiga - UEFA Coef: 25th
            325: 1445,  # Latvian Higher League
            326: 1445,  # Meistriliiga (Estonia)
            361: 1450,  # Erovnuli Liga (Georgia)
            350: 1440,  # Macedonian First League
            355: 1440,  # Albanian Superliga
            
            # === TIER 5: EMERGING EUROPEAN LEAGUES (1410-1435) ===
            367: 1435,  # Bulgarian First League
            385: 1430,  # Lithuanian A Lyga
            395: 1425,  # Belarusian Premier League
            340: 1420,  # Moldovan National Division
            375: 1415,  # Montenegrin First League
            380: 1410,  # Armenian Premier League
            
            # === TIER 6: MAJOR SOUTH AMERICAN LEAGUES (1465-1485) ===
            71: 1485,   # Brasileirão (Brazil) - CONMEBOL strongest
            128: 1480,  # Argentine Primera División - Historic powerhouse
            286: 1475,  # Colombian Primera A - Strong league
            294: 1470,  # Chilean Primera División
            298: 1465,  # Ecuadorian Serie A
            309: 1460,  # Peruvian Primera División
            315: 1455,  # Venezuelan Primera División
            305: 1450,  # Uruguayan Primera División
            320: 1445,  # Paraguayan División Profesional
            312: 1440,  # Bolivian División Profesional
            
            # === TIER 7: NORTH AMERICAN LEAGUES (1450-1480) ===
            253: 1480,  # MLS (USA & Canada) - Growing rapidly
            262: 1470,  # Liga MX (Mexico) - Strong CONCACAF
            274: 1450,  # Liga Nacional (Guatemala)
            277: 1445,  # Primera División (Costa Rica)
            280: 1440,  # Liga Nacional (Honduras)
            285: 1435,  # Primera División (El Salvador)
            290: 1430,  # Liga Panameña
            295: 1425,  # Primera División (Nicaragua)
            
            # === TIER 8: MAJOR ASIAN LEAGUES (1440-1470) ===
            98: 1470,   # J1 League (Japan) - AFC strongest
            292: 1465,  # K League 1 (South Korea) - Strong AFC
            169: 1460,  # Chinese Super League - Investment league
            302: 1455,  # A-League (Australia) - Growing league
            316: 1450,  # Indian Super League - Emerging market
            321: 1445,  # Thai League 1 - Southeast Asia leader
            327: 1440,  # V.League 1 (Vietnam)
            332: 1435,  # Malaysia Super League
            337: 1430,  # Singapore Premier League
            342: 1425,  # Indonesian Liga 1
            
            # === TIER 9: MIDDLE EASTERN LEAGUES (1430-1460) ===
            271: 1460,  # Israeli Premier League - Strong Middle East
            392: 1455,  # Saudi Pro League - Heavy investment
            383: 1450,  # UAE Pro League - Gulf powerhouse
            388: 1445,  # Qatar Stars League - World Cup host
            393: 1440,  # Kuwait Premier League
            398: 1435,  # Iraqi Premier League
            403: 1430,  # Lebanese Premier League
            
            # === TIER 10: AFRICAN LEAGUES (1420-1450) ===
            233: 1450,  # Egyptian Premier League - CAF leader
            384: 1445,  # South African Premier Division
            390: 1440,  # Moroccan Botola Pro
            395: 1435,  # Tunisian Ligue Professionnelle 1
            400: 1430,  # Algerian Ligue Professionnelle 1
            405: 1425,  # Nigerian Professional Football League
            410: 1420,  # Ghanaian Premier League
            415: 1415,  # Kenyan Premier League
            
            # === TIER 11: SMALLER EUROPEAN & OTHER LEAGUES (1400-1430) ===
            358: 1430,  # Premier League Bosnia - Regional league
            420: 1425,  # Finnish Veikkausliiga
            425: 1420,  # Icelandic Úrvalsdeild
            430: 1415,  # Faroese Effodeildin
            435: 1410,  # Andorran Primera Divisió
            440: 1405,  # San Marinese Campionato
            445: 1400,  # Gibraltarian National League
        }

        # Default league tier value for unknown leagues (midpoint between tiers)
        self.default_league_elo = 1465
        
        # Performance monitoring
        self.metrics = METRICS
        self.last_save_time = time.time()
        self.operations_since_save = 0
        
        # Load existing metrics if available
        self._load_metrics()
          # Update startup metrics
        ratings = getattr(self.elo_rating.elo_system, 'ratings', {})
        logger.info(f"ELO system initialized with {len(ratings)} teams")
    
    def _track_operation_time(self, operation_name: str, start_time: float) -> None:
        """Track the execution time of an operation"""
        duration = time.time() - start_time
        self.metrics.operation_times[operation_name].append(duration)
        logger.debug(f"{operation_name} took {duration:.3f} seconds")

    def _track_rating_adjustment(self, team_id: int, old_rating: float, new_rating: float) -> None:
        """Track rating adjustments for analysis"""
        adjustment = new_rating - old_rating
        self.metrics.rating_adjustments[str(team_id)].append(adjustment)
        logger.debug(f"Team {team_id} rating adjusted by {adjustment:.1f} points")    
    
    def _update_rating_distributions(self) -> None:
        """Update rating distribution statistics"""
        ratings_dict = getattr(self.elo_rating.elo_system, 'ratings', {})
        ratings = list(ratings_dict.values())
        ratings = list(ratings_dict.values())
        if ratings:
            self.metrics.rating_distributions[datetime.now().isoformat()] = {
                'mean': statistics.mean(ratings),
                'median': statistics.median(ratings),
                'std_dev': statistics.stdev(ratings) if len(ratings) > 1 else 0,
                'min': min(ratings),
                'max': max(ratings)
            }

    def _save_metrics(self) -> None:
        """Save performance metrics to file"""
        metrics_file = os.path.join(os.path.dirname(self.ratings_file), 'elo_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(asdict(self.metrics), f, indent=2)
        logger.info("Performance metrics saved to file")

    def _load_metrics(self) -> None:
        """Load performance metrics from file"""
        metrics_file = os.path.join(os.path.dirname(self.ratings_file), 'elo_metrics.json')
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = PerformanceMetrics(**data)
                logger.info("Performance metrics loaded from file")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")

    def get_team_rating(self, team_id: int, league_id: Optional[int] = None) -> float:
        """Get a team's ELO rating, automatically adding new teams with intelligent defaults"""
        start = time.time()
        
        try:
            # Try to get cached rating
            if team_id in TEAM_INFO_CACHE:
                METRICS.cache_hits += 1
                rating = TEAM_INFO_CACHE[team_id]["rating"]
                last_match = TEAM_INFO_CACHE[team_id].get("last_match", None)
                matches_played = TEAM_INFO_CACHE[team_id].get("matches_played", 0)
                
                # Calculate days since last match if available
                days_inactive = 0
                if last_match:
                    days_inactive = (datetime.now() - datetime.strptime(last_match, "%Y-%m-%d")).days
                
                # Get rating reliability
                reliability = self.edge_case_handler.calculate_rating_reliability(
                    matches_played=matches_played,
                    days_inactive=days_inactive
                )
                
                # Apply reliability adjustments
                if reliability < 1.0:
                    logger.info(f"Rating reliability for team {team_id} is {reliability:.2f}")
                    # Blend with default rating based on reliability
                    default_rating = self._get_default_rating(league_id)
                    rating = (rating * reliability) + (default_rating * (1 - reliability))
                
                return self.edge_case_handler.validate_rating(rating)
            
            METRICS.cache_misses += 1
            
            # Try to get rating from base system
            try:
                rating = self.elo_rating.get_team_rating(team_id)
            except Exception as e:
                logger.info(f"Team {team_id} not found in base system, adding automatically")
                rating = self._add_new_team(team_id, league_id)
                METRICS.new_teams_added += 1
            
            # Validate and store rating
            rating = self.edge_case_handler.validate_rating(rating)
            TEAM_INFO_CACHE[team_id] = {
                "rating": rating,
                "last_update": datetime.now().isoformat(),
                "matches_played": 0
            }
            
            return rating
            
        finally:
            # Record operation time
            end = time.time()
            METRICS.operation_times["get_rating"].append(end - start)
            METRICS.total_operations += 1
    
    def _calculate_smart_default_rating(self, team_id: int, league_id: Optional[int] = None) -> float:
        """
        Calculate an intelligent default rating for a new team based on context.
        
        Args:
            team_id: The team ID
            league_id: Optional league ID for more accurate default rating
            
        Returns:
            A realistic starting ELO rating
        """
        # Start with base default
        base_rating = self.elo_rating.elo_system.initial_rating  # 1500.0
        
        # If we have league information, use it for better defaults
        if league_id is not None:
            # Check if we have a predefined tier for this league
            if league_id in self.league_tiers:
                base_rating = self.league_tiers[league_id]
            else:
                # Use our default league value for unknown leagues
                base_rating = self.default_league_elo
            
            # Calculate standard deviation based on league tier
            # Lower-tier leagues typically have more variance
            if base_rating >= 1540:  # Top tier leagues
                std_dev = 15.0  # Tighter spread for top leagues
            elif base_rating >= 1500:  # Second tier
                std_dev = 20.0  # More variance for mid-tier
            elif base_rating >= 1470:  # Third tier
                std_dev = 25.0  # Even more variance
            else:  # Fourth tier or lower
                std_dev = 30.0  # Highest variance for lower tiers
                
            # Add controlled random variation based on league tier
            import random
            random.seed(team_id)  # Use team ID as seed for consistency
            variation = random.gauss(0, std_dev)
            
            # Limit the maximum variation to 2 standard deviations
            variation = max(min(variation, 2 * std_dev), -2 * std_dev)
            
            # Apply the variation to base rating
            base_rating += variation
            
            # Ensure rating stays within reasonable bounds for the league tier
            # Don't let new teams start too high or too low relative to their league
            min_rating = base_rating - 3 * std_dev
            max_rating = base_rating + 3 * std_dev
            base_rating = max(min(base_rating, max_rating), min_rating)
            
        # Round to reasonable precision (1 decimal place)
        return round(base_rating, 1)
    def _save_ratings(self) -> None:
        """Save the updated ratings using database adapter or fallback to JSON"""
        try:
            ratings = {}
            if hasattr(self.elo_rating.elo_system, 'ratings'):
                # Convert int keys to str for database compatibility
                ratings = {str(k): float(v) for k, v in self.elo_rating.elo_system.ratings.items()}
            
            if self.use_database and self.database_adapter:
                # Use database adapter for saving
                success = self.database_adapter.save_team_ratings(ratings)
                if success:
                    logger.info(f"Saved {len(ratings)} team ratings to database")
                else:
                    logger.error("Failed to save ratings to database, falling back to JSON")
                    self._save_ratings_json(ratings)
            else:
                # Fallback to JSON file
                self._save_ratings_json(ratings)
                
        except Exception as e:
            logger.error(f"Error saving ratings: {e}")
            # Final fallback to JSON
            try:
                ratings = {str(k): float(v) for k, v in getattr(self.elo_rating.elo_system, 'ratings', {}).items()}
                self._save_ratings_json(ratings)
            except Exception as fallback_error:
                logger.error(f"Fallback save also failed: {fallback_error}")
    
    def _save_ratings_json(self, ratings: Dict[str, Any]) -> None:
        """Fallback method to save ratings to JSON file"""
        try:
            os.makedirs(os.path.dirname(self.ratings_file), exist_ok=True)
            
            data = {
                'ratings': ratings,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
            with open(self.ratings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(ratings)} team ratings to {self.ratings_file} (JSON fallback)")
        except Exception as e:
            logger.error(f"Error saving ratings to JSON: {e}")
    
    def get_match_probabilities(self, home_team_id: Optional[int], away_team_id: Optional[int], 
                               league_id: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Get win, draw, loss probabilities for a match with auto-team-addition.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID for context
            
        Returns:
            Tuple of (win_prob, draw_prob, loss_prob)
        """
        safe_home_id = home_team_id if home_team_id is not None else 0
        safe_away_id = away_team_id if away_team_id is not None else 0
        
        # Get ratings, adding teams if needed
        home_rating = self.get_team_rating(safe_home_id, league_id)
        away_rating = self.get_team_rating(safe_away_id, league_id)
        
        # Apply league-specific home advantage if available
        if league_id is not None and league_id in self.elo_rating.elo_system.league_adjustments:
            home_adv_mult = self.elo_rating.elo_system.league_adjustments[league_id].get("home_advantage_mult", 1.0)
            effective_home_rating = home_rating + (self.elo_rating.elo_system.home_advantage * home_adv_mult)
        else:
            effective_home_rating = home_rating + self.elo_rating.elo_system.home_advantage
        
        # Calculate win probability using standard Elo formula
        win_prob = 1.0 / (1.0 + 10 ** ((away_rating - effective_home_rating) / 400.0))
        
        # Draw probability is highest when teams are evenly matched
        draw_prob = 0.32 - (abs(win_prob - 0.5) * 0.2)
        
        # Loss probability (remaining probability)
        loss_prob = 1.0 - win_prob - draw_prob
        
        # Ensure probabilities are in valid range
        win_prob = max(0.01, min(0.99, win_prob))
        draw_prob = max(0.01, min(0.98, draw_prob))
        loss_prob = max(0.01, min(0.99, loss_prob))
        
        # Normalize to ensure they sum to 1
        total = win_prob + draw_prob + loss_prob
        win_prob /= total
        draw_prob /= total
        loss_prob /= total
        
        return win_prob, draw_prob, loss_prob
    
    def get_expected_goal_diff(self, home_team_id: Optional[int], away_team_id: Optional[int],
                              league_id: Optional[int] = None) -> float:
        """
        Get expected goal difference based on ELO ratings with auto-team-addition.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: Optional league ID for context
            
        Returns:
            Expected goal difference (positive favors home team)
        """
        safe_home_id = home_team_id if home_team_id is not None else 0
        safe_away_id = away_team_id if away_team_id is not None else 0
        
        # Get ratings, adding teams if needed
        home_rating = self.get_team_rating(safe_home_id, league_id)
        away_rating = self.get_team_rating(safe_away_id, league_id)
        
        # Raw Elo difference
        raw_elo_diff = home_rating - away_rating
        
        # Convert to expected goal difference (empirical conversion factor)
        # Each 100 Elo points is roughly 0.15 goals difference
        expected_goal_diff = raw_elo_diff * 0.15 / 100.0
        
        return expected_goal_diff

    def force_load_ratings(self) -> None:
        """Force reload of ratings from database or file"""
        if self.use_database and self.database_adapter:
            self._load_ratings_from_database()
        else:
            self.elo_rating.force_load_ratings()

    def _load_ratings_from_database(self):
        """Load ratings from database and convert to proper format"""
        if self.database_adapter:
            try:
                ratings = self.database_adapter.get_all_ratings()
                # Convert str keys to int for internal use
                converted_ratings = {int(k): float(v) for k, v in ratings.items()}
                self.elo_rating.elo_system.ratings = converted_ratings  # Access through elo_system
                logger.info("Successfully loaded ratings from database")
            except Exception as e:
                logger.error(f"Failed to load ratings from database: {e}")

    def save_ratings(self):
        """Save ratings to both JSON and database if enabled"""
        if self.use_database and self.database_adapter:
            try:
                # Convert int keys to str for database storage
                ratings = {str(k): float(v) for k, v in self.elo_rating.elo_system.ratings.items()}
                success = self.database_adapter.save_team_ratings(ratings)
                if not success:
                    logger.error("Failed to save ratings to database")
            except Exception as e:
                logger.error(f"Error saving to database: {e}")

        # Always save JSON backup
        ratings_data = {
            "ratings": {str(k): float(v) for k, v in self.elo_rating.elo_system.ratings.items()},
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
        os.makedirs(os.path.dirname(self.ratings_file), exist_ok=True)
        with open(self.ratings_file, 'w') as f:
            json.dump(ratings_data, f, indent=2)

    def get_team_stats(self, team_id: int, limit: Optional[int] = None) -> Dict[str, Any]:
        """Get team statistics including match history"""
        stats = {
            'rating': self.elo_rating.elo_system.ratings.get(team_id, self.elo_rating.elo_system.initial_rating),
            'wins': getattr(self.elo_rating, 'wins', {}).get(team_id, 0),
            'losses': getattr(self.elo_rating, 'losses', {}).get(team_id, 0),
            'draws': getattr(self.elo_rating, 'draws', {}).get(team_id, 0)
        }
        
        if self.use_database and self.database_adapter:
            try:
                db_stats = self.database_adapter.get_team_metadata(team_id)
                stats.update(db_stats)
            except Exception as e:
                logger.error(f"Error getting team stats: {e}")
                
        return stats

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system-wide statistics"""
        stats = {
            'total_teams': len(self.elo_rating.elo_system.ratings),
            'average_rating': statistics.mean(self.elo_rating.elo_system.ratings.values()) if self.elo_rating.elo_system.ratings else 1500.0,
            'last_updated': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Add database statistics if available
        if self.use_database and self.database_adapter:
            try:
                db_stats = self.database_adapter.get_database_statistics()
                stats.update(db_stats)
            except Exception as e:
                logger.error(f"Error getting database stats: {e}")
                
        return stats
    
    def get_league_tier(self, league_id: Optional[int]) -> str:
        """
        Get the tier classification of a league based on its ELO rating.
        
        Args:
            league_id: The league ID to classify
            
        Returns:
            A string representing the league tier ("Top", "Second", "Third", "Fourth", "Unknown")
        """
        if league_id is None or league_id not in self.league_tiers:
            return "Unknown"
        
        league_rating = self.league_tiers[league_id]
        if league_rating >= 1540:
            return "Top"
        elif league_rating >= 1500:
            return "Second"
        elif league_rating >= 1470:
            return "Third"
        else:
            return "Fourth"
    
    def _get_default_rating(self, league_id: Optional[int] = None) -> float:
        """Get a sensible default rating for a new team based on league context"""
        try:
            STANDARD_DEFAULT = 1500.0
            RATING_VARIANCE = 50.0  # Small random variance to avoid identical ratings
            
            # If no league context, return standard default with small random adjustment
            if not league_id:
                return STANDARD_DEFAULT + random.uniform(-RATING_VARIANCE/2, RATING_VARIANCE/2)
                
            # Get league tier if available
            tier = self.get_league_tier(league_id)
            
            # Adjust base rating by tier
            tier_adjustments = {
                'top': 100.0,
                'second': 50.0,
                'third': 0.0,
                'fourth': -50.0,
                'other': -100.0
            }
            
            base_rating = STANDARD_DEFAULT - 50.0  # Start slightly below standard
            tier_bonus = tier_adjustments.get(tier, 0.0)
            
            # Calculate final rating with small random variance
            final_rating = base_rating + tier_bonus + random.uniform(-RATING_VARIANCE/2, RATING_VARIANCE/2)
            
            return min(1600.0, max(1400.0, final_rating))  # Constrain to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating default rating: {e}")
            return STANDARD_DEFAULT

    def _add_new_team(self, team_id: int, league_id: Optional[int] = None) -> float:
        """Add a new team to the system with an appropriate starting rating"""
        rating = self._get_default_rating(league_id)
        
        # Add to base system
        self.elo_rating.elo_system.set_rating(team_id, rating)
        
        # Add to cache with metadata
        TEAM_INFO_CACHE[team_id] = {
            "rating": rating,
            "added_date": datetime.now().isoformat(),
            "league_id": league_id,
            "matches_played": 0,
            "last_match": None
        }
        
        logger.info(f"Added new team {team_id} with initial rating {rating}")
        return rating

    def _save_match_result(self, match_data: Dict[str, Any], home_rating_before: float, 
                          away_rating_before: float, home_rating_after: float, 
                          away_rating_after: float) -> None:
        """Save match result to database if database integration is enabled"""
        if not (self.use_database and self.database_adapter):
            return
        
        try:
            # Extract match information
            match_id = match_data.get('match_id')
            if match_id is None:
                # Generate a match ID if not provided
                match_id = int(f"{match_data['home_team_id']}{match_data['away_team_id']}{int(time.time())}")
            
            match_date = match_data.get('match_date')
            if match_date is None:
                match_date = datetime.now()
            elif isinstance(match_date, str):
                match_date = datetime.fromisoformat(match_date)
            
            # Calculate K-factor used (approximate)
            rating_change = abs(home_rating_after - home_rating_before)
            k_factor = rating_change / max(0.1, abs(home_rating_before - away_rating_before) / 400.0) if rating_change > 0 else 32.0
            
            # Prepare match result dictionary
            match_result_dict = {
                'match_id': match_id,
                'home_team_id': match_data['home_team_id'],
                'away_team_id': match_data['away_team_id'],
                'home_score': match_data['home_score'],
                'away_score': match_data['away_score'],
                'match_date': match_date,
                'league_id': match_data.get('league_id', 0),
                'season': match_data.get('season', 2025),
                'home_rating_before': home_rating_before,
                'away_rating_before': away_rating_before,
                'home_rating_after': home_rating_after,
                'away_rating_after': away_rating_after,
                'k_factor': k_factor
            }
            
            # Save match result to database using dict argument
            success = self.database_adapter.save_match_result(match_result_dict)
            
            if success:
                logger.debug(f"Saved match result to database: {match_id}")
            else:
                logger.warning(f"Failed to save match result to database: {match_id}")
                
        except Exception as e:
            logger.error(f"Error saving match result to database: {e}")

    def update_ratings(self, match_data: Dict[str, Any]) -> Tuple[float, float]:
        """Update ratings for a pair of teams based on a match result"""
        start = time.time()
        
        try:
            home_id = match_data['home_team_id']
            away_id = match_data['away_team_id']
            home_goals = match_data['home_goals']
            away_goals = match_data['away_goals']
            match_importance = match_data.get('match_importance', 1.0)
            
            # Get ratings before update for database saving
            home_rating_before = self.get_team_rating(home_id, match_data.get('league_id'))
            away_rating_before = self.get_team_rating(away_id, match_data.get('league_id'))
            
            # Update ratings using underlying ELO system
            home_new, away_new = self.elo_rating.elo_system.update_ratings(
                home_id=home_id,
                away_id=away_id,
                home_goals=home_goals,
                away_goals=away_goals,
                match_importance=match_importance
            )
            
            # Save match result to database
            self._save_match_result(match_data, home_rating_before, away_rating_before, home_new, away_new)
            
            # Track metrics
            self._track_rating_adjustment(home_id, home_rating_before, home_new)
            self._track_rating_adjustment(away_id, away_rating_before, away_new)
            
            # Save updated ratings
            self._save_ratings()
            
            return home_new, away_new
            
        finally:
            # Record operation time
            end = time.time()
            METRICS.operation_times["update_rating"].append(end - start)
            METRICS.total_operations += 1
    
    def get_match_prediction(self, home_id: int, away_id: int, league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get match prediction with edge case handling and validation"""
        try:
            # Initialize prediction dictionary
            prediction = {}
            
            # Get ratings for both teams, automatically adding if needed
            home_rating = self.get_team_rating(home_id, league_id)
            away_rating = self.get_team_rating(away_id, league_id)
            
            # Calculate rating difference with home advantage
            HOME_ADVANTAGE = 35.0  # Standard home advantage in ELO points
            rating_diff = (home_rating + HOME_ADVANTAGE) - away_rating
            
            # Calculate win probability using standard ELO formula
            home_win_prob = 1.0 / (1.0 + 10 ** (-rating_diff / 400.0))
            
            # Calculate draw probability based on rating difference
            # Draw probability is highest when teams are evenly matched
            max_draw_prob = 0.32  # Maximum draw probability
            rating_diff_factor = abs(rating_diff) / 400.0  # Normalize rating difference
            draw_prob = max_draw_prob * (1.0 - min(1.0, rating_diff_factor))
            
            # Away win probability (remaining probability)
            away_win_prob = 1.0 - home_win_prob - draw_prob
            
            # Ensure probabilities are valid
            if away_win_prob < 0:
                draw_prob += away_win_prob  # Adjust draw prob if away prob negative
                away_win_prob = 0
                
            # Expected goal difference based on rating difference
            expected_goal_diff = rating_diff / 250.0  # Rough approximation
            
            # Base prediction values
            base_prediction = {
                'home_win_prob': round(home_win_prob, 3),
                'draw_prob': round(draw_prob, 3),
                'away_win_prob': round(away_win_prob, 3),
                'expected_goals_diff': round(expected_goal_diff, 2)
            }
            prediction.update(base_prediction)
            
            # Add ELO rating information
            prediction['elo_ratings'] = {
                'home_team_rating': round(home_rating, 1),
                'away_team_rating': round(away_rating, 1),
                'rating_diff': round(rating_diff, 1),
                'home_advantage': HOME_ADVANTAGE
            }
            
            # Add reliability metrics
            prediction['reliability'] = {
                'data_quality': 'high',
                'confidence': 'normal',
                'rating_age': 'current'
            }
            
            # Validate prediction if edge case handler is available
            if hasattr(self, 'edge_case_handler'):
                prediction, warnings = self.edge_case_handler.validate_prediction(prediction)
                if warnings:
                    prediction['warnings'] = warnings
        
            return prediction
        
        except Exception as e:
            logger.error(f"Error getting match prediction: {e}")
            return {
                'home_win_prob': 0.333,
                'draw_prob': 0.334,
                'away_win_prob': 0.333,
                'expected_goals_diff': 0.0,
                'error': str(e)
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health metrics and statistics"""
        try:
            total_time = time.time() - self.start_time
            
            # Calculate rating distribution statistics
            ratings = [info["rating"] for info in TEAM_INFO_CACHE.values()]
            if ratings:
                avg_rating = statistics.mean(ratings)
                std_rating = statistics.stdev(ratings)
                rating_range = (min(ratings), max(ratings))
            else:
                avg_rating = std_rating = 0
                rating_range = (0, 0)
            
            # Calculate operation timings
            avg_times = {
                op: statistics.mean(times) if times else 0
                for op, times in METRICS.operation_times.items()
            }
            
            # Calculate rating adjustment statistics
            adjustments = METRICS.rating_adjustments
            avg_adjustments = {
                k: statistics.mean(v) if v else 0
                for k, v in adjustments.items()
            }
            
            # Collect system health metrics
            health_metrics = {
                "uptime_seconds": total_time,
                "total_operations": METRICS.total_operations,
                "new_teams_added": METRICS.new_teams_added,
                "cache_performance": {
                    "hits": METRICS.cache_hits,
                    "misses": METRICS.cache_misses,
                    "hit_ratio": METRICS.cache_hits / (METRICS.cache_hits + METRICS.cache_misses) if (METRICS.cache_hits + METRICS.cache_misses) > 0 else 0
                },
                "rating_statistics": {
                    "average": avg_rating,
                    "std_dev": std_rating,
                    "range": rating_range
                },
                "operation_timing": {
                    "averages": avg_times
                },
                "rating_adjustments": {
                    "averages": avg_adjustments
                },
                "teams_in_cache": len(TEAM_INFO_CACHE)
            }
              # Send to monitoring dashboard if enabled
            if self.monitoring_enabled and self.performance_dashboard:
                # Collect ELO metrics and store both system and ELO metrics
                elo_metrics = self.performance_dashboard.collect_elo_metrics(self)
                system_metrics = self.performance_dashboard.collect_system_metrics()
                self.performance_dashboard.store_metrics(system_metrics, elo_metrics)
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error generating system health metrics: {e}")
            return {"error": str(e)}
        
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get database statistics for monitoring and health checks"""
        if not (self.use_database and self.database_adapter):
            return {'database_enabled': False, 'mode': 'json'}
        
        try:
            return self.database_adapter.get_database_statistics()
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'database_enabled': True, 'error': str(e)}
    def get_team_match_history(self, team_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent match history for a team from database"""
        if not (self.use_database and self.database_adapter):
            return []
        
        try:
            # Convert team_id to int for database adapter
            return self.database_adapter.get_team_match_history(int(team_id), limit)
        except Exception as e:
            logger.error(f"Error getting team match history: {e}")
            return []
    
    def create_backup(self) -> str:
        """Create a backup of the current data"""
        if self.use_database and self.database_adapter:
            try:
                return self.database_adapter.create_backup()
            except Exception as e:
                logger.error(f"Error creating database backup: {e}")
                return ""
        else:
            # Create backup of JSON file
            try:
                if os.path.exists(self.ratings_file):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = f"{self.ratings_file}.backup_{timestamp}"
                    import shutil
                    shutil.copy2(self.ratings_file, backup_path)
                    logger.info(f"JSON backup created: {backup_path}")
                    return backup_path
                else:
                    logger.warning("No JSON file to backup")
                    return ""
            except Exception as e:
                logger.error(f"Error creating JSON backup: {e}")
                return ""

    def close(self):
        """Clean up resources including database connections and monitoring threads"""
        try:
            # Close database adapter
            if self.use_database and self.database_adapter:
                self.database_adapter.close()
                logger.info("Database adapter closed")
            
            # Stop monitoring systems
            if self.monitoring_enabled:
                if self.performance_dashboard:
                    self.performance_dashboard.stop_monitoring()
                    logger.info("Performance monitoring stopped")
                
                if self.alert_manager:
                    # Alert manager doesn't have a stop method, but we can clean up
                    logger.info("Alert manager cleanup completed")
            
            logger.info("AutoUpdatingEloRating system cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup on object deletion"""
        try:
            self.close()
        except:
            pass  # Ignore errors in destructor

# Helper function for integration with existing code
def get_elo_data_with_auto_rating(home_team_id: Optional[int], away_team_id: Optional[int], 
                                league_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get complete ELO data for a match with automatic team addition.
    
    Args:
        home_team_id: Home team ID
        away_team_id: Away team ID
        league_id: League ID for context
        
    Returns:
        Dictionary with ratings, probabilities and expected goal diff
    """
    # Create our enhanced auto-rating system
    auto_elo = AutoUpdatingEloRating()
    
    # Safe handling of None values
    safe_home_id = home_team_id if home_team_id is not None else 0
    safe_away_id = away_team_id if away_team_id is not None else 0
    
    # Get ratings (this will automatically add teams if needed)
    home_elo = auto_elo.get_team_rating(safe_home_id, league_id)
    away_elo = auto_elo.get_team_rating(safe_away_id, league_id)
    elo_diff = home_elo - away_elo
    
    # Get probabilities
    win_prob, draw_prob, loss_prob = auto_elo.get_match_probabilities(
        safe_home_id, safe_away_id, league_id
    )
    
    # Get expected goal difference
    expected_goal_diff = auto_elo.get_expected_goal_diff(
        safe_home_id, safe_away_id, league_id
    )
    
    # Return results dictionary matching the original format
    return {
        'home_elo': round(home_elo, 1),
        'away_elo': round(away_elo, 1),
        'elo_diff': round(elo_diff, 1),
        'elo_win_probability': round(win_prob, 3),
        'elo_draw_probability': round(draw_prob, 3),
        'elo_loss_probability': round(loss_prob, 3),
        'elo_expected_goal_diff': round(expected_goal_diff, 2),
        'league_id': league_id
    }

class AutoUpdatingELO:
    """
    Complete Auto-Updating ELO System with Database Backend
    
    This is the main entry point for the enhanced ELO system that provides:
    - Automatic team addition
    - Database backend with SQLite/PostgreSQL support
    - Advanced monitoring and alerting
    - Performance metrics and health monitoring
    - Intelligent rating initialization
    - Match result tracking and history
    """
    
    def __init__(self, 
                 ratings_file: str = 'data/team_elo_ratings.json',
                 use_database: bool = True,
                 database_config: Optional[DatabaseConfig] = None,
                 enable_monitoring: bool = True,
                 monitoring_interval: int = 300,
                 alert_config: Optional[Dict[str, Any]] = None,
                 edge_case_config: Optional[EdgeCaseConfig] = None):
        """
        Initialize the complete Auto-Updating ELO system
        
        Args:
            ratings_file: Path to JSON ratings file (used as fallback or migration source)
            use_database: Whether to use database backend (recommended)
            database_config: Database configuration (uses SQLite default if None)
            enable_monitoring: Enable performance monitoring and alerting
            monitoring_interval: Monitoring check interval in seconds
            alert_config: Alert system configuration
            edge_case_config: Edge case handling configuration
        """
        logger.info("Initializing Auto-Updating ELO System with database backend...")
        
        # Initialize the core ELO system
        self.elo_system = AutoUpdatingEloRating(
            ratings_file=ratings_file,
            edge_case_config=edge_case_config,
            enable_monitoring=enable_monitoring,
            monitoring_interval=monitoring_interval,
            use_database=use_database,
            database_config=database_config
        )
        
        # Configure alerts if provided
        if alert_config and self.elo_system.alert_manager:
            self._configure_alerts(alert_config)
        
        logger.info("Auto-Updating ELO System initialized successfully")
    
    def _configure_alerts(self, alert_config: Dict[str, Any]):
        """Configure the alert system with user-provided settings"""
        try:
            if not self.elo_system.alert_manager:
                return
            
            # Update notification config
            if 'smtp_server' in alert_config:
                self.elo_system.alert_manager.notification_config.smtp_server = alert_config['smtp_server']
            if 'smtp_port' in alert_config:
                self.elo_system.alert_manager.notification_config.smtp_port = alert_config['smtp_port']
            if 'email_user' in alert_config:
                self.elo_system.alert_manager.notification_config.smtp_username = alert_config['email_user']
            if 'email_password' in alert_config:
                self.elo_system.alert_manager.notification_config.smtp_password = alert_config['email_password']
            if 'recipient_email' in alert_config:
                self.elo_system.alert_manager.notification_config.email_recipients = [alert_config['recipient_email']]
                self.elo_system.alert_manager.notification_config.email_enabled = True
            
            logger.info("Alert system configured successfully")
            
        except Exception as e:
            logger.error(f"Error configuring alerts: {e}")
    
    def process_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a match and update ELO ratings
        
        Args:
            match_data: Dictionary containing match information:
                - home_team_id: ID of home team
                - away_team_id: ID of away team  
                - home_goals: Goals scored by home team
                - away_goals: Goals scored by away team
                - league_id: Optional league ID
                - match_date: Optional match date
                - match_importance: Optional importance factor (default 1.0)
        
        Returns:
            Dictionary with updated ratings and match information
        """
        try:
            # Get ratings before update
            home_rating_before = self.elo_system.get_team_rating(
                match_data['home_team_id'], 
                match_data.get('league_id')
            )
            away_rating_before = self.elo_system.get_team_rating(
                match_data['away_team_id'], 
                match_data.get('league_id')
            )
            
            # Update ratings
            home_rating_after, away_rating_after = self.elo_system.update_ratings(match_data)
            
            return {
                'home_team_id': match_data['home_team_id'],
                'away_team_id': match_data['away_team_id'],
                'home_goals': match_data['home_goals'],
                'away_goals': match_data['away_goals'],
                'home_rating_before': home_rating_before,
                'away_rating_before': away_rating_before,
                'home_rating_after': home_rating_after,
                'away_rating_after': away_rating_after,
                'rating_change_home': home_rating_after - home_rating_before,
                'rating_change_away': away_rating_after - away_rating_before,
                'match_processed': True
            }
            
        except Exception as e:
            logger.error(f"Error processing match: {e}")
            return {
                'match_processed': False,
                'error': str(e)
            }
    
    def get_team_rating(self, team_id: int, league_id: Optional[int] = None) -> float:
        """Get current ELO rating for a team"""
        return self.elo_system.get_team_rating(team_id, league_id)
    
    def get_match_prediction(self, home_team_id: int, away_team_id: int, 
                           league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get match prediction including probabilities and expected goals"""
        return self.elo_system.get_match_prediction(home_team_id, away_team_id, league_id)
    
    def get_team_match_history(self, team_id: int, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent match history for a team"""
        return self.elo_system.get_team_match_history(team_id, limit)