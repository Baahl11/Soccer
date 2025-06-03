"""
Database Integration Adapter for Auto-Updating ELO System
Seamlessly integrates the new database backend with the existing ELO system.

Author: GitHub Copilot
Date: May 27, 2025
Version: 2.0.0
"""

import json
import logging
import shutil
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from elo_database_backend import (
    DatabaseBackend, DatabaseConfig, DatabaseType,
    TeamRating, MatchResult, LeagueInfo
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_integration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseIntegrationAdapter:
    """
    Adapter that provides backward compatibility while transitioning
    from JSON file storage to database backend.
    """
    def __init__(self, use_database: bool = True, 
                json_file_path: str = "data/team_elo_ratings.json",
                 database_config: Optional[Union[DatabaseConfig, Dict[str, Any]]] = None):
        """Initialize the integration adapter"""
        self.use_database = use_database
        self.json_file_path = json_file_path
        
        # Add normalization settings
        self.rating_settings = {
            'max_rating': 1499.0,
            'min_rating': 1000.0,
            'default_rating': 1400.0,  # Start slightly below 1500
            'ensure_probabilities': True
        }
        
        # Initialize database backend if enabled
        if self.use_database:
            # Convert dict to DatabaseConfig if needed
            if database_config is None:
                self.db_config = DatabaseConfig(
                    db_type=DatabaseType.SQLITE,
                    sqlite_path="data/elo_system.db"
                )
            elif isinstance(database_config, dict):
                self.db_config = self._convert_dict_to_config(database_config)
            else:
                self.db_config = database_config
            
            self.database = DatabaseBackend(self.db_config)
            # Migrate from JSON if database is empty and JSON exists
            self._check_migration_needed()        
        else:
            self.database = None
            logger.info(f"Database integration adapter initialized (database_mode: {use_database})")
    
    def _convert_dict_to_config(self, config_dict: Dict[str, Any]) -> DatabaseConfig:
        """Convert dictionary configuration to DatabaseConfig object and ensure consistent formats"""
        db_type_str = config_dict.get('type', 'sqlite')
        
        # Convert string to DatabaseType enum
        if db_type_str.lower() == 'sqlite':
            db_type = DatabaseType.SQLITE
        elif db_type_str.lower() == 'postgresql':
            db_type = DatabaseType.POSTGRESQL
        else:
            logger.warning(f"Unknown database type '{db_type_str}', defaulting to SQLite")
            db_type = DatabaseType.SQLITE
        
        # Ensure consistent rating formats by normalizing config
        base_config = DatabaseConfig(
            db_type=db_type,
            sqlite_path=config_dict.get('database', 'data/elo_system.db'),
            postgres_host=config_dict.get('host', 'localhost'),
            postgres_port=config_dict.get('port', 5432),
            postgres_database=config_dict.get('database', 'elo_system'),
            postgres_user=config_dict.get('user', 'elo_user'),
            postgres_password=config_dict.get('password', ''),
            backup_enabled=config_dict.get('backup_enabled', True),
            backup_interval_hours=config_dict.get('backup_interval_hours', 6)
        )
          # Store normalization settings in instance
        self.rating_normalization = {
            'normalize_ratings': True,
            'max_rating': 1499.0,
            'min_rating': 1000.0,
            'ensure_probabilities': True
        }
        
        return base_config

    def _check_migration_needed(self):
        """Check if migration from JSON to database is needed"""
        if not self.use_database or not self.database:
            return
            
        if not Path(self.json_file_path).exists():
            logger.info("No JSON file found, starting with empty database")
            return
        
        # Check if database is empty
        if not self.database:
            return
        
        stats = self.database.get_database_statistics()
        if stats.get('total_teams', 0) == 0:
            logger.info("Database is empty, checking for JSON migration...")
            
            try:
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                if json_data.get('ratings'):
                    logger.info(f"Found {len(json_data['ratings'])} teams in JSON, starting migration...")
                    success = self.migrate_from_json()
                    if success:
                        # Create backup of original JSON file
                        backup_path = f"{self.json_file_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        Path(self.json_file_path).rename(backup_path)
                        logger.info(f"JSON file backed up to {backup_path}")
                        
            except Exception as e:
                logger.error(f"Error checking migration: {e}")
    def migrate_from_json(self) -> bool:
        """Migrate existing JSON data to database"""
        if not self.use_database:
            logger.warning("Cannot migrate: database mode not enabled")
            return False
        
        if not self.database:
            logger.error("Cannot migrate: database not initialized")
            return False
        
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            teams_migrated = 0
            
            # Migrate team ratings
            for team_id_str, team_data in json_data.get('ratings', {}).items():
                try:
                    team_id = int(team_id_str)
                    
                    # Handle different JSON formats
                    if isinstance(team_data, dict):
                        rating = team_data.get('rating', 1500.0)
                        team_name = team_data.get('name', f"Team_{team_id}")
                        last_updated = team_data.get('last_updated')
                    else:
                        # Simple format: just rating value
                        rating = float(team_data)
                        team_name = f"Team_{team_id}"
                        last_updated = None
                    
                    team_rating = TeamRating(
                        team_id=team_id,
                        team_name=team_name,
                        rating=rating,
                        last_match_date=datetime.fromisoformat(last_updated) if last_updated else None
                    )
                    
                    if self.database.save_team_rating(team_rating):
                        teams_migrated += 1
                    
                except Exception as e:
                    logger.warning(f"Error migrating team {team_id_str}: {e}")
                    continue
            
            # Migrate league tier information if available
            leagues_migrated = 0
            for league_id_str, tier_info in json_data.get('league_tiers', {}).items():
                try:
                    league_id = int(league_id_str)
                    
                    if isinstance(tier_info, dict):
                        tier = tier_info.get('tier', 1)
                        elo_range = tier_info.get('elo_range', [1500, 1600])
                        name = tier_info.get('name', f"League_{league_id}")
                        country = tier_info.get('country', 'Unknown')
                    else:
                        # Simple format: just tier number
                        tier = int(tier_info)
                        elo_range = [1500, 1600]  # Default range
                        name = f"League_{league_id}"
                        country = 'Unknown'
                    
                    league_info = LeagueInfo(
                        league_id=league_id,
                        league_name=name,
                        country=country,
                        tier=tier,
                        elo_range=(elo_range[0], elo_range[1])
                    )
                    
                    if self.database.save_league_info(league_info):
                        leagues_migrated += 1
                        
                except Exception as e:
                    logger.warning(f"Error migrating league {league_id_str}: {e}")
                    continue
            
            logger.info(f"Migration completed: {teams_migrated} teams, {leagues_migrated} leagues")
            return teams_migrated > 0
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False

    def _init_db_backend(self, config: DatabaseConfig) -> None:
        """Initialize database backend with performance optimizations"""
        try:
            self.db_backend = DatabaseBackend(config)
            
            # Enable performance optimizations for SQLite
            if config.db_type == DatabaseType.SQLITE:
                with self.db_backend.get_connection() as conn:
                    cursor = conn.cursor()
                    # Set pragmas for better performance
                    cursor.execute('PRAGMA journal_mode=WAL')  # Write-Ahead Logging
                    cursor.execute('PRAGMA synchronous=NORMAL')  # Speeds up writes
                    cursor.execute('PRAGMA cache_size=-2000')  # Use 2MB cache
                    cursor.execute('PRAGMA temp_store=MEMORY')  # Store temp tables in memory
                    cursor.execute('PRAGMA mmap_size=2147483648')  # 2GB memory map
                    conn.commit()
                    
            logger.info(f"Database backend initialized with {config.db_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database backend: {e}")
            raise

    def _batch_save_ratings(self, ratings: List[TeamRating], batch_size: int = 100) -> int:
        """Save ratings in batches for better performance"""
        if not self.database:
            logger.error("Database not initialized")
            return 0
            
        saved_count = 0
        for i in range(0, len(ratings), batch_size):
            batch = ratings[i:i + batch_size]
            try:
                # Save each rating individually since we don't have batch save
                for team_rating in batch:
                    if self.database.save_team_rating(team_rating):
                        saved_count += 1
            except Exception as e:
                logger.error(f"Error saving batch {i//batch_size}: {e}")
                continue
                
        return saved_count

    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        if not self.use_database or not self.database:
            return {
                'mode': 'json',
                'teams': 0,
                'matches': 0
            }

        try:
            return self.database.get_database_statistics()
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            stats = {'error': str(e)}
            return stats

    def save_team_ratings(self, ratings: Dict[str, Any]) -> bool:
        """Save team ratings to database or JSON file"""
        if not self.use_database or not self.database:
            try:
                # Save to JSON file
                data = {
                    'ratings': ratings,
                    'last_updated': datetime.now().strftime("%Y-%m-%d")
                }
                os.makedirs(os.path.dirname(self.json_file_path), exist_ok=True)
                with open(self.json_file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return True
            except Exception as e:
                logger.error(f"Error saving to JSON file: {e}")
                return False
                
        # Database mode
        try:
            for team_id_str, rating_data in ratings.items():
                try:
                    team_id = int(team_id_str)
                    if isinstance(rating_data, dict):
                        rating = rating_data.get('rating', 1500.0)
                        team_name = rating_data.get('name', f"Team_{team_id}")
                        metadata = rating_data.get('metadata', {})
                    else:
                        rating = float(rating_data)
                        team_name = f"Team_{team_id}"
                        metadata = {}
                        
                    team_rating = TeamRating(
                        team_id=team_id,
                        team_name=team_name,
                        rating=rating,
                        last_match_date=datetime.now(),
                        metadata=metadata
                    )
                    
                    # Save each rating individually to avoid batch complexity
                    if not self.database.save_team_rating(team_rating):
                        logger.warning(f"Failed to save rating for team {team_id}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Skipping invalid team ID {team_id_str}: {e}")
                    continue
                    
            return True
            
        except Exception as e:
            logger.error(f"Error saving ratings to database: {e}")
            return False

    def get_all_ratings(self) -> Dict[str, float]:
        """Get all team ratings from database or JSON file"""
        if not self.use_database or not self.database:
            try:
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return {str(k): float(v) for k, v in data.get('ratings', {}).items()}
            except Exception as e:
                logger.error(f"Error loading ratings from JSON: {e}")
                return {}

        try:
            teams = self.database.get_all_team_ratings()
            return {str(team.team_id): float(team.rating) for team in teams.values()}
        except Exception as e:
            logger.error(f"Error getting all ratings from database: {e}")
            return {}

    def save_match_result(self, match_data: Dict[str, Any]) -> bool:
        """Save a match result to the database"""
        if not self.use_database or not self.database:
            return False

        try:
            # Create MatchResult object
            match = MatchResult(
                match_id=int(match_data.get('match_id', 0)),
                home_team_id=int(match_data['home_team_id']),
                away_team_id=int(match_data['away_team_id']),
                home_score=int(match_data['home_score']),
                away_score=int(match_data['away_score']),
                match_date=datetime.fromisoformat(match_data.get('match_date', datetime.now().isoformat())),
                league_id=int(match_data.get('league_id', 0)),
                season=int(match_data.get('season', datetime.now().year)),
                home_rating_before=float(match_data.get('home_rating_before', 1500.0)),
                away_rating_before=float(match_data.get('away_rating_before', 1500.0)),
                home_rating_after=float(match_data.get('home_rating_after', 0.0)),
                away_rating_after=float(match_data.get('away_rating_after', 0.0)),
                rating_change_home=float(match_data.get('rating_change', 0.0))
            )
            
            success = self.database.save_match_result(match)
            if success:
                logger.debug(f"Match {match.match_id} saved successfully")
            return success
        except Exception as e:
            logger.error(f"Error saving match result: {e}")
            return False

    def get_team_match_history(self, team_id: int, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get match history for a team"""
        if not self.use_database or not self.database:
            return []

        try:
            matches = self.database.get_team_match_history(team_id, limit or 50)
            
            matches_list = [
                {
                    'match_id': match.match_id,
                    'opponent_id': match.away_team_id if team_id == match.home_team_id else match.home_team_id,
                    'home_away': 'home' if team_id == match.home_team_id else 'away',
                    'score': f"{match.home_score}-{match.away_score}",
                    'match_date': match.match_date.isoformat(),
                    'rating_before': match.home_rating_before if team_id == match.home_team_id else match.away_rating_before,
                    'rating_after': match.home_rating_after if team_id == match.home_team_id else match.away_rating_after,
                    'rating_change': match.rating_change_home if team_id == match.home_team_id else -match.rating_change_home
                }
                for match in matches
            ]
            return matches_list
            
        except Exception as e:
            logger.error(f"Error getting team match history: {e}")
            return []

    def create_backup(self) -> str:
        """Create a backup of the current ratings data"""
        if not self.use_database or not self.database:
            try:
                backup_dir = Path('data/backups')
                backup_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f'team_elo_ratings_{timestamp}.json'
                
                shutil.copy2(self.json_file_path, backup_path)
                logger.info(f"Created JSON backup at {backup_path}")
                return str(backup_path)
            except Exception as e:
                logger.error(f"Error creating JSON backup: {e}")
                return ""

        try:
            backup_path = self.database.create_backup()
            if backup_path:
                logger.info(f"Created database backup at {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return ""

    def close(self) -> None:
        """Properly close database connections"""
        try:
            if self.use_database and self.database:
                self.database.close()
                logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
        finally:
            logger.info("Database integration adapter closed")

    def get_team_metadata(self, team_id: int) -> Dict[str, Any]:
        """
        Get metadata for a specific team from the database.
        
        Args:
            team_id: The ID of the team
            
        Returns:
            Dict containing team metadata or empty dict if not found
        """
        if not self.use_database or not self.database:
            logger.warning("Database not enabled, cannot retrieve team metadata")
            return {}
            
        try:
            team_rating = self.database.get_team_rating(team_id)
            if team_rating and team_rating.metadata:
                return team_rating.metadata
            return {}
            
        except Exception as e:
            logger.error(f"Error retrieving metadata for team {team_id}: {e}")
            return {}

if __name__ == "__main__":
    # Example usage and testing
    print("Testing Database Integration Adapter...")
    
    # Test with database mode
    adapter = DatabaseIntegrationAdapter(use_database=True)
    
    # Test saving and loading ratings
    test_ratings = {
        '1': {'name': 'Real Madrid', 'rating': 1600.0, 'matches_played': 10},
        '2': {'name': 'Barcelona', 'rating': 1580.0, 'matches_played': 9},
        '3': {'name': 'Atletico Madrid', 'rating': 1550.0, 'matches_played': 8}
    }
    
    print("Saving test ratings...")
    success = adapter.save_team_ratings(test_ratings)
    print(f"Save success: {'✓' if success else '✗'}")
    
    print("Loading ratings...")
    loaded_ratings = adapter.get_all_ratings()
    print(f"Loaded {len(loaded_ratings)} team ratings")
    
    # Test match result saving
    print("Saving test match result...")
    match_result = {
        'match_id': 2001,
        'home_team_id': 1,
        'away_team_id': 2,
        'home_score': 2,
        'away_score': 1,
        'match_date': datetime.now().isoformat(),
        'league_id': 39,
        'season': 2025,
        'home_rating_before': 1600.0,
        'away_rating_before': 1580.0,
        'home_rating_after': 1610.0,
        'away_rating_after': 1570.0,
        'rating_change': 10.0
    }
    match_success = adapter.save_match_result(match_result)
    print(f"Match save success: {'✓' if match_success else '✗'}")
    
    # Get match history
    print("Getting match history...")
    history = adapter.get_team_match_history(1, limit=5)
    print(f"Found {len(history)} matches in history")
    
    # Get statistics
    stats = adapter.get_database_statistics()
    print(f"Database statistics: {stats}")
    
    # Create backup
    backup_path = adapter.create_backup()
    print(f"Backup created: {backup_path}")
    
    adapter.close()
    print("Database integration adapter test completed")
