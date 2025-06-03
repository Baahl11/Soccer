"""
Database Backend Implementation for Auto-Updating ELO System
Replaces JSON file storage with robust SQLite/PostgreSQL database backend.

Author: GitHub Copilot
Date: May 27, 2025
Version: 2.0.0
"""

import sqlite3
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
from pathlib import Path
import os
from enum import Enum

# PostgreSQL support (optional)
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/database_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Supported database types"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"

@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    db_type: DatabaseType = DatabaseType.SQLITE
    sqlite_path: str = "data/elo_system.db"
    
    # PostgreSQL settings
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_database: str = "elo_system"
    postgres_user: str = "elo_user"
    postgres_password: str = ""
    
    # Connection pool settings
    max_connections: int = 10
    connection_timeout: int = 30
    retry_attempts: int = 3
    backup_enabled: bool = True
    backup_interval_hours: int = 6

@dataclass
class TeamRating:
    """Team rating data structure for database storage"""
    team_id: int
    team_name: str = ""  # Make team_name optional with empty string default
    rating: float = 1500.0  # Default rating
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    last_match_date: Optional[datetime] = None
    league_id: Optional[int] = None
    season: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MatchResult:
    """Match result data structure for database storage"""
    match_id: int
    home_team_id: int
    away_team_id: int
    home_score: int
    away_score: int
    match_date: datetime
    league_id: int
    season: int
    home_rating_before: float
    away_rating_before: float
    home_rating_after: float
    away_rating_after: float
    rating_change_home: float
    rating_change_away: Optional[float] = None
    k_factor_used: Optional[float] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LeagueInfo:
    """League information data structure"""
    league_id: int
    league_name: str
    country: str
    tier: int
    elo_range: Tuple[float, float]
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class DatabaseBackend:
    """
    Advanced database backend for ELO rating system with support for
    SQLite and PostgreSQL, connection pooling, and automated backups.
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize the database backend"""
        self.config = config
        self.connection_pool = []
        self.pool_lock = threading.Lock()
        self.backup_thread = None
        self.backup_running = False

        # Initialize persistent connection for methods using self.connection
        if self.config.db_type == DatabaseType.SQLITE:
            import sqlite3
            self.connection = sqlite3.connect(self.config.sqlite_path, check_same_thread=False)
            self.connection.row_factory = sqlite3.Row
        else:
            self.connection = None  # For PostgreSQL or others, to be implemented

        # Initialize database
        self._initialize_database()
        
        # Start backup process if enabled
        if self.config.backup_enabled:
            self._start_backup_process()
        
        logger.info(f"Database backend initialized with {config.db_type.value}")
    
    def _initialize_database(self):
        """Initialize database schema based on database type"""
        if self.config.db_type == DatabaseType.SQLITE:
            self._initialize_sqlite()
        elif self.config.db_type == DatabaseType.POSTGRESQL:
            self._initialize_postgresql()
        else:
            raise ValueError(f"Unsupported database type: {self.config.db_type}")
    
    def _initialize_sqlite(self):
        """Initialize SQLite database and schema"""
        # Ensure directory exists
        db_path = Path(self.config.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.config.sqlite_path) as conn:
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Create tables
            self._create_tables_sqlite(conn)
            
            # Create indexes for performance
            self._create_indexes_sqlite(conn)
        
        logger.info(f"SQLite database initialized at {self.config.sqlite_path}")
    
    def _create_tables_sqlite(self, conn: sqlite3.Connection):
        """Create SQLite tables"""
        
        # Teams table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY,
                team_name TEXT NOT NULL UNIQUE,
                rating REAL NOT NULL DEFAULT 1500.0,
                matches_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                goals_for INTEGER DEFAULT 0,
                goals_against INTEGER DEFAULT 0,
                last_match_date TEXT,
                league_id INTEGER,
                season INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (league_id) REFERENCES leagues (league_id)
            )
        ''')
        
        # Matches table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                home_team_id INTEGER NOT NULL,
                away_team_id INTEGER NOT NULL,
                home_score INTEGER NOT NULL,
                away_score INTEGER NOT NULL,
                match_date TEXT NOT NULL,
                league_id INTEGER NOT NULL,
                season INTEGER NOT NULL,
                home_rating_before REAL NOT NULL,
                away_rating_before REAL NOT NULL,
                home_rating_after REAL NOT NULL,
                away_rating_after REAL NOT NULL,
                rating_change_home REAL NOT NULL,
                rating_change_away REAL NOT NULL,
                k_factor_used REAL NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (league_id) REFERENCES leagues (league_id)
            )
        ''')
        
        # Leagues table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY,
                league_name TEXT NOT NULL,
                country TEXT NOT NULL,
                tier INTEGER NOT NULL,
                elo_range_min REAL NOT NULL,
                elo_range_max REAL NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Rating history table for tracking changes over time
        conn.execute('''
            CREATE TABLE IF NOT EXISTS rating_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                change_amount REAL NOT NULL,
                match_id INTEGER,
                timestamp TEXT NOT NULL,
                reason TEXT,
                FOREIGN KEY (team_id) REFERENCES teams (team_id),
                FOREIGN KEY (match_id) REFERENCES matches (match_id)
            )
        ''')
        
        # System metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                metric_value REAL,
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        logger.info("SQLite tables created successfully")
    
    def _create_indexes_sqlite(self, conn: sqlite3.Connection):
        """Create SQLite indexes for performance optimization"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_teams_rating ON teams(rating)",
            "CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_id)",
            "CREATE INDEX IF NOT EXISTS idx_teams_last_match ON teams(last_match_date)",
            "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(match_date)",
            "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id)",
            "CREATE INDEX IF NOT EXISTS idx_matches_league ON matches(league_id, season)",
            "CREATE INDEX IF NOT EXISTS idx_rating_history_team ON rating_history(team_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_system_metrics_type_time ON system_metrics(metric_type, timestamp)"
        ]
        for index_sql in indexes:
            conn.execute(index_sql)
        
        logger.info("SQLite indexes created successfully")
    
    def _initialize_postgresql(self):
        """Initialize PostgreSQL database and schema"""
        if not POSTGRES_AVAILABLE:
            raise ImportError("psycopg2 not available. Install with: pip install psycopg2-binary")
        
        # Implementation for PostgreSQL would go here
        # For now, we'll focus on SQLite as it's more universally available
        raise NotImplementedError("PostgreSQL support will be implemented in future version")
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = None
        try:
            if self.config.db_type == DatabaseType.SQLITE:
                conn = sqlite3.connect(self.config.sqlite_path)
                conn.row_factory = sqlite3.Row  # Enable dict-like access
                yield conn
            else:
                raise NotImplementedError("PostgreSQL connection pooling not yet implemented")
        finally:
            if conn:
                conn.close()    
    def save_team_rating(self, team_rating: TeamRating) -> bool:
        """Save or update team rating in database"""
        try:
            with self.get_connection() as conn:
                now = datetime.now().isoformat()
                metadata_json = json.dumps(team_rating.metadata)
                
                # Check if team exists
                cursor = conn.execute(
                    "SELECT team_id FROM teams WHERE team_id = ?", 
                    (team_rating.team_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    # Update existing team rating
                    conn.execute(
                        """UPDATE teams SET 
                            rating = ?, matches_played = ?, wins = ?, draws = ?, 
                            losses = ?, goals_for = ?, goals_against = ?, 
                            last_match_date = ?, updated_at = ?, metadata = ? 
                        WHERE team_id = ?""",
                        (
                            team_rating.rating, team_rating.matches_played, team_rating.wins,
                            team_rating.draws, team_rating.losses, team_rating.goals_for,
                            team_rating.goals_against,
                            team_rating.last_match_date.isoformat() if team_rating.last_match_date else None,
                            now, metadata_json, team_rating.team_id
                        )
                    )
                else:
                    # Insert new team rating
                    conn.execute(
                        """INSERT INTO teams (
                            team_id, team_name, rating, matches_played, wins,
                            draws, losses, goals_for, goals_against,
                            last_match_date, created_at, updated_at, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            team_rating.team_id, team_rating.team_name, team_rating.rating,
                            team_rating.matches_played, team_rating.wins, team_rating.draws,
                            team_rating.losses, team_rating.goals_for, team_rating.goals_against,
                            team_rating.last_match_date.isoformat() if team_rating.last_match_date else None,
                            now, now, metadata_json
                        )
                    )
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving team rating for team {team_rating.team_id}: {e}")
            return False
    def get_team_rating(self, team_id: int) -> Optional[TeamRating]:
        """Retrieve team rating from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM teams WHERE team_id = ?", 
                    (team_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    return TeamRating(
                        team_id=row['team_id'],
                        team_name=row['team_name'],
                        rating=row['rating'],
                        matches_played=row['matches_played'],
                        wins=row['wins'],
                        draws=row['draws'],
                        losses=row['losses'],
                        goals_for=row['goals_for'],
                        goals_against=row['goals_against'],
                        last_match_date=datetime.fromisoformat(row['last_match_date']) if row['last_match_date'] else None,
                        league_id=row['league_id'],
                        season=row['season'],
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        metadata=json.loads(row['metadata']) if row['metadata'] else {}
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error retrieving team rating for team {team_id}: {e}")
            return None
    def save_match_result(self, match: MatchResult) -> bool:
        """Save match result to database"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT INTO matches (
                        match_id, home_team_id, away_team_id,
                        home_score, away_score, match_date,
                        league_id, season,
                        home_rating_before, away_rating_before,
                        home_rating_after, away_rating_after,
                        rating_change_home,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    match.match_id,
                    match.home_team_id,
                    match.away_team_id,
                    match.home_score,
                    match.away_score,
                    match.match_date.isoformat(),
                    match.league_id,
                    match.season,
                    match.home_rating_before,
                    match.away_rating_before,
                    match.home_rating_after,
                    match.away_rating_after,
                    match.rating_change_home,
                    datetime.now().isoformat()
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Error saving match result: {e}")
            return False

    def get_team_match_history(self, team_id: int, limit: int = 50) -> List[MatchResult]:
        """Get match history for a team"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT * FROM matches 
                    WHERE home_team_id = ? OR away_team_id = ?
                    ORDER BY match_date DESC LIMIT ?
                ''', (team_id, team_id, limit))

                matches = []
                for row in cursor.fetchall():
                    match = MatchResult(
                        match_id=row['match_id'],
                        home_team_id=row['home_team_id'],
                        away_team_id=row['away_team_id'],
                        home_score=row['home_score'],
                        away_score=row['away_score'],
                        match_date=datetime.fromisoformat(row['match_date']),
                        league_id=row['league_id'],
                        season=row['season'],
                        home_rating_before=row['home_rating_before'],
                        away_rating_before=row['away_rating_before'],
                        home_rating_after=row['home_rating_after'],
                        away_rating_after=row['away_rating_after'],
                        rating_change_home=row['rating_change_home']
                    )
                    matches.append(match)
                return matches
        except Exception as e:
            logger.error(f"Error getting match history: {e}")
            return []

    def get_all_team_ratings(self) -> Dict[int, TeamRating]:
        """Get all team ratings from database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('SELECT * FROM teams')
                teams = {}
                for row in cursor.fetchall():
                    team = TeamRating(
                        team_id=row['team_id'],
                        team_name=row.get('team_name', f'Team_{row["team_id"]}'),
                        rating=row['rating'],
                        matches_played=row.get('matches_played', 0),
                        wins=row.get('wins', 0),
                        draws=row.get('draws', 0),
                        losses=row.get('losses', 0),
                        goals_for=row.get('goals_for', 0),
                        goals_against=row.get('goals_against', 0),
                        last_match_date=datetime.fromisoformat(row['last_match_date']) if row.get('last_match_date') else None,
                        league_id=row.get('league_id'),
                        season=row.get('season')
                    )
                    teams[team.team_id] = team
                return teams
        except Exception as e:
            logger.error(f"Error getting all team ratings: {e}")
            return {}
    def save_league_info(self, league: LeagueInfo) -> bool:
        """Save league information to database"""
        try:
            with self.get_connection() as conn:
                league.updated_at = datetime.now()
                
                conn.execute('''
                    INSERT OR REPLACE INTO leagues (
                        league_id, league_name, country, tier,
                        elo_range_min, elo_range_max, is_active,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    league.league_id, league.league_name, league.country, league.tier,
                    league.elo_range[0], league.elo_range[1], league.is_active,
                    league.created_at.isoformat(), league.updated_at.isoformat()
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error saving league info {league.league_id}: {e}")
            return False
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            stats = {}
            with self.get_connection() as conn:
                # Team statistics
                cursor = conn.execute("SELECT COUNT(*) as count FROM teams")
                stats['total_teams'] = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(*) as count FROM teams WHERE last_match_date IS NOT NULL")
                stats['active_teams'] = cursor.fetchone()['count']
                
                # Match statistics
                cursor = conn.execute("SELECT COUNT(*) as count FROM matches")
                stats['total_matches'] = cursor.fetchone()['count']
                
                cursor = conn.execute("SELECT COUNT(DISTINCT league_id) as count FROM matches")
                stats['leagues_with_matches'] = cursor.fetchone()['count']
                
                # Rating statistics
                cursor = conn.execute("SELECT MIN(rating) as min, MAX(rating) as max, AVG(rating) as avg FROM teams")
                rating_stats = cursor.fetchone()
                stats['rating_range'] = {
                    'min': rating_stats['min'],
                    'max': rating_stats['max'],
                    'average': rating_stats['avg']
                }
                
                # Database file size (SQLite only)
                if self.config.db_type == DatabaseType.SQLITE:
                    db_size = Path(self.config.sqlite_path).stat().st_size / (1024 * 1024)  # MB
                    stats['database_size_mb'] = round(db_size, 2)
                
                stats['generated_at'] = datetime.now().isoformat()
                
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {'error': str(e)}
    
    def _start_backup_process(self):
        """Start automated backup process"""
        if self.config.db_type != DatabaseType.SQLITE:
            logger.warning("Automated backup only supported for SQLite currently")
            return
        
        self.backup_running = True
        self.backup_thread = threading.Thread(
            target=self._backup_loop,
            daemon=True
        )
        self.backup_thread.start()
        logger.info(f"Backup process started with {self.config.backup_interval_hours}h interval")
    
    def _backup_loop(self):
        """Automated backup loop"""
        while self.backup_running:
            try:
                time.sleep(self.config.backup_interval_hours * 3600)  # Convert hours to seconds
                if self.backup_running:
                    self.create_backup()
            except Exception as e:
                logger.error(f"Error in backup loop: {e}")
    
    def create_backup(self) -> str:
        """Create a database backup"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = Path("data/backups")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            backup_path = backup_dir / f"elo_database_backup_{timestamp}.db"
            
            if self.config.db_type == DatabaseType.SQLITE:
                # SQLite backup using backup API
                with sqlite3.connect(self.config.sqlite_path) as source:
                    with sqlite3.connect(str(backup_path)) as backup:
                        source.backup(backup)
                
                logger.info(f"Database backup created: {backup_path}")
                return str(backup_path)
            else:
                raise NotImplementedError("Backup not implemented for PostgreSQL yet")
                
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return ""
    
    def migrate_from_json(self, json_file_path: str) -> bool:
        """Migrate existing JSON data to database"""
        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
            
            # Migrate team ratings
            teams_migrated = 0
            for team_id_str, team_data in json_data.get('ratings', {}).items():
                team_id = int(team_id_str)
                
                team_rating = TeamRating(
                    team_id=team_id,
                    team_name=team_data.get('name', f"Team_{team_id}"),
                    rating=team_data.get('rating', 1500.0),
                    matches_played=team_data.get('matches_played', 0),
                    last_match_date=datetime.fromisoformat(team_data['last_updated']) if team_data.get('last_updated') else None
                )
                
                if self.save_team_rating(team_rating):
                    teams_migrated += 1
            
            logger.info(f"Successfully migrated {teams_migrated} teams from JSON to database")
            return True
            
        except Exception as e:
            logger.error(f"Error migrating from JSON: {e}")
            return False
    def batch_save_ratings(self, ratings: List[TeamRating], batch_size: int = 100) -> int:
        """Save a batch of ratings efficiently"""
        saved = 0
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create temporary table for batch insert
                cursor.execute("""
                    CREATE TEMP TABLE IF NOT EXISTS temp_ratings (
                        team_id INTEGER PRIMARY KEY,
                        rating REAL
                    )
                """)
                
                # Insert into temp table
                for i in range(0, len(ratings), batch_size):
                    batch = ratings[i:i + batch_size]
                    cursor.executemany(
                        'INSERT OR REPLACE INTO temp_ratings (team_id, rating) VALUES (?, ?)',
                        [(r.team_id, r.rating) for r in batch]
                    )
                    saved += len(batch)
                
                # Use efficient UPSERT from temp table
                cursor.execute("""
                    INSERT OR REPLACE INTO team_ratings (team_id, rating)
                    SELECT team_id, rating FROM temp_ratings
                """)
                
                # Cleanup
                cursor.execute('DROP TABLE temp_ratings')
                conn.commit()
                
            return saved
            
        except Exception as e:
            logger.error(f"Error in batch save: {e}")            
            return 0

    def get_all_ratings(self) -> List[TeamRating]:
        """Get all team ratings efficiently"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                # Use efficient indexed query
                cursor.execute('SELECT team_id, team_name, rating FROM teams')
                return [TeamRating(
                    team_id=row['team_id'],
                    team_name=row['team_name'],
                    rating=row['rating']
                ) for row in cursor.fetchall()]
                   
        except Exception as e:
            logger.error(f"Error getting ratings: {e}")
            return []

    def close(self) -> None:
        """Close database connection properly"""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
                delattr(self, 'connection')
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
            
    def __del__(self):
        """Ensure connection is closed on deletion"""
        self.close()

if __name__ == "__main__":
    # Example usage and testing
    config = DatabaseConfig(
        db_type=DatabaseType.SQLITE,
        sqlite_path="data/test_elo_system.db",
        backup_enabled=True,
        backup_interval_hours=1  # For testing
    )
    
    db = DatabaseBackend(config)
    
    # Test team rating operations
    test_team = TeamRating(
        team_id=1,
        team_name="Test Team FC",
        rating=1600.0,
        matches_played=10,
        wins=6,
        draws=2,
        losses=2
    )
    
    print("Testing database operations...")
    
    # Save team
    success = db.save_team_rating(test_team)
    print(f"Save team: {'✓' if success else '✗'}")
    
    # Retrieve team
    retrieved_team = db.get_team_rating(1)
    print(f"Retrieve team: {'✓' if retrieved_team else '✗'}")
    if retrieved_team:
        print(f"  Team: {retrieved_team.team_name}, Rating: {retrieved_team.rating}")
    
    # Test match result
    test_match = MatchResult(
        match_id=1001,
        home_team_id=1,
        away_team_id=2,
        home_score=2,
        away_score=1,
        match_date=datetime.now(),
        league_id=39,
        season=2025,
        home_rating_before=1600.0,
        away_rating_before=1550.0,
        home_rating_after=1615.0,
        away_rating_after=1535.0,
        rating_change_home=15.0,
        rating_change_away=-15.0,
        k_factor_used=32.0
    )
    
    success = db.save_match_result(test_match)
    print(f"Save match: {'✓' if success else '✗'}")
    
    # Get statistics
    stats = db.get_database_statistics()
    print(f"Database stats: {stats}")
    
    # Create backup
    backup_path = db.create_backup()
    print(f"Backup created: {backup_path}")
    
    db.close()
    print("Database backend test completed")
