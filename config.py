import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv()

# API Configuration
API_FOOTBALL_KEY = os.getenv('API_FOOTBALL_KEY', '')  # Valor por defecto vacío
API_BASE_URL = 'https://v3.football.api-sports.io'  # URL directa de api-football.com
API_HOST = 'v3.football.api-sports.io'  # Host fijo para API-Football.com

# Configuración de Odds (usa la misma API)
ODDS_ENDPOINTS = {
    "pre_match": "/odds",
    "live": "/odds/live",
    "bookmakers": "/odds/bookmakers",
    "bets": "/odds/bets"
}

# Data Directory for Corner Models
DATA_DIR = "data"

# Database Configuration
DB_PATH = "soccer_cache.sqlite"
SCHEMA_VERSION = "1.2.0"

# Model Configuration
MODEL_PATH = "models/fnn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
INPUT_DIM = 14

# Prediction Configuration
MIN_MATCHES_FOR_FORM = 3
FORM_WINDOW_SIZE = 5
DEFAULT_CONFIDENCE = 0.7

# Odds Analysis Configuration
ODDS_CONFIG = {
    "min_edge": 2.0,  # Minimum edge percentage to consider value
    "min_efficiency": 0.90,  # Minimum market efficiency
    "max_margin": 0.08,  # Maximum bookmaker margin
    "min_odds": 1.10,  # Minimum odds to consider
    "max_odds": 15.0,  # Maximum odds to consider
    "markets": [
        "match_winner",
        "over_under",
        "btts",
        "asian_handicap",
        "corners",
        "cards"
    ],
    "update_interval": timedelta(minutes=15),  # How often to update odds
    "stale_threshold": timedelta(hours=1)  # When odds are considered stale
}

# Value Detection Configuration
VALUE_CONFIG = {
    "min_value": 0.02,  # Minimum value percentage (2%)
    "max_exposure": 0.1,  # Maximum bankroll exposure per bet
    "kelly_fraction": 0.5,  # Conservative Kelly criterion
    "min_prob": 0.05,  # Minimum probability threshold
    "max_prob": 0.95  # Maximum probability threshold
}

# Metrics Configuration
METRICS_CONFIG = {
    "prediction_window": timedelta(days=30),  # Default window for accuracy metrics
    "roi_window": timedelta(days=90),  # Window for ROI calculation
    "min_predictions": 10,  # Minimum predictions for accuracy calculation
    "confidence_levels": [0.7, 0.8, 0.9],  # Confidence thresholds for analysis
    "track_markets": ["match_winner", "over_under", "btts"]  # Markets to track
}

# Backup Configuration
BACKUP_CONFIG = {
    "base_dir": "backup",
    "max_backups": 5,  # Maximum number of backups to keep
    "backup_interval": timedelta(days=1),
    "compression": True,
    "critical_files": [
        "models/fnn_model.pkl",
        "models/scaler.pkl",
        "config.py",
        "business_rules.py",
        "predictions.py",
        "features.py"
    ],
    "backup_db": True,  # Whether to include database in backups
    "auto_cleanup": True  # Automatically remove old backups
}

# Cache Configuration
CACHE_CONFIG = {
    "predictions": timedelta(hours=2),      # Aumentado a 2 horas para reducir API calls
    "fixtures": timedelta(hours=24),       # Aumentado a 24 horas para partidos analizados
    "odds": timedelta(minutes=15),         # Mantener 15 min para odds en tiempo real
    "weather": timedelta(hours=3),
    "injuries": timedelta(hours=6),
    "form": timedelta(hours=12),
    "standings": timedelta(days=1),
    "lineups": timedelta(hours=1),
    "events": timedelta(minutes=5),
    "league_data": timedelta(days=7),
    "analyzed_matches": timedelta(hours=24)  # Cache específico para partidos analizados
}

# Odds API Constants
ODDS_BOOKMAKERS_PRIORITY = [1, 6, 8, 2]  # Bookmakers prioritarios por ID
ODDS_DEFAULT_MARKETS = [1, 2, 3]  # Match Winner, Home/Away, Over/Under
