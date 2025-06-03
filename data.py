# data.py
import requests
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import pandas as pd
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import time
import os
import hashlib
import sys

# Importar CacheManager para integración
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cache_manager import CacheManager

logger = logging.getLogger(__name__)

# Instancia global de CacheManager - será inicializada por app.py
_cache_manager = None

def set_cache_manager(cache_manager: CacheManager):
    """Set the global cache manager instance."""
    global _cache_manager
    _cache_manager = cache_manager
    logger.info("Cache manager set in data.py")

def get_cache_manager() -> Optional[CacheManager]:
    """Get the global cache manager instance."""
    return _cache_manager

# Create API instance at module level
_api_instance = None

def get_api_instance():
    """Get or create the FootballAPI instance."""
    global _api_instance
    if _api_instance is None:
        _api_instance = FootballAPI()
    return _api_instance

# Interface functions for app.py
def get_fixture_data(fixture_id: int) -> Dict[str, Any]:
    """Get fixture data by ID."""
    api = get_api_instance()
    params = {'id': fixture_id}
    return api._make_request('fixtures', params)

def get_lineup_data(fixture_id: int) -> Dict[str, Any]:
    """Get lineup data for a fixture."""
    api = get_api_instance()
    params = {'fixture': fixture_id}
    return api._make_request('fixtures/lineups', params)

def get_fixture_statistics(fixture_id: int) -> pd.DataFrame:
    """Get statistics for a fixture and convert to DataFrame."""
    api = get_api_instance()
    params = {'fixture': fixture_id}
    data = api._make_request('fixtures/statistics', params)
    
    if not data or 'response' not in data or not data['response']:
        return pd.DataFrame()
    
    # Process statistics into DataFrame format
    stats_list = []
    for team_stats in data['response']:
        team_id = team_stats.get('team', {}).get('id', 0)
        team_name = team_stats.get('team', {}).get('name', '')
        
        stats_dict = {'team_id': team_id, 'team_name': team_name}
        
        # Extract statistics
        for stat in team_stats.get('statistics', []):
            key = stat.get('type', '').lower().replace(' ', '_')
            value = stat.get('value', 0)
            
            # Convert percentage strings to float
            if isinstance(value, str) and '%' in value:
                value = float(value.replace('%', '').strip())
            # Default value for None
            if value is None:
                value = 0
                
            stats_dict[key] = value
            
        stats_list.append(stats_dict)
    
    return pd.DataFrame(stats_list)

def get_fixtures_filtered(
    league_id: int, 
    season: int, 
    status: str = "NS", 
    days_range: int = 7,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """Get fixtures filtered by league, season, status and date range."""
    api = get_api_instance()
    
    # Calculate date range
    today = datetime.now()
    from_date = today.strftime('%Y-%m-%d')
    to_date = (today + timedelta(days=days_range)).strftime('%Y-%m-%d')
    
    params = {
        'league': league_id,
        'season': season,
        'status': status,
        'from': from_date,
        'to': to_date
    }
    
    data = api._make_request('fixtures', params)
    fixtures = data.get('response', [])
    
    # Format fixtures into simplified format
    result = []
    for i, fix in enumerate(fixtures):
        if i >= limit:
            break
            
        fixture = fix.get('fixture', {})
        teams = fix.get('teams', {})
        
        result.append({
            'fixture_id': fixture.get('id'),
            'date': fixture.get('date'),
            'home_team': teams.get('home', {}).get('name'),
            'away_team': teams.get('away', {}).get('name'),
            'home_team_id': teams.get('home', {}).get('id'),
            'away_team_id': teams.get('away', {}).get('id'),
        })
    
    return result

def get_fixture_players(fixture_id: int) -> Dict[str, Any]:
    """Get player statistics for a fixture."""
    api = get_api_instance()
    params = {'fixture': fixture_id}
    return api._make_request('fixtures/players', params)

def get_team_statistics(team_id: int, league_id: int, season: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas de un equipo para una liga y temporada específicas.
    
    Args:
        team_id: ID del equipo
        league_id: ID de la liga
        season: Temporada (ej: "2024")
        
    Returns:
        Diccionario con estadísticas del equipo incluyendo:
        - corners_per_game: promedio de corners por partido
        - cards_per_game: promedio de tarjetas por partido
        - home_corners_for: promedio de corners a favor como local
        - away_corners_against: promedio de corners en contra como visitante
        etc.
    """
    try:
        api = get_api_instance()
        params = {
            'team': team_id,
            'league': league_id,
            'season': season
        }
        
        # Obtener estadísticas del equipo
        stats = api.get_team_stats(team_id)
        
        # Si no hay datos, retornar valores predeterminados
        if not stats:
            return {
                'corners_per_game': 5.2,
                'cards_per_game': 2.0,
                'home_corners_for': 5.5,
                'away_corners_against': 4.5,
                'home_corners_against': 5.0,
                'away_corners_for': 4.5
            }
            
        # Calcular estadísticas de corners
        total_matches = stats.get('fixtures', {}).get('played', {}).get('total', 1)
        total_corners = stats.get('statistics', {}).get('corners', {}).get('total', {}).get('total', total_matches * 5.2)
        corners_per_game = total_corners / max(1, total_matches)
        
        # Calcular estadísticas de tarjetas
        total_yellows = stats.get('cards', {}).get('yellow', {}).get('total', total_matches * 2)
        total_reds = stats.get('cards', {}).get('red', {}).get('total', total_matches * 0.1)
        cards_per_game = (total_yellows + total_reds) / max(1, total_matches)
        
        # Separar estadísticas de local/visitante
        home_matches = stats.get('fixtures', {}).get('played', {}).get('home', total_matches/2)
        away_matches = stats.get('fixtures', {}).get('played', {}).get('away', total_matches/2)
        
        home_corners = stats.get('statistics', {}).get('corners', {}).get('home', {}).get('total', home_matches * 5.5)
        away_corners = stats.get('statistics', {}).get('corners', {}).get('away', {}).get('total', away_matches * 4.5)
        
        return {
            'corners_per_game': corners_per_game,
            'cards_per_game': cards_per_game,
            'home_corners_for': home_corners / max(1, home_matches),
            'away_corners_against': away_corners / max(1, away_matches),
            'home_corners_against': (total_corners - home_corners) / max(1, away_matches),
            'away_corners_for': (total_corners - away_corners) / max(1, home_matches)
        }
        
    except Exception as e:
        logger.error(f"Error getting team statistics: {e}")
        return {
            'corners_per_game': 5.2,
            'cards_per_game': 2.0,
            'home_corners_for': 5.5,
            'away_corners_against': 4.5,
            'home_corners_against': 5.0,
            'away_corners_for': 4.5
        }

class DataCache:
    def __init__(self, cache_dir: str = 'cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.db_path = self.cache_dir / 'football_data.db'
        self._init_db()
    
    def _init_db(self):
        """Initialize DB for fixtures/statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Verificar si la tabla match_statistics existe
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='match_statistics'")
            table_exists = cursor.fetchone() is not None
            
            # Si la tabla existe pero tiene una estructura antigua, eliminarla
            if table_exists:
                try:
                    # Verificar si la tabla tiene la columna home_stats
                    cursor = conn.execute("PRAGMA table_info(match_statistics)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'home_stats' not in columns or 'away_stats' not in columns:
                        conn.execute("DROP TABLE match_statistics")
                        table_exists = False
                except Exception as e:
                    conn.execute("DROP TABLE match_statistics")
                    table_exists = False
            
            # Crear tablas si no existen
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fixtures (
                    fixture_id INTEGER PRIMARY KEY,
                    league_id INTEGER,
                    season INTEGER,
                    date TEXT,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    home_goals INTEGER,
                    away_goals INTEGER,
                    home_score TEXT,
                    away_score TEXT,
                    timestamp DATETIME
                )
            ''')
            
            if not table_exists:
                conn.execute('''
                    CREATE TABLE match_statistics (
                        fixture_id INTEGER PRIMARY KEY,
                        league_id INTEGER,
                        season INTEGER,
                        home_team_id INTEGER,
                        away_team_id INTEGER,
                        home_goals INTEGER,
                        away_goals INTEGER,
                        home_stats TEXT,
                        away_stats TEXT,
                        timestamp DATETIME
                    )
                ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS league_seasons (
                    league_id INTEGER,
                    season INTEGER,
                    fixtures_count INTEGER,
                    last_update DATETIME,
                    PRIMARY KEY (league_id, season)
                )
            ''')

    def _get_cache_file(self, endpoint: str, params: Dict[str, Any]) -> Path:
        """Create a filename based on endpoint + params hash."""
        params_str = json.dumps(params, sort_keys=True)
        md5hash = hashlib.md5(params_str.encode('utf-8')).hexdigest()
        # Create subdirectories based on endpoint
        cache_path = self.cache_dir / endpoint.split('/')[0]
        cache_path.mkdir(exist_ok=True)
        return cache_path / f"{endpoint.replace('/', '_')}_{md5hash}.json"

    def get(self, endpoint: str, params: Dict[str, Any], max_age_hours: int = 24) -> Optional[Dict]:
        """Retrieve cached API data (JSON file)."""
        cache_file = self._get_cache_file(endpoint, params)
        if not cache_file.exists():
            return None

        try:
            with cache_file.open("r", encoding="utf-8") as f:
                cached = json.load(f)
            timestamp = datetime.fromisoformat(cached["timestamp"])
            if datetime.now() - timestamp < timedelta(hours=max_age_hours):
                return cached["data"]
        except (json.JSONDecodeError, KeyError, ValueError):
            return None
        return None

    def set(self, endpoint: str, params: Dict[str, Any], data: Dict):
        """Save cached API data (JSON file)."""
        cache_file = self._get_cache_file(endpoint, params)
        payload = {
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f)
    
    def save_fixtures(self, fixtures: List[Dict[str, Any]], league_id: int, season: int):
        """Guarda partidos y actualiza el conteo de la temporada"""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Guardar partidos
            for fixture in fixtures:
                fixture_data = {
                    'fixture_id': fixture.get('fixture', {}).get('id'),
                    'league_id': league_id,
                    'season': season,
                    'date': fixture.get('fixture', {}).get('date'),
                    'home_team_id': fixture.get('teams', {}).get('home', {}).get('id'),
                    'away_team_id': fixture.get('teams', {}).get('away', {}).get('id'),
                    'home_goals': fixture.get('goals', {}).get('home'),
                    'away_goals': fixture.get('goals', {}).get('away'),
                    'home_score': json.dumps(fixture.get('score', {}).get('home', {})),
                    'away_score': json.dumps(fixture.get('score', {}).get('away', {})),
                    'timestamp': timestamp
                }
                
                conn.execute('''
                    INSERT OR REPLACE INTO fixtures 
                    (fixture_id, league_id, season, date, home_team_id, away_team_id,
                     home_goals, away_goals, home_score, away_score, timestamp)
                    VALUES 
                    (:fixture_id, :league_id, :season, :date, :home_team_id, :away_team_id,
                     :home_goals, :away_goals, :home_score, :away_score, :timestamp)
                ''', fixture_data)
            
            # Actualizar conteo de temporada
            conn.execute('''
                INSERT OR REPLACE INTO league_seasons 
                (league_id, season, fixtures_count, last_update)
                VALUES (?, ?, ?, ?)
            ''', (league_id, season, len(fixtures), timestamp))
    
    def get_fixtures(self, league_id: int, season: int) -> pd.DataFrame:
        """Obtiene partidos almacenados para una liga y temporada"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM fixtures 
                WHERE league_id = ? AND season = ?
            '''
            return pd.read_sql_query(query, conn, params=(league_id, season))
    
    def save_match_statistics(self, fixture_id: int, league_id: int, season: int,
                            home_team_id: int, away_team_id: int,
                            home_goals: int, away_goals: int,
                            home_stats: Dict[str, Any], away_stats: Dict[str, Any]):
        """Guarda estadísticas de un partido"""
        timestamp = datetime.now().isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO match_statistics 
                (fixture_id, league_id, season, home_team_id, away_team_id,
                 home_goals, away_goals, home_stats, away_stats, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (fixture_id, league_id, season, home_team_id, away_team_id,
                  home_goals, away_goals, json.dumps(home_stats), json.dumps(away_stats),
                  timestamp))
    
    def get_match_statistics(self, league_id: int, season: int) -> pd.DataFrame:
        """Obtiene estadísticas almacenadas para una liga y temporada"""
        with sqlite3.connect(self.db_path) as conn:
            query = '''
                SELECT * FROM match_statistics 
                WHERE league_id = ? AND season = ?
            '''
            df = pd.read_sql_query(query, conn, params=(league_id, season))
            if not df.empty:
                df['home_stats'] = df['home_stats'].apply(json.loads)
                df['away_stats'] = df['away_stats'].apply(json.loads)
            return df
    
    def has_complete_season(self, league_id: int, season: int, max_age_hours: int = 24) -> bool:
        """Verifica si tenemos todos los datos de una temporada"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT fixtures_count, last_update 
                FROM league_seasons 
                WHERE league_id = ? AND season = ?
            ''', (league_id, season))
            
            result = cursor.fetchone()
            if not result:
                return False
                
            count, last_update = result
            last_update_time = datetime.fromisoformat(last_update)
            
            # Si los datos son muy antiguos, considerarlos incompletos
            if datetime.now() - last_update_time > timedelta(hours=max_age_hours):
                return False
                
            # Verificar si tenemos un número razonable de partidos
            expected_matches = {
                39: 380,  # Premier League
                140: 380, # La Liga
                135: 380, # Serie A
                78: 340,  # Bundesliga
                61: 380   # Ligue 1
            }
            
            return count >= expected_matches.get(league_id, 0) * 0.9

class FootballAPI:
    BASE_URL = "https://v3.football.api-sports.io"
    API_KEY = os.getenv("API_FOOTBALL_KEY")
    RATE_LIMIT = 30  # requests per minute
    
    def __init__(self):
        self.session = requests.Session()
        api_key = self.API_KEY or ""
        self.session.headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': api_key
        }
        self.cache = get_cache_manager()
        self.last_request_time = time.time()
        self.request_times = []  # Track last minute of requests
        
    def _respect_rate_limit(self):
        """Ensure we respect the requests per minute limit."""
        min_delay = 60.0 / self.RATE_LIMIT  # minimum time between requests in seconds
        current_time = time.time()

        # Clean up old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if current_time - t < 60]

        # Calculate time since last request
        if self.request_times:
            time_since_last_request = current_time - self.request_times[-1]
            if time_since_last_request < min_delay:
                time_to_wait = min_delay - time_since_last_request
                time.sleep(time_to_wait)

        # If we have too many requests, wait until we're under the limit
        if len(self.request_times) >= self.RATE_LIMIT:
            oldest_request = self.request_times[0]
            wait_time = 60 - (current_time - oldest_request)
            if wait_time > 0:
                time.sleep(wait_time)

        # Add current request timestamp
        current_time = time.time()
        self.request_times.append(current_time)
        self.last_request_time = current_time

    # Alias para mantener compatibilidad con las pruebas
    def get_team_statistics(self, team_id: int, league_id: int, season: str) -> Dict[str, Any]:
        """Alias para get_team_stats para mantener compatibilidad."""
        return self.get_team_stats(team_id)
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     max_age_hours: int = 24, cache_bypass: bool = False) -> Dict[str, Any]:
        """Hace una petición a la API usando el caché"""
        params = params or {}
        
        # Intentar obtener del caché primero, a menos que se omita
        if not cache_bypass and self.cache:
            # Crear una clave única basada en el endpoint y los parámetros
            cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
            cached_data = self.cache.get_data(cache_key)
            if cached_data is not None:
                logger.info(f"Usando datos en caché para {endpoint}")
                return cached_data
        
        # Si no está en caché o se omite, respetar el rate limit
        self._respect_rate_limit()
            
        # Hacer la petición
        response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params)
        
        if response.status_code == 200:
            data = response.json()
            # Guardar en caché
            if self.cache:
                # Usar la misma clave para guardar los datos
                cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
                self.cache.set_data(cache_key, data)
            return data
        else:
            logger.error(f"Error en petición a {endpoint}: {response.status_code}")
            return {"response": []}
    
    def get_fixtures(self, league_id: int, season: int) -> List[Dict[str, Any]]:
        """Obtiene partidos para una liga y temporada."""
        # Si ya tenemos los datos completos en caché, usarlos
        if self.cache and self.has_complete_season(league_id, season):
            logger.info(f"Usando datos completos en caché para liga {league_id} temporada {season}")
            # Convert pandas records to standard Dict[str, Any] with proper typing
            records = self.cache.get_data(f'fixtures_{league_id}_{season}')
            if records and isinstance(records, list):
                return records
            elif records and isinstance(records, dict) and 'response' in records:
                return records.get('response', [])
            return []  # Return empty list as fallback
        
        # Si no, hacer la petición a la API
        params = {
            'league': league_id,
            'season': season
        }
        
        logger.info(f"Obteniendo datos de la API para liga {league_id} temporada {season}")
        data = self._make_request('fixtures', params)
        fixtures = data.get('response', [])
        
        # Guardar en caché si obtuvimos datos
        if fixtures and self.cache:
            self.cache.set_data(f'fixtures_{league_id}_{season}', {"response": fixtures})
        
        return fixtures
    
    def get_historical_data(self, league_id: int, season: int, 
                          force_refresh: bool = False) -> pd.DataFrame:
        """Obtiene datos históricos para entrenamiento de forma eficiente."""
        try:
            # Si no forzamos actualización, intentar obtener del caché
            if not force_refresh and self.cache:
                # Usar clave consistente para los datos históricos
                cache_key = f'historical_data_{league_id}_{season}'
                cached_data = self.cache.get_data(cache_key)
                if cached_data is not None:
                    logger.info(f"Usando datos en caché para liga {league_id} temporada {season}")
                    # Convertir a DataFrame si está en formato de lista de diccionarios
                    if isinstance(cached_data, list):
                        return pd.DataFrame(cached_data)
                    elif isinstance(cached_data, dict) and 'records' in cached_data:
                        return pd.DataFrame(cached_data['records'])
                    elif isinstance(cached_data, pd.DataFrame):
                        return cached_data
                    else:
                        # Si no podemos convertir, retornar DataFrame vacío
                        logger.warning(f"Formato de datos en caché no reconocido para {cache_key}")
                        return pd.DataFrame()
            
            # Si no hay datos en caché o forzamos actualización, obtener de la API
            fixtures = self.get_fixtures(league_id, season)
            if not fixtures:
                logger.warning(f"No se encontraron partidos para liga {league_id} temporada {season}")
                return pd.DataFrame()
                
            logger.info(f"Obteniendo datos de {len(fixtures)} partidos para liga {league_id}")
            
            all_matches = []
            # Procesar en lotes de 10 partidos
            fixture_batches = [fixtures[i:i+10] for i in range(0, len(fixtures), 10)]
            
            for i, batch in enumerate(fixture_batches):
                logger.info(f"Procesando lote {i+1}/{len(fixture_batches)}")
                batch_matches = []
                
                for fixture in batch:
                    try:
                        match_data = self._process_fixture(fixture)
                        if match_data:
                            batch_matches.append(match_data)
                    except Exception as e:
                        logger.warning(f"Error procesando partido {fixture.get('fixture', {}).get('id')}: {e}")
                        continue
                
                all_matches.extend(batch_matches)
                
                # Pequeña pausa entre lotes para no sobrecargar la API
                if i < len(fixture_batches) - 1:
                    time.sleep(1)
            
            if all_matches:  # Verificar que hay partidos procesados
                df = pd.DataFrame(all_matches)
                logger.info(f"Se procesaron exitosamente {len(df)} partidos")
                
                # Guardar en caché - guardamos como una estructura compatible con DataFrame
                if self.cache:
                    cache_key = f'historical_data_{league_id}_{season}'
                    # Convertir DataFrame a formato de diccionario para caché
                    records_dict = {'records': df.to_dict('records')}
                    self.cache.set_data(cache_key, records_dict)
                    
                return df
            else:
                logger.warning(f"No se pudieron procesar partidos para liga {league_id} temporada {season}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
            return pd.DataFrame()
    
    def _process_fixture(self, fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesa un partido individual y sus estadísticas."""
        try:
            fixture_id = fixture.get('fixture', {}).get('id')
            if not fixture_id:
                return None
            
            # Datos básicos del partido
            match_data = {
                'fixture_id': fixture_id,
                'date': fixture.get('fixture', {}).get('date'),
                'home_team_id': fixture.get('teams', {}).get('home', {}).get('id'),
                'away_team_id': fixture.get('teams', {}).get('away', {}).get('id'),
                'home_goals': fixture.get('goals', {}).get('home', 0),
                'away_goals': fixture.get('goals', {}).get('away', 0)
            }
            
            # Verificar campos requeridos
            for key in ['home_team_id', 'away_team_id']:
                if not match_data.get(key):
                    logger.warning(f"Campo requerido '{key}' no encontrado en fixture {fixture_id}")
                    return None
            
            # Obtener estadísticas del partido
            try:
                stats = self._get_fixture_statistics(fixture_id)
                
                # Inicializar estadísticas por defecto
                home_stats = self._get_default_stats()
                away_stats = self._get_default_stats()
                
                # Rellenar con datos reales si están disponibles
                if stats:
                    for team_stats in stats:
                        team_id = team_stats.get('team', {}).get('id')
                        if not team_id:
                            continue
                        
                        is_home = team_id == match_data['home_team_id']
                        team_dict = self._process_team_stats(team_stats)
                        
                        if is_home:
                            home_stats.update(team_dict)
                        else:
                            away_stats.update(team_dict)
                
                # Añadir las estadísticas procesadas al match_data
                match_data['home_stats'] = home_stats
                match_data['away_stats'] = away_stats
                
                # Guardar en caché
                if self.cache:
                    # Crear clave única para las estadísticas de este partido
                    stat_key = f'match_statistics_{fixture_id}'
                    stat_data = {
                        'fixture_id': fixture_id,
                        'league_id': fixture.get('league', {}).get('id'),
                        'season': fixture.get('league', {}).get('season'),
                        'home_team_id': match_data['home_team_id'],
                        'away_team_id': match_data['away_team_id'],
                        'home_goals': match_data['home_goals'],
                        'away_goals': match_data['away_goals'],
                        'home_stats': home_stats,
                        'away_stats': away_stats,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.cache.set_data(stat_key, stat_data)
                
                return match_data
                
            except Exception as stats_error:
                logger.warning(f"Error obteniendo estadísticas para fixture {fixture_id}: {stats_error}")
                return None
            
        except Exception as e:
            logger.warning(f"Error procesando partido {fixture.get('fixture', {}).get('id')}: {e}")
            return None
    
    def _get_default_stats(self) -> Dict[str, float]:
        """Retorna estadísticas por defecto para un equipo."""
        return {
            'shots_on_goal': 0.0,
            'shots_off_goal': 0.0,
            'total_shots': 0.0,
            'blocked_shots': 0.0,
            'shots_insidebox': 0.0,
            'shots_outsidebox': 0.0,
            'fouls': 0.0,
            'corner_kicks': 0.0,
            'offsides': 0.0,
            'ball_possession': 50.0,
            'yellow_cards': 0.0,
            'red_cards': 0.0,
            'goalkeeper_saves': 0.0,
            'total_passes': 0.0,
            'passes_accurate': 0.0,
            'passes_percentage': 0.0
        }
    
    def _get_fixture_statistics(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtiene estadísticas de un partido."""
        params = {'fixture': fixture_id}
        
        # Crear la ruta del directorio para estadísticas
        stats_dir = Path('cache') / 'fixtures' / 'statistics'
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        data = self._make_request('fixtures/statistics', params)
        return data.get('response', [])
    
    def _process_team_stats(self, team_stats: Dict[str, Any]) -> Dict[str, float]:
        """Procesa las estadísticas de un equipo."""
        stats_dict = self._get_default_stats()
        
        # Mapeo de nombres de estadísticas de la API a nuestras claves
        key_mapping = {
            'shots on goal': 'shots_on_goal',
            'shots off goal': 'shots_off_goal',
            'total shots': 'total_shots',
            'blocked shots': 'blocked_shots',
            'shots insidebox': 'shots_insidebox',
            'shots inside box': 'shots_insidebox',
            'shots outsidebox': 'shots_outsidebox',
            'shots outside box': 'shots_outsidebox',
            'fouls': 'fouls',
            'corner kicks': 'corner_kicks',
            'offsides': 'offsides',
            'ball possession': 'ball_possession',
            'yellow cards': 'yellow_cards',
            'red cards': 'red_cards',
            'goalkeeper saves': 'goalkeeper_saves',
            'total passes': 'total_passes',
            'passes accurate': 'passes_accurate',
            'passes %': 'passes_percentage'
        }
        
        for stat in team_stats.get('statistics', []):
            api_key = stat.get('type', '').lower()
            our_key = key_mapping.get(api_key)
            
            if not our_key:
                continue  # Ignorar estadísticas que no reconocemos
                
            value = stat.get('value', 0)
            
            # Convertir valores a float
            try:
                if isinstance(value, str):
                    if '%' in value:
                        value = float(value.replace('%', ''))
                    else:
                        value = float(value)
                elif value is None:
                    value = 0.0
                else:
                    value = float(value)  # Convertir cualquier otro tipo a float
                    
                stats_dict[our_key] = value  # Asignar el valor ya convertido a float
            except (ValueError, TypeError):
                stats_dict[our_key] = 0.0  # Valor por defecto si hay error de conversión
        
        return stats_dict

    def get_team_stats(self, team_id: int) -> Dict[str, Any]:
        """Get team statistics."""
        params = {'team': team_id}
        data = self._make_request('teams/statistics', params)
        
        # Handle both list and dict responses
        stats = data.get('response', {})
        if isinstance(stats, list) and stats:
            stats = stats[0]
        elif isinstance(stats, list):
            stats = {}
            
        # Procesar estadísticas para asegurar que tenemos todos los campos requeridos
        fixtures_data = stats.get('fixtures', {})
        if isinstance(fixtures_data, list) and fixtures_data:
            fixtures_data = fixtures_data[0]
        elif isinstance(fixtures_data, list):
            fixtures_data = {}
            
        total_matches = fixtures_data.get('played', {}).get('total', 1)
        total_corners = stats.get('corners', {}).get('total', {}).get('total', total_matches * 5)
        
        processed_stats = {
            'corners_per_game': total_corners / max(total_matches, 1),
            'cards_per_game': (
                stats.get('cards', {}).get('yellow', {}).get('total', 0) +
                stats.get('cards', {}).get('red', {}).get('total', 0)
            ) / max(total_matches, 1),
            'home_corners_for': stats.get('corners', {}).get('home', total_corners/2) / max(total_matches/2, 1),
            'away_corners_against': stats.get('corners', {}).get('away', total_corners/2) / max(total_matches/2, 1)
        }
        
        return processed_stats

    def get_team_matches(self, team_id: int, limit: int = 5, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent matches for a team."""
        params: Dict[str, Any] = {
            'team': team_id,
            'last': limit
        }
        if end_date:
            params['date'] = end_date

        data = self._make_request('fixtures', params)
        return data.get('response', [])

    def get_league_standings(self, league_id: int, team_id: Optional[int] = None) -> Dict[str, Any]:
        """Get league standings."""
        params = {'league': league_id}
        data = self._make_request('standings', params)
        standings = data.get('response', [{}])[0].get('league', {}).get('standings', [[]])[0]
        
        if team_id:
            for team in standings:
                if team.get('team', {}).get('id') == team_id:
                    return team
        return {}

    def get_team_tactical_stats(self, team_id: int) -> Dict[str, Any]:
        """Get team tactical statistics."""
        params = {'team': team_id}
        data = self._make_request('teams/statistics', params)
        tactics = data.get('response', {}).get('tactics', {})
        
        return {
            'possession': tactics.get('possession', 50),
            'pressing_intensity': tactics.get('pressing_intensity', 0.5),
            'buildup_speed': tactics.get('buildup_speed', 0.5),
            'width': tactics.get('width', 0.5),
            'defensive_line': tactics.get('defensive_line', 0.5),
            'counter_attacks': tactics.get('counter_attacks', 0.5)
        }

    def get_league_stats(self, league_id: int) -> Dict[str, Any]:
        """Get league statistics."""
        params = {'league': league_id}
        data = self._make_request('leagues', params)
        stats = data.get('response', [{}])[0]
        
        return {
            'goals_per_game': stats.get('goals_per_game', 2.7),
            'home_win_rate': stats.get('home_win_rate', 0.45),
            'competitiveness': stats.get('competitiveness', 0.5),
            'tier': stats.get('tier', 0.5)
        }

    def get_team_xg_stats(self, team_id: int) -> Dict[str, Any]:
        """Get team expected goals statistics."""
        params = {'team': team_id}
        data = self._make_request('teams/statistics', params)
        stats = data.get('response', {})
        
        return {
            'xg_for': stats.get('xg_for', 1.4),
            'xg_against': stats.get('xg_against', 1.2),
            'xg_ratio': stats.get('xg_ratio', 1.0)
        }

    def get_player_statistics(self, player_id: int, season: int, league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get detailed player statistics for a season.
        
        Args:
            player_id: The ID of the player
            season: The season year (e.g., 2024)
            league_id: Optional league ID to filter statistics
            
        Returns:
            Dictionary containing player statistics
        """
        params = {
            'id': player_id,
            'season': season
        }
        if league_id:
            params['league'] = league_id
            
        return self._make_request('players', params)

    def get_live_odds(self, fixture_id: Optional[int] = None, league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get live (in-play) odds for fixtures.
        
        Args:
            fixture_id: Optional specific fixture ID
            league_id: Optional league ID to filter odds
            
        Returns:
            Dictionary containing live odds data
        """
        params = {}
        if fixture_id:
            params['fixture'] = fixture_id
        if league_id:
            params['league'] = league_id
            
        return self._make_request('odds/live', params)

    def get_live_odds_bets(self, bet_id: Optional[int] = None) -> Dict[str, Any]:
        """Get available bet types for live odds.
        
        Args:
            bet_id: Optional specific bet ID to filter
            
        Returns:
            Dictionary containing available live bet types
        """
        params = {}
        if bet_id:
            params['id'] = bet_id
            
        return self._make_request('odds/live/bets', params)

    def get_coach_info(self, coach_id: Optional[int] = None, team_id: Optional[int] = None, search: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed coach information.
        
        Args:
            coach_id: Optional ID of specific coach
            team_id: Optional team ID to get current coach
            search: Optional coach name to search
            
        Returns:
            Dictionary containing coach information including team history and trophies
        """
        params = {}
        if coach_id:
            params['id'] = coach_id
        if team_id:
            params['team'] = team_id
        if search:
            params['search'] = search
            
        return self._make_request('coachs', params)

    def get_transfers(self, player_id: Optional[int] = None, team_id: Optional[int] = None) -> Dict[str, Any]:
        """Get transfer history.
        
        Args:
            player_id: Optional player ID to get their transfer history
            team_id: Optional team ID to get their transfer history
            
        Returns:
            Dictionary containing transfer information
        """
        params = {}
        if player_id:
            params['player'] = player_id
        if team_id:
            params['team'] = team_id
            
        return self._make_request('transfers', params)

    def get_fixture_players(self, fixture_id: int) -> Dict[str, Any]:
        """Get player statistics for a fixture.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            Dictionary containing player statistics for the fixture
        """
        params = {'fixture': fixture_id}
        return self._make_request('fixtures/players', params)

    def has_complete_season(self, league_id: int, season: int) -> bool:
        """Verifica si tenemos datos completos para una liga y temporada en el caché."""
        if not self.cache:
            return False
            
        # Intentar obtener datos desde el caché
        cache_key = f'fixtures_{league_id}_{season}'
        fixtures = self.cache.get_data(cache_key)
        
        if not fixtures:
            return False
            
        # Verificar si hay suficientes partidos en la temporada
        expected_matches = {
            39: 380,  # Premier League
            140: 380, # La Liga
            135: 380, # Serie A
            78: 340,  # Bundesliga
            61: 380   # Ligue 1
        }
        
        # Comprobar si tenemos al menos el 90% de los partidos esperados
        min_fixtures = expected_matches.get(league_id, 0) * 0.9
        return len(fixtures) >= min_fixtures
        
def get_upcoming_fixtures(days_ahead: int = 3) -> List[Dict[str, Any]]:
    """
    Get upcoming fixture data for the specified number of days ahead.
    
    Args:
        days_ahead: Number of days ahead to fetch fixtures for
        
    Returns:
        List of fixture data dictionaries
    """
    try:
        api = get_api_instance()
        
        # Calculate date range
        today = datetime.now().date()
        end_date = today + timedelta(days=days_ahead)
        
        # Format dates for API request
        from_date = today.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")
        
        # Request fixtures in date range
        params = {
            'from': from_date,
            'to': to_date,
            'status': 'NS'  # Not Started
        }
        
        response = api._make_request('fixtures', params)
        
        if response and "response" in response:
            return response.get("response", [])
        else:
            logger.warning(f"No upcoming fixtures found for next {days_ahead} days")
            return []
            
    except Exception as e:
        logger.error(f"Error getting upcoming fixtures: {e}")
        return []

