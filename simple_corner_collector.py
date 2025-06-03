"""
Módulo para recolectar datos de córners y formaciones de partidos de fútbol.
"""
import time
import logging
import json
import sys
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime
from tactical_analysis import FormationAnalyzer
from data_cache import DataCache
from batch_processor import BatchProcessor
import json  # Added for JSON handling in validate_api_key

class FootballDataCollector:
    """Clase para recolectar datos de córners de la API de fútbol."""
    
    def __init__(self, api_key: str, api_base_url: str, api_host: str,
                 batch_size: int = 3,
                 min_request_interval: float = 5.0,
                 max_retries: int = 3):
        """
        Inicializar el recolector con credenciales de API y configuración.
        
        Args:
            api_key: API key para autenticación
            api_base_url: URL base de la API
            api_host: Host de la API
            batch_size: Tamaño del lote para procesar partidos
            min_request_interval: Intervalo mínimo entre requests (segundos)
            max_retries: Número máximo de reintentos por request
        """          
        # Configurar logging primero
        logging.basicConfig(
            filename='corner_collection.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)        # Configurar API
        self.api_key = api_key.strip()
        self.api_base_url = api_base_url.rstrip('/')
        self.api_host = api_host          
        self.headers = {
            'x-apisports-key': self.api_key  # API key directa de api-football.com
        }
        self.logger.debug(f"Initialized with API key: {self.api_key[:8]}...")
        
        # Configuración de componentes
        self.formation_analyzer = FormationAnalyzer()
        self.cache = DataCache()  # Using default cache settings
        self.batch_processor = BatchProcessor(batch_size=batch_size, max_workers=1)
        
        # Configuración de rate limiting
        self.last_request_time = 0
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Hacer una petición a la API con caché y manejo de rate limiting.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            
        Returns:
            Respuesta de la API
        """
        params_dict = params if params is not None else {}
        
        # Intentar obtener del caché primero
        cached_data = self.cache.get(endpoint, params_dict)
        if cached_data is not None:
            self.logger.info(f"Cache hit para {endpoint}")
            return cached_data

        # Control de rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting: esperando {sleep_time:.2f} segundos")
            time.sleep(sleep_time)
        
        # Reintentos con backoff exponencial
        for attempt in range(self.max_retries):
            try:
                url = f"{self.api_base_url}/{endpoint}"
                self.logger.debug(f"Request {attempt + 1}/{self.max_retries}: {url}")
                response = requests.get(url, headers=self.headers, params=params_dict)
                self.last_request_time = time.time()

                if response.status_code == 200:
                    # Get rate limit info from headers
                    remaining = response.headers.get('x-ratelimit-remaining', '0')
                    reset_time = response.headers.get('x-ratelimit-reset', '0')
                    
                    # Log rate limit status
                    self.logger.debug(f"Rate limit remaining: {remaining}")
                    if int(remaining) < 5:
                        self.logger.warning(f"Rate limit running low: {remaining} requests remaining")
                    
                    data = response.json()
                    self.cache.set(endpoint, params_dict, data)
                    return data
                    
                elif response.status_code == 429:  # Rate limit
                    # Get reset time from headers or use exponential backoff
                    reset_time = response.headers.get('x-ratelimit-reset')
                    if reset_time:
                        wait_time = max(1, int(reset_time) - int(time.time()))
                        self.logger.warning(f"Rate limit alcanzado, esperando hasta reset: {wait_time}s")
                    else:
                        wait_time = (2 ** attempt) * 5  # Espera exponencial
                        self.logger.warning(f"Rate limit alcanzado, esperando {wait_time}s")
                    time.sleep(wait_time)
                    continue
                    
                else:
                    self.logger.error(f"Error {response.status_code} en {url}")
                    self.logger.error(f"Headers: {self.headers}")
                    self.logger.error(f"Params: {params_dict}")
                    try:
                        error_data = response.json()
                        self.logger.error(f"Error Response: {error_data}")
                    except:
                        self.logger.error(f"Response Text: {response.text}")
                    raise Exception(f"API request failed: {response.status_code}")
            
            except Exception as e:
                if attempt == self.max_retries - 1:  # Último intento
                    self.logger.error(f"Error final en {endpoint}: {str(e)}")
                    raise
                wait_time = (2 ** attempt) * 5
                self.logger.warning(f"Error en intento {attempt + 1}, reintentando en {wait_time}s")
                time.sleep(wait_time)
        
        raise Exception("Máximo número de reintentos alcanzado")
    
    def get_fixtures(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """Obtener lista de partidos."""
        params = {'league': league_id, 'season': season}
        response = self._make_request('fixtures', params)
        return response.get('response', [])

    def get_fixture_lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtener datos de alineación de un partido."""
        params = {'fixture': fixture_id}
        response = self._make_request('fixtures/lineups', params)
        return response.get('response', [])

    def get_fixture_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtener estadísticas de un partido."""
        params = {'fixture': fixture_id}
        response = self._make_request('fixtures/statistics', params)
        if not response or 'response' not in response:
            self.logger.warning(f"No statistics data available for fixture {fixture_id}")
            return []
        stats = response.get('response', [])
        if not stats:
            self.logger.warning(f"Empty statistics data for fixture {fixture_id}")
            return []
        return stats
        
    def collect_corner_data(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """
        Recolectar datos de córners y formaciones.
        
        Args:
            league_id: ID de la liga
            season: Temporada (ej: "2023")
        
        Returns:
            Lista de eventos de córner con datos de formación
        """
        try:
            self.logger.info(f"Iniciando recolección para liga {league_id}, temporada {season}")
            fixtures = self.get_fixtures(league_id, season)
            
            if not fixtures:
                self.logger.warning("No se encontraron partidos")
                return []
            
            # For testing, limit to first 5 fixtures
            if len(fixtures) > 5:
                self.logger.info("Modo prueba: limitando a 5 partidos")
                fixtures = fixtures[:5]
            
            self.logger.info(f"Procesando {len(fixtures)} partidos")
            
            # Procesar partidos en lotes
            corner_data = []
            desc = f"Procesando partidos de liga {league_id}"
            results = self.batch_processor.process_batches(fixtures, self._process_single_fixture, desc)
            
            for result in results:
                if result is not None:
                    corner_data.append(result)
                    self.logger.info(f"Procesado partido {result['fixture_id']}: Home {result['home_corners']} - Away {result['away_corners']}")

            self.logger.info(f"Recolección completada. Total eventos: {len(corner_data)}")
            return corner_data
            
        except Exception as e:
            self.logger.error(f"Error en collect_corner_data: {str(e)}")
            return []

    def _process_single_fixture(self, fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single fixture to extract corner and formation data."""
        if not isinstance(fixture, dict):
            return None
        
        fixture_data = fixture.get('fixture', {})
        if not isinstance(fixture_data, dict):
            return None
        
        fixture_id = fixture_data.get('id')
        if not fixture_id:
            return None
        
        try:
            # Obtener estadísticas
            stats = self.get_fixture_stats(fixture_id)
            
            # Obtener formaciones
            lineups = self.get_fixture_lineups(fixture_id)
            if not isinstance(lineups, list) or len(lineups) < 2:
                self.logger.warning(f"Datos de alineación inválidos para partido {fixture_id}")
                formation_analysis = {
                    'home_formation': 'Unknown',
                    'away_formation': 'Unknown',
                    'tactical_analysis': 'No disponible - datos inválidos'
                }
            else:
                formation_analysis = self.analyze_formations(lineups)
            
            # Procesar estadísticas de córners
            home_corners = 0
            away_corners = 0
            
            for team_stats in stats:
                if not isinstance(team_stats, dict):
                    continue
                
                statistics = team_stats.get('statistics', [])
                if not isinstance(statistics, list):
                    continue
                
                for stat in statistics:
                    if not isinstance(stat, dict):
                        continue
                    
                    if stat.get('type') == 'Corner Kicks':
                        value = stat.get('value', '0')
                        if value and value != 'null':
                            try:
                                corners = int(value)
                                teams = fixture.get('teams', {})
                                home_team = teams.get('home', {})
                                team_info = team_stats.get('team', {})
                                
                                if team_info.get('id') == home_team.get('id'):
                                    home_corners = corners
                                else:
                                    away_corners = corners
                            except (ValueError, TypeError):
                                continue

            # Si hay córners, crear evento
            if home_corners > 0 or away_corners > 0:
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                return {                    'fixture_id': fixture_id,
                    'home_team_id': teams.get('home', {}).get('id'),
                    'away_team_id': teams.get('away', {}).get('id'),
                    'home_corners': home_corners,
                    'away_corners': away_corners,
                    'score_home': goals.get('home'),
                    'score_away': goals.get('away'),
                    'date': fixture_data.get('date'),
                    'formations': formation_analysis
                }
        
        except Exception as e:
            self.logger.error(f"Error procesando partido {fixture_id}: {str(e)}")
            return None
        
        return None

    def analyze_formations(self, lineups_response: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza las formaciones de ambos equipos y sus implicaciones tácticas.
        
        Args:
            lineups_response: Lista de alineaciones de los equipos
            
        Returns:
            Diccionario con análisis detallado de las formaciones
        """
        if not isinstance(lineups_response, list) or len(lineups_response) < 2:
            self.logger.warning("Datos de alineación inválidos o incompletos")
            return {
                'home_formation': 'Unknown',
                'away_formation': 'Unknown',
                'tactical_analysis': 'No disponible - datos incompletos'
            }

        # Identificar formaciones
        home_formation = self.formation_analyzer.identify_formation(lineups_response[0] if isinstance(lineups_response[0], dict) else {})
        away_formation = self.formation_analyzer.identify_formation(lineups_response[1] if isinstance(lineups_response[1], dict) else {})

        # Analizar el matchup táctico
        formation_analysis = self.formation_analyzer.analyze_formation_matchup(home_formation, away_formation)
        
        # Enriquecer el análisis con detalles adicionales
        result = {
            'home_formation': home_formation,
            'away_formation': away_formation,
            'tactical_analysis': formation_analysis.get('analysis', 'No disponible'),
            'predicted_possession': formation_analysis.get('possession_prediction'),
            'tactical_advantages': formation_analysis.get('advantages', []),
            'team_names': {
                'home': lineups_response[0].get('team', {}).get('name', 'Unknown') if isinstance(lineups_response[0], dict) else 'Unknown',
                'away': lineups_response[1].get('team', {}).get('name', 'Unknown') if isinstance(lineups_response[1], dict) else 'Unknown'
            }
        }
        
        self.logger.info(f"Análisis de formación completado: {home_formation} vs {away_formation}")
        return result    
    def _validate_api_key(self) -> bool:
        """Validar que la API key tenga acceso a los endpoints necesarios."""
        try:
            url = f"{self.api_base_url}/status"
            print(f"Validando API key con URL: {url}")  # Print for immediate feedback
            self.logger.debug(f"Validando API key con URL: {url}")
            self.logger.debug(f"Headers: {self.headers}")
            
            response = requests.get(url, headers=self.headers)
            print(f"Status code: {response.status_code}")  # Print for immediate feedback
            self.logger.debug(f"Status code: {response.status_code}")
            
            # Try to get response content
            try:
                data = response.json()
                print(f"Response data: {json.dumps(data, indent=2)}")  # Print for immediate feedback
                self.logger.debug(f"Response data: {json.dumps(data, indent=2)}")
                
                if response.status_code == 200:
                    account = data.get('response', {}).get('account', {})
                    subscription = data.get('response', {}).get('subscription', {})                    
                    request_stats = data.get('response', {}).get('requests', {})
                    
                    print(f"Account: {account.get('firstname')} {account.get('lastname')}")
                    print(f"Subscription: {subscription.get('plan')} (Active: {subscription.get('active')})")
                    print(f"Requests: {request_stats.get('current')}/{request_stats.get('limit_day')} today")
                    
                    return True
            except json.JSONDecodeError:
                print(f"Invalid JSON response: {response.text}")  # Print for immediate feedback
                self.logger.error(f"Invalid JSON response: {response.text}")
                return False
                
            error_msg = data.get('message', 'Unknown error')
            print(f"API key inválida: {error_msg}")  # Print for immediate feedback
            self.logger.error(f"API key inválida: {error_msg}")
            return False
            
        except Exception as e:
            print(f"Error validando API key: {str(e)}")  # Print for immediate feedback
            self.logger.error(f"Error validando API key: {str(e)}")
            return False

if __name__ == '__main__':
    try:
        from config import API_FOOTBALL_KEY, API_BASE_URL, API_HOST
    except ImportError:
        print("Error: No se pudo importar la configuración")
        sys.exit(1)

    if not API_FOOTBALL_KEY:
        print("Error: API key no encontrada")
        sys.exit(1)
        
    try:
        collector = FootballDataCollector(API_FOOTBALL_KEY, API_BASE_URL, API_HOST)
        
        # Validar API key antes de continuar
        if not collector._validate_api_key():
            print("Error: API key inválida o sin acceso a los endpoints necesarios")
            sys.exit(1)
            
        # Premier League 2023
        league_id = 39
        season = "2023"
        
        print(f"Iniciando recolección para liga {league_id}, temporada {season}")
        corner_data = collector.collect_corner_data(league_id, season)
        
        # Guardar resultados
        if corner_data:
            filename = f"corner_data_{league_id}_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(corner_data, f, indent=2)
            print(f"\nDatos guardados en {filename}")
            print(f"Total eventos: {len(corner_data)}")
        else:
            print("\nNo se encontraron datos")
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        sys.exit(1)
