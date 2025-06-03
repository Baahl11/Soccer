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

class FootballDataCollector:
    """Clase para recolectar datos de córners de la API de fútbol."""
    
    def __init__(self, api_key: str, api_base_url: str, api_host: str,
                 batch_size: int = 3,
                 min_request_interval: float = 5.0,
                 max_retries: int = 3):
        """Inicializar el recolector con credenciales de API."""
        self.api_key = api_key.strip()
        self.api_base_url = api_base_url.rstrip('/')
        self.api_host = api_host
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        }
        self.formation_analyzer = FormationAnalyzer()
        self.cache = DataCache()
        self.batch_processor = BatchProcessor(batch_size=batch_size, max_workers=1)
        self.last_request_time = 0
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        
        logging.basicConfig(
            filename='corner_collection.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _validate_api_key(self) -> bool:
        """Validar que la API key tenga acceso a los endpoints necesarios."""
        try:
            response = requests.get(
                f"{self.api_base_url}/status",
                headers=self.headers
            )
            if response.status_code == 200:
                return True
            error_msg = response.json().get('message', 'Unknown error')
            self.logger.error(f"API key inválida: {error_msg}")
            return False
        except Exception as e:
            self.logger.error(f"Error validando API key: {str(e)}")
            return False

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Hacer una petición a la API con caché y rate limiting."""
        params_dict = params if params is not None else {}
        
        cached_data = self.cache.get(endpoint, params_dict)
        if cached_data is not None:
            self.logger.info(f"Cache hit para {endpoint}")
            return cached_data

        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            self.logger.debug(f"Rate limiting: esperando {sleep_time:.2f} segundos")
            time.sleep(sleep_time)
        
        for attempt in range(self.max_retries):
            try:
                url = f"{self.api_base_url}/{endpoint}"
                self.logger.debug(f"Request {attempt + 1}/{self.max_retries}: {url}")
                response = requests.get(url, headers=self.headers, params=params_dict)
                self.last_request_time = time.time()

                if response.status_code == 200:
                    data = response.json()
                    self.cache.set(endpoint, params_dict, data)
                    return data
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 5
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
                if attempt == self.max_retries - 1:
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

    def get_fixture_stats(self, fixture_id: int) -> Dict[str, Any]:
        """Obtener estadísticas de un partido."""
        params = {'fixture': fixture_id}
        response = self._make_request('fixtures/statistics', params)
        return response.get('response', [])

    def get_fixture_lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtener datos de alineación de un partido."""
        params = {'fixture': fixture_id}
        response = self._make_request('fixtures/lineups', params)
        return response.get('response', [])

    def analyze_formations(self, lineups_response: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar las formaciones de ambos equipos."""
        try:
            if not isinstance(lineups_response, list) or len(lineups_response) < 2:
                self.logger.warning("Datos de alineación inválidos o incompletos")
                return {
                    'home_formation': 'Unknown',
                    'away_formation': 'Unknown',
                    'tactical_analysis': 'No disponible - datos incompletos'
                }

            home_formation = self.formation_analyzer.identify_formation(lineups_response[0])
            away_formation = self.formation_analyzer.identify_formation(lineups_response[1])
            formation_analysis = self.formation_analyzer.analyze_formation_matchup(
                home_formation, away_formation
            )
            
            result = {
                'home_formation': home_formation,
                'away_formation': away_formation,
                'tactical_analysis': formation_analysis.get('analysis', 'No disponible'),
                'predicted_possession': formation_analysis.get('possession_prediction'),
                'tactical_advantages': formation_analysis.get('advantages', []),
                'team_names': {
                    'home': lineups_response[0].get('team', {}).get('name', 'Unknown'),
                    'away': lineups_response[1].get('team', {}).get('name', 'Unknown')
                }
            }
            
            self.logger.info(f"Análisis completado: {home_formation} vs {away_formation}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error en análisis: {str(e)}")
            return {
                'home_formation': 'Error',
                'away_formation': 'Error',
                'tactical_analysis': f'Error en análisis: {str(e)}'
            }

    def collect_corner_data(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """Recolectar datos de córners y formaciones."""
        try:
            self.logger.info(f"Iniciando recolección para liga {league_id}, temporada {season}")
            fixtures = self.get_fixtures(league_id, season)
            if not fixtures:
                self.logger.warning("No se encontraron partidos")
                return []
                
            self.logger.info(f"Encontrados {len(fixtures)} partidos")
            
            def process_fixture(fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
                if not isinstance(fixture, dict):
                    return None
                    
                fixture_data = fixture.get('fixture', {})
                if not isinstance(fixture_data, dict):
                    return None
                    
                fixture_id = fixture_data.get('id')
                if not fixture_id:
                    return None

                try:
                    stats = self.get_fixture_stats(fixture_id)
                    lineups = self.get_fixture_lineups(fixture_id)
                    
                    if not isinstance(lineups, list):
                        self.logger.warning(f"Datos de alineación inválidos para partido {fixture_id}")
                        formation_analysis = {
                            'home_formation': 'Unknown',
                            'away_formation': 'Unknown',
                            'tactical_analysis': 'No disponible - datos inválidos'
                        }
                    else:
                        formation_analysis = self.analyze_formations(lineups)
                    
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

                    if home_corners > 0 or away_corners > 0:
                        teams = fixture.get('teams', {})
                        goals = fixture.get('goals', {})
                        
                        return {
                            'fixture_id': fixture_id,
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

            corner_data = []
            desc = f"Procesando partidos de liga {league_id}"
            results = self.batch_processor.process_batches(fixtures, process_fixture, desc)
            
            for result in results:
                if result is not None:
                    corner_data.append(result)
                    self.logger.info(
                        f"Procesado partido {result['fixture_id']}: "
                        f"Home {result['home_corners']} - Away {result['away_corners']}"
                    )

            self.logger.info(f"Recolección completada. Total eventos: {len(corner_data)}")
            return corner_data
            
        except Exception as e:
            self.logger.error(f"Error en recolección: {str(e)}")
            return []

if __name__ == '__main__':
    from config import API_FOOTBALL_KEY, API_BASE_URL, API_HOST
    
    if not API_FOOTBALL_KEY:
        print("Error: API key no encontrada")
        sys.exit(1)
        
    try:
        collector = FootballDataCollector(API_FOOTBALL_KEY, API_BASE_URL, API_HOST)
        
        if not collector._validate_api_key():
            print("Error: API key inválida o sin acceso a los endpoints necesarios")
            sys.exit(1)
            
        league_id = 39  # Premier League
        season = "2023"
        
        print(f"Iniciando recolección para liga {league_id}, temporada {season}")
        
        fixtures = collector.get_fixtures(league_id, season)
        if not fixtures:
            print("No se encontraron partidos")
            sys.exit(1)
            
        if len(fixtures) > 5:
            print("Modo prueba: limitando a 5 partidos")
            fixtures = fixtures[:5]
            
        corner_data = collector.collect_corner_data(league_id, season)
        
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
