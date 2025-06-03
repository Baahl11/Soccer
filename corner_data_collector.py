"""
Módulo para recolectar datos de córners de partidos de fútbol.
"""
import logging
import time
import json
import sys
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime
from formation_analyzer import FormationAnalyzer
from data_cache import DataCache
from batch_processor import BatchProcessor

class FootballDataCollector:
    """Clase para recolectar datos de córners de la API de fútbol."""
    
    def __init__(self, api_key: str = '', api_base_url: str = '', api_host: str = ''):
        """
        Inicializar el recolector con credenciales de API.
        
        Args:
            api_key: Clave de la API
            api_base_url: URL base de la API
            api_host: Host de la API
            
        Raises:
            ValueError: Si no se proporciona una clave de API válida
        """
        if not api_key:
            raise ValueError("Se requiere una clave de API válida")
        if not api_base_url:
            api_base_url = 'https://api-football-v1.p.rapidapi.com/v3'
        if not api_host:
            api_host = 'api-football-v1.p.rapidapi.com'

        # Verificar formato de la API key
        if not api_key or '"' in api_key:
            raise ValueError("API key inválida o mal formateada")
        
        # Configurar logging
        logging.basicConfig(
            filename='corner_data_collector.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        self.api_key = api_key.strip()  # Eliminar espacios en blanco
        self.api_base_url = api_base_url.rstrip('/')  # Eliminar slash final si existe
        self.api_host = api_host
        self.headers = {
            'x-rapidapi-key': self.api_key,
            'x-rapidapi-host': self.api_host
        }
        
        # Rate limiting
        self.min_request_interval = 5.0  # 5 seconds between requests
        self.last_request_time = 0
        
        # Initialize components
        self.formation_analyzer = FormationAnalyzer()
        self.cache = DataCache()
        self.batch_processor = BatchProcessor(batch_size=3, max_workers=1)

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> Dict[str, Any]:
        """
        Hacer una petición a la API con caché y manejo de rate limiting.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            max_retries: Número máximo de reintentos
            
        Returns:
            Respuesta de la API
        """
        params_dict = params if params is not None else {}
        
        # Intentar obtener del caché primero
        cached_data = self.cache.get(endpoint, params_dict)
        if cached_data is not None:
            return cached_data

        # Control de rate limiting
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            time.sleep(sleep_time)
        
        # Reintentos con backoff exponencial
        for attempt in range(max_retries):
            try:
                url = f"{self.api_base_url}/{endpoint}"
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    params=params_dict,
                    # Deshabilitar paginación
                    stream=True
                )
                self.last_request_time = time.time()
                if response.status_code == 200:
                    data = response.json()
                    self.cache.set(endpoint, params_dict, data)
                    return data
                elif response.status_code == 429:  # Rate limit
                    wait_time = (2 ** attempt) * 5  # Espera exponencial: 5s, 10s, 20s
                    print(f"\nRate limit alcanzado, esperando {wait_time} segundos...")
                    time.sleep(wait_time)
                    continue
                else:
                    # Imprimir información de debug sin paginación
                    print("\nError en la llamada a la API:")
                    print(f"URL: {url}")
                    print(f"Headers: {self.headers}")
                    print(f"Params: {params_dict}")
                    print(f"Status Code: {response.status_code}")
                    try:
                        error_data = response.json()
                        print(f"Error Response: {error_data}")
                    except:
                        print(f"Response Text: {response.text}")
                    raise Exception(f"API request failed: {response.status_code}")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (2 ** attempt) * 5
                print(f"\nError en intento {attempt + 1}, reintentando en {wait_time} segundos...")
                time.sleep(wait_time)
        
        raise Exception("Máximo número de reintentos alcanzado")

    def _make_api_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the football API.
        
        Args:
            endpoint: API endpoint to call
            params: Query parameters (optional)
            
        Returns:
            API response as a dictionary
        """
        url = f"{self.api_base_url}/{endpoint}"
        if params is None:
            params = {}
        
        try:
            # Respect rate limiting
            time_since_last = time.time() - self.last_request_time
            if time_since_last < self.min_request_interval:
                time.sleep(self.min_request_interval - time_since_last)
            
            response = requests.get(url, headers=self.headers, params=params)
            self.last_request_time = time.time()
            
            if response.status_code == 429:
                logging.warning("Rate limit reached, waiting...")
                time.sleep(65)  # Wait 65 seconds before retrying
                return self._make_api_request(endpoint, params)
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logging.error(f"API request error for {endpoint}: {str(e)}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error for {endpoint}: {str(e)}")
            return {}

    def _process_corner_events(
        self,
        fixture: Dict[str, Any],
        statistics: List[Dict[str, Any]],
        events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Procesar eventos de córner de un partido.
        
        Args:
            fixture: Datos del partido
            statistics: Estadísticas del partido
            events: Eventos del partido
            
        Returns:
            Lista de eventos de córner procesados
        """
        result = []
        
        # Extraer estadísticas de córners
        home_corners = 0
        away_corners = 0
        
        # Procesar estadísticas
        for team_stats in statistics:
            team_id = team_stats.get('team', {}).get('id')
            if not team_id:
                continue
                
            for stat in team_stats.get('statistics', []):
                if stat.get('type') == 'Corner Kicks':
                    value = stat.get('value')
                    if value is None or value == 'null':
                        continue
                        
                    try:
                        corner_count = int(value)
                    except (ValueError, TypeError):
                        continue
                        
                    if team_id == fixture['teams']['home']['id']:
                        home_corners = corner_count
                    else:
                        away_corners = corner_count
        
        # Obtener y analizar formaciones
        match_id = fixture['fixture']['id']
        formation_data = self.collect_formation_data(match_id)
        
        home_team_id = fixture['teams']['home']['id']
        away_team_id = fixture['teams']['away']['id']
        
        home_formation = formation_data.get(str(home_team_id), {})
        away_formation = formation_data.get(str(away_team_id), {})
        
        # Calcular features tácticas
        tactical_features = self.calculate_formation_features(home_formation, away_formation)
        
        # Si encontramos corners en las estadísticas, crear un evento por cada corner
        if home_corners > 0 or away_corners > 0:
            # Crear eventos para corners del equipo local
            for i in range(home_corners):
                event_data = {
                    'fixture_id': fixture['fixture']['id'],
                    'timestamp': -1,  # No tenemos el minuto exacto
                    'team_id': fixture['teams']['home']['id'],
                    'is_home': True,
                    'total_home_corners': home_corners,
                    'total_away_corners': away_corners,
                    'score_home': fixture.get('goals', {}).get('home', 0),
                    'score_away': fixture.get('goals', {}).get('away', 0),
                    'home_formation': home_formation.get('formation'),
                    'away_formation': away_formation.get('formation'),
                    'wing_play_advantage': tactical_features.get('wing_play_advantage', 0),
                    'possession_advantage': tactical_features.get('possession_advantage', 0),
                    'pressing_advantage': tactical_features.get('pressing_advantage', 0),
                    'positional_dominance': tactical_features.get('positional_dominance', 1.0)
                }
                result.append(event_data)
            
            # Crear eventos para corners del equipo visitante
            for i in range(away_corners):
                event_data = {
                    'fixture_id': fixture['fixture']['id'],
                    'timestamp': -1,  # No tenemos el minuto exacto
                    'team_id': fixture['teams']['away']['id'],
                    'is_home': False,
                    'total_home_corners': home_corners,
                    'total_away_corners': away_corners,
                    'score_home': fixture.get('goals', {}).get('home', 0),
                    'score_away': fixture.get('goals', {}).get('away', 0),
                    'home_formation': home_formation.get('formation'),
                    'away_formation': away_formation.get('formation'),
                    'wing_play_advantage': -tactical_features.get('wing_play_advantage', 0),  # Invertir para equipo visitante
                    'possession_advantage': -tactical_features.get('possession_advantage', 0),  # Invertir para equipo visitante
                    'pressing_advantage': -tactical_features.get('pressing_advantage', 0),  # Invertir para equipo visitante
                    'positional_dominance': 1/tactical_features.get('positional_dominance', 1.0)  # Invertir para equipo visitante
                }
                result.append(event_data)
        
        return result

    def get_fixtures(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """
        Obtener partidos con caché.
        
        Args:
            league_id: ID de la liga
            season: Temporada
            
        Returns:
            Lista de partidos
        """
        params = {'league': league_id, 'season': season}
        response = self._make_request('fixtures', params)
        return response.get('response', [])

    def get_fixture_data(self, fixture_id: int) -> Dict[str, Any]:
        """
        Obtener datos combinados de un partido.
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Datos del partido incluyendo estadísticas y formaciones
        """
        # Obtener estadísticas usando el endpoint correcto
        stats = self._make_request('fixtures/statistics', {'fixture': fixture_id})
        
        # Obtener formaciones
        formations = self.get_lineup_formation(fixture_id)
        
        return {
            'fixture_id': fixture_id,
            'statistics': stats.get('response', []),
            'events': [],  # Ya no necesitamos los eventos ya que usamos las estadísticas
            'formations': formations
        }

    def collect_corner_data(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """
        Collect corner and formation data for matches in a specific league and season.
        
        Args:
            league_id: ID of the league
            season: Season year (e.g., "2023")
            
        Returns:
            List of processed match data
        """
        try:
            # Get fixtures for the league and season
            params = {
                'league': league_id,
                'season': season
            }
            fixtures_response = self._make_api_request('fixtures', params)
            
            if not fixtures_response or 'response' not in fixtures_response:
                logging.error(f"No fixtures found for league {league_id}, season {season}")
                return []
            
            fixtures = fixtures_response['response']
            logging.info(f"Found {len(fixtures)} fixtures for league {league_id}, season {season}")
            
            # Process matches in batches
            processed_matches = []
            for fixture in fixtures:
                try:
                    match_data = self.process_match(fixture)
                    if match_data:
                        processed_matches.append(match_data)
                except Exception as e:
                    logging.error(f"Error processing match {fixture.get('fixture', {}).get('id')}: {str(e)}")
                    continue
            
            logging.info(f"Successfully processed {len(processed_matches)} matches")
            return processed_matches
            
        except Exception as e:
            logging.error(f"Error collecting corner data: {str(e)}")
            return []

    def get_lineup_formation(self, fixture_id: int) -> Dict[str, Any]:
        """
        Obtener las formaciones de los equipos para un partido.
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Diccionario con formaciones de local y visitante
        """
        lineup_data = self._make_request('fixtures/lineups', {'fixture': fixture_id})
        formations = {
            'home': None,
            'away': None
        }
        
        for team_lineup in lineup_data.get('response', []):
            team_id = team_lineup.get('team', {}).get('id')
            formation = team_lineup.get('formation')
            
            if not team_id or not formation:
                continue
                
            if team_id == lineup_data.get('response', [{}])[0].get('team', {}).get('id'):
                formations['home'] = formation
            else:
                formations['away'] = formation
        
        return formations

    def collect_formation_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect and process formation data for a match.
        
        Args:
            match_data: Raw match data from the API
            
        Returns:
            Dictionary with processed formation features
        """
        try:
            home_formation = match_data.get('teams', {}).get('home', {}).get('formation', '')
            away_formation = match_data.get('teams', {}).get('away', {}).get('formation', '')
            
            if not home_formation or not away_formation:
                logging.warning(f"Missing formation data for match {match_data.get('fixture', {}).get('id')}")
                return {}
                
            # Get tactical features from FormationAnalyzer
            formation_features = self.formation_analyzer.predict_corner_impact(home_formation, away_formation)
            
            # Add raw formation data
            formation_features.update({
                'home_formation': home_formation,
                'away_formation': away_formation
            })
            
            # Get detailed tactical analysis
            home_analysis = self.formation_analyzer.analyze_formation(home_formation)
            away_analysis = self.formation_analyzer.analyze_formation(away_formation)
            
            # Add tactical indices
            formation_features.update({
                'home_wing_play': home_analysis['wing_play'],
                'home_possession': home_analysis['possession'],
                'home_pressing': home_analysis['pressing'],
                'away_wing_play': away_analysis['wing_play'],
                'away_possession': away_analysis['possession'],
                'away_pressing': away_analysis['pressing']
            })
            
            return formation_features
            
        except Exception as e:
            logging.error(f"Error processing formation data: {str(e)}")
            return {}
    
    def calculate_formation_features(self, home_formation: Dict[str, Any], away_formation: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula features basadas en el matchup de formaciones.
        
        Args:
            home_formation: Datos de formación del equipo local
            away_formation: Datos de formación del equipo visitante
            
        Returns:
            Dict con features calculadas
        """
        if not home_formation or not away_formation:
            return {}
            
        try:
            home_indices = home_formation["tactical_indices"]
            away_indices = away_formation["tactical_indices"]
            
            # Calcular ventajas tácticas
            wing_play_advantage = home_indices["wing_play"] - away_indices["wing_play"]
            possession_advantage = home_indices["possession"] - away_indices["possession"]
            pressing_advantage = home_indices["pressing"] - away_indices["pressing"]
            
            # Calcular probabilidad de dominancia posicional
            positional_dominance = (
                (home_indices["possession"] * 0.4) +
                (home_indices["pressing"] * 0.3) +
                (home_indices["wing_play"] * 0.3)
            ) / (
                (away_indices["possession"] * 0.4) +
                (away_indices["pressing"] * 0.3) +
                (away_indices["wing_play"] * 0.3)
            )
            
            return {
                "wing_play_advantage": wing_play_advantage,
                "possession_advantage": possession_advantage,
                "pressing_advantage": pressing_advantage,
                "positional_dominance": positional_dominance
            }
            
        except Exception as e:
            logging.error(f"Error al calcular features de formación: {str(e)}")
            return {}

    def process_match(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single match's data to extract corner and formation information.
        
        Args:
            match_data: Raw match data from the API
            
        Returns:
            Dictionary with processed match data including corners and formations
        """
        try:
            fixture_id = match_data.get('fixture', {}).get('id')
            if not fixture_id:
                logging.warning("Match data missing fixture ID")
                return {}
            
            # Get formation data
            home_team = match_data.get('teams', {}).get('home', {})
            away_team = match_data.get('teams', {}).get('away', {})
            
            formations = {
                'home_formation': match_data.get('teams', {}).get('home', {}).get('formation'),
                'away_formation': match_data.get('teams', {}).get('away', {}).get('formation')
            }
            
            # Use FormationAnalyzer to get tactical features
            if formations['home_formation'] and formations['away_formation']:
                tactical_features = self.formation_analyzer.predict_corner_impact(
                    formations['home_formation'], 
                    formations['away_formation']
                )
                formations.update(tactical_features)
                
                # Get detailed formation analysis
                home_analysis = self.formation_analyzer.analyze_formation(formations['home_formation'])
                away_analysis = self.formation_analyzer.analyze_formation(formations['away_formation'])
                
                formations.update({
                    'home_wing_play': home_analysis['wing_play'],
                    'home_possession': home_analysis['possession'],
                    'home_pressing': home_analysis['pressing'],
                    'away_wing_play': away_analysis['wing_play'],
                    'away_possession': away_analysis['possession'],
                    'away_pressing': away_analysis['pressing']
                })
            
            # Get corner statistics
            stats = self.get_fixture_statistics(fixture_id)
            
            # Combine all data
            return {
                'match_id': fixture_id,
                'date': match_data.get('fixture', {}).get('date'),
                'league_id': match_data.get('league', {}).get('id'),
                'home_team_id': home_team.get('id'),
                'away_team_id': away_team.get('id'),
                'home_team_name': home_team.get('name'),
                'away_team_name': away_team.get('name'),
                'home_corners': stats['home_corners'],
                'away_corners': stats['away_corners'],
                'total_corners': stats['total_corners'],
                **formations
            }
            
        except Exception as e:
            logging.error(f"Error processing match {match_data.get('fixture', {}).get('id')}: {str(e)}")
            return {}

    def get_fixture_statistics(self, fixture_id: int) -> Dict[str, Any]:
        """
        Get statistics for a specific fixture, focusing on corners.
        
        Args:
            fixture_id: The ID of the fixture
            
        Returns:
            Dictionary with corner statistics
        """
        try:
            # Try to get from cache first
            endpoint = 'fixtures/statistics'
            params = {'fixture': fixture_id}
            cached_stats = self.cache.get(endpoint, params)
            if cached_stats:
                return self._extract_corners_from_stats(cached_stats)
            
            # Make API request if not in cache
            response = self._make_api_request(endpoint, params)
            
            if not response or 'response' not in response:
                return {'home_corners': 0, 'away_corners': 0, 'total_corners': 0}
            
            stats = response['response']
            stats_data = self._extract_corners_from_stats(stats)
            
            return stats_data
            
        except Exception as e:
            logging.error(f"Error getting fixture statistics for {fixture_id}: {str(e)}")
            return {'home_corners': 0, 'away_corners': 0, 'total_corners': 0}
    
    def _extract_corners_from_stats(self, stats: Dict[str, Any] | List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract corner statistics from raw stats data.
        
        Args:
            stats: Statistics data either as a dictionary or list of dictionaries
            
        Returns:
            Dictionary with corner statistics
        """
        home_corners = 0
        away_corners = 0
        
        # Handle both direct API response and cached data formats
        stats_list = stats if isinstance(stats, list) else stats.get('response', [])
        
        for team_stats in stats_list:
            team_type = 'home' if team_stats.get('team', {}).get('home', False) else 'away'
            statistics = team_stats.get('statistics', [])
            
            for stat in statistics:
                if isinstance(stat, dict) and stat.get('type') == 'Corner Kicks':
                    value = stat.get('value', '0')
                    # Handle cases where value might be 'None' or other non-numeric values
                    try:
                        corner_count = int(str(value).replace('None', '0'))
                        if team_type == 'home':
                            home_corners = corner_count
                        else:
                            away_corners = corner_count
                    except (ValueError, TypeError):
                        logging.warning(f"Invalid corner value for {team_type} team: {value}")
        
        return {
            'home_corners': home_corners,
            'away_corners': away_corners,
            'total_corners': home_corners + away_corners
        }

class FormationDataCollector:
    """Recolector de datos de formación para predicción de corners"""
    
    def __init__(self):
        self.formation_weights = {
            '3-5-2': {'wing_attack': 0.8, 'high_press': 0.6, 'possession': 0.5},
            '4-3-3': {'wing_attack': 0.7, 'high_press': 0.7, 'possession': 0.6},
            '4-4-2': {'wing_attack': 0.5, 'high_press': 0.5, 'possession': 0.5},
            '4-2-3-1': {'wing_attack': 0.6, 'high_press': 0.6, 'possession': 0.7},
            '5-3-2': {'wing_attack': 0.4, 'high_press': 0.3, 'possession': 0.4},
            '5-4-1': {'wing_attack': 0.3, 'high_press': 0.2, 'possession': 0.3}
        }
    
    def format_formation(self, formation_string: Optional[str]) -> str:
        """Normaliza el formato de la formación"""
        if not formation_string or not isinstance(formation_string, str):
            return '4-4-2'  # formación por defecto
        
        # Limpiar y normalizar formato
        formation = formation_string.replace(' ', '')
        
        # Si es un formato numérico sin guiones (ej: 442)
        if formation.isdigit() and len(formation) == 3:
            return f"{formation[0]}-{formation[1]}-{formation[2]}"
        
        parts = formation.split('-')
        if len(parts) != 3:
            return '4-4-2'
            
        # Validar que cada parte sea un número
        if not all(part.isdigit() for part in parts):
            return '4-4-2'
            
        return '-'.join(parts)
    
    def get_formation_features(self, formation: str) -> Dict[str, float]:
        """Extrae características tácticas de una formación"""
        formation = self.format_formation(formation)
        if formation not in self.formation_weights:
            return self.formation_weights['4-4-2']
        return self.formation_weights[formation]
    
    def calculate_matchup_advantage(self, home_formation: str, away_formation: str) -> float:
        """Calcula la ventaja táctica entre dos formaciones"""
        home_features = self.get_formation_features(home_formation)
        away_features = self.get_formation_features(away_formation)
        
        # Calcular ventaja basada en diferencias tácticas
        wing_advantage = home_features['wing_attack'] - away_features['wing_attack']
        press_advantage = home_features['high_press'] - away_features['high_press']
        possession_advantage = home_features['possession'] - away_features['possession']
        
        # Ponderación de factores
        total_advantage = (wing_advantage * 0.5 + 
                         press_advantage * 0.3 + 
                         possession_advantage * 0.2)
        
        return round(total_advantage, 3)
    
    def enrich_match_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enriquece los datos del partido con análisis de formación"""
        # Crear copia para no modificar original
        enriched = match_data.copy()
        
        # Obtener y validar formaciones
        home_formation = self.format_formation(match_data.get('home_formation'))
        away_formation = self.format_formation(match_data.get('away_formation'))
        
        # Obtener características tácticas
        home_features = self.get_formation_features(home_formation)
        away_features = self.get_formation_features(away_formation)
        
        # Convertir formaciones a IDs numéricos
        try:
            home_formation_id = int(home_formation.replace('-', ''))
            away_formation_id = int(away_formation.replace('-', ''))
        except (ValueError, AttributeError):
            home_formation_id = 442
            away_formation_id = 442
        
        # Añadir features de formación
        enriched.update({
            'home_formation_id': home_formation_id,
            'away_formation_id': away_formation_id,
            'home_wing_attack': home_features['wing_attack'],
            'home_high_press': home_features['high_press'],
            'home_possession': home_features['possession'],
            'away_wing_attack': away_features['wing_attack'],
            'away_high_press': away_features['high_press'],
            'away_possession': away_features['possession'],
            'formation_advantage': self.calculate_matchup_advantage(home_formation, away_formation)
        })
        
        return enriched

if __name__ == '__main__':
    from config import API_FOOTBALL_KEY as API_KEY, API_BASE_URL, API_HOST
    import logging
    import json
    import sys
    from datetime import datetime
    
    # Configurar logging
    logging.basicConfig(
        filename=f'corner_data_collector.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Verificar que tenemos la clave de API
    if not API_KEY:
        print("Error: No se encontró la clave de API. Por favor configura la variable de entorno 'API_FOOTBALL_KEY'")
        sys.exit(1)
    
    try:
        # Inicializar el recolector
        collector = FootballDataCollector(
            api_key=API_KEY,
            api_base_url=API_BASE_URL
        )
        
        # Liga Premier League y temporada actual
        league_id = 39  # Premier League
        season = "2023"
        
        logging.info(f"Iniciando recolección de datos para liga {league_id}, temporada {season}")
        print(f"Iniciando recolección de datos...")
        
        # Recolectar datos
        corner_data = collector.collect_corner_data(
            league_id=league_id,
            season=season
        )
        
        # Guardar resultados
        import json
        output_file = f"corner_data_{league_id}_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(corner_data, f, indent=2)
        
        logging.info(f"Datos guardados en {output_file}")
        print(f"\nRecolección completada. Se encontraron {len(corner_data)} eventos de córner.")
        print(f"Datos guardados en: {output_file}")
        
    except Exception as e:
        logging.error(f"Error durante la recolección: {str(e)}", exc_info=True)
        print(f"\nError durante la recolección: {str(e)}")
        raise
