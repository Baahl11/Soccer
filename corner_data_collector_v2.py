"""
Módulo para recolectar datos de córners y formaciones de partidos de fútbol.
"""
import logging
import time
import json
import sys
from typing import Dict, Any, List, Optional, Tuple
import requests
from datetime import datetime
from formation_analyzer import FormationAnalyzer
from data_cache import DataCache
from batch_processor import BatchProcessor

# League IDs constants
PREMIER_LEAGUE_ID = 39
LA_LIGA_ID = 140
SERIE_A_ID = 135
BUNDESLIGA_ID = 78
LIGUE_1_ID = 61

SUPPORTED_LEAGUES = [
    (PREMIER_LEAGUE_ID, "Premier League"),
    (LA_LIGA_ID, "La Liga"),
    (SERIE_A_ID, "Serie A"),
    (BUNDESLIGA_ID, "Bundesliga"),
    (LIGUE_1_ID, "Ligue 1")
]

class FootballDataCollector:
    """Clase para recolectar datos de córners y formaciones de la API de fútbol."""
    
    def __init__(self, api_key: str, api_base_url: str, api_host: str):
        """Inicializa el collector con credenciales de API."""
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.api_host = api_host
        self.headers = {
            'x-apisports-key': self.api_key
        }
        
        # Inicializar componentes
        self.formation_analyzer = FormationAnalyzer()
        self.cache = DataCache()
        self.batch_processor = BatchProcessor()
        
        # Configurar logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler('corner_collection.log')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
    def collect_corner_data(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """
        Recolectar datos de córners y formaciones.
        
        Args:
            league_id: ID de la liga
            season: Temporada (ej. "2023")
            
        Returns:
            Lista de datos de córners con información táctica
        """
        # Obtener lista de partidos
        fixtures = self.get_fixtures(league_id, season)
        if not fixtures:
            self.logger.warning(f"No se encontraron partidos para liga {league_id}")
            return []
            
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
        
    def _process_single_fixture(self, fixture: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Procesa un partido individual para extraer datos de córners y formaciones."""
        if not isinstance(fixture, dict):
            return None
            
        fixture_id = fixture.get('fixture', {}).get('id')
        if not fixture_id:
            return None
            
        try:
            # Obtener estadísticas y formaciones
            stats = self.get_fixture_stats(fixture_id)
            lineups = self.get_fixture_lineups(fixture_id)
            
            # Analizar formaciones
            if not lineups or len(lineups) < 2:
                self.logger.warning(f"Datos de alineación inválidos para partido {fixture_id}")
                formation_analysis = {
                    'home_formation': 'Unknown',
                    'away_formation': 'Unknown',
                    'tactical_indices': {}
                }
            else:
                formation_analysis = self.analyze_formations(lineups)
                formation_analysis['tactical_indices'] = self.calculate_formation_features(formation_analysis)
                
            # Procesar estadísticas de córners
            corners_data = self._process_corner_statistics(fixture, stats, formation_analysis)
            
            return corners_data
            
        except Exception as e:
            self.logger.error(f"Error procesando partido {fixture_id}: {str(e)}")
            return None
            
    def get_fixtures(self, league_id: int, season: str) -> List[Dict[str, Any]]:
        """Obtiene la lista de partidos de una liga y temporada."""
        response = self._make_api_request(
            'fixtures',
            {'league': league_id, 'season': season}
        )
        return response.get('response', [])
        
    def get_fixture_stats(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtiene estadísticas de un partido."""
        response = self._make_api_request(
            'fixtures/statistics',
            {'fixture': fixture_id}
        )
        return response.get('response', [])
        
    def get_fixture_lineups(self, fixture_id: int) -> List[Dict[str, Any]]:
        """Obtiene alineaciones y formaciones de un partido."""
        response = self._make_api_request(
            'fixtures/lineups',
            {'fixture': fixture_id}
        )
        
        lineups = response.get('response', [])
        if not lineups:
            self.logger.debug(f"No se encontraron alineaciones para el partido {fixture_id}")
            return []
            
        if len(lineups) < 2:
            self.logger.debug(f"Alineaciones incompletas para el partido {fixture_id}: {len(lineups)} equipos")
            return []
            
        # Validar formaciones
        for lineup in lineups:
            formation = lineup.get('formation')
            team_name = lineup.get('team', {}).get('name', 'Unknown')
            if not formation:
                self.logger.debug(f"Formación no disponible para {team_name} en partido {fixture_id}")
            elif not self._is_valid_formation(formation):
                self.logger.debug(f"Formación inválida {formation} para {team_name} en partido {fixture_id}")
                lineup['formation'] = None
                
        return lineups
        
    def _make_api_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Realiza una petición a la API con manejo de errores y rate limiting."""
        try:
            url = f"{self.api_base_url}/{endpoint}"
              # Intentar obtener de caché
            cache_key = f"{endpoint}_{json.dumps(params, sort_keys=True)}"
            cached = self.cache.get(cache_key, params)
            if cached:
                return cached
                  # Realizar request
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                self.cache.set(cache_key, data=data, params=params)
                return data
            else:
                self.logger.error(f"Error API {response.status_code}: {response.text}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error en request {endpoint}: {str(e)}")
            return {}
            
    def analyze_formations(self, lineups: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza las formaciones de ambos equipos.
        
        Args:
            lineups: Lista de alineaciones de ambos equipos
            
        Returns:
            Dict con formaciones y análisis táctico
        """
        analysis = {
            'home_formation': 'Unknown',
            'away_formation': 'Unknown',
            'home_team_id': None,
            'away_team_id': None
        }
        
        if not lineups or len(lineups) < 2:
            return analysis
            
        # Identificar equipo local y visitante
        home_team_id = lineups[0].get('team', {}).get('id')
        away_team_id = lineups[1].get('team', {}).get('id')
        
        if not home_team_id or not away_team_id:
            return analysis
            
        analysis['home_team_id'] = home_team_id
        analysis['away_team_id'] = away_team_id
        
        # Procesar formaciones
        for lineup in lineups:
            formation = lineup.get('formation')
            team_id = lineup.get('team', {}).get('id')
            
            if not formation or not self._is_valid_formation(formation):
                continue
                
            if team_id == home_team_id:
                analysis['home_formation'] = formation
            elif team_id == away_team_id:
                analysis['away_formation'] = formation
                
        return analysis
        
    def calculate_formation_features(self, formation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calcula features tácticas basadas en las formaciones."""
        features = {}
        
        for side in ['home', 'away']:
            formation = formation_analysis.get(f'{side}_formation')
            if formation and formation != 'Unknown':
                try:
                    numbers = formation.split('-')
                    def_num, mid_num, fwd_num = map(int, numbers)
                    
                    features[f'{side}_wing_attack'] = self._calculate_wing_attack_index(def_num, mid_num, fwd_num)
                    features[f'{side}_high_press'] = self._calculate_high_press_index(def_num, mid_num, fwd_num)
                    features[f'{side}_possession'] = self._calculate_possession_index(def_num, mid_num, fwd_num)
                except ValueError:
                    features.update({
                        f'{side}_wing_attack': 0.5,
                        f'{side}_high_press': 0.5,
                        f'{side}_possession': 0.5
                    })
            else:
                features.update({
                    f'{side}_wing_attack': 0.5,
                    f'{side}_high_press': 0.5,
                    f'{side}_possession': 0.5
                })
                
        # Calcular ventaja táctica
        features['formation_advantage'] = self._calculate_formation_advantage(
            formation_analysis['home_formation'],
            formation_analysis['away_formation']
        )
        
        return features
        
    def _calculate_wing_attack_index(self, defenders: int, midfielders: int, forwards: int) -> float:
        """Calcula índice de ataque por bandas."""
        base = 0.5
        
        if defenders == 3:
            base += 0.2  # Formaciones con 3 centrales suelen usar carrileros
        elif defenders == 5:
            base += 0.15  # Formaciones con 5 defensas suelen usar laterales ofensivos
            
        if midfielders >= 4:
            base -= 0.1  # Muchos mediocampistas indican juego más central
            
        if forwards == 3:
            base += 0.1  # Tres delanteros suelen incluir extremos
            
        return min(max(base, 0), 1)
        
    def _calculate_high_press_index(self, defenders: int, midfielders: int, forwards: int) -> float:
        """Calcula índice de presión alta."""
        base = forwards * 0.2  # Más delanteros = más presión
        
        if defenders <= 3:
            base += 0.2  # Menos defensores indica presión alta
            
        base += min(midfielders * 0.1, 0.3)  # Mediocampistas influyen en la presión
        
        return min(max(base, 0), 1)
        
    def _calculate_possession_index(self, defenders: int, midfielders: int, forwards: int) -> float:
        """Calcula índice de posesión."""
        base = midfielders * 0.15  # Más mediocampistas = más posesión
        
        if forwards <= 2:
            base += 0.2  # Menos delanteros suele indicar enfoque en posesión
            
        if defenders >= 4:
            base += 0.1  # Más defensores puede indicar build-up desde atrás
            
        return min(max(base, 0), 1)
        
    def _calculate_formation_advantage(self, home_formation: str, away_formation: str) -> float:
        """Calcula ventaja táctica entre formaciones."""
        if home_formation == 'Unknown' or away_formation == 'Unknown':
            return 0
            
        try:
            home_def, home_mid, home_fwd = map(int, home_formation.split('-'))
            away_def, away_mid, away_fwd = map(int, away_formation.split('-'))
            
            midfield_control = (home_mid - away_mid) * 0.4
            attacking_presence = (home_fwd - away_def) * 0.3
            defensive_stability = (home_def - away_fwd) * 0.3
            
            advantage = midfield_control + attacking_presence + defensive_stability
            return max(min(advantage, 1), -1)
            
        except ValueError:
            return 0
            
    def _process_corner_statistics(self, fixture: Dict[str, Any], 
                                 stats: List[Dict[str, Any]],
                                 formation_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa estadísticas de córners de un partido."""
        home_corners = away_corners = 0
        
        # Extraer corners de estadísticas
        for team_stats in stats:
            for stat in team_stats.get('statistics', []):
                if stat.get('type') == 'Corner Kicks':
                    value = stat.get('value', 0)
                    if isinstance(value, str):
                        try:
                            value = int(value)
                        except ValueError:
                            value = 0
                            
                    if team_stats.get('team', {}).get('id') == fixture['teams']['home']['id']:
                        home_corners = value
                    else:
                        away_corners = value
                        
        return {
            'fixture_id': fixture['fixture']['id'],
            'home_team_id': fixture['teams']['home']['id'],
            'away_team_id': fixture['teams']['away']['id'],
            'home_corners': home_corners,
            'away_corners': away_corners,
            'total_corners': home_corners + away_corners,
            'home_formation': formation_analysis['home_formation'],
            'away_formation': formation_analysis['away_formation'],
            'tactical_indices': formation_analysis['tactical_indices']
        }
    
    def _is_valid_formation(self, formation: str) -> bool:
        """
        Valida si una formación es válida.
        
        Args:
            formation: String con la formación (ej: "4-3-3")
            
        Returns:
            bool: True si la formación es válida
        """
        try:
            # Verificar formato básico
            if not formation or not isinstance(formation, str):
                return False
                
            # Dividir en números
            numbers = formation.split('-')
            if len(numbers) != 3:  # Debe tener 3 líneas (def-mid-fwd)
                return False
                
            # Convertir a enteros y validar
            def_num, mid_num, fwd_num = map(int, numbers)
            
            # Validaciones de lógica de fútbol
            if def_num + mid_num + fwd_num != 10:  # Total debe ser 10 (+ portero = 11)
                return False
                
            if not (2 <= def_num <= 5):  # Entre 2 y 5 defensas
                return False
                
            if not (2 <= mid_num <= 5):  # Entre 2 y 5 mediocampistas
                return False
                
            if not (1 <= fwd_num <= 3):  # Entre 1 y 3 delanteros
                return False
                
            return True
            
        except (ValueError, AttributeError):
            return False

    def collect_multiple_leagues(self, season: str, leagues: Optional[List[Tuple[int, str]]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recolecta datos de córners y formaciones para múltiples ligas.
        
        Args:
            season: Temporada (ej. "2023")
            leagues: Lista opcional de tuplas (league_id, league_name). Si no se proporciona,
                    se usan todas las ligas soportadas.
            
        Returns:
            Dict con datos de córners por liga
        """
        if leagues is None:
            leagues = SUPPORTED_LEAGUES
            
        all_data = {}
        for league_id, league_name in leagues:
            self.logger.info(f"Iniciando recolección para {league_name} (ID: {league_id})")
            try:
                league_data = self.collect_corner_data(league_id, season)
                
                # Guardar datos en archivo JSON
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"corner_data_{league_id}_{season}_{timestamp}.json"
                
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(league_data, f, indent=2)
                    self.logger.info(f"Datos guardados en {filename}")
                except Exception as e:
                    self.logger.error(f"Error guardando datos de {league_name}: {str(e)}")
                
                all_data[league_name] = league_data
                
            except Exception as e:
                self.logger.error(f"Error recolectando datos de {league_name}: {str(e)}")
            
            # Pausa entre ligas para evitar rate limiting
            time.sleep(2)
            
        return all_data

if __name__ == '__main__':
    from config import API_FOOTBALL_KEY, API_BASE_URL, API_HOST
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('corner_data_collector.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        if not API_FOOTBALL_KEY:
            raise ValueError("No se encontró la API key")
            
        collector = FootballDataCollector(API_FOOTBALL_KEY, API_BASE_URL, API_HOST)
        
        # Validar conexión
        if not collector._make_api_request('status', {}):
            raise ConnectionError("No se pudo conectar con la API")
            
        # Recolectar datos de Premier League 2023
        league_id = 39  # Premier League
        season = "2023"
        
        logger.info(f"Iniciando recolección para liga {league_id}, temporada {season}")
        corner_data = collector.collect_corner_data(league_id, season)
        
        if corner_data:
            # Guardar resultados
            filename = f"corner_data_{league_id}_{season}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(corner_data, f, indent=2)
            logger.info(f"Datos guardados en {filename}")
            logger.info(f"Total eventos: {len(corner_data)}")
        else:
            logger.warning("No se encontraron datos")
            
    except Exception as e:
        logger.error(f"Error durante la ejecución: {str(e)}", exc_info=True)
        sys.exit(1)
