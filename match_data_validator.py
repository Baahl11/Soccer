"""
Match Data Validator

Este módulo implementa validadores para detectar inconsistencias en los datos de partidos
de fútbol, aplicable a cualquier liga y conjunto de datos.

Autor: Equipo de Desarrollo
Fecha: Mayo 25, 2025
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuración de logging
logger = logging.getLogger(__name__)

class MatchDataValidator:
    """
    Validador de datos de partidos que detecta inconsistencias y anomalías en los datos
    provenientes de diferentes ligas y fuentes.
    """
    
    def __init__(self):
        """Inicializa el validador de datos de partidos."""
        self.known_issues = set()  # Almacena hashes de problemas ya identificados
        self.fix_suggestions = {}  # Almacena sugerencias de correcciones
    
    def validate_match_data(self, match_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida un conjunto de datos de partido y devuelve un informe con inconsistencias.
        
        Args:
            match_data: Datos del partido a validar
            
        Returns:
            Informe de validación con inconsistencias detectadas
        """
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'fixes_applied': [],
            'match_id': match_data.get('fixture_id') or match_data.get('id')
        }
        
        # Validar datos básicos
        self._validate_basic_info(match_data, validation_results)
        
        # Validar estadísticas
        if 'statistics' in match_data:
            self._validate_statistics(match_data['statistics'], validation_results)
            
        # Validar resultados y goles
        self._validate_score_data(match_data, validation_results)
        
        # Validar datos de equipos e historial
        self._validate_team_data(match_data, validation_results)
        
        # Validar datos tácticos
        self._validate_tactical_data(match_data, validation_results)
        
        # Validar probabilidades de apuestas
        self._validate_odds_data(match_data, validation_results)
        
        # Verificar coherencia interna
        self._check_internal_coherence(match_data, validation_results)
        
        # Marcar el resultado como no válido si hay errores
        if validation_results['errors']:
            validation_results['valid'] = False
            
        return validation_results
    
    def validate_multiple_matches(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Valida múltiples partidos y detecta anomalías en el conjunto.
        
        Args:
            matches: Lista de datos de partidos
            
        Returns:
            Informe de validación global y estadísticas de anomalías
        """
        individual_results = []
        global_issues = {
            'duplicate_matches': [],
            'anomalies': [],
            'weather_repetition': [],
            'corner_anomalies': [],
            'card_anomalies': []
        }
        
        # Validar cada partido individualmente
        for match in matches:
            result = self.validate_match_data(match)
            individual_results.append(result)
        
        # Buscar problemas a nivel de conjunto
        self._detect_duplicate_matches(matches, global_issues)
        self._detect_statistical_anomalies(matches, global_issues)
        self._detect_weather_repetition(matches, global_issues)
        self._detect_corner_card_anomalies(matches, global_issues)
        
        # Preparar resultados
        summary = {
            'total_matches': len(matches),
            'valid_matches': sum(1 for r in individual_results if r['valid']),
            'invalid_matches': sum(1 for r in individual_results if not r['valid']),
            'total_errors': sum(len(r['errors']) for r in individual_results),
            'total_warnings': sum(len(r['warnings']) for r in individual_results),
            'fixes_applied': sum(len(r['fixes_applied']) for r in individual_results),
            'individual_results': individual_results,
            'global_issues': global_issues
        }
        
        return summary
    
    def _validate_basic_info(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida la información básica del partido."""
        # Verificar que existen los IDs necesarios
        required_ids = ['fixture_id', 'home_team_id', 'away_team_id', 'league_id']
        alternative_ids = {'fixture_id': ['id', 'match_id'], 
                          'home_team_id': ['teams.home.id'], 
                          'away_team_id': ['teams.away.id'],
                          'league_id': ['league.id']}
        
        for id_field in required_ids:
            if not self._get_nested_value(match_data, id_field, alternative_ids.get(id_field, [])):
                results['errors'].append(f"Missing required ID: {id_field}")
            
        # Verificar fecha válida
        match_date = self._get_nested_value(
            match_data, 
            'match_date', 
            ['date', 'fixture.date', 'datetime']
        )
        
        if not match_date:
            results['errors'].append("Missing match date")
        elif isinstance(match_date, str):
            try:
                datetime.fromisoformat(match_date.replace('Z', '+00:00'))
            except ValueError:
                results['errors'].append(f"Invalid date format: {match_date}")
        elif isinstance(match_date, datetime):
            # Accept datetime object as valid
            pass
        else:
            results['errors'].append(f"Invalid type for match date: {type(match_date)}")
    
    def _validate_statistics(self, statistics: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida las estadísticas del partido."""
        # Verificar formato de estadísticas
        if not isinstance(statistics, dict) and not isinstance(statistics, list):
            results['errors'].append(f"Invalid statistics format: {type(statistics)}")
            return
            
        # Convertir a formato estándar si es una lista
        stats_dict = {}
        if isinstance(statistics, list):
            for team_stats in statistics:
                if isinstance(team_stats, dict):
                    team = team_stats.get('team')
                    if isinstance(team, dict):
                        team_id = team.get('id')
                    else:
                        team_id = None
                else:
                    team_id = None
                if team_id and isinstance(team_stats, dict):
                    stats_dict[str(team_id)] = team_stats.get('statistics', {})
            statistics = stats_dict
            
        # Validar valores numéricos en estadísticas
        for team_id, team_stats in statistics.items():
            if isinstance(team_stats, dict):
                for stat_name, stat_value in team_stats.items():
                    if self._should_be_numeric(stat_name) and not self._is_valid_numeric(stat_value):
                        results['warnings'].append(
                            f"Invalid numeric value for {stat_name} (team {team_id}): {stat_value}"
                        )
    
    def _validate_score_data(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida los datos de resultados y goles."""
        # Verificar coherencia entre goles y resultado
        home_goals = self._get_nested_value(
            match_data, 
            'home_goals', 
            ['goals.home', 'score.fulltime.home']
        )
        
        away_goals = self._get_nested_value(
            match_data, 
            'away_goals', 
            ['goals.away', 'score.fulltime.away']
        )
        
        winner = self._get_nested_value(
            match_data, 
            'winner', 
            ['teams.winner', 'match_winner']
        )
        
        # Validar resultado
        if home_goals is not None and away_goals is not None and winner is not None:
            expected_winner = 'home' if home_goals > away_goals else ('away' if away_goals > home_goals else 'draw')
            if str(winner).lower() != expected_winner.lower() and winner != expected_winner:
                results['errors'].append(
                    f"Inconsistent result: score {home_goals}-{away_goals} but winner is {winner}"
                )
                
        # Validar goles de jugadores vs. goles totales
        if 'events' in match_data and isinstance(match_data['events'], list):
            goals_from_events = {'home': 0, 'away': 0}
            
            for event in match_data['events']:
                if event.get('type') == 'Goal' or event.get('detail') == 'Goal':
                    team_side = self._determine_team_side(event.get('team_id'), match_data)
                    if team_side:
                        goals_from_events[team_side] += 1
                        
            # Comparar con goles totales (permitiendo autogoles)
            if (home_goals is not None and abs(goals_from_events['home'] - home_goals) > 1) or \
               (away_goals is not None and abs(goals_from_events['away'] - away_goals) > 1):
                results['warnings'].append(
                    f"Inconsistent goal count: reported {home_goals}-{away_goals}, but events show {goals_from_events['home']}-{goals_from_events['away']}"
                )
    
    def _validate_team_data(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida datos de equipos e historial."""
        # Verificar que los datos de equipo son consistentes
        home_team_id = self._get_nested_value(
            match_data, 
            'home_team_id', 
            ['teams.home.id']
        )
        
        away_team_id = self._get_nested_value(
            match_data, 
            'away_team_id', 
            ['teams.away.id']
        )
        
        # Validar head-to-head
        h2h = self._get_nested_value(match_data, 'head_to_head', ['h2h'])
        
        if h2h and isinstance(h2h, dict) and 'matches' in h2h and isinstance(h2h['matches'], list):
            # Verificar equipos en partidos h2h
            for h2h_match in h2h['matches']:
                h2h_home = self._get_nested_value(h2h_match, 'home_team_id', ['teams.home.id'])
                h2h_away = self._get_nested_value(h2h_match, 'away_team_id', ['teams.away.id'])
                
                if h2h_home and h2h_away:
                    if h2h_home not in [home_team_id, away_team_id] or h2h_away not in [home_team_id, away_team_id]:
                        results['errors'].append(
                            f"H2H match contains different teams: {h2h_home} vs {h2h_away}"
                        )
    
    def _validate_tactical_data(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida datos tácticos del partido."""
        # Verificar datos de alineación
        for side in ['home', 'away']:
            formation = self._get_nested_value(
                match_data, 
                f'{side}_formation', 
                [f'lineups.{side}.formation', f'formations.{side}']
            )
            
            if formation and not self._is_valid_formation(formation):
                results['warnings'].append(
                    f"Invalid {side} formation format: {formation}"
                )
    
    def _validate_odds_data(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Valida datos de probabilidades y apuestas."""
        # Verificar datos de odds si existen
        odds = self._get_nested_value(match_data, 'odds', ['bookmakers'])
        
        if odds and isinstance(odds, (list, dict)):
            # Convertir a lista estándar
            odds_list = odds if isinstance(odds, list) else [odds]
            
            for bookie in odds_list:
                # Validar probabilidades sumando a ~1.0
                if isinstance(bookie, dict) and 'bets' in bookie:
                    for bet in bookie['bets']:
                        if bet.get('name') == '1x2' or bet.get('name') == 'Match Winner':
                            values = []
                            for odd in bet.get('values', []):
                                try:
                                    if 'odd' in odd:
                                        # Convertir cuota a probabilidad implícita
                                        odd_val = float(odd['odd'])
                                        if odd_val > 1:  # Odds europeas
                                            values.append(1 / odd_val)
                                except (ValueError, TypeError):
                                    results['warnings'].append(
                                        f"Invalid odd format: {odd.get('odd')}"
                                    )
                                    
                            # Verificar suma de probabilidades (con margen para overround)
                            if values and len(values) >= 3:
                                prob_sum = sum(values)
                                if prob_sum < 0.9 or prob_sum > 1.2:
                                    results['warnings'].append(
                                        f"Suspicious odds distribution, implied probabilities sum to {prob_sum:.2f}"
                                    )
    
    def _check_internal_coherence(self, match_data: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Verifica coherencia interna entre diferentes aspectos de los datos."""
        # Verificar Elo ratings vs. probabilidades
        elo_home = self._get_nested_value(
            match_data, 
            'elo_ratings.home', 
            ['home_elo', 'ratings.home.elo']
        )
        
        elo_away = self._get_nested_value(
            match_data, 
            'elo_ratings.away', 
            ['away_elo', 'ratings.away.elo']
        )
        
        home_win_prob = self._get_nested_value(
            match_data, 
            'probabilities.home_win', 
            ['home_win_probability', 'predictions.home']
        )
        
        away_win_prob = self._get_nested_value(
            match_data, 
            'probabilities.away_win', 
            ['away_win_probability', 'predictions.away']
        )
        
        # Si tenemos tanto ratings Elo como probabilidades, verificar coherencia
        if elo_home and elo_away and home_win_prob and away_win_prob:
            elo_diff = elo_home - elo_away
            prob_diff = home_win_prob - away_win_prob
            
            # En general, una ventaja de 100 puntos Elo debería corresponder
            # aproximadamente a 60-65% de probabilidad de victoria
            expected_prob_diff = self._elo_diff_to_win_prob_diff(elo_diff)
            actual_prob_diff = prob_diff
            
            # Si la diferencia es muy grande, marcar como incoherente
            if abs(expected_prob_diff - actual_prob_diff) > 0.25:  # 25% de umbral
                results['errors'].append(
                    f"Inconsistency between Elo ratings ({elo_diff:+.0f} difference) " +
                    f"and win probabilities ({actual_prob_diff*100:+.0f}% difference)"
                )
    
    def _detect_duplicate_matches(self, matches: List[Dict[str, Any]], issues: Dict[str, Any]) -> None:
        """Detecta partidos duplicados en la lista."""
        match_signatures = {}
        
        for match in matches:
            # Crear firma del partido basada en equipos y fecha
            home_id = self._get_nested_value(match, 'home_team_id', ['teams.home.id'])
            away_id = self._get_nested_value(match, 'away_team_id', ['teams.away.id'])
            date = self._get_nested_value(match, 'match_date', ['date', 'fixture.date'])
            
            if home_id and away_id and date:
                # Normalizar fecha a solo día
                if isinstance(date, str):
                    try:
                        date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                        date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
                
                signature = f"{home_id}-{away_id}-{date}"
                match_id = match.get('fixture_id') or match.get('id')
                
                if signature in match_signatures:
                    issues['duplicate_matches'].append({
                        'signature': signature,
                        'matches': [match_signatures[signature], match_id]
                    })
                else:
                    match_signatures[signature] = match_id
    
    def _detect_statistical_anomalies(self, matches: List[Dict[str, Any]], issues: Dict[str, Any]) -> None:
        """Detecta anomalías estadísticas en el conjunto de partidos."""
        # Recolectar estadísticas para análisis
        stats_by_league = {}
        
        for match in matches:
            league_id = self._get_nested_value(match, 'league_id', ['league.id'])
            
            if not league_id:
                continue
                
            if league_id not in stats_by_league:
                stats_by_league[league_id] = {
                    'corners': [],
                    'cards': [],
                    'goals': [],
                    'xg': []
                }
                
            # Recolectar estadísticas
            corners = self._extract_total_corners(match)
            cards = self._extract_total_cards(match)
            goals = self._extract_total_goals(match)
            xg = self._extract_total_xg(match)
            
            if corners is not None:
                stats_by_league[league_id]['corners'].append(corners)
            if cards is not None:
                stats_by_league[league_id]['cards'].append(cards)
            if goals is not None:
                stats_by_league[league_id]['goals'].append(goals)
            if xg is not None:
                stats_by_league[league_id]['xg'].append(xg)
        
        # Analizar anomalías por liga
        for league_id, league_stats in stats_by_league.items():
            for stat_type, values in league_stats.items():
                if len(values) >= 10:  # Necesitamos suficientes datos para análisis
                    anomalies = self._detect_outliers(values)
                    
                    if anomalies:
                        issues['anomalies'].append({
                            'league_id': league_id,
                            'stat_type': stat_type,
                            'anomalies': anomalies,
                            'mean': np.mean(values),
                            'std': np.std(values)
                        })
    
    def _detect_weather_repetition(self, matches: List[Dict[str, Any]], issues: Dict[str, Any]) -> None:
        """Detecta repetición de datos de clima entre diferentes ciudades."""
        weather_by_location = {}
        
        for match in matches:
            # Extraer datos de clima y ubicación
            weather = self._get_nested_value(match, 'weather', ['weather_data'])
            city = self._get_nested_value(
                match,
                'venue.city',
                ['city', 'venue_city', 'weather.city']
            )
            
            if weather and city:
                # Crear firma del clima
                w_condition = str(self._get_nested_value(weather, 'condition', ['main']))
                w_temp = str(self._get_nested_value(weather, 'temperature', ['temp']))
                w_wind = str(self._get_nested_value(weather, 'wind_speed', ['wind', 'wind.speed']))
                
                weather_sig = f"{w_condition}-{w_temp}-{w_wind}"
                match_id = match.get('fixture_id') or match.get('id')
                
                if city not in weather_by_location:
                    weather_by_location[city] = []
                    
                weather_by_location[city].append({
                    'signature': weather_sig,
                    'match_id': match_id,
                    'date': self._get_nested_value(match, 'match_date', ['date', 'fixture.date'])
                })
        
        # Verificar repeticiones entre ciudades
        checked = set()
        
        for city1, weather1 in weather_by_location.items():
            for city2, weather2 in weather_by_location.items():
                if city1 == city2 or f"{city1}-{city2}" in checked or f"{city2}-{city1}" in checked:
                    continue
                    
                checked.add(f"{city1}-{city2}")
                
                # Buscar firmas idénticas entre ciudades diferentes
                for w1 in weather1:
                    for w2 in weather2:
                        if w1['signature'] == w2['signature'] and w1['date'] == w2['date']:
                            issues['weather_repetition'].append({
                                'city1': city1,
                                'city2': city2,
                                'match1': w1['match_id'],
                                'match2': w2['match_id'],
                                'date': w1['date'],
                                'weather_signature': w1['signature']
                            })
                            break
    
    def _detect_corner_card_anomalies(self, matches: List[Dict[str, Any]], issues: Dict[str, Any]) -> None:
        """Detecta anomalías específicas en datos de corners y tarjetas."""
        for match in matches:
            match_id = match.get('fixture_id') or match.get('id')
            corner_issues = self._check_corner_stats(match)
            card_issues = self._check_card_stats(match)
            
            if corner_issues:
                issues['corner_anomalies'].append({
                    'match_id': match_id,
                    'issues': corner_issues
                })
                
            if card_issues:
                issues['card_anomalies'].append({
                    'match_id': match_id,
                    'issues': card_issues
                })
    
    # Métodos auxiliares
    def _get_nested_value(self, data: Dict[str, Any], path: str, alternatives: Optional[List[str]] = None) -> Any:
        """Obtiene un valor anidado de un diccionario usando rutas con puntos."""
        if alternatives is None:
            alternatives = []
            
        # Primero intenta la ruta principal
        paths_to_try = [path] + alternatives
        
        for path in paths_to_try:
            current = data
            found = True
            
            for key in path.split('.'):
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    found = False
                    break
                    
            if found:
                return current
                
        return None
    
    def _should_be_numeric(self, stat_name: str) -> bool:
        """Determina si una estadística debería ser numérica."""
        numeric_stats = [
            'corners', 'shots', 'passes', 'goal', 'fouls', 'possession',
            'saves', 'offsides', 'yellow', 'red', 'penalties', 'substitutions'
        ]
        
        return any(term in stat_name.lower() for term in numeric_stats)
    
    def _is_valid_numeric(self, value: Any) -> bool:
        """Verifica si un valor puede ser tratado como numérico."""
        if value is None:
            return False
            
        if isinstance(value, (int, float)):
            return True
            
        if isinstance(value, str):
            # Manejar formatos como "10 (2)" o "10%"
            value = value.split('(')[0].strip()
            value = value.rstrip('%').strip()
            
            try:
                float(value)
                return True
            except ValueError:
                return False
                
        return False
    
    def _determine_team_side(self, team_id: Any, match_data: Dict[str, Any]) -> Optional[str]:
        """Determina si un equipo es local o visitante."""
        home_id = self._get_nested_value(match_data, 'home_team_id', ['teams.home.id'])
        away_id = self._get_nested_value(match_data, 'away_team_id', ['teams.away.id'])
        
        if str(team_id) == str(home_id):
            return 'home'
        elif str(team_id) == str(away_id):
            return 'away'
        else:
            return None
    
    def _is_valid_formation(self, formation: str) -> bool:
        """Verifica si una formación tiene formato válido."""
        if not formation:
            return False
            
        # Verificar formato con guiones (ej: 4-4-2)
        if '-' in formation:
            parts = formation.split('-')
            
            # Validar que todas las partes son numéricas y suman 10 (+ portero = 11)
            try:
                numbers = [int(part) for part in parts]
                return sum(numbers) == 10
            except ValueError:
                return False
        
        return False
        
    def _elo_diff_to_win_prob_diff(self, elo_diff: float) -> float:
        """Convierte diferencia de Elo a diferencia de probabilidades de victoria."""
        # Basado en la fórmula Elo estándar
        p_home = 1 / (1 + 10 ** (-elo_diff / 400))
        p_away = 1 - p_home
        return p_home - p_away
    
    def _extract_total_corners(self, match: Dict[str, Any]) -> Optional[int]:
        """Extrae el número total de saques de esquina del partido."""
        # Buscar en varias ubicaciones posibles
        corners_home = self._get_nested_value(
            match, 
            'statistics.corners.home', 
            [
                'stats.corners.home', 
                'statistics.home.corner_kicks', 
                'corners_for.home'
            ]
        )
        
        corners_away = self._get_nested_value(
            match, 
            'statistics.corners.away',
            [
                'stats.corners.away',
                'statistics.away.corner_kicks',
                'corners_for.away'
            ]
        )
        
        # Convertir a enteros si son cadenas
        if isinstance(corners_home, str):
            try:
                corners_home = int(corners_home.split('(')[0].strip())
            except (ValueError, AttributeError):
                corners_home = None
                
        if isinstance(corners_away, str):
            try:
                corners_away = int(corners_away.split('(')[0].strip())
            except (ValueError, AttributeError):
                corners_away = None
        
        if corners_home is not None and corners_away is not None:
            return corners_home + corners_away
        else:
            return None
    
    def _extract_total_cards(self, match: Dict[str, Any]) -> Optional[int]:
        """Extrae el número total de tarjetas del partido."""
        # Buscar tarjetas amarillas
        yellows_home = self._get_nested_value(
            match, 
            'statistics.yellow_cards.home', 
            [
                'stats.yellow.home',
                'statistics.home.yellow_cards'
            ]
        )
        
        yellows_away = self._get_nested_value(
            match, 
            'statistics.yellow_cards.away', 
            [
                'stats.yellow.away',
                'statistics.away.yellow_cards'
            ]
        )
        
        # Buscar tarjetas rojas
        reds_home = self._get_nested_value(
            match, 
            'statistics.red_cards.home', 
            [
                'stats.red.home',
                'statistics.home.red_cards'
            ]
        )
        
        reds_away = self._get_nested_value(
            match, 
            'statistics.red_cards.away', 
            [
                'stats.red.away',
                'statistics.away.red_cards'
            ]
        )
        
        # Convertir valores a enteros
        try:
            yellows_home = int(yellows_home) if yellows_home is not None else 0
            yellows_away = int(yellows_away) if yellows_away is not None else 0
            reds_home = int(reds_home) if reds_home is not None else 0
            reds_away = int(reds_away) if reds_away is not None else 0
            
            return yellows_home + yellows_away + reds_home + reds_away
        except (ValueError, TypeError):
            return None
    
    def _extract_total_goals(self, match: Dict[str, Any]) -> Optional[int]:
        """Extrae el número total de goles del partido."""
        home_goals = self._get_nested_value(
            match, 
            'home_goals', 
            ['goals.home', 'score.fulltime.home']
        )
        
        away_goals = self._get_nested_value(
            match, 
            'away_goals', 
            ['goals.away', 'score.fulltime.away']
        )
        
        if home_goals is not None and away_goals is not None:
            try:
                return int(home_goals) + int(away_goals)
            except (ValueError, TypeError):
                return None
        else:
            return None
    
    def _extract_total_xg(self, match: Dict[str, Any]) -> Optional[float]:
        """Extrae el total de expected goals del partido."""
        home_xg = self._get_nested_value(
            match, 
            'home_xg', 
            ['xG.home', 'statistics.home.expected_goals']
        )
        
        away_xg = self._get_nested_value(
            match, 
            'away_xg', 
            ['xG.away', 'statistics.away.expected_goals']
        )
        
        if home_xg is not None and away_xg is not None:
            try:
                return float(home_xg) + float(away_xg)
            except (ValueError, TypeError):
                return None
        else:
            return None
    
    def _detect_outliers(self, values: List[float], z_threshold: float = 2.5) -> List[Dict[str, Any]]:
        """Detecta valores atípicos usando puntuaciones z."""
        if not values:
            return []
            
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:  # Evitar división por cero
            return []
            
        z_scores = np.abs((values_array - mean) / std)
        outliers = []
        
        for i, z in enumerate(z_scores):
            if z > z_threshold:
                outliers.append({
                    'value': values[i],
                    'z_score': float(z),
                    'index': i
                })
                
        return outliers
    
    def _check_corner_stats(self, match: Dict[str, Any]) -> List[str]:
        """Verifica anomalías específicas en estadísticas de córners."""
        issues = []
        
        corners_home = self._get_nested_value(
            match, 
            'statistics.corners.home', 
            [
                'stats.corners.home', 
                'statistics.home.corner_kicks'
            ]
        )
        
        corners_away = self._get_nested_value(
            match, 
            'statistics.corners.away',
            [
                'stats.corners.away',
                'statistics.away.corner_kicks'
            ]
        )
        
        # Verificar valores extremos
        if corners_home is not None:
            try:
                corners_home_val = int(str(corners_home).split('(')[0].strip())
                if corners_home_val > 20:
                    issues.append(f"Unusually high corner count for home team: {corners_home_val}")
                if corners_home_val == 0:
                    issues.append("Zero corners for home team - potential data issue")
            except (ValueError, AttributeError):
                issues.append(f"Invalid corner format for home team: {corners_home}")
                
        if corners_away is not None:
            try:
                corners_away_val = int(str(corners_away).split('(')[0].strip())
                if corners_away_val > 20:
                    issues.append(f"Unusually high corner count for away team: {corners_away_val}")
                if corners_away_val == 0:
                    issues.append("Zero corners for away team - potential data issue")
            except (ValueError, AttributeError):
                issues.append(f"Invalid corner format for away team: {corners_away}")
                
        return issues
    
    def _check_card_stats(self, match: Dict[str, Any]) -> List[str]:
        """Verifica anomalías específicas en estadísticas de tarjetas."""
        issues = []
        
        yellows_home = self._get_nested_value(
            match, 
            'statistics.yellow_cards.home', 
            [
                'stats.yellow.home',
                'statistics.home.yellow_cards'
            ]
        )
        
        yellows_away = self._get_nested_value(
            match, 
            'statistics.yellow_cards.away', 
            [
                'stats.yellow.away',
                'statistics.away.yellow_cards'
            ]
        )
        
        reds_home = self._get_nested_value(
            match, 
            'statistics.red_cards.home', 
            [
                'stats.red.home',
                'statistics.home.red_cards'
            ]
        )
        
        reds_away = self._get_nested_value(
            match, 
            'statistics.red_cards.away', 
            [
                'stats.red.away',
                'statistics.away.red_cards'
            ]
        )
        
        # Verificar valores
        try:
            if yellows_home is not None:
                yellows_home = int(yellows_home)
                if yellows_home > 8:
                    issues.append(f"Unusually high yellow cards for home team: {yellows_home}")
                    
            if yellows_away is not None:
                yellows_away = int(yellows_away)
                if yellows_away > 8:
                    issues.append(f"Unusually high yellow cards for away team: {yellows_away}")
                    
            if reds_home is not None:
                reds_home = int(reds_home)
                if reds_home > 3:
                    issues.append(f"Unusually high red cards for home team: {reds_home}")
                    
            if reds_away is not None:
                reds_away = int(reds_away)
                if reds_away > 3:
                    issues.append(f"Unusually high red cards for away team: {reds_away}")
        except (ValueError, TypeError):
            issues.append("Invalid card data format")
            
        return issues


# Función de conveniencia para usar el validador
def validate_matches(matches: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Valida una lista de partidos y devuelve informe completo de anomalías.
    
    Args:
        matches: Lista de datos de partidos
        
    Returns:
        Informe de validación
    """
    validator = MatchDataValidator()
    return validator.validate_multiple_matches(matches)


def validate_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """
    Valida un partido individual.
    
    Args:
        match: Datos del partido
        
    Returns:
        Informe de validación
    """
    validator = MatchDataValidator()
    return validator.validate_match_data(match)
