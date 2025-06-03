# team_form.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from data import FootballAPI
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

# League season structures
CALENDAR_YEAR_LEAGUES = [39, 103, 128, 207]  # Leagues that run in calendar year (Jan-Dec)
SPLIT_YEAR_LEAGUES = [40, 39, 135, 61, 78]  # European leagues that split across years (Aug-May)

def get_active_season(league_id: int) -> int:
    """
    Determine the active season for a specific league based on its structure and current date
    
    Args:
        league_id: ID of the league
        
    Returns:
        Active season year as integer
    """
    try:
        # Inicializar la API
        api = FootballAPI()
        
        # Try to get current season from API first
        params = {"id": league_id, "current": "true"}
        league_data = api._make_request('leagues', params)
        
        if league_data and "response" in league_data and league_data["response"]:
            for league in league_data["response"]:
                if "seasons" in league and league["seasons"]:
                    for season in league["seasons"]:
                        if season.get("current", False):
                            return season.get("year", datetime.now().year)
        
        # Fallback to date-based logic if API doesn't return current season
        current_date = datetime.now()
        current_month = current_date.month
        current_year = current_date.year
        
        # Calendar year leagues (Jan-Dec: MLS, Nordic leagues, etc.)
        if league_id in CALENDAR_YEAR_LEAGUES:
            return current_year
        
        # Split year leagues (Aug-May: European leagues)
        elif league_id in SPLIT_YEAR_LEAGUES:
            # If we're in second half of year (Jul-Dec), we're in season that starts this year
            # If we're in first half (Jan-Jun), we're in season that started previous year
            if current_month >= 7:  # July or later
                return current_year
            else:  # January through June
                return current_year - 1
                
        # Default - use current year
        else:
            return current_year
            
    except Exception as e:
        logger.error(f"Error determining active season: {e}")
        return datetime.now().year

def validate_season(league_id: int, requested_season: int) -> int:
    """
    Validate requested season against active season for the league
    
    Args:
        league_id: ID of the league
        requested_season: Season year requested
        
    Returns:
        Valid season year to use
    """
    try:
        active_season = get_active_season(league_id)
        
        # If requested season is in the future, use active season
        if requested_season > active_season:
            logger.warning(f"Season {requested_season} is in the future for league {league_id}, using {active_season} instead")
            return active_season
            
        # Don't go too far in the past (max 5 years back)
        if requested_season < active_season - 5:
            logger.warning(f"Season {requested_season} is too old for league {league_id}, using {active_season-5} instead")
            return active_season - 5
            
        # Season is valid
        return requested_season
        
    except Exception as e:
        logger.error(f"Error validating season: {e}")
        return datetime.now().year

class FormAnalyzer:
    """
    Analiza la forma reciente de los equipos basándose en sus resultados anteriores.
    """
    
    def __init__(self):
        self.api = FootballAPI()
    
    def get_team_form(self, team_id: int, last_matches: int = 5) -> Dict[str, float]:
        """
        Obtiene métricas de forma basadas en los últimos partidos jugados por el equipo.
        
        Args:
            team_id: ID del equipo
            last_matches: Número de partidos recientes a considerar
            
        Returns:
            Diccionario con métricas de forma
        """
        try:
            # Obtener los últimos partidos del equipo
            params = {'team': team_id, 'last': str(last_matches)}
            fixtures_data = self.api._make_request('fixtures', params)
            fixtures = fixtures_data.get('response', [])
            
            if not fixtures:
                logger.warning(f"No se encontraron partidos recientes para el equipo {team_id}")
                return {'form_score': 0.5}
            
            # Analizar resultados recientes
            points = 0
            max_points = last_matches * 3  # Máximo de puntos posibles (3 por victoria)
            
            for fixture in fixtures:
                match_result = self._get_match_result(fixture, team_id)
                
                if match_result == 'W':
                    points += 3
                elif match_result == 'D':
                    points += 1
            
            # Calcular puntuación de forma (normalizada entre 0 y 1)
            form_score = points / max_points if max_points > 0 else 0.5
            
            return {'form_score': form_score}
            
        except Exception as e:
            logger.error(f"Error obteniendo forma del equipo {team_id}: {e}")
            return {'form_score': 0.5}
    
    def get_team_form_metrics(self, team_id: int, last_matches: int = 5) -> Dict[str, float]:
        """
        Obtiene métricas avanzadas de forma basadas en los últimos partidos del equipo.
        
        Args:
            team_id: ID del equipo
            last_matches: Número de partidos recientes a considerar
            
        Returns:
            Diccionario con métricas avanzadas de forma
        """
        try:
            # Obtener los últimos partidos del equipo
            params = {'team': team_id, 'last': str(last_matches)}
            fixtures_data = self.api._make_request('fixtures', params)
            fixtures = fixtures_data.get('response', [])
            
            if not fixtures:
                logger.warning(f"No se encontraron partidos recientes para el equipo {team_id}")
                return self._get_default_form_metrics()
            
            # Inicializar contadores
            total_matches = len(fixtures)
            wins = 0
            draws = 0
            losses = 0
            goals_scored = 0
            goals_conceded = 0
            clean_sheets = 0
            failed_to_score = 0
            
            # Analizar cada partido
            for fixture in fixtures:
                match_result = self._get_match_result(fixture, team_id)
                goals = self._get_goals(fixture, team_id)
                
                # Actualizar contadores
                if match_result == 'W':
                    wins += 1
                elif match_result == 'D':
                    draws += 1
                else:  # 'L'
                    losses += 1
                
                goals_scored += goals['scored']
                goals_conceded += goals['conceded']
                
                if goals['conceded'] == 0:
                    clean_sheets += 1
                if goals['scored'] == 0:
                    failed_to_score += 1
            
            # Calcular métricas
            metrics = {}
            
            # Porcentajes de resultados
            metrics['recent_win_rate'] = wins / total_matches if total_matches > 0 else 0.33
            metrics['recent_draw_rate'] = draws / total_matches if total_matches > 0 else 0.33
            metrics['recent_loss_rate'] = losses / total_matches if total_matches > 0 else 0.34
            
            # Promedios de goles
            metrics['recent_avg_goals_scored'] = goals_scored / total_matches if total_matches > 0 else 1.0
            metrics['recent_avg_goals_conceded'] = goals_conceded / total_matches if total_matches > 0 else 1.0
            
            # Diferencia de goles promedio
            metrics['recent_avg_goal_diff'] = (goals_scored - goals_conceded) / total_matches if total_matches > 0 else 0.0
            
            # Partidos sin recibir goles y sin marcar
            metrics['recent_clean_sheet_rate'] = clean_sheets / total_matches if total_matches > 0 else 0.2
            metrics['recent_failed_to_score_rate'] = failed_to_score / total_matches if total_matches > 0 else 0.2
            
            # Puntuación de forma (3 puntos por victoria, 1 por empate)
            points = (wins * 3) + draws
            max_points = total_matches * 3
            metrics['form_score'] = points / max_points if max_points > 0 else 0.5
            
            # Normalización de métricas para consistencia (excepto las de goles promedios)
            for key, value in metrics.items():
                # Las métricas de promedio de goles y diferencia no se normalizan
                if key not in ['recent_avg_goals_scored', 'recent_avg_goals_conceded', 'recent_avg_goal_diff']:
                    # Asegurar que todas las demás métricas estén entre 0 y 1
                    metrics[key] = max(0.0, min(1.0, value))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas de forma del equipo {team_id}: {e}")
            return self._get_default_form_metrics()
    
    def get_consistency_score(self, team_id: int, last_matches: int = 10) -> float:
        """
        Calcula una puntuación de consistencia basada en la variabilidad 
        de los resultados recientes de un equipo.
        
        Args:
            team_id: ID del equipo
            last_matches: Número de partidos recientes a considerar
            
        Returns:
            Puntuación de consistencia entre 0 y 1 (0 = muy inconsistente, 1 = muy consistente)
        """
        try:
            # Obtener resultados recientes
            params = {'team': str(team_id), 'last': str(last_matches)}
            fixtures_data = self.api._make_request('fixtures', params)
            fixtures = fixtures_data.get('response', [])
            
            if not fixtures:
                logger.warning(f"No se encontraron partidos recientes para el equipo {team_id}")
                return 0.5  # Valor neutro por defecto
            
            # Extraer resultados (1 = victoria, 0 = empate, -1 = derrota)
            results = []
            for fixture in fixtures:
                # Solo partidos completados
                if fixture.get('fixture', {}).get('status', {}).get('short') != 'FT':
                    continue
                    
                teams = fixture.get('teams', {})
                is_home = teams.get('home', {}).get('id') == team_id
                home_win = teams.get('home', {}).get('winner') == True
                away_win = teams.get('away', {}).get('winner') == True
                
                if (is_home and home_win) or (not is_home and away_win):
                    results.append(1)  # Victoria
                elif (is_home and away_win) or (not is_home and home_win):
                    results.append(-1)  # Derrota
                else:
                    results.append(0)  # Empate
            
            if not results:
                return 0.5
                
            # Calcular consistencia basada en la variabilidad de resultados
            # Más variaciones en resultados = menos consistencia
            changes = 0
            for i in range(1, len(results)):
                if results[i] != results[i-1]:
                    changes += 1
            
            # Normalizar (0 = muy inconsistente, 1 = muy consistente)
            max_possible_changes = len(results) - 1
            consistency = 1.0 - (changes / max_possible_changes if max_possible_changes > 0 else 0)
            
            return consistency
            
        except Exception as e:
            logger.error(f"Error calculando puntuación de consistencia: {e}")
            return 0.5
    
    def _get_match_result(self, fixture: Dict, team_id: int) -> str:
        """
        Determina el resultado del partido para el equipo especificado.
        
        Args:
            fixture: Datos del partido
            team_id: ID del equipo
            
        Returns:
            'W' para victoria, 'D' para empate, 'L' para derrota
        """
        try:
            teams = fixture.get('teams', {})
            home_team = teams.get('home', {})
            away_team = teams.get('away', {})
            
            home_id = home_team.get('id', 0)
            away_id = away_team.get('id', 0)
            
            goals = fixture.get('goals', {})
            home_goals = goals.get('home', 0)
            away_goals = goals.get('away', 0)
            
            # Determinar si el equipo es local o visitante
            is_home = home_id == team_id
            
            # Determinar resultado
            if home_goals > away_goals:
                return 'W' if is_home else 'L'
            elif home_goals < away_goals:
                return 'L' if is_home else 'W'
            else:
                return 'D'
        
        except Exception as e:
            logger.error(f"Error determinando resultado: {e}")
            return 'D'  # Valor por defecto
    
    def _get_goals(self, fixture: Dict, team_id: int) -> Dict[str, int]:
        """
        Obtiene los goles marcados y recibidos por el equipo en el partido.
        
        Args:
            fixture: Datos del partido
            team_id: ID del equipo
            
        Returns:
            Diccionario con goles marcados y recibidos
        """
        try:
            teams = fixture.get('teams', {})
            home_team = teams.get('home', {})
            away_team = teams.get('away', {})
            
            home_id = home_team.get('id', 0)
            away_id = away_team.get('id', 0)
            
            goals = fixture.get('goals', {})
            home_goals = goals.get('home', 0) or 0  # Manejar None
            away_goals = goals.get('away', 0) or 0  # Manejar None
            
            # Determinar si el equipo es local o visitante
            is_home = home_id == team_id
            
            if is_home:
                return {'scored': home_goals, 'conceded': away_goals}
            else:
                return {'scored': away_goals, 'conceded': home_goals}
        
        except Exception as e:
            logger.error(f"Error obteniendo goles: {e}")
            return {'scored': 0, 'conceded': 0}  # Valores por defecto
    
    def _get_default_form_metrics(self) -> Dict[str, float]:
        """
        Devuelve valores predeterminados para métricas de forma.
        
        Returns:
            Diccionario con valores predeterminados
        """
        return {
            'recent_win_rate': 0.33,
            'recent_draw_rate': 0.33,
            'recent_loss_rate': 0.34,
            'recent_avg_goals_scored': 1.0,
            'recent_avg_goals_conceded': 1.0,
            'recent_avg_goal_diff': 0.0,
            'recent_clean_sheet_rate': 0.2,
            'recent_failed_to_score_rate': 0.2,
            'form_score': 0.5
        }

def get_team_form(team_id: int, league_id: int, season: Optional[int] = None, last_matches: int = 5) -> Dict[str, Any]:
    """
    Get recent form data for a team using real data
    
    Args:
        team_id: Team ID 
        league_id: League ID
        season: Optional season year, defaults to active season for the league if not provided
        last_matches: Number of matches to analyze
        
    Returns:
        Dictionary with team form information
    """
    try:
        # If season not provided, use the active season for this league
        if season is None:
            season = get_active_season(league_id)
        else:
            # Validate requested season
            season = validate_season(league_id, season)
            
        logger.info(f"Getting form for team {team_id} in league {league_id}, season {season}")
        
        # Initialize API
        api = FootballAPI()
        
        # Get recent matches for the team in this league and season
        params = {
            'team': team_id, 
            'league': league_id,
            'season': season,
            'last': last_matches,
            'status': 'FT'  # Only finished matches
        }
        
        # Hacer petición a la API
        fixtures_data = api._make_request('fixtures', params)
        fixtures = fixtures_data.get('response', [])
        
        if not fixtures:
            # If no fixtures found for current season, try previous season
            if season == get_active_season(league_id):
                logger.warning(f"No fixtures found for team {team_id} in current season {season}, trying previous season")
                previous_season = season - 1
                return get_team_form(team_id, league_id, previous_season, last_matches)
            else:
                logger.warning(f"No fixtures found for team {team_id} in season {season}")
                return {}
            
        # Inicializar contadores
        stats = {
            'matches_played': len(fixtures),
            'wins': 0,
            'draws': 0,
            'losses': 0,
            'goals_scored': 0,
            'goals_conceded': 0,
            'clean_sheets': 0,
            'failed_to_score': 0,
            'corners_for': 0,
            'corners_against': 0,
            'cards_yellow': 0,
            'cards_red': 0,
            'fouls_committed': 0,
            'fouls_drawn': 0
        }
        
        # Analizar cada partido
        for fixture in fixtures:
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})
            score = fixture.get('score', {})
            stats_data = fixture.get('statistics', [])
            
            # Identificar si el equipo jugó como local o visitante
            is_home = teams.get('home', {}).get('id') == team_id
            team_goals = goals.get('home' if is_home else 'away', 0)
            opponent_goals = goals.get('away' if is_home else 'home', 0)
            
            # Actualizar resultados
            if team_goals > opponent_goals:
                stats['wins'] += 1
            elif team_goals < opponent_goals:
                stats['losses'] += 1
            else:
                stats['draws'] += 1
                
            # Actualizar goles
            stats['goals_scored'] += team_goals
            stats['goals_conceded'] += opponent_goals
            
            if opponent_goals == 0:
                stats['clean_sheets'] += 1
            if team_goals == 0:
                stats['failed_to_score'] += 1
                
            # Procesar estadísticas detalladas
            for stat in stats_data:
                if stat.get('team', {}).get('id') == team_id:
                    type_name = stat.get('type', '').lower()
                    value = stat.get('value', 0)
                    
                    if 'corner' in type_name:
                        stats['corners_for'] += value
                    elif 'fouls' in type_name:
                        stats['fouls_committed'] += value
                    elif 'yellow' in type_name:
                        stats['cards_yellow'] += value
                    elif 'red' in type_name:
                        stats['cards_red'] += value
        
        # Calcular promedios
        matches = max(1, stats['matches_played'])
        form_data = {
            'matches_played': matches,
            'win_percentage': stats['wins'] / matches,
            'draw_percentage': stats['draws'] / matches,
            'loss_percentage': stats['losses'] / matches,
            'avg_goals_scored': stats['goals_scored'] / matches,
            'avg_goals_conceded': stats['goals_conceded'] / matches,
            'clean_sheet_percentage': stats['clean_sheets'] / matches,
            'fail_to_score_percentage': stats['failed_to_score'] / matches,
            'avg_corners_for': stats['corners_for'] / matches,
            'avg_corners_against': stats['corners_against'] / matches,
            'avg_cards': (stats['cards_yellow'] + stats['cards_red']) / matches,
            'avg_fouls': stats['fouls_committed'] / matches,
            'form_score': (3 * stats['wins'] + stats['draws']) / (3 * matches),
            'season_used': season  # Add the season that was actually used
        }
        
        return form_data
        
    except Exception as e:
        logger.error(f"Error getting team form: {e}")
        return {}

def get_head_to_head_analysis(team1_id: int, team2_id: int, league_id: Optional[int] = None, season: Optional[int] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Get head-to-head analysis between two teams using real data
    
    Args:
        team1_id: First team ID
        team2_id: Second team ID
        league_id: Optional league ID for contextual season validation
        season: Optional season to filter H2H matches
        limit: Number of matches to analyze (default: 10)
        
    Returns:
        Dictionary with head-to-head analysis
    """
    try:
        logger.info(f"Getting head-to-head analysis for teams {team1_id} and {team2_id}")
        
        # Initialize the API
        api = FootballAPI()
        
        # Prepare parameters for h2h request
        params = {
            'h2h': f"{team1_id}-{team2_id}",
            'last': limit
        }
        
        # Add season filter if provided and valid
        if season is not None and league_id is not None:
            validated_season = validate_season(league_id, season)
            params['season'] = validated_season
        
        h2h_data = api._make_request('fixtures/headtohead', params)
        fixtures = h2h_data.get('response', [])
        
        if not fixtures:
            logger.warning(f"No se encontraron encuentros directos entre equipos {team1_id} y {team2_id}")
            # If no matches found with current season, try without season filter
            if 'season' in params:
                logger.info(f"Trying to get H2H data without season filter")
                del params['season']
                h2h_data = api._make_request('fixtures/headtohead', params)
                fixtures = h2h_data.get('response', [])
                
                if not fixtures:
                    return {
                        "team1_id": team1_id,
                        "team2_id": team2_id,
                        "total_matches": 0,
                        "team1_wins": 0,
                        "team2_wins": 0,
                        "draws": 0,
                        "team1_goals": 0,
                        "team2_goals": 0,
                        "average_goals_per_match": 0
                    }
        
        # Analyze results
        total_matches = len(fixtures)
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = 0
        team2_goals = 0
        total_corners = 0
        total_cards = 0
        total_fouls = 0
        last_matches = []
        
        for fixture in fixtures:
            teams = fixture.get('teams', {})
            goals = fixture.get('goals', {})
            
            # Determine result
            if teams.get('home', {}).get('id') == team1_id:
                team1_is_home = True
                team1_match_goals = goals.get('home', 0)
                team2_match_goals = goals.get('away', 0)
            else:
                team1_is_home = False
                team1_match_goals = goals.get('away', 0)
                team2_match_goals = goals.get('home', 0)
            
            # Count result
            if team1_match_goals > team2_match_goals:
                team1_wins += 1
                result = "Team1 Win"
            elif team1_match_goals < team2_match_goals:
                team2_wins += 1
                result = "Team2 Win"
            else:
                draws += 1
                result = "Draw"
            
            # Count goals
            team1_goals += team1_match_goals
            team2_goals += team2_match_goals
            
            # Create match object
            fixture_date = fixture.get('fixture', {}).get('date', '')
            match_data = {
                "date": fixture_date,
                "result": result,
                "score": f"{team1_match_goals}-{team2_match_goals}",
                "team1_is_home": team1_is_home
            }
            
            # Try to get additional match statistics
            try:
                fixture_id = fixture.get('fixture', {}).get('id')
                stats_params = {'fixture': fixture_id}
                stats_data = api._make_request('fixtures/statistics', stats_params)
                
                if stats_data and 'response' in stats_data:
                    match_corners = 0
                    match_cards = 0
                    match_fouls = 0
                    
                    for team_stats in stats_data['response']:
                        if 'statistics' in team_stats:
                            for stat in team_stats['statistics']:
                                if stat.get('type') == 'Corner Kicks':
                                    try:
                                        val = stat.get('value', '0')
                                        # Handle cases where val is None or empty
                                        if val is None or val == '':
                                            val = 0
                                        # Convert to integer
                                        if isinstance(val, int):
                                            match_corners += val
                                        else:
                                            # Safer conversion handling empty strings and non-numeric characters
                                            val_str = str(val).strip()
                                            if val_str.isdigit():
                                                match_corners += int(val_str)
                                            elif val_str and val_str[0].isdigit():
                                                # Try to extract just the number part if format is like "10 (2)"
                                                digits = ''.join(c for c in val_str if c.isdigit())
                                                match_corners += int(digits) if digits else 0
                                            else:
                                                match_corners += 0
                                    except Exception as e:
                                        logger.warning(f"Error processing corners: {e}")
                                elif stat.get('type') in ['Yellow Cards', 'Red Cards']:
                                    try:
                                        val = stat.get('value', '0')
                                        # Handle cases where val is None or empty
                                        if val is None or val == '':
                                            val = 0
                                        # Convert to integer
                                        if isinstance(val, int):
                                            match_cards += val
                                        else:
                                            # Safer conversion handling empty strings and non-numeric characters
                                            val_str = str(val).strip()
                                            if val_str.isdigit():
                                                match_cards += int(val_str)
                                            elif val_str and val_str[0].isdigit():
                                                # Try to extract just the number part
                                                digits = ''.join(c for c in val_str if c.isdigit())
                                                match_cards += int(digits) if digits else 0
                                            else:
                                                match_cards += 0
                                    except Exception as e:
                                        logger.warning(f"Error processing cards: {e}")
                                elif stat.get('type') == 'Fouls':
                                    try:
                                        val = stat.get('value', '0')
                                        # Handle cases where val is None or empty
                                        if val is None or val == '':
                                            val = 0
                                        # Convert to integer
                                        if isinstance(val, int):
                                            match_fouls += val
                                        else:
                                            # Safer conversion handling empty strings and non-numeric characters
                                            val_str = str(val).strip()
                                            if val_str.isdigit():
                                                match_fouls += int(val_str)
                                            elif val_str and val_str[0].isdigit():
                                                # Try to extract just the number part 
                                                digits = ''.join(c for c in val_str if c.isdigit())
                                                match_fouls += int(digits) if digits else 0
                                            else:
                                                match_fouls += 0
                                    except Exception as e:
                                        logger.warning(f"Error processing fouls: {e}")
                    
                    total_corners += match_corners
                    total_cards += match_cards
                    total_fouls += match_fouls
                    
                    match_data["corners"] = match_corners
                    match_data["cards"] = match_cards
                    match_data["fouls"] = match_fouls
            except Exception as stats_error:
                logger.warning(f"Error getting statistics: {stats_error}")
            
            last_matches.append(match_data)
        
        # Calculate averages
        total_goals = team1_goals + team2_goals
        average_goals = total_goals / total_matches if total_matches > 0 else 0
        average_corners = total_corners / total_matches if total_matches > 0 else 0
        average_cards = total_cards / total_matches if total_matches > 0 else 0
        average_fouls = total_fouls / total_matches if total_matches > 0 else 0
        
        # Create response object
        h2h_analysis = {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "total_matches": total_matches,
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "draws": draws,
            "team1_goals": team1_goals,
            "team2_goals": team2_goals,
            "average_goals_per_match": average_goals,
            "average_corners_per_match": average_corners,
            "average_cards_per_match": average_cards,
            "average_fouls_per_match": average_fouls,
            "last_matches": last_matches
        }
        
        return h2h_analysis
        
    except Exception as e:
        logger.error(f"Error getting head-to-head analysis: {e}")
        return {
            "team1_id": team1_id,
            "team2_id": team2_id,
            "total_matches": 0,
            "team1_wins": 0,
            "team2_wins": 0,
            "draws": 0,
            "team1_goals": 0,
            "team2_goals": 0,
            "average_goals_per_match": 0,
            "error": str(e)
        }
