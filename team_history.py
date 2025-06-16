# team_history.py
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from data import FootballAPI

logger = logging.getLogger(__name__)

class HistoricalAnalyzer:
    """
    Analiza el historial de enfrentamientos entre equipos y su desempeño histórico.
    """
    
    def __init__(self):
        self.api = FootballAPI()
        self._cache = {}
    
    def get_head_to_head_stats(self, home_team_id: int, away_team_id: int, 
                             last_matches: int = 10) -> Dict[str, Any]:
        """
        Obtiene estadísticas de enfrentamientos directos entre dos equipos.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            last_matches: Número de enfrentamientos a considerar
            
        Returns:
            Diccionario con estadísticas de enfrentamientos directos
        """
        cache_key = f"h2h_{home_team_id}_{away_team_id}_{last_matches}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Obtener enfrentamientos directos
            params = {'h2h': f"{home_team_id}-{away_team_id}", 'last': str(last_matches)}
            h2h_data = self.api._make_request('fixtures/headtohead', params)
            fixtures = h2h_data.get('response', [])
            
            if not fixtures:
                logger.warning(f"No se encontraron enfrentamientos directos entre equipos {home_team_id} y {away_team_id}")
                return self._get_default_h2h_stats()
            
            # Inicializar contadores
            total_matches = len(fixtures)
            home_wins = 0
            away_wins = 0
            draws = 0
            home_goals = 0
            away_goals = 0
            
            # Analizar cada partido
            for fixture in fixtures:
                teams = fixture.get('teams', {})
                goals = fixture.get('goals', {})
                
                home_team = teams.get('home', {})
                away_team = teams.get('away', {})
                
                home_id = home_team.get('id', 0)
                away_id = away_team.get('id', 0)
                
                home_score = goals.get('home', 0) or 0  # Manejar None
                away_score = goals.get('away', 0) or 0  # Manejar None
                
                # Solo contar partidos completados
                if fixture.get('fixture', {}).get('status', {}).get('short') != 'FT':
                    continue
                
                # Determinar resultado
                if home_score > away_score:
                    if home_id == home_team_id:
                        home_wins += 1
                    else:
                        away_wins += 1
                elif home_score < away_score:
                    if home_id == home_team_id:
                        away_wins += 1
                    else:
                        home_wins += 1
                else:
                    draws += 1
                
                # Sumar goles (ajustando para consistencia con los IDs de equipos)
                if home_id == home_team_id and away_id == away_team_id:
                    home_goals += home_score
                    away_goals += away_score
                else:
                    home_goals += away_score
                    away_goals += home_score
            
            # Recalcular total de partidos completados
            total_matches = home_wins + away_wins + draws
            
            # Calcular estadísticas
            stats = {}
            
            # Porcentajes de resultados
            stats['h2h_home_win_rate'] = home_wins / total_matches if total_matches > 0 else 0.4
            stats['h2h_away_win_rate'] = away_wins / total_matches if total_matches > 0 else 0.3
            stats['h2h_draw_rate'] = draws / total_matches if total_matches > 0 else 0.3
            
            # Promedios de goles
            stats['h2h_avg_home_goals'] = home_goals / total_matches if total_matches > 0 else 1.5
            stats['h2h_avg_away_goals'] = away_goals / total_matches if total_matches > 0 else 1.0
            
            # Superioridad relativa (diferencia de victorias normalizada)
            superiority = (home_wins - away_wins) / total_matches if total_matches > 0 else 0.1
            stats['h2h_home_superiority'] = min(max(superiority + 0.5, 0), 1)  # Normalizado entre 0 y 1
            
            # Número de enfrentamientos (para evaluar confianza)
            stats['h2h_match_count'] = total_matches
            
            self._cache[cache_key] = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas H2H: {e}")
            return self._get_default_h2h_stats()
    
    def get_season_stats(self, home_team_id: int, away_team_id: int) -> Dict[str, float]:
        """Get season average statistics for both teams."""
        cache_key = f"season_{home_team_id}_{away_team_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Calculate season averages...
        stats = {
            "home_team_avg_goals": 1.8,  # These would normally be calculated from actual data
            "away_team_avg_goals": 1.4,
            "home_team_avg_conceded": 1.1,
            "away_team_avg_conceded": 1.6
        }
        
        self._cache[cache_key] = stats
        return stats

    def get_league_average_goals(self, is_home: bool = True) -> float:
        """Get league-wide average goals for home/away teams."""
        # These would normally be calculated from actual league data
        return 1.6 if is_home else 1.3

    def get_historic_performance(self, team_id: int, league_id: int, 
                               season: str, last_seasons: int = 3) -> Dict[str, float]:
        """
        Obtiene métricas de rendimiento histórico de un equipo en una liga.
        
        Args:
            team_id: ID del equipo
            league_id: ID de la liga
            season: Temporada actual (formato: YYYY)
            last_seasons: Número de temporadas anteriores a considerar
            
        Returns:
            Diccionario con métricas de rendimiento histórico
        """
        try:
            metrics = {}
            seasons_data = []
            
            # Obtener temporadas anteriores
            current_year = int(season)
            previous_seasons = [str(current_year - i) for i in range(1, last_seasons + 1)]
            
            # Recopilar datos de cada temporada
            for prev_season in previous_seasons:
                # Obtener posición final en la liga
                params = {'league': str(league_id), 'season': prev_season, 'team': str(team_id)}
                standings_data = self.api._make_request('standings', params)
                
                season_data = self._extract_season_performance(standings_data, team_id)
                if season_data:
                    seasons_data.append(season_data)
            
            if not seasons_data:
                logger.warning(f"No se encontraron datos históricos para el equipo {team_id} en la liga {league_id}")
                return self._get_default_historic_metrics()
            
            # Calcular promedios
            avg_data = {}
            for key in seasons_data[0].keys():
                values = [season[key] for season in seasons_data]
                avg_data[key] = sum(values) / len(values)
            
            # Construir métricas finales
            metrics['historic_rank_percentile'] = 1.0 - (avg_data.get('rank_percentile', 0.5))
            metrics['historic_win_rate'] = avg_data.get('win_rate', 0.33)
            metrics['historic_draw_rate'] = avg_data.get('draw_rate', 0.33)
            metrics['historic_loss_rate'] = avg_data.get('loss_rate', 0.34)
            metrics['historic_goals_scored_pg'] = avg_data.get('goals_scored_pg', 1.2)
            metrics['historic_goals_conceded_pg'] = avg_data.get('goals_conceded_pg', 1.2)
            metrics['historic_clean_sheet_rate'] = avg_data.get('clean_sheet_rate', 0.2)
            metrics['historic_home_advantage'] = avg_data.get('home_advantage', 0.2)
            
            # Calcular tendencia (mejora o deterioro)
            if len(seasons_data) > 1:
                # Ordenar por temporada (más reciente primero)
                sorted_data = sorted(seasons_data, key=lambda x: x.get('season'), reverse=True)
                
                # Calcular tendencia en la posición (positiva = mejora)
                recent_rank = sorted_data[0].get('rank_percentile', 0.5)
                older_rank = sorted_data[-1].get('rank_percentile', 0.5)
                metrics['historic_trend'] = older_rank - recent_rank  # Valor positivo = mejora
            else:
                metrics['historic_trend'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo rendimiento histórico: {e}")
            return self._get_default_historic_metrics()
    
    def _extract_season_performance(self, standings_data: Dict, team_id: int) -> Optional[Dict[str, float]]:
        """
        Extrae métricas de rendimiento de una temporada específica.
        
        Args:
            standings_data: Datos de clasificación
            team_id: ID del equipo
            
        Returns:
            Diccionario con métricas de rendimiento o None si no hay datos
        """
        try:
            response = standings_data.get('response', [])
            
            if not response:
                return None
                
            league_data = response[0]
            league_standings = league_data.get('league', {}).get('standings', [])
            
            if not league_standings:
                return None
            
            # Puede haber múltiples grupos, buscar en todos
            team_standing = None
            total_teams = 0
            
            for group in league_standings:
                total_teams = max(total_teams, len(group))
                
                for team_data in group:
                    if team_data.get('team', {}).get('id') == team_id:
                        team_standing = team_data
                        break
                
                if team_standing:
                    break
            
            if not team_standing:
                return None
            
            # Extraer datos relevantes
            season = league_data.get('league', {}).get('season')
            rank = team_standing.get('rank', 0)
            points = team_standing.get('points', 0)
            
            stats = team_standing.get('all', {})
            played = stats.get('played', 0)
            win = stats.get('win', 0)
            draw = stats.get('draw', 0)
            lose = stats.get('lose', 0)
            goals_for = stats.get('goals', {}).get('for', 0)
            goals_against = stats.get('goals', {}).get('against', 0)
            
            home_stats = team_standing.get('home', {})
            away_stats = team_standing.get('away', {})
            
            home_points = home_stats.get('win', 0) * 3 + home_stats.get('draw', 0)
            away_points = away_stats.get('win', 0) * 3 + away_stats.get('draw', 0)
            
            # Calcular métricas
            performance = {}
            performance['season'] = season
            performance['rank'] = rank
            performance['rank_percentile'] = rank / total_teams if total_teams > 0 else 0.5
            
            if played > 0:
                performance['win_rate'] = win / played
                performance['draw_rate'] = draw / played
                performance['loss_rate'] = lose / played
                performance['goals_scored_pg'] = goals_for / played
                performance['goals_conceded_pg'] = goals_against / played
                performance['clean_sheet_rate'] = home_stats.get('clean_sheet', 0) / played
                
                # Ventaja de local (diferencia entre puntos en casa vs fuera)
                total_points = points
                home_advantage = (home_points - away_points) / total_points if total_points > 0 else 0
                performance['home_advantage'] = max(min(home_advantage + 0.5, 1.0), 0.0)  # Normalizar entre 0 y 1
            else:
                # Valores predeterminados
                performance['win_rate'] = 0.33
                performance['draw_rate'] = 0.33
                performance['loss_rate'] = 0.34
                performance['goals_scored_pg'] = 1.2
                performance['goals_conceded_pg'] = 1.2
                performance['clean_sheet_rate'] = 0.2
                performance['home_advantage'] = 0.2
            
            return performance
            
        except Exception as e:
            logger.error(f"Error extrayendo rendimiento de temporada: {e}")
            return None
    
    def _get_default_h2h_stats(self) -> Dict[str, Any]:
        """
        Devuelve valores predeterminados para estadísticas H2H.
        
        Returns:
            Diccionario con valores predeterminados
        """
        return {
            'h2h_home_win_rate': 0.4,
            'h2h_away_win_rate': 0.3,
            'h2h_draw_rate': 0.3,
            'h2h_avg_home_goals': 1.5,
            'h2h_avg_away_goals': 1.0,
            'h2h_home_superiority': 0.5,
            'h2h_match_count': 0
        }
    
    def _get_default_historic_metrics(self) -> Dict[str, float]:
        """
        Devuelve valores predeterminados para métricas históricas.
        
        Returns:
            Diccionario con valores predeterminados
        """
        return {
            'historic_rank_percentile': 0.5,
            'historic_win_rate': 0.33,
            'historic_draw_rate': 0.33,
            'historic_loss_rate': 0.34,
            'historic_goals_scored_pg': 1.2,
            'historic_goals_conceded_pg': 1.2,
            'historic_clean_sheet_rate': 0.2,
            'historic_home_advantage': 0.2,
            'historic_trend': 0.0
        }
    
    def get_data_quality_score(self, home_team_id: int, away_team_id: int) -> float:
        """
        Calcula una puntuación de calidad de datos basada en la cantidad y actualidad
        de los datos históricos disponibles para dos equipos.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Puntuación de calidad de datos entre -0.1 y 0.1
        """
        h2h_stats = self.get_head_to_head_stats(home_team_id, away_team_id)
        total_matches = h2h_stats.get("total_matches", 0)
        
        # Más partidos = mejor calidad
        if total_matches >= 5:
            return 0.1
        elif total_matches >= 3:
            return 0.05
        else:
            return 0.0

    def get_league_stability(self, league_id: int, season: str) -> Dict[str, float]:
        """
        Calcula métricas de estabilidad de una liga basadas en datos históricos.
        
        Args:
            league_id: ID de la liga
            season: Temporada actual
            
        Returns:
            Diccionario con métricas de estabilidad
        """
        try:
            stability_metrics = {
                'position_volatility': 0.3,  # Qué tanto cambian las posiciones entre temporadas
                'scoring_consistency': 0.7,  # Consistencia en goles marcados
                'home_advantage': 0.6,  # Impacto del factor local
                'top_team_dominance': 0.5,  # Dominancia de equipos top
                'upset_frequency': 0.2  # Frecuencia de resultados inesperados
            }
            return stability_metrics
        except Exception as e:
            logger.error(f"Error getting league stability: {e}")
            return {
                'position_volatility': 0.3,
                'scoring_consistency': 0.5,
                'home_advantage': 0.6,
                'top_team_dominance': 0.5,
                'upset_frequency': 0.3
            }    
    def get_team_stats(self, team_id: int, league_id: int, season: str) -> Dict[str, Any]:
        """
        Get detailed team statistics for a specific league and season.
        
        Args:
            team_id: ID of the team
            league_id: ID of the league
            season: Season
            
        Returns:
            Dictionary containing team statistics
        """
        try:
            # Use the unified team_statistics module
            from team_statistics import get_team_statistics
            stats = get_team_statistics(team_id, league_id, season)
            
            if not stats:
                return self._get_default_team_stats()
            
            # Convert stats to historical format
            return {
                'avg_goals_scored': stats.get('goals_per_game', 1.5),
                'avg_goals_conceded': stats.get('goals_conceded_per_game', 1.2),
                'clean_sheets': stats.get('clean_sheets', 5),
                'failed_to_score': stats.get('failed_to_score', 3),
                'form_score': stats.get('form_score', 0.5),
                'win_rate': stats.get('win_rate', 0.33),                'draw_rate': stats.get('draw_rate', 0.33),
                'loss_rate': stats.get('loss_rate', 0.34),
                'corners_per_game': stats.get('corners_per_game', 5.0),
                'cards_per_game': stats.get('cards_per_game', 2.0)
            }
        except Exception as e:
            logger.error(f"Error getting team stats: {e}")
            return self._get_default_team_stats()

    def _get_default_team_stats(self) -> Dict[str, Any]:
        """
        Returns default statistics when no data is available.
        """
        return {
            'avg_goals_scored': 1.5,
            'avg_goals_conceded': 1.2,
            'clean_sheets': 5,
            'failed_to_score': 3,
            'form_score': 0.5,
            'win_rate': 0.33,
            'draw_rate': 0.33,
            'loss_rate': 0.34,
            'corners_per_game': 5.0,
            'cards_per_game': 2.0
        }

    def get_team_matches(self, team_id: int) -> List[Dict[str, Any]]:
        """Get historical matches for a team across seasons"""
        try:
            all_matches = []
            seasons = range(2020, 2025)  # Last 5 seasons
            
            # Get matches for each season without passing season parameter
            matches = self.api.get_team_matches(
                team_id=team_id,
                limit=100  # Get a good sample size
            )
            all_matches.extend(matches)
            
            # Sort by date descending
            return sorted(
                all_matches,
                key=lambda x: x.get('date', '1900-01-01'),
                reverse=True
            )
            
        except Exception as e:
            logger.error(f"Error getting historical matches for team {team_id}: {e}")
            return []
