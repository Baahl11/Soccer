import logging
from typing import Dict, Any, Optional
from data import FootballAPI
from team_form import get_team_form
import numpy as np

logger = logging.getLogger(__name__)

def get_team_statistics(team_id: int, league_id: int, season: Optional[int] = None) -> Dict[str, float]:
    """
    Obtener estadísticas detalladas de un equipo incluyendo promedios de goles, córners, tarjetas, etc.
    
    Args:
        team_id: ID del equipo
        league_id: ID de la liga
        season: Temporada (opcional)
        
    Returns:
        Diccionario con estadísticas del equipo
    """
    try:
        # Obtener estadísticas base del equipo
        api = FootballAPI()
        stats_response = api._make_request('teams/statistics', {
            'team': team_id,
            'league': league_id,
            'season': season
        })
        
        if not stats_response or not stats_response.get('response'):
            logger.warning(f"No statistics found for team {team_id}")
            return {}
            
        stats = stats_response['response']
        
        # Extraer estadísticas relevantes
        goals = stats.get('goals', {})
        cards = stats.get('cards', {})
        fixtures = stats.get('fixtures', {})
        
        total_matches = fixtures.get('played', {}).get('total', 0)
        if total_matches == 0:
            logger.warning(f"No matches found for team {team_id}")
            return {}
            
        # Calcular promedios por partido
        avg_stats = {
            'matches_played': total_matches,
            'avg_goals_scored': goals.get('for', {}).get('average', {}).get('total', 1.5),
            'avg_goals_conceded': goals.get('against', {}).get('average', {}).get('total', 1.2),
            'avg_corners_for': stats.get('corners', {}).get('for', {}).get('average', 5.0),
            'avg_corners_against': stats.get('corners', {}).get('against', {}).get('average', 4.5),
            'avg_cards': (cards.get('yellow', {}).get('total', 30) + 
                         cards.get('red', {}).get('total', 2)) / max(1, total_matches),
            'avg_fouls': stats.get('fouls', {}).get('committed', {}).get('average', 11.0),
            'clean_sheets_ratio': stats.get('clean_sheet', {}).get('total', 5) / max(1, total_matches),
            'failed_to_score_ratio': stats.get('failed_to_score', {}).get('total', 5) / max(1, total_matches)
        }
        
        # Obtener datos de forma reciente
        recent_form = get_team_form(team_id, league_id, season, last_matches=5)
        if recent_form:
            # Combinar con datos de forma reciente (dar más peso a la forma reciente)
            form_weight = 0.7
            hist_weight = 0.3
            
            avg_stats['avg_goals_scored'] = (
                form_weight * recent_form.get('avg_goals_scored', avg_stats['avg_goals_scored']) +
                hist_weight * avg_stats['avg_goals_scored']
            )
            
            avg_stats['avg_goals_conceded'] = (
                form_weight * recent_form.get('avg_goals_conceded', avg_stats['avg_goals_conceded']) +
                hist_weight * avg_stats['avg_goals_conceded']
            )
            
            avg_stats['form_score'] = recent_form.get('form_score', 0.5)
            
        return avg_stats
        
    except Exception as e:
        logger.error(f"Error getting team statistics: {e}")
        return {}

def get_team_foul_tendency(team_id: int, default: float = 1.0) -> float:
    """
    Calcula la tendencia de un equipo a cometer faltas en relación a la media de la liga.
    
    Args:
        team_id: ID del equipo
        default: Valor por defecto si no hay datos
        
    Returns:
        Factor de tendencia de faltas (>1 significa más faltas que el promedio)
    """
    try:
        # Obtener estadísticas del equipo
        api = FootballAPI()
        stats_response = api._make_request('teams/statistics', {
            'team': team_id,
            'last': 10  # Últimos 10 partidos para tendencia reciente
        })
        
        if not stats_response or not stats_response.get('response'):
            logger.warning(f"No statistics found for team {team_id} for foul tendency")
            return default
            
        stats = stats_response['response']
        
        # Extraer estadísticas de faltas
        fouls = stats.get('fouls', {})
        avg_fouls = fouls.get('committed', {}).get('average', 11.0)
        
        # Liga promedio es aproximadamente 11 faltas por partido
        league_avg = 11.0
        
        # Calcular factor de tendencia (normalizado)
        if league_avg > 0:
            foul_tendency = avg_fouls / league_avg
            # Limitar a un rango razonable
            return max(0.7, min(1.3, foul_tendency))
        
        return default
        
    except Exception as e:
        logger.error(f"Error calculando tendencia de faltas para equipo {team_id}: {e}")
        return default