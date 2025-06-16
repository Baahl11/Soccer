import logging
from typing import Dict, Any, Optional
from data import FootballAPI
from team_form import get_team_form
import numpy as np

logger = logging.getLogger(__name__)

def get_team_statistics(team_id: int, league_id: int, season: Optional[int] = None) -> Dict[str, float]:
    """
    Get detailed team statistics including averages for goals, corners, cards, etc.
    
    Args:
        team_id: ID of the team
        league_id: ID of the league
        season: Optional season year
        
    Returns:
        Dictionary with team statistics    """
    try:
        # Get base team statistics using the API client
        from data import get_api_instance
        api = get_api_instance()
        stats = api.get_team_stats(team_id, league_id, season)
        
        if not stats:
            logger.warning(f"No statistics found for team {team_id}")
            return {}
            
        # Convert base stats into detailed metrics
        total_matches = stats.get('matches_played', 0)
        if total_matches == 0:
            logger.warning(f"No matches found for team {team_id}")
            return {}
            
        # Calculate per-match averages 
        avg_stats = {
            'matches_played': total_matches,
            'corners_per_game': stats.get('corners_per_game', 5.0),
            'cards_per_game': stats.get('cards_per_game', 2.0),
            'home_corners_for': stats.get('home_corners_for', 5.5),
            'away_corners_against': stats.get('away_corners_against', 4.5)
        }
        
        # Get recent form data to enhance statistics
        recent_form = get_team_form(team_id, league_id, season, last_matches=5)
        if recent_form:
            # Combine with recent form data (weighing recent form more heavily)
            form_weight = 0.7
            hist_weight = 0.3
            
            # Update core stats with form-weighted values
            for key in ['corners_per_game', 'cards_per_game']:
                if key in recent_form:
                    avg_stats[key] = (
                        form_weight * recent_form[key] +
                        hist_weight * avg_stats[key]
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
        from data import get_api_instance
        api = get_api_instance()
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