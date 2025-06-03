#!/usr/bin/env python3
"""
Dynamic xG Calculator

Calcula valores Expected Goals (xG) específicos para cada equipo 
basándose en sus estadísticas de forma, calidad ofensiva/defensiva 
y contexto del enfrentamiento.
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

def calculate_team_xg(
    team_form: Dict[str, Any],
    opponent_form: Dict[str, Any],
    is_home: bool = True,
    league_id: int = 39,
    h2h_data: Dict[str, Any] = None
) -> float:
    """
    Calcula el Expected Goals (xG) específico para un equipo.
    
    Args:
        team_form: Datos de forma del equipo
        opponent_form: Datos de forma del rival
        is_home: Si el equipo juega en casa
        league_id: ID de la liga (para ajustes específicos)
        h2h_data: Datos históricos head-to-head
        
    Returns:
        Valor xG calculado para el equipo
    """
    
    # Valores base por defecto
    base_xg = 1.2 if is_home else 1.0
    
    if not team_form or not opponent_form:
        logger.warning("Datos de forma insuficientes, usando xG base")
        return base_xg
    
    try:
        # 1. CAPACIDAD OFENSIVA DEL EQUIPO
        team_attack = team_form.get('avg_goals_scored', 1.2)
        
        # 2. DEBILIDAD DEFENSIVA DEL RIVAL  
        opponent_defense = opponent_form.get('avg_goals_conceded', 1.2)
        
        # 3. CÁLCULO BASE: Ataque vs Defensa
        # Fórmula: 60% ataque propio + 40% debilidad defensiva rival
        base_calculation = (team_attack * 0.6) + (opponent_defense * 0.4)
        
        # 4. AJUSTES CONTEXTUALES
        
        # a) Ventaja de local
        if is_home:
            home_advantage = 1.15  # 15% boost para equipo local
        else:
            home_advantage = 0.90   # 10% penalización para visitante
        
        # b) Ajuste por forma reciente
        team_form_score = team_form.get('form_score', 50) / 100.0  # Normalizar a 0-1
        form_multiplier = 0.85 + (team_form_score * 0.3)  # Rango: 0.85 - 1.15
        
        # c) Ajuste por calidad de liga
        league_multipliers = {
            39: 1.10,   # Premier League (alta calidad)
            140: 1.08,  # La Liga  
            135: 1.06,  # Serie A
            78: 1.05,   # Bundesliga
            61: 1.04,   # Ligue 1
        }
        league_multiplier = league_multipliers.get(league_id, 1.0)
        
        # d) Ajuste por historial H2H (si disponible)
        h2h_multiplier = 1.0
        if h2h_data and h2h_data.get('total_matches', 0) >= 3:
            # Calcular promedio de goles en enfrentamientos directos
            if is_home:
                h2h_avg = h2h_data.get('team1_goals', 0) / max(1, h2h_data.get('total_matches', 1))
            else:
                h2h_avg = h2h_data.get('team2_goals', 0) / max(1, h2h_data.get('total_matches', 1))
            
            # Ajustar ligeramente basándose en historial
            if h2h_avg > 1.5:
                h2h_multiplier = 1.1  # Historial ofensivo
            elif h2h_avg < 0.8:
                h2h_multiplier = 0.9  # Historial defensivo
        
        # 5. CÁLCULO FINAL
        calculated_xg = (
            base_calculation * 
            home_advantage * 
            form_multiplier * 
            league_multiplier * 
            h2h_multiplier
        )
        
        # 6. LÍMITES DE SEGURIDAD
        # Asegurar que el xG esté en un rango realista
        final_xg = max(0.3, min(4.5, calculated_xg))
        
        logger.debug(f"xG Calculation - Base: {base_calculation:.2f}, "
                    f"Home: {home_advantage:.2f}, Form: {form_multiplier:.2f}, "
                    f"League: {league_multiplier:.2f}, H2H: {h2h_multiplier:.2f}, "
                    f"Final: {final_xg:.2f}")
        
        return round(final_xg, 2)
        
    except Exception as e:
        logger.error(f"Error calculating xG: {e}")
        return base_xg

def calculate_match_xg(
    home_team_id: int,
    away_team_id: int,
    home_form: Dict[str, Any],
    away_form: Dict[str, Any],
    league_id: int = 39,
    h2h_data: Dict[str, Any] = None
) -> Tuple[float, float]:
    """
    Calcula valores xG para ambos equipos en un enfrentamiento.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante  
        home_form: Forma del equipo local
        away_form: Forma del equipo visitante
        league_id: ID de la liga
        h2h_data: Datos head-to-head
        
    Returns:
        Tuple (home_xg, away_xg)
    """
    
    # Calcular xG para equipo local
    home_xg = calculate_team_xg(
        team_form=home_form,
        opponent_form=away_form,
        is_home=True,
        league_id=league_id,
        h2h_data=h2h_data
    )
    
    # Calcular xG para equipo visitante
    away_xg = calculate_team_xg(
        team_form=away_form,
        opponent_form=home_form,
        is_home=False,
        league_id=league_id,
        h2h_data=h2h_data
    )
    
    logger.info(f"Match xG calculated - Home({home_team_id}): {home_xg}, "
               f"Away({away_team_id}): {away_xg}")
    
    return home_xg, away_xg

def validate_xg_calculation():
    """Función de prueba para validar el cálculo de xG."""
    
    # Datos de prueba
    strong_team_form = {
        'avg_goals_scored': 2.1,
        'avg_goals_conceded': 0.8,
        'form_score': 80
    }
    
    weak_team_form = {
        'avg_goals_scored': 0.9,
        'avg_goals_conceded': 1.8,
        'form_score': 30
    }
    
    # Prueba 1: Equipo fuerte vs débil
    home_xg, away_xg = calculate_match_xg(
        home_team_id=1,
        away_team_id=2,
        home_form=strong_team_form,
        away_form=weak_team_form,
        league_id=39
    )
    
    print(f"Prueba 1 - Fuerte(local) vs Débil(visitante): {home_xg} vs {away_xg}")
    
    # Prueba 2: Equipos equilibrados
    balanced_form = {
        'avg_goals_scored': 1.4,
        'avg_goals_conceded': 1.3,
        'form_score': 55
    }
    
    home_xg2, away_xg2 = calculate_match_xg(
        home_team_id=3,
        away_team_id=4,
        home_form=balanced_form,
        away_form=balanced_form,
        league_id=39
    )
    
    print(f"Prueba 2 - Equilibrado vs Equilibrado: {home_xg2} vs {away_xg2}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validate_xg_calculation()
