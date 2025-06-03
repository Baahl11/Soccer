"""
Script para probar y visualizar el sistema de rating ELO.

Este script demuestra el sistema de ELO ratings, creando visualizaciones
y realizando pruebas con datos históricos.
"""

import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from typing import Dict, List
from datetime import datetime
from team_elo_rating import TeamEloRating, EloRatingTracker
from elo_rating_updater import update_elo_ratings_from_results, create_team_rating_report

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Diccionario de prueba de mapeo ID a nombres de equipos
TEAM_NAMES = {
    39: "Manchester City",
    40: "Liverpool",
    33: "Manchester United",
    42: "Arsenal",
    47: "Tottenham",
    49: "Chelsea",
    44: "Wolverhampton",
    46: "Leicester",
    45: "Everton",
    51: "Brighton",
    
    541: "Real Madrid",
    529: "Barcelona",
    530: "Atlético Madrid",
    532: "Villarreal",
    546: "Sevilla",
    548: "Real Sociedad",
    
    496: "Juventus",
    505: "Inter",
    489: "AC Milan",
    492: "Napoli",
    487: "Roma",
    
    157: "Bayern Munich",
    165: "Borussia Dortmund",
    173: "RB Leipzig",
    169: "Leverkusen",
}

def generate_test_data(n_matches: int = 50, start_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Genera datos de prueba para actualizar ratings ELO
    
    Args:
        n_matches: Número de partidos a generar
        start_date: Fecha inicial para los partidos
        
    Returns:
        DataFrame con resultados de partidos simulados
    """
    np.random.seed(42)  # Para reproducibilidad
    
    # Seleccionar equipos para nuestros datos simulados
    team_ids = list(TEAM_NAMES.keys())
    n_teams = len(team_ids)
    
    # Crear arrays para datos
    home_teams = []
    away_teams = []
    home_goals = []
    away_goals = []
    league_ids = []
    match_dates = []
    
    # Generar datos aleatorios pero realistas
    for i in range(n_matches):
        # Seleccionar equipos distintos
        home_idx = np.random.randint(0, n_teams)
        away_idx = np.random.randint(0, n_teams)
        while away_idx == home_idx:
            away_idx = np.random.randint(0, n_teams)
            
        home_id = team_ids[home_idx]
        away_id = team_ids[away_idx]
        
        # Generar goles (distribución Poisson)
        home_team_strength = np.random.uniform(0.8, 1.8)
        away_team_strength = np.random.uniform(0.6, 1.5)
        home_goal = np.random.poisson(home_team_strength * 1.4)  # Ventaja local
        away_goal = np.random.poisson(away_team_strength)
        
        # Determinar liga (usamos lógica simple por ID de equipo)
        if home_id < 100:
            league_id = 39  # Premier League
        elif home_id < 400:
            league_id = 78  # Bundesliga
        elif home_id < 500:
            league_id = 135  # Serie A
        elif home_id < 550:
            league_id = 140  # La Liga
        else:
            league_id = 61  # Ligue 1
            
        # Generar fecha (incrementando desde la fecha inicial)
        match_date = pd.Timestamp(start_date) + pd.Timedelta(days=i // 5)
        
        # Añadir a listas
        home_teams.append(home_id)
        away_teams.append(away_id)
        home_goals.append(home_goal)
        away_goals.append(away_goal)
        league_ids.append(league_id)
        match_dates.append(match_date)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'home_team_id': home_teams,
        'away_team_id': away_teams,
        'home_goals': home_goals,
        'away_goals': away_goals,
        'league_id': league_ids,
        'match_date': match_dates
    })
    
    return df

def test_rating_evolution():
    """Prueba la evolución de ratings con datos simulados"""
    logger.info("Generando datos de prueba para evolución de ratings ELO...")
    
    # Generar datos
    matches_df = generate_test_data(100)
    
    # Iniciar sistema ELO
    elo = TeamEloRating()
    
    # Actualizar ratings con datos
    elo = update_elo_ratings_from_results(matches_df, elo, save_history=True)
    
    # Crear directorio para reportes
    os.makedirs("reports", exist_ok=True)
    
    # Crear reporte y visualización
    report_path = create_team_rating_report(list(TEAM_NAMES.keys()), TEAM_NAMES)
    
    logger.info(f"Reporte creado en: {report_path}")
    
    # Imprimir algunos ratings para verificación
    top_teams = [39, 40, 541, 529, 496, 157]
    logger.info("Ratings finales para equipos seleccionados:")
    for team_id in top_teams:
        name = TEAM_NAMES.get(team_id, f"Team {team_id}")
        rating = elo.get_rating(team_id)
        logger.info(f"{name:<20}: {rating:.1f}")

def print_win_probabilities():
    """Muestra ejemplos de probabilidades de victoria según ratings"""
    logger.info("\nProbabilidades de victoria según ratings ELO:")
    
    # Cargar sistema ELO
    elo = TeamEloRating()
    
    # Definir algunos encuentros interesantes para mostrar
    matchups = [
        (39, 40, 39),    # Man City vs Liverpool (Premier)
        (541, 529, 140),  # Real Madrid vs Barcelona (La Liga)
        (496, 505, 135),  # Juventus vs Inter (Serie A)
        (39, 529, None),  # Man City vs Barcelona (sin liga/neutral)
        (541, 39, None),  # Real Madrid vs Man City (sin liga/neutral)
        (157, 541, 2),    # Bayern vs Real Madrid (Champions - ID 2)
    ]
    
    # Mostrar probabilidades para cada partido
    for home_id, away_id, league_id in matchups:
        home_name = TEAM_NAMES.get(home_id, f"Team {home_id}")
        away_name = TEAM_NAMES.get(away_id, f"Team {away_id}")
        league_name = "neutral" if league_id is None else f"league {league_id}"
        
        # Obtener probabilidad
        home_win_prob = elo.get_win_probability(home_id, away_id, league_id)
        draw_prob = 0.32 - (abs(home_win_prob - 0.5) * 0.2)
        away_win_prob = 1 - home_win_prob - draw_prob
        
        # Obtener ratings
        home_rating = elo.get_rating(home_id)
        away_rating = elo.get_rating(away_id)
        
        logger.info(f"{home_name} ({home_rating:.0f}) vs {away_name} ({away_rating:.0f}) - {league_name}")
        logger.info(f"  Victoria local: {home_win_prob:.1%}")
        logger.info(f"  Empate:         {draw_prob:.1%}")
        logger.info(f"  Victoria visit: {away_win_prob:.1%}")
        logger.info("")

if __name__ == "__main__":
    logger.info("== TEST DEL SISTEMA DE RATING ELO ==")
    
    # Probar evolución de ratings
    test_rating_evolution()
    
    # Mostrar probabilidades de victoria
    print_win_probabilities()
    
    logger.info("Pruebas completadas.")
