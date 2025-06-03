"""
Script simplificado para probar las correcciones de ELO y expected_goal_diff
"""

import json
from team_elo_rating import EloRating
import traceback
import math

def test_elo_expected_goal_diff():
    """Prueba que el expected_goal_diff se calcule correctamente"""
    try:
        print("==== Inicio de prueba ELO expected_goal_diff ====")
        
        # Inicializar el sistema ELO
        print("Inicializando sistema ELO...")
        elo_system = EloRating()
        elo_system.force_load_ratings()
        
        # Obtener valores para equipos específicos
        home_team_id = 541  # Real Madrid
        away_team_id = 529  # Barcelona
        
        print(f"Usando equipos: {home_team_id} (Real Madrid) vs {away_team_id} (Barcelona)")
        
        # Obtener ratings
        home_elo = elo_system.get_team_rating(home_team_id)
        away_elo = elo_system.get_team_rating(away_team_id)
        elo_diff = home_elo - away_elo
        
        print(f"Home team (Real Madrid) ELO: {home_elo}")
        print(f"Away team (Barcelona) ELO: {away_elo}")
        print(f"ELO difference: {elo_diff}")
        
        # Obtener expected_goal_diff
        print("Calculando expected_goal_diff...")
        expected_goal_diff = elo_system.get_expected_goal_diff(home_team_id, away_team_id)
        print(f"Expected goal difference: {expected_goal_diff}")
        
        # Verificar que no sea None
        if expected_goal_diff is None:
            print("⚠️ expected_goal_diff es None")
        elif math.isnan(expected_goal_diff):
            print("⚠️ expected_goal_diff es NaN")
        else:
            print(f"✅ expected_goal_diff calculado correctamente: {expected_goal_diff}")
        
        # Obtener probabilidades de victoria/empate/derrota
        win_prob, draw_prob, loss_prob = elo_system.get_match_probabilities(home_team_id, away_team_id)
        print(f"Probabilidades: Victoria: {win_prob:.3f}, Empate: {draw_prob:.3f}, Derrota: {loss_prob:.3f}")
        
        print("==== Fin de prueba ELO expected_goal_diff ====")
    
    except Exception as e:
        print(f"Error durante la prueba: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_elo_expected_goal_diff()
