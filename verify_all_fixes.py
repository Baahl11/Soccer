"""
Script final para verificar todas las correcciones
"""

import json
import sys
from predictions import make_global_prediction
from team_elo_rating import EloRating
from tactical_integration import get_simplified_tactical_analysis

def main():
    try:
        print("Verificando correcciones en el sistema de predicción...")
        
        # 1. Verificar ratings ELO
        print("\n=== Verificación de ratings ELO ===")
        elo = EloRating()
        elo.force_load_ratings()
        home_id, away_id = 541, 529  # Real Madrid vs Barcelona
        home_elo = elo.get_team_rating(home_id)
        away_elo = elo.get_team_rating(away_id)
        expected_goal_diff = elo.get_expected_goal_diff(home_id, away_id)
        
        print(f"Real Madrid (ID: {home_id}) ELO: {home_elo}")
        print(f"Barcelona (ID: {away_id}) ELO: {away_elo}")
        print(f"Diferencia ELO: {home_elo - away_elo}")
        print(f"Expected Goal Diff: {expected_goal_diff}")
        
        # 2. Verificar análisis táctico
        print("\n=== Verificación de análisis táctico ===")
        tactical = get_simplified_tactical_analysis(home_id, away_id)
        
        if tactical:
            print("Análisis táctico obtenido correctamente")
            print(f"Claves: {list(tactical.keys())}")
        else:
            print("No se pudo obtener análisis táctico")
        
        # 3. Verificar todo junto en la predicción global
        print("\n=== Verificación de la predicción global ===")
        pred = make_global_prediction(12345)  # Fixture ID ficticio
        
        # Comprobar que tenemos los campos necesarios
        has_elo = "elo_ratings" in pred and pred["elo_ratings"].get("home_elo") != pred["elo_ratings"].get("away_elo")
        has_expected_goal_diff = "elo_expected_goal_diff" in pred and pred["elo_expected_goal_diff"] is not None
        has_tactical = "tactical_analysis" in pred and len(pred["tactical_analysis"]) > 0
        
        print(f"¿Ratings ELO diferentes? {'Sí' if has_elo else 'No'}")
        print(f"¿Expected Goal Diff válido? {'Sí' if has_expected_goal_diff else 'No'}")
        print(f"¿Análisis táctico presente? {'Sí' if has_tactical else 'No'}")
        
        # 4. Resumen
        print("\n=== Resumen de correcciones ===")
        if has_elo and has_expected_goal_diff and has_tactical:
            print("Todas las correcciones implementadas correctamente")
        else:
            print("Algunas correcciones no están funcionando:")
            if not has_elo: print("- Los ratings ELO no son diferentes")
            if not has_expected_goal_diff: print("- El expected_goal_diff no es válido")
            if not has_tactical: print("- Falta el análisis táctico")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
