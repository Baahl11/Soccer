"""
Script para verificar las correcciones y guardar el resultado en JSON
"""

import json
from predictions import make_global_prediction

def main():
    try:
        # Obtener una predicción
        fixture_id = 12345  # ID ficticio
        prediction = make_global_prediction(fixture_id)
        
        # Guardar la predicción en un archivo JSON
        with open('fix_verification_result.json', 'w', encoding='utf-8') as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)
        
        # Crear un informe estructurado
        report = {
            "elo_verification": {
                "home_elo": prediction.get("elo_ratings", {}).get("home_elo"),
                "away_elo": prediction.get("elo_ratings", {}).get("away_elo"),
                "elo_diff": prediction.get("elo_ratings", {}).get("elo_diff"),
                "are_different": prediction.get("elo_ratings", {}).get("home_elo") != prediction.get("elo_ratings", {}).get("away_elo")
            },
            "expected_goal_diff_verification": {
                "value": prediction.get("elo_expected_goal_diff"),
                "is_valid": prediction.get("elo_expected_goal_diff") is not None
            },
            "tactical_analysis_verification": {
                "is_present": "tactical_analysis" in prediction,
                "keys": list(prediction.get("tactical_analysis", {}).keys()) if "tactical_analysis" in prediction else []
            },
            "all_fixes_implemented": False  # Se actualizará basado en las verificaciones
        }
        
        # Verificar si todas las correcciones están implementadas
        report["all_fixes_implemented"] = (
            report["elo_verification"]["are_different"] and
            report["expected_goal_diff_verification"]["is_valid"] and
            report["tactical_analysis_verification"]["is_present"]
        )
        
        # Guardar el informe en un archivo JSON
        with open('fix_verification_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print("Verificación completada, resultados guardados en 'fix_verification_result.json' y 'fix_verification_report.json'")
        
    except Exception as e:
        print(f"Error durante la verificación: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
