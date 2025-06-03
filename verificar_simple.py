"""
Script final simple de verificación para confirmar que las predicciones tienen variabilidad adecuada.
"""
import json
from predictions import make_global_prediction

def main():
    print("\n===== VERIFICACIÓN FINAL DE SOLUCIÓN =====")
    print("Generando múltiples predicciones para comparar variabilidad...\n")
    
    # Lista de partidos para comprobar
    fixture_ids = [
        1208382, 1208374, 1208373, 1208380, 1208375,  # Partidos originales
        1208385, 1208386, 1208387, 1208388, 1208389   # Partidos adicionales
    ]
    
    results = []
    unique_values = set()
    
    for fixture_id in fixture_ids:
        prediction = make_global_prediction(fixture_id)
        
        # Extractar valores principales
        home_goals = round(prediction.get("predicted_home_goals", 0), 2)
        away_goals = round(prediction.get("predicted_away_goals", 0), 2)
        total = round(prediction.get("total_goals", 0), 2)
        method = prediction.get("method", "unknown")
        
        # Guardar valores únicos para análisis
        key = f"{home_goals}-{away_goals}"
        unique_values.add(key)
        
        results.append({
            "fixture_id": fixture_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "total": total,
            "method": method
        })
        
        print(f"Partido #{fixture_id}: {home_goals} - {away_goals} (Total: {total})")
        print(f"  Método: {method}\n")
    
    # Verificar variabilidad
    print("\n===== ANÁLISIS DE VARIABILIDAD =====")
    print(f"Total de predicciones: {len(results)}")
    print(f"Valores únicos: {len(unique_values)}")
    
    if len(unique_values) == len(results):
        print("\n✅ ÉXITO: Todas las predicciones son diferentes!")
    else:
        print("\n❌ ERROR: Algunas predicciones siguen siendo idénticas.")
    
    # Guardar resultados en JSON
    with open("resultados_finales.json", "w", encoding="utf-8") as f:
        json.dump({"predicciones": results}, f, indent=2)
    
    print("\nResultados guardados en resultados_finales.json")

if __name__ == "__main__":
    main()
