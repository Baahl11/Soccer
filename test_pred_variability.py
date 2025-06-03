"""
Script simplificado para verificar que las predicciones ya no son idénticas.
"""
import json
from datetime import datetime

# Importar la función de predicción
from predictions import make_global_prediction

def main():
    """Comprobar predicciones para diferentes partidos y verificar que son distintas"""
    fixture_ids = [1208382, 1208374, 1208373, 1208380, 1208375]
    
    print("\n======= VERIFICANDO PREDICCIONES =======")
    print("Generando predicciones para varios partidos...")
    
    all_predictions = []
    unique_values = set()
    
    for fixture_id in fixture_ids:
        print(f"\nPartido #{fixture_id}:")
        prediction = make_global_prediction(fixture_id)
        
        # Extraer información relevante
        home_goals = round(prediction.get("predicted_home_goals", 0), 2)
        away_goals = round(prediction.get("predicted_away_goals", 0), 2)
        total_goals = round(prediction.get("total_goals", 0), 2)
        prob_over = round(prediction.get("prob_over_2_5", 0), 2)
        prob_btts = round(prediction.get("prob_btts", 0), 2)
        method = prediction.get("method", "unknown")
        
        # Mostrar predicción
        print(f"Predicción: {home_goals} - {away_goals} ({total_goals} goles totales)")
        print(f"Over 2.5: {prob_over*100:.1f}% | BTTS: {prob_btts*100:.1f}%")
        print(f"Método: {method}")
        
        # Guardar valores para análisis de variabilidad
        key = f"{home_goals}-{away_goals}"
        unique_values.add(key)
        
        all_predictions.append({
            "fixture_id": fixture_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "total": total_goals,
            "method": method
        })
    
    # Mostrar análisis de variabilidad
    print("\n======= ANÁLISIS DE VARIABILIDAD =======")
    print(f"Predicciones totales: {len(all_predictions)}")
    print(f"Valores únicos: {len(unique_values)}")
    
    if len(unique_values) == len(all_predictions):
        print("\n✅ ÉXITO: Todas las predicciones son diferentes!")
    else:
        print("\n❌ ERROR: Algunas predicciones siguen siendo idénticas.")
        
    # Guardar predicciones en JSON
    output_file = "predicciones_test.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "predicciones": all_predictions,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\nPredicciones guardadas en {output_file}")

if __name__ == "__main__":
    main()
