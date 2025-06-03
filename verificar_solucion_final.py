"""
Script final de verificación para confirmar que las predicciones tienen variabilidad adecuada.
"""
import json
import pandas as pd
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
    
    for fixture_id in fixture_ids:
        prediction = make_global_prediction(fixture_id)
        
        # Extractar valores principales
        result = {
            "fixture_id": fixture_id,
            "home_goals": round(prediction.get("predicted_home_goals", 0), 2),
            "away_goals": round(prediction.get("predicted_away_goals", 0), 2),
            "total": round(prediction.get("total_goals", 0), 2),
            "over_2.5": round(prediction.get("prob_over_2_5", 0), 2),
            "btts": round(prediction.get("prob_btts", 0), 2),
            "method": prediction.get("method", "unknown")
        }
        
        results.append(result)
        
        print(f"Partido #{fixture_id}: {result['home_goals']} - {result['away_goals']} (Total: {result['total']})")
        print(f"  Método: {result['method']}, Over 2.5: {result['over_2.5']*100:.0f}%, BTTS: {result['btts']*100:.0f}%\n")
    
    # Crear DataFrame para mejor análisis
    df = pd.DataFrame(results)
    
    # Verificar variabilidad
    print("\n===== ANÁLISIS DE VARIABILIDAD =====")
    print(f"Total de predicciones: {len(df)}")
    print(f"Valores únicos de goles locales: {df['home_goals'].nunique()} de {len(df)}")
    print(f"Valores únicos de goles visitantes: {df['away_goals'].nunique()} de {len(df)}")
    print(f"Valores únicos de goles totales: {df['total'].nunique()} de {len(df)}")
    
    # Estadísticas descriptivas
    print("\n===== ESTADÍSTICAS DESCRIPTIVAS =====")
    print(df[['home_goals', 'away_goals', 'total', 'over_2.5', 'btts']].describe().round(3))
    
    # Guardar resultados en JSON
    with open("resultados_finales.json", "w", encoding="utf-8") as f:
        json.dump({"predicciones": results}, f, indent=2)
    
    print("\n✅ Verificación completa. Resultados guardados en resultados_finales.json")

if __name__ == "__main__":
    main()
