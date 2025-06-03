"""
Script simplificado para comprobar la variabilidad en las predicciones.
"""
import sys
import logging
import json
from typing import Dict, Any, List
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar la función de predicción
from predictions import make_global_prediction

def main():
    """Verificar y mostrar predicciones para varios partidos"""
    fixture_ids = [1208382, 1208374, 1208373, 1208380, 1208375]
    
    print("Generando predicciones para varios partidos...")
    all_predictions = []
    
    for fixture_id in fixture_ids:
        print(f"\nProcesando partido {fixture_id}...")
        prediction = make_global_prediction(fixture_id)
        
        # Extraer información relevante
        home_goals = prediction.get("predicted_home_goals", 0)
        away_goals = prediction.get("predicted_away_goals", 0)
        total_goals = prediction.get("total_goals", 0)
        prob_over = prediction.get("prob_over_2_5", 0)
        prob_btts = prediction.get("prob_btts", 0)
        method = prediction.get("method", "unknown")
        
        print(f"Resultado: {home_goals:.2f} - {away_goals:.2f} ({total_goals:.2f} goles totales)")
        print(f"Over 2.5: {prob_over*100:.1f}% | BTTS: {prob_btts*100:.1f}%")
        print(f"Método: {method}")
        
        # Guardar predicción
        all_predictions.append(prediction)
    
    # Guardar predicciones en JSON para análisis
    output_file = "match_predictions_sample.json"
    predictions_object = {
        "match_predictions": all_predictions,
        "generated_at": datetime.now().isoformat()
    }
    
    with open(output_file, "w") as f:
        json.dump(predictions_object, f, indent=2)
    
    print(f"\nPredicciones guardadas en {output_file}")

if __name__ == "__main__":
    main()
