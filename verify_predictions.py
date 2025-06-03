"""
Script para verificar que el modelo FNN fixed está funcionando correctamente en la predicción.
"""

import os
import sys
import json
import numpy as np
import joblib
import logging
import pandas as pd
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Importar funciones necesarias (adaptar según estructura de proyecto)
from predictions import make_global_prediction

def test_predictions_variability():
    """Verificar que las predicciones para diferentes partidos son diferentes"""
    logger.info("Verificando variabilidad en predicciones de partidos...")
    
    try:
        # Lista de IDs de partidos para probar
        fixture_ids = [1208382, 1208374, 1208373, 1208380, 1208375]
        
        # Obtener predicciones para cada partido
        predictions = []
        for fixture_id in fixture_ids:
            logger.info(f"Obteniendo predicción para partido {fixture_id}...")
            pred = make_global_prediction(fixture_id)
            
            # Guardar valores principales para comparación
            pred_values = {
                'fixture_id': fixture_id,
                'home_goals': round(pred.get('predicted_home_goals', 0), 2),
                'away_goals': round(pred.get('predicted_away_goals', 0), 2),
                'total_goals': round(pred.get('total_goals', 0), 2),
                'over_2_5': round(pred.get('prob_over_2_5', 0), 2),
                'btts': round(pred.get('prob_btts', 0), 2),
                'home_team': pred.get('home_team_id', 0),
                'away_team': pred.get('away_team_id', 0),
                'method': pred.get('method', '')
            }
            predictions.append(pred_values)
        
        # Mostrar predicciones
        logger.info("\nPREDICCIONES OBTENIDAS:")
        print("\n---------------------------------")
        for p in predictions:
            print(f"Partido {p['fixture_id']} ({p['home_team']} vs {p['away_team']}):")
            print(f"  - Home: {p['home_goals']} goles")
            print(f"  - Away: {p['away_goals']} goles") 
            print(f"  - Total: {p['total_goals']} goles")
            print(f"  - Over 2.5: {p['over_2_5']*100:.1f}%")
            print(f"  - BTTS: {p['btts']*100:.1f}%")
            print(f"  - Método: {p['method']}")
            print("---------------------------------")
        
        # Verificar variabilidad
        # Agrupar predicciones por valores idénticos
        unique_values = {}
        for p in predictions:
            key = f"{p['home_goals']}-{p['away_goals']}"
            if key not in unique_values:
                unique_values[key] = []
            unique_values[key].append(p['fixture_id'])
        
        # Contar cuántos valores únicos hay
        num_unique = len(unique_values)
        
        logger.info(f"\nVariabilidad: {num_unique} valores diferentes en {len(predictions)} predicciones")
        
        # Si tenemos algún duplicado, mostrar detalle
        if num_unique < len(predictions):
            logger.warning("Se encontraron duplicados en las predicciones:")
            for key, fixture_list in unique_values.items():
                if len(fixture_list) > 1:
                    logger.warning(f"  Valor {key}: {len(fixture_list)} partidos - IDs: {fixture_list}")
        
        # Criterio para determinar si la variabilidad es suficiente
        if num_unique >= len(predictions) * 0.8:  # Al menos 80% de valores únicos
            logger.info("✓ VARIABILIDAD ACEPTABLE: Las predicciones muestran suficiente variabilidad")
            return True
        else:
            logger.warning("✗ VARIABILIDAD INSUFICIENTE: Las predicciones son demasiado similares")
            return False
    
    except Exception as e:
        logger.error(f"Error verificando predicciones: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def save_predictions_to_json():
    """Guardar muestra de predicciones en un archivo JSON para referencia"""
    logger.info("Guardando muestra de predicciones en archivo JSON...")
    
    try:
        # Lista de IDs de partidos
        fixture_ids = [1208382, 1208374, 1208373, 1208380, 1208375]
        
        # Obtener predicciones completas
        predictions_list = []
        for fixture_id in fixture_ids:
            pred = make_global_prediction(fixture_id)
            predictions_list.append(pred)
        
        # Crear objeto JSON
        predictions_obj = {
            "match_predictions": predictions_list,
            "timestamp": datetime.now().isoformat(),
            "version": "fixed_1.0"
        }
        
        # Guardar archivo
        output_file = "match_predictions_sample.json"
        with open(output_file, 'w') as f:
            json.dump(predictions_obj, f, indent=2)
        
        logger.info(f"Predicciones guardadas en {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error guardando predicciones: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando verificación de predicciones con modelo mejorado...")
    
    # Verificar variabilidad en predicciones
    variability_ok = test_predictions_variability()
    
    # Si las predicciones tienen variabilidad, guardar una muestra en JSON
    if variability_ok:
        save_predictions_to_json()
        logger.info("Verificación completa. El sistema está funcionando correctamente.")
    else:
        logger.warning("Verificación fallida. Revisar implementación del modelo.")
