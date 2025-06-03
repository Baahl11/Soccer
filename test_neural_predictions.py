"""
Script para diagnosticar el problema de predicciones duplicadas en la red neuronal.
Este script realiza pruebas con datos de diferente variabilidad para ver si el modelo
produce siempre los mismos resultados o responde a las diferencias de entrada.
"""

import numpy as np
import joblib
import logging
import os
import sys
from fnn_model import FeedforwardNeuralNetwork

# Configuración de logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])

logger = logging.getLogger(__name__)

def test_model_predictions():
    try:
        # Cargar el scaler
        logger.info("Cargando scaler...")
        scaler = joblib.load('models/scaler.pkl')
        logger.info(f'Scaler n_features_in_: {scaler.n_features_in_}')
        
        # Cargar el modelo desde .pkl
        logger.info("Cargando modelo neural desde .pkl...")
        model_dict = joblib.load('models/fnn_model.pkl')
        fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict.get('input_dim', 14))
        fnn_model.model.set_weights(model_dict['weights'])
        
        # Crear algunos datos de prueba con pequeñas variaciones
        logger.info("Generando datos de prueba con variaciones pequeñas...")
        test_features = []
        
        # 5 equipos con variaciones pequeñas
        base_features = np.array([1.2, 0.8, 0.6, 0.4, 1.1, 0.9, 0.5, 0.3, 1.0, 0.9, 0.5, 2.5, 1.2, 1.1])
        
        for i in range(5):
            # Añadir pequeñas variaciones aleatorias
            variation = np.random.uniform(-0.1, 0.1, 14)
            features = base_features + variation
            test_features.append(features)
        
        # Escalar los datos
        scaled_features = scaler.transform(np.array(test_features))
        
        # Obtener predicciones
        predictions = fnn_model.predict(scaled_features)
        
        # Mostrar resultados
        logger.info('Predicciones con variaciones pequeñas:')
        for i, pred in enumerate(predictions):
            logger.info(f'Equipo {i+1}: Home XG = {pred[0]:.3f}, Away XG = {pred[1]:.3f}')
            
        # Verificar si hay predicciones idénticas
        unique_predictions = set([(round(p[0], 3), round(p[1], 3)) for p in predictions])
        logger.info(f'Número de predicciones únicas: {len(unique_predictions)} de {len(predictions)}')
        logger.info(f'Predicciones únicas: {unique_predictions}')
        
        # Crear datos muy diferentes para verificar si el modelo responde a cambios grandes
        logger.info("\nGenerando datos de prueba con variaciones extremas...")
        extreme_features = []
        
        # 3 equipos con características muy contrastantes
        strong_team = np.array([2.5, 0.4, 0.9, 0.7, 0.7, 1.8, 0.2, 0.1, 2.2, 0.5, 0.8, 3.0, 2.5, 0.7])
        average_team = np.array([1.2, 0.9, 0.5, 0.4, 1.0, 1.0, 0.5, 0.3, 1.0, 1.0, 0.5, 2.5, 1.2, 1.0])
        weak_team = np.array([0.6, 1.9, 0.1, 0.1, 1.9, 0.5, 0.8, 0.6, 0.4, 1.8, 0.3, 1.8, 0.6, 1.9])
        
        extreme_features.append(strong_team)
        extreme_features.append(average_team)
        extreme_features.append(weak_team)
        
        # Escalar los datos
        scaled_extreme = scaler.transform(np.array(extreme_features))
        
        # Obtener predicciones
        extreme_predictions = fnn_model.predict(scaled_extreme)
        
        # Mostrar resultados
        logger.info('Predicciones para equipos con diferencias extremas:')
        logger.info(f'Equipo fuerte: Home XG = {extreme_predictions[0][0]:.3f}, Away XG = {extreme_predictions[0][1]:.3f}')
        logger.info(f'Equipo medio:  Home XG = {extreme_predictions[1][0]:.3f}, Away XG = {extreme_predictions[1][1]:.3f}')
        logger.info(f'Equipo débil:  Home XG = {extreme_predictions[2][0]:.3f}, Away XG = {extreme_predictions[2][1]:.3f}')
        
        # Verificar si hay respuesta a cambios extremos
        home_diff = abs(extreme_predictions[0][0] - extreme_predictions[2][0])
        away_diff = abs(extreme_predictions[0][1] - extreme_predictions[2][1])
        
        logger.info(f'Diferencia en XG local entre equipos fuerte y débil: {home_diff:.3f}')
        logger.info(f'Diferencia en XG visitante entre equipos fuerte y débil: {away_diff:.3f}')
        
        if home_diff < 0.1 and away_diff < 0.1:
            logger.warning("PROBLEMA DETECTADO: El modelo produce resultados muy similares incluso con datos muy diferentes")
        else:
            logger.info("El modelo parece responder a diferencias significativas en los datos")
            
    except Exception as e:
        logger.error(f'Error en test_model_predictions: {e}')

if __name__ == '__main__':
    test_model_predictions()
