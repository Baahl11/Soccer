import numpy as np
import joblib
import logging
import os
import sys
from fnn_model import FeedforwardNeuralNetwork

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_predictions():
    try:
        # Cargar el scaler
        scaler = joblib.load('models/scaler.pkl')
        logger.info(f'Scaler n_features_in_: {scaler.n_features_in_}')
        
        # Cargar el modelo desde .pkl
        model_dict = joblib.load('models/fnn_model.pkl')
        fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict.get('input_dim', 14))
        fnn_model.model.set_weights(model_dict['weights'])
        
        # Crear algunos datos de prueba con pequeñas variaciones
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
        logger.info('Predicciones:')
        for i, pred in enumerate(predictions):
            logger.info(f'Equipo {i+1}: Home XG = {pred[0]:.2f}, Away XG = {pred[1]:.2f}')
            
        # Verificar si hay predicciones idénticas
        unique_predictions = set([(round(p[0], 2), round(p[1], 2)) for p in predictions])
        logger.info(f'Número de predicciones únicas: {len(unique_predictions)} de {len(predictions)}')
        logger.info(f'Predicciones únicas: {unique_predictions}')
        
        # Crear datos muy diferentes para verificar si el modelo responde a cambios grandes
        extreme_features = []
        
        # 2 equipos con características muy contrastantes
        strong_team = np.array([2.5, 0.4, 0.9, 0.7, 0.7, 1.8, 0.2, 0.1, 2.2, 0.5, 0.8, 3.0, 2.5, 0.7])
        weak_team = np.array([0.6, 1.9, 0.1, 0.1, 1.9, 0.5, 0.8, 0.6, 0.4, 1.8, 0.3, 1.8, 0.6, 1.9])
        
        extreme_features.append(strong_team)
        extreme_features.append(weak_team)
        
        # Escalar los datos
        scaled_extreme = scaler.transform(np.array(extreme_features))
        
        # Obtener predicciones
        extreme_predictions = fnn_model.predict(scaled_extreme)
        
        # Mostrar resultados
        logger.info('\nPredicciones para equipos extremos:')
        logger.info(f'Equipo fuerte: Home XG = {extreme_predictions[0][0]:.2f}, Away XG = {extreme_predictions[0][1]:.2f}')
        logger.info(f'Equipo débil: Home XG = {extreme_predictions[1][0]:.2f}, Away XG = {extreme_predictions[1][1]:.2f}')
            
    except Exception as e:
        logger.error(f'Error en test_model_predictions: {e}')

if __name__ == '__main__':
    test_model_predictions()

