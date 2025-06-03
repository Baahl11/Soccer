"""
Script para validar que la solución implementada para las predicciones duplicadas
funciona correctamente.
"""

import numpy as np
import logging
import sys
import os
import pandas as pd
from fnn_model import FeedforwardNeuralNetwork
from fnn_model_fixed import FeedforwardNeuralNetworkFixed
import joblib
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"validate_solution_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def generate_test_data(n_samples=10):
    """Genera datos de prueba variados para validar los modelos."""
    # Crear datos de prueba basados en patrones realistas de fútbol
    test_data = []
    
    # Equipos con diferentes perfiles - [home_stats, away_stats]
    team_profiles = [
        # Equipo fuerte en casa vs débil fuera
        [2.0, 0.6, 0.8, 0.6, 0.8, 1.8, 0.3, 0.1, 1.9, 1.3, 0.7, 2.5, 2.0, 0.8],
        # Equipo débil en casa vs fuerte fuera
        [0.8, 1.4, 0.3, 0.2, 1.9, 0.7, 0.7, 0.5, 0.95, 1.65, 0.3, 2.5, 0.8, 1.9],
        # Equipos parejos de nivel medio
        [1.3, 1.1, 0.5, 0.3, 1.4, 1.0, 0.5, 0.3, 1.2, 1.2, 0.5, 2.5, 1.3, 1.4],
        # Equipos del mismo nivel muy defensivos
        [0.8, 0.6, 0.4, 0.5, 0.7, 0.5, 0.4, 0.5, 0.7, 0.6, 0.5, 1.5, 0.8, 0.7],
        # Equipos del mismo nivel muy ofensivos
        [2.2, 1.8, 0.5, 0.2, 2.1, 1.9, 0.6, 0.2, 2.0, 2.0, 0.5, 3.5, 2.2, 2.1]
    ]
    
    # Usar los perfiles base y añadir variaciones
    for i in range(n_samples):
        profile_idx = i % len(team_profiles)
        base = np.array(team_profiles[profile_idx])
        
        # Aplicar variación aleatoria para más diversidad
        noise = np.random.uniform(-0.15, 0.15, size=base.shape) 
        features = base + noise
        
        # Asegurar valores positivos y realistas
        features = np.clip(features, 0.1, 3.0)
        test_data.append(features)
        
    return np.array(test_data)

def test_models():
    """Compara el modelo original y el modelo mejorado."""
    try:
        # Verificar existencia de modelos
        if not os.path.exists('models/fnn_model.pkl'):
            logger.error("No se encuentra el modelo original en models/fnn_model.pkl")
            return False
        
        if not os.path.exists('models/scaler.pkl'):
            logger.error("No se encuentra el scaler en models/scaler.pkl")
            return False
        
        # Cargar el scaler
        logger.info("Cargando scaler...")
        scaler = joblib.load('models/scaler.pkl')
        
        # Cargar modelo original
        logger.info("Cargando modelo original...")
        model_dict_orig = joblib.load('models/fnn_model.pkl')
        
        if not isinstance(model_dict_orig, dict) or 'input_dim' not in model_dict_orig:
            logger.error("El modelo original no tiene el formato esperado")
            return False
        
        # Crear instancia del modelo original
        orig_model = FeedforwardNeuralNetwork(input_dim=model_dict_orig['input_dim'])
        if 'weights' in model_dict_orig:
            orig_model.model.set_weights(model_dict_orig['weights'])
        
        # Intentar cargar el modelo mejorado si existe
        fixed_model = None
        if os.path.exists('models/fnn_model_fixed.pkl'):
            logger.info("Cargando modelo mejorado...")
            try:
                model_dict_fixed = joblib.load('models/fnn_model_fixed.pkl')
                if isinstance(model_dict_fixed, dict) and 'input_dim' in model_dict_fixed:
                    fixed_model = FeedforwardNeuralNetworkFixed(input_dim=model_dict_fixed['input_dim'])
                    if 'weights' in model_dict_fixed:
                        fixed_model.load_weights(model_dict_fixed['weights'])
            except Exception as e:
                logger.error(f"Error cargando modelo mejorado: {e}")
        else:
            logger.warning("Modelo mejorado no disponible. Sólo se probará el modelo original.")
        
        # Generar datos de prueba
        logger.info("Generando datos de prueba...")
        test_data = generate_test_data(n_samples=20)
        
        # Escalar datos
        test_data_scaled = scaler.transform(test_data)
        
        # Evaluar modelo original
        logger.info("\nEvaluando modelo original:")
        orig_predictions = orig_model.predict(test_data_scaled)
        
        # Analizar predicciones originales
        orig_rounded = [(round(p[0], 3), round(p[1], 3)) for p in orig_predictions]
        orig_unique = set(orig_rounded)
        logger.info(f"Predicciones originales: {orig_rounded}")
        logger.info(f"Número de predicciones únicas: {len(orig_unique)} de {len(orig_rounded)}")
        
        # Crear tabla para mostrar las predicciones originales
        logger.info("\nTabla de predicciones del modelo original:")
        for i, pred in enumerate(orig_predictions):
            logger.info(f"Ejemplo {i+1:2d}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
        
        # Evaluar modelo mejorado si está disponible
        if fixed_model is not None:
            logger.info("\nEvaluando modelo mejorado:")
            fixed_predictions = fixed_model.predict(test_data_scaled)
            
            # Analizar predicciones mejoradas
            fixed_rounded = [(round(p[0], 3), round(p[1], 3)) for p in fixed_predictions]
            fixed_unique = set(fixed_rounded)
            logger.info(f"Predicciones mejoradas: {fixed_rounded}")
            logger.info(f"Número de predicciones únicas: {len(fixed_unique)} de {len(fixed_rounded)}")
            
            # Crear tabla para mostrar las predicciones mejoradas
            logger.info("\nTabla de predicciones del modelo mejorado:")
            for i, pred in enumerate(fixed_predictions):
                logger.info(f"Ejemplo {i+1:2d}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
            
            # Comparar variabilidad
            orig_std = np.std([p[0] for p in orig_predictions]), np.std([p[1] for p in orig_predictions])
            fixed_std = np.std([p[0] for p in fixed_predictions]), np.std([p[1] for p in fixed_predictions])
            
            logger.info("\nComparación de variabilidad (desviación estándar):")
            logger.info(f"  Modelo original: Home σ = {orig_std[0]:.4f}, Away σ = {orig_std[1]:.4f}")
            logger.info(f"  Modelo mejorado: Home σ = {fixed_std[0]:.4f}, Away σ = {fixed_std[1]:.4f}")
            
            # Veredicto final
            if len(fixed_unique) > len(orig_unique):
                logger.info("\nRESULTADO: El modelo mejorado muestra mayor variabilidad en las predicciones.")
                if len(fixed_unique) >= len(orig_rounded) * 0.9:
                    logger.info("ÉXITO: El modelo mejorado genera predicciones únicas para la mayoría de los casos.")
                else:
                    logger.info("PARCIAL: El modelo mejorado mejora la variabilidad, pero aún tiene limitaciones.")
            else:
                logger.warning("\nRESULTADO: El modelo mejorado no muestra mayor variabilidad que el original.")
                
        return True
        
    except Exception as e:
        logger.error(f"Error durante la validación: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando validación de modelos para problema de predicciones duplicadas")
    test_models()
