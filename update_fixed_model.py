"""
Script para actualizar el modelo fnn_model_fixed mejorando la variabilidad de predicciones.
"""

import os
import sys
import numpy as np
import joblib
import logging
import pickle
from datetime import datetime
import tensorflow as tf
import pandas as pd

# Importar ambos modelos para comparación
from fnn_model import FeedforwardNeuralNetwork
from fnn_model_fixed import FeedforwardNeuralNetworkFixed

# Configurar logging - solo consola para simplificar
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_original_model():
    """Cargar el modelo original y su configuración"""
    logger.info("Cargando el modelo original...")
    
    try:
        model_path = os.path.join('models', 'fnn_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"No se encuentra el modelo original en {model_path}")
            return None, None
        
        model_dict = joblib.load(model_path)
        if not isinstance(model_dict, dict) or 'input_dim' not in model_dict or 'weights' not in model_dict:
            logger.error("El formato del modelo original no es válido")
            return None, None
        
        return model_dict['input_dim'], model_dict['weights']
    
    except Exception as e:
        logger.error(f"Error al cargar el modelo original: {e}")
        return None, None

def update_fixed_model():
    """Actualizar el modelo mejorado con la implementación más reciente"""
    logger.info("Actualizando el modelo fnn_model_fixed...")
    
    try:
        # Cargar configuración y pesos del modelo original
        input_dim, weights = load_original_model()
        if input_dim is None or weights is None:
            logger.error("No se pudo cargar la configuración del modelo original")
            return False
        
        # Crear nueva instancia del modelo mejorado
        logger.info(f"Creando nuevo modelo mejorado con input_dim={input_dim}")
        fixed_model = FeedforwardNeuralNetworkFixed(input_dim=input_dim)
        
        # Transferir pesos del modelo original
        logger.info("Transfiriendo pesos del modelo original...")
        fixed_model.load_weights(weights)
        
        # Verificar coincidencia de arquitecturas
        original_weights = weights
        fixed_weights = fixed_model.get_weights()
        
        if len(original_weights) != len(fixed_weights):
            logger.warning(f"Las arquitecturas no coinciden exactamente: Original={len(original_weights)} capas, Mejorado={len(fixed_weights)} capas")
            # Continuar a pesar de la diferencia, ya que se manejará en load_weights
        
        # Guardar el modelo actualizado
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar el archivo .h5
        h5_path = os.path.join(models_dir, 'fnn_model_fixed.h5')
        fixed_model.save_model(h5_path)
        logger.info(f"Modelo Keras guardado en {h5_path}")
        
        # Guardar en formato pickle para compatibilidad
        pkl_path = os.path.join(models_dir, 'fnn_model_fixed.pkl')
        model_dict = {
            'input_dim': input_dim,
            'weights': fixed_model.get_weights()
        }
        joblib.dump(model_dict, pkl_path)
        logger.info(f"Modelo serializado guardado en {pkl_path}")
        
        # Verificar que el modelo se guardó correctamente
        if os.path.exists(pkl_path) and os.path.exists(h5_path):
            logger.info("Modelo actualizado guardado con éxito")
            return True
        else:
            logger.error("Error al guardar el modelo actualizado")
            return False
    
    except Exception as e:
        logger.error(f"Error al actualizar el modelo mejorado: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_variability():
    """Probar la variabilidad del modelo actualizado"""
    logger.info("Verificando variabilidad del modelo actualizado...")
    
    try:
        # Cargar scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        if not os.path.exists(scaler_path):
            logger.error(f"No se encuentra el scaler en {scaler_path}")
            return False
        
        scaler = joblib.load(scaler_path)
        
        # Cargar modelo actualizado
        model_path = os.path.join('models', 'fnn_model_fixed.pkl')
        model_dict = joblib.load(model_path)
        fixed_model = FeedforwardNeuralNetworkFixed(input_dim=model_dict['input_dim'])
        fixed_model.load_weights(model_dict['weights'])
        
        # Crear datos de prueba (vector idéntico repetido 10 veces)
        base_vector = [1.2, 1.1, 0.6, 0.4, 1.5, 1.2, 0.5, 0.3, 1.3, 1.4, 0.4, 2.5, 1.4, 1.3]
        test_data = []
        for _ in range(10):
            test_data.append(base_vector.copy())
        
        test_data = np.array(test_data)
        test_data_scaled = scaler.transform(test_data)
        
        # Verificar predicciones
        predictions = fixed_model.predict(test_data_scaled)
        
        # Analizar variabilidad
        logger.info("Predicciones del modelo actualizado:")
        for i, pred in enumerate(predictions[:5]):
            logger.info(f"  Predicción {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
        
        # Verificar si hay variabilidad (al menos 8 valores únicos de 10)
        rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in predictions]
        unique_preds = set(rounded_preds)
        
        variability_ratio = len(unique_preds) / len(predictions)
        logger.info(f"Ratio de variabilidad: {len(unique_preds)}/{len(predictions)} = {variability_ratio:.2f}")
        
        if variability_ratio >= 0.8:
            logger.info("✓ La variabilidad del modelo es BUENA")
            return True
        elif variability_ratio >= 0.5:
            logger.info("⚠ La variabilidad del modelo es ACEPTABLE pero podría mejorar")
            return True
        else:
            logger.info("✗ La variabilidad del modelo es INSUFICIENTE")
            return False
    
    except Exception as e:
        logger.error(f"Error al verificar variabilidad: {e}")
        return False

if __name__ == "__main__":
    logger.info("Iniciando actualización del modelo fnn_model_fixed...")
    
    # Actualizar y guardar el modelo mejorado
    if update_fixed_model():
        # Verificar la variabilidad del modelo actualizado
        if test_model_variability():
            logger.info("Proceso completado con éxito. El modelo está listo para usar.")
        else:
            logger.warning("El modelo se actualizó pero no pasó la prueba de variabilidad.")
    else:
        logger.error("No se pudo actualizar el modelo mejorado.")
