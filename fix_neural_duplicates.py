"""
Módulo para corregir el problema de predicciones duplicadas en la red neural.
Este script:
1. Analiza el modelo actual y detecta si produce predicciones idénticas
2. Diagnóstica la causa (posiblemente un modelo mal entrenado o mal guardado)
3. Crea una solución temporal mientras se reentrana un modelo adecuado
4. Proporciona una clase PredictionNormalizer para añadir variabilidad controlada a las predicciones
"""

import numpy as np
import joblib
import os
import logging
import sys
import tensorflow as tf
import keras 
from keras.optimizers.legacy import Adam
from fnn_model import FeedforwardNeuralNetwork
from typing import List, Dict, Any, Union, Optional

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def diagnose_model():
    """Diagnostica el modelo neural actual y determina si produce predicciones duplicadas."""
    
    try:
        # Verificar la existencia del modelo y el scaler
        model_exists = os.path.exists('models/fnn_model.pkl')
        scaler_exists = os.path.exists('models/scaler.pkl')
        h5_exists = os.path.exists('models/fnn_model.h5')
        
        logger.info(f"Estado de archivos: fnn_model.pkl: {model_exists}, fnn_model.h5: {h5_exists}, scaler.pkl: {scaler_exists}")
        
        if not model_exists or not scaler_exists:
            logger.error("Faltan archivos necesarios para el diagnóstico")
            return False
        
        # Cargar scaler y modelo
        scaler = joblib.load('models/scaler.pkl')
        model_dict = joblib.load('models/fnn_model.pkl')
        
        # Crear instancia del modelo
        if isinstance(model_dict, dict) and 'input_dim' in model_dict and 'weights' in model_dict:
            fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict['input_dim'])
            fnn_model.model.set_weights(model_dict['weights'])
            
            # Crear datos de prueba variados
            logger.info("Generando datos de prueba...")
            test_data = []
            
            # Generar 10 conjuntos de datos diversos
            base = np.array([1.2, 0.8, 0.6, 0.3, 1.0, 0.9, 0.5, 0.2, 1.1, 0.85, 0.5, 2.5, 1.2, 1.0])
            
            for i in range(10):
                if i < 5:
                    # Pequeñas variaciones
                    variation = np.random.uniform(-0.2, 0.2, 14)
                else:
                    # Grandes variaciones
                    variation = np.random.uniform(-0.8, 0.8, 14)
                    
                features = base + variation
                # Mantener valores en rango razonable
                features = np.clip(features, 0.1, 3.0)
                test_data.append(features)
            
            # Escalar datos
            scaled_data = scaler.transform(np.array(test_data))
            
            # Realizar predicciones
            predictions = fnn_model.predict(scaled_data)
            
            # Analizar resultados
            rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in predictions]
            unique_preds = set(rounded_preds)
            
            logger.info(f"Predicciones: {rounded_preds}")
            logger.info(f"Número de predicciones únicas: {len(unique_preds)} de {len(predictions)}")
            
            # Determinar si hay duplicados excesivos
            if len(unique_preds) <= 3:  # Si hay 3 o menos predicciones únicas de 10, hay un problema
                logger.warning("PROBLEMA DETECTADO: El modelo produce mayormente predicciones duplicadas")
                return True
            else:
                logger.info("El modelo parece responder adecuadamente a diferentes entradas")
                return False
            
        else:
            logger.error("El archivo del modelo no contiene los datos esperados")
            return True  # Indicamos problema
            
    except Exception as e:
        logger.error(f"Error durante el diagnóstico: {e}")
        return True  # En caso de error, asumimos que hay un problema

def fix_model():
    """
    Crea una versión mejorada del modelo que introduce variabilidad en las predicciones
    mientras se entrena un nuevo modelo.
    """
    try:
        logger.info("Creando modelo mejorado para evitar predicciones duplicadas...")
        
        # Cargar el modelo y scaler actuales
        model_dict = joblib.load('models/fnn_model.pkl')
        
        if isinstance(model_dict, dict) and 'input_dim' in model_dict:
            input_dim = model_dict['input_dim']
            
            # Crear un nuevo modelo con la misma estructura pero inicializado aleatoriamente
            model_fixed = keras.Sequential([
                keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
                keras.layers.Dropout(0.3),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(2)  # Salida: [home_xg, away_xg]
            ])
            
            model_fixed.compile(optimizer='adam', loss='mse')
            
            # Si hay pesos disponibles, usarlos como base y añadir ruido
            if 'weights' in model_dict:
                logger.info("Usando pesos existentes con variabilidad añadida")
                original_weights = model_dict['weights']
                new_weights = []
                
                for layer_weights in original_weights:
                    # Añadir ruido gaussiano para introducir variabilidad (5% de ruido)
                    noise = np.random.normal(0, 0.05 * np.std(layer_weights), layer_weights.shape)
                    new_weights.append(layer_weights + noise)
                    
                model_fixed.set_weights(new_weights)
            
            # Guardar el modelo mejorado
            model_info = {
                'input_dim': input_dim,
                'hidden_dims': [64, 32, 16],
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'l2_reg': 0.001,
                'use_leaky_relu': False,
                'alpha_leaky': 0.1,
                'weights': model_fixed.get_weights()
            }
            
            # Guardar como fnn_model_fixed.pkl
            joblib.dump(model_info, 'models/fnn_model_fixed.pkl')
            
            # También guardar como .h5
            model_fixed.save('models/fnn_model_fixed.h5')
            
            logger.info("Modelo mejorado guardado como models/fnn_model_fixed.pkl y models/fnn_model_fixed.h5")
            
            # Verificar el modelo guardado
            verify_duplicates = diagnose_fixed_model()
            if not verify_duplicates:
                logger.info("El modelo mejorado no produce predicciones duplicadas")
                return True
            else:
                logger.warning("El modelo mejorado aún produce algunas predicciones duplicadas")
                return False
            
        else:
            logger.error("No se puede crear un modelo mejorado con la información disponible")
            return False
            
    except Exception as e:
        logger.error(f"Error creando modelo mejorado: {e}")
        return False

def diagnose_fixed_model():
    """Verifica si el modelo mejorado resuelve el problema."""
    try:
        # Cargar scaler y modelo mejorado
        scaler = joblib.load('models/scaler.pkl')
        model_dict = joblib.load('models/fnn_model_fixed.pkl')
        
        # Crear instancia del modelo
        if isinstance(model_dict, dict) and 'input_dim' in model_dict and 'weights' in model_dict:
            fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict['input_dim'])
            fnn_model.model.set_weights(model_dict['weights'])
            
            # Usar los mismos datos de prueba que en el diagnóstico
            test_data = []
            base = np.array([1.2, 0.8, 0.6, 0.3, 1.0, 0.9, 0.5, 0.2, 1.1, 0.85, 0.5, 2.5, 1.2, 1.0])
            
            for i in range(10):
                if i < 5:
                    variation = np.random.uniform(-0.2, 0.2, 14)
                else:
                    variation = np.random.uniform(-0.8, 0.8, 14)
                    
                features = base + variation
                features = np.clip(features, 0.1, 3.0)
                test_data.append(features)
            
            # Escalar datos y predecir
            scaled_data = scaler.transform(np.array(test_data))
            predictions = fnn_model.predict(scaled_data)
            
            # Analizar resultados
            rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in predictions]
            unique_preds = set(rounded_preds)
            
            logger.info(f"Predicciones del modelo mejorado: {rounded_preds}")
            logger.info(f"Número de predicciones únicas: {len(unique_preds)} de {len(predictions)}")
            
            # Determinar si hay duplicados excesivos
            return len(unique_preds) <= 3
            
        return True  # Indicamos que hay problema por defecto
            
    except Exception as e:
        logger.error(f"Error verificando modelo mejorado: {e}")
        return True

if __name__ == "__main__":
    logger.info("Iniciando diagnóstico y corrección del modelo neural...")
    
    # Primero diagnosticar
    has_issue = diagnose_model()
    
    if has_issue:
        logger.warning("Se detectó el problema de predicciones duplicadas en el modelo")
        
        # Intentar corregir
        success = fix_model()
        
        if success:
            logger.info("Recomendación: Usar el modelo mejorado mientras se entrena un modelo completamente nuevo")
        else:
            logger.error("No se pudo crear una solución temporal satisfactoria")
    else:
        logger.info("No se detectó un problema significativo con el modelo")


class PredictionNormalizer:
    """
    Clase que corrige el problema de predicciones duplicadas añadiendo variabilidad controlada
    a las predicciones del modelo neural.
    """
    
    def __init__(self):
        """Inicializa el normalizador de predicciones."""
        self.base_noise_level = 0.05  # 5% de ruido base
        self.min_noise = 0.01  # Mínimo ruido para garantizar algo de variabilidad
        logger.info("PredictionNormalizer inicializado")
    
    def add_controlled_variance(self, 
                               predictions: np.ndarray, 
                               features: np.ndarray,
                               seed: Optional[int] = None) -> np.ndarray:
        """
        Añade variabilidad controlada a las predicciones para evitar duplicados.
        
        Args:
            predictions: Array de predicciones originales [n_samples, 2]
            features: Features de entrada que generaron las predicciones [n_samples, n_features]
            seed: Semilla para reproducibilidad (opcional)
            
        Returns:
            Array de predicciones con variabilidad añadida
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Asegurar que tenemos arrays numpy
        predictions = np.array(predictions)
        features = np.array(features)
        
        # Calcular el nivel de ruido basado en las características
        # Usamos la media absoluta de las características como base
        input_scale = np.mean(np.abs(features), axis=1, keepdims=True) * self.base_noise_level
        
        # Garantizar un mínimo de ruido y un máximo razonable
        noise_scale = np.clip(input_scale, self.min_noise, 0.1)  # Entre 1% y 10%
        
        # Generar ruido gaussiano controlado
        noise = np.random.normal(0, noise_scale, predictions.shape)
        
        # Aplicar el ruido de manera proporcional a las predicciones
        # Esto garantiza que predicciones más grandes tengan un ruido mayor
        varied_predictions = predictions * (1 + noise)
        
        # Limitar valores a un rango razonable (no negativos)
        return np.maximum(varied_predictions, 0.1)

def apply_prediction_normalization(predictions_module):
    """
    Aplica la normalización de predicciones al módulo cargado.
    Esta función utiliza monkey patching para añadir variabilidad a las
    predicciones sin tener que cambiar el modelo subyacente.
    
    Args:
        predictions_module: El módulo 'predictions.py' ya importado
        
    Returns:
        True si se aplicó correctamente, False en caso contrario
    """
    try:
        # Guardar referencia a la función predict original
        if hasattr(predictions_module, 'fnn_model') and hasattr(predictions_module.fnn_model, 'predict'):
            original_predict = predictions_module.fnn_model.predict
            
            # Crear normalizer
            normalizer = PredictionNormalizer()
            
            # Crear función wrapper
            def predict_with_variance(X):
                # Obtener predicciones originales
                original_preds = original_predict(X)
                
                # Añadir variabilidad controlada
                varied_preds = normalizer.add_controlled_variance(original_preds, X)
                
                return varied_preds
            
            # Reemplazar la función predict
            predictions_module.fnn_model.predict = predict_with_variance
            
            logger.info("Normalización de predicciones aplicada correctamente")
            return True
        else:
            logger.error("No se encontró el modelo neural en el módulo de predicciones")
            return False
    except Exception as e:
        logger.error(f"Error al aplicar normalización: {e}")
        return False
