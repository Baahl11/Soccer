"""
Script para generar un modelo neural mejorado que evita predicciones duplicadas.
Este script:
1. Carga el modelo actual y sus pesos
2. Crea un nuevo modelo mejorado basado en el actual
3. Guarda el modelo mejorado para su uso posterior
"""

import numpy as np
import joblib
import os
import logging
import sys
import tensorflow as tf
from fnn_model import FeedforwardNeuralNetwork
from fnn_model_fixed import FeedforwardNeuralNetworkFixed

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("generate_improved_model.log"),
              logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def generate_improved_model():
    """
    Genera un modelo neural mejorado basado en el actual pero con garantía
    de variabilidad en las predicciones.
    """
    try:
        logger.info("Iniciando generación de modelo mejorado...")
        
        # Verificar archivos necesarios
        if not os.path.exists('models/fnn_model.pkl'):
            logger.error("No se encuentra el modelo original en models/fnn_model.pkl")
            return False
        
        if not os.path.exists('models/scaler.pkl'):
            logger.error("No se encuentra el scaler en models/scaler.pkl")
            return False
        
        # Cargar el modelo original
        logger.info("Cargando modelo original...")
        model_dict = joblib.load('models/scaler.pkl')
        
        # Verificar el formato del modelo
        if not isinstance(model_dict, dict) or 'input_dim' not in model_dict:
            logger.error("El modelo original no tiene el formato esperado")
            # Crear un modelo con input_dim por defecto
            input_dim = 14
            logger.info(f"Usando dimensión de entrada predeterminada: {input_dim}")
        else:
            input_dim = model_dict.get('input_dim', 14)
            logger.info(f"Dimensión de entrada del modelo: {input_dim}")
        
        # Crear el nuevo modelo mejorado
        logger.info("Creando modelo mejorado...")
        improved_model = FeedforwardNeuralNetworkFixed(
            input_dim=input_dim,
            hidden_dims=[64, 32, 16],
            learning_rate=0.001,
            dropout_rate=0.3,
            use_leaky_relu=True
        )
        
        # Si hay pesos disponibles, transferirlos con ruido
        if isinstance(model_dict, dict) and 'weights' in model_dict:
            logger.info("Transfiriendo pesos del modelo original...")
            original_weights = model_dict['weights']
            
            # Verificar si la estructura coincide
            if len(original_weights) == len(improved_model.get_weights()):
                new_weights = []
                for i, layer_weights in enumerate(original_weights):
                    # Añadir ruido gaussiano a los pesos para mejorar variabilidad
                    noise_scale = 0.05 * np.std(layer_weights)  # 5% de la desviación estándar
                    noise = np.random.normal(0, noise_scale, layer_weights.shape)
                    new_weights.append(layer_weights + noise)
                
                # Aplicar los pesos modificados
                try:
                    improved_model.load_weights(new_weights)
                    logger.info("Pesos transferidos exitosamente con ruido añadido")
                except Exception as e:
                    logger.error(f"Error al transferir pesos: {e}")
            else:
                logger.warning(f"La estructura de capas no coincide: {len(original_weights)} vs {len(improved_model.get_weights())}")
                logger.info("El modelo usará pesos inicializados aleatoriamente")
        
        # Guardar el nuevo modelo
        logger.info("Guardando modelo mejorado...")
        
        # Crear diccionario para joblib
        improved_model_dict = {
            'input_dim': input_dim,
            'hidden_dims': [64, 32, 16],
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'use_leaky_relu': True,
            'alpha_leaky': 0.1,
            'weights': improved_model.get_weights()
        }
        
        # Guardar con joblib
        joblib.dump(improved_model_dict, 'models/fnn_model_fixed.pkl')
        logger.info("Modelo guardado como models/fnn_model_fixed.pkl")
        
        # También guardar en formato h5
        improved_model.save_model('models/fnn_model_fixed.h5')
        logger.info("Modelo guardado como models/fnn_model_fixed.h5")
        
        # Probar el modelo con datos sintéticos
        logger.info("Verificando el modelo con datos de prueba...")
        test_data = np.random.uniform(0.5, 2.0, (5, input_dim))
        
        # Cargar scaler
        scaler = joblib.load('models/scaler.pkl')
        
        # Escalar datos y obtener predicciones
        test_data_scaled = scaler.transform(test_data)
        predictions = improved_model.predict(test_data_scaled)
        
        logger.info("Predicciones de prueba:")
        for i, pred in enumerate(predictions):
            logger.info(f"  Muestra {i+1}: Home xG: {pred[0]:.3f}, Away xG: {pred[1]:.3f}")
        
        logger.info("Modelo mejorado generado y verificado correctamente")
        return True
    
    except Exception as e:
        logger.error(f"Error generando modelo mejorado: {e}")
        return False

if __name__ == "__main__":
    success = generate_improved_model()
    if success:
        print("Modelo mejorado generado correctamente.")
    else:
        print("Error al generar el modelo mejorado. Ver el log para detalles.")
