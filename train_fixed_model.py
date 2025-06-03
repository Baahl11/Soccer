"""
Script para crear un nuevo modelo FNN mejorado desde cero.
"""

import os
import sys
import numpy as np
import joblib
import logging
import tensorflow as tf
import pandas as pd
from datetime import datetime

# Importar la clase para el modelo mejorado
from fnn_model_fixed import FeedforwardNeuralNetworkFixed

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data():
    """Crear datos sintéticos de entrenamiento"""
    logger.info("Generando datos sintéticos para entrenamiento...")
    
    np.random.seed(42)  # Para reproducibilidad
    
    # Generar X (características) sintético
    n_samples = 1000
    n_features = 14  # Mismo número que usa el modelo original
    
    # Crear características base con patrones
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Crear objetivos (y) con correlaciones a X
    # Crear patrones: algunas características aumentan goles en casa, otras visitantes
    home_weights = np.array([0.4, 0.3, 0.2, 0.1, 0.5, 0.3, 0.1, 0.0, 0.4, 0.2, 0.1, 0.3, 0.2, 0.1])
    away_weights = np.array([0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.2, 0.3, 0.4, 0.1, 0.1, 0.2])
    
    # Calcular valores base
    home_goals = np.maximum(0.1, 1.0 + X @ home_weights + np.random.normal(0, 0.5, n_samples))
    away_goals = np.maximum(0.1, 0.8 + X @ away_weights + np.random.normal(0, 0.5, n_samples))
    
    # Convertir a escala esperada (goals)
    y = np.column_stack([home_goals, away_goals])
    
    # Dividir en entrenamiento (70%) y validación (30%)
    split_idx = int(n_samples * 0.7)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    logger.info(f"Datos generados: {X_train.shape[0]} muestras de entrenamiento, {X_val.shape[0]} de validación")
    
    return X_train, y_train, X_val, y_val

def train_new_model():
    """Entrenar un nuevo modelo desde cero"""
    logger.info("Creando y entrenando nuevo modelo mejorado...")
    
    try:
        # Generar datos sintéticos para entrenamiento
        X_train, y_train, X_val, y_val = create_synthetic_data()
        
        # Crear nuevo modelo
        input_dim = X_train.shape[1]
        model = FeedforwardNeuralNetworkFixed(
            input_dim=input_dim,
            hidden_dims=[32, 16, 8],  # Arquitectura más simple
            learning_rate=0.001,
            dropout_rate=0.2
        )
        
        # Entrenar modelo
        logger.info("Entrenando modelo...")
        print("\n===== INICIO ENTRENAMIENTO =====")
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=20,  # Reducido para que sea más rápido
            batch_size=32,
            verbose=1
        )
        print("===== FIN ENTRENAMIENTO =====\n")
        
        # Guardar modelo
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Guardar en formato h5
        h5_path = os.path.join(models_dir, 'fnn_model_fixed.h5')
        model.save_model(h5_path)
        
        # Guardar en formato pickle
        pkl_path = os.path.join(models_dir, 'fnn_model_fixed.pkl')
        model_dict = {
            'input_dim': input_dim,
            'weights': model.get_weights()
        }
        joblib.dump(model_dict, pkl_path)
        
        logger.info(f"Modelo guardado en {h5_path} y {pkl_path}")
        
        # Crear un scaler simple si no existe
        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            joblib.dump(scaler, scaler_path)
            logger.info(f"Nuevo scaler creado y guardado en {scaler_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error al entrenar modelo: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_variability():
    """Probar la variabilidad del modelo actualizado"""
    logger.info("\nVerificando variabilidad del modelo nuevo...")
    
    try:
        # Cargar scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        if not os.path.exists(scaler_path):
            logger.error(f"No se encuentra el scaler en {scaler_path}")
            return False
        
        scaler = joblib.load(scaler_path)
        
        # Cargar modelo entrenado - recrear con la misma configuración exacta
        model_path = os.path.join('models', 'fnn_model_fixed.pkl')
        model_dict = joblib.load(model_path)
        
        # IMPORTANTE: Usar exactamente los mismos parámetros que en train_new_model
        fixed_model = FeedforwardNeuralNetworkFixed(
            input_dim=model_dict['input_dim'],
            hidden_dims=[32, 16, 8],  # Debe coincidir exactamente
            learning_rate=0.001,
            dropout_rate=0.2
        )
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
        logger.info("Predicciones del modelo:")
        for i, pred in enumerate(predictions[:5]):
            logger.info(f"  Predicción {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
        
        # Verificar si hay variabilidad
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
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Iniciando creación de nuevo modelo FNN mejorado...")
    
    # Crear y entrenar nuevo modelo
    if train_new_model():
        # Verificar la variabilidad del modelo
        if test_model_variability():
            logger.info("Proceso completado con éxito. El nuevo modelo está listo para usar.")
        else:
            logger.warning("El modelo se entrenó pero no pasó la prueba de variabilidad.")
    else:
        logger.error("No se pudo crear y entrenar el nuevo modelo.")
