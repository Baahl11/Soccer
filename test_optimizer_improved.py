"""
Script para probar el optimizador de hiperparámetros mejorado que resuelve errores de tipo.
Este script realiza una prueba rápida del optimizador con datos sintéticos.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from hyperparameter_optimizer_improved import OptunaOptimizer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reducir mensajes de TensorFlow
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_optimizer_with_synthetic_data():
    """
    Prueba el optimizador con datos sintéticos para verificar su funcionamiento básico.
    """
    logger.info("Iniciando prueba de optimizer mejorado con datos sintéticos...")
    
    # Crear directorio temporal para tests
    test_dir = 'models/test_optimizer'
    os.makedirs(test_dir, exist_ok=True)
    
    # Generar datos sintéticos
    logger.info("Generando datos sintéticos...")
    input_dim = 10
    n_samples = 100
    
    # Features aleatorias
    X = np.random.rand(n_samples, input_dim)
    
    # Target: una función simple y = 0.5*x1 + 0.3*x2 + ruido
    y = 0.5 * X[:, 0:1] + 0.3 * X[:, 1:2] + 0.1 * np.random.randn(n_samples, 1)
    # Duplicar para simular home_goals y away_goals
    y = np.hstack([y, y * 0.7])
    
    # Dividir en train/val
    train_size = int(0.7 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Inicializar optimizador con configuración mínima
    logger.info("Inicializando optimizador...")
    optimizer = OptunaOptimizer(
        study_name="test_synthetic",
        n_trials=2,  # Mínimo para verificar funcionamiento
        save_dir=test_dir
    )
    
    # Verificar inicialización
    assert optimizer.study_name == "test_synthetic"
    assert optimizer.n_trials == 2
    assert optimizer.save_dir == test_dir
    logger.info("Inicialización del optimizador correcta")
    
    # Probar creación de modelo FNN
    logger.info("Probando creación de modelo FNN personalizado...")
    test_params = {
        'input_dim': input_dim,
        'hidden_layers': 2,
        'neurons': 32,
        'dropout_rate': 0.2,
        'learning_rate': 0.001,
        'batch_size': 64,
        'activation': 'relu',
        'l2_regularization': 0.001
    }
    
    model = optimizer._create_custom_fnn(test_params)
    assert model is not None
    assert hasattr(model, 'model')
    
    # Probar predicción con el modelo
    dummy_input = np.random.rand(1, input_dim)
    prediction = model.predict(dummy_input)
    assert isinstance(prediction, np.ndarray)
    assert prediction.shape == (1, 2)
    logger.info("Creación y predicción del modelo correcta")
    
    # Probar optimización con muy pocos trials (solo para verificar que no hay errores de tipo)
    logger.info("Ejecutando prueba rápida de optimización (2 trials)...")
    try:
        result = optimizer.optimize_fnn(
            X_train, y_train, X_val, y_val, early_stopping_rounds=3
        )
        
        # Verificar estructura del resultado
        assert isinstance(result, dict)
        assert "best_params" in result
        assert "best_value" in result
        assert "model_path" in result
        assert "study_path" in result
        
        logger.info("Optimización básica completada sin errores")
        return True
    
    except Exception as e:
        logger.error(f"Error en la optimización: {e}")
        return False

if __name__ == "__main__":
    success = test_optimizer_with_synthetic_data()
    
    if success:
        print("\n✅ El optimizador mejorado funciona correctamente sin errores de tipo")
    else:
        print("\n❌ Hubo problemas con el optimizador mejorado")
