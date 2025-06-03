# test_hyperparameter_optimizer.py
import unittest
import sys
import os
import numpy as np
import logging

# Añadir el directorio padre al path para importar los módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hyperparameter_optimizer_improved import OptunaOptimizer
import tensorflow as tf

class TestHyperparameterOptimizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up para todas las pruebas"""
        # Desactivar logging durante las pruebas
        logging.disable(logging.CRITICAL)
        
        # Reducir mensajes de TensorFlow
        tf.get_logger().setLevel('ERROR')
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        
        # Crear directorio temporal para tests
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_models')
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Inicializar optimizador con configuración mínima para tests
        cls.optimizer = OptunaOptimizer(
            study_name="test_optimization",
            n_trials=2,  # Mínimo para tests
            save_dir=cls.test_dir
        )

    @classmethod
    def tearDownClass(cls):
        """Cleanup después de todas las pruebas"""
        # Reactivar logging
        logging.disable(logging.NOTSET)
        
        # Opcionalmente eliminar directorio de test
        # Comentado para permitir inspección manual de resultados
        # import shutil
        # shutil.rmtree(cls.test_dir)
    
    def test_optimizer_initialization(self):
        """Probar inicialización del optimizador"""
        self.assertEqual(self.optimizer.study_name, "test_optimization")
        self.assertEqual(self.optimizer.n_trials, 2)
        self.assertEqual(self.optimizer.save_dir, self.test_dir)
        self.assertIsNotNone(self.optimizer.timestamp)
    
    def test_custom_fnn_creation(self):
        """Probar creación de modelo FNN personalizado"""
        # Parámetros de prueba
        params = {
            'input_dim': 10,
            'hidden_layers': 2,
            'neurons': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 64,
            'activation': 'relu',
            'l2_regularization': 0.001
        }
        
        # Crear modelo
        model = self.optimizer._create_custom_fnn(params)
        
        # Verificar modelo
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.model)
        
        # Probar predicción
        dummy_input = np.random.rand(1, 10)
        prediction = model.predict(dummy_input)
        
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1, 2))  # Debe predecir home_goals y away_goals
    
    def test_ensemble_weights_optimization(self):
        """Probar optimización de pesos para ensemble"""
        # Crear modelos simulados para testing
        def mock_model_1(X):
            return np.ones((len(X), 2)) * 1.5
            
        def mock_model_2(X):
            return np.ones((len(X), 2)) * 2.0
            
        def mock_model_3(X):
            return np.ones((len(X), 2)) * 0.5
        
        # Datos simulados
        X_val = np.random.rand(10, 5)
        y_val = np.ones((10, 2)) * 1.5  # Target ideal (coincide con modelo 1)
        
        # Ejecutar optimización con pocos trials para test
        models = [mock_model_1, mock_model_2, mock_model_3]
        names = ['model_1', 'model_2', 'model_3']
        
        weights = self.optimizer.optimize_ensemble_weights(
            models=models,
            model_names=names,
            X_val=X_val,
            y_val=y_val
        )
        
        # Verificar resultado
        self.assertIsInstance(weights, dict)
        for name in names:
            self.assertIn(name, weights)
            self.assertIsInstance(weights[name], float)
        
        # La suma de pesos debe ser aproximadamente 1
        total = sum(weights.values())
        self.assertAlmostEqual(total, 1.0, places=1)
        
        # El modelo 1 debería tener el mayor peso ya que coincide con target
        self.assertGreater(weights['model_1'], weights['model_3'])

    # Nota: No probamos optimize_fnn completo porque requeriría demasiado tiempo
    # En un test real, sería mejor hacer mock de study.optimize

if __name__ == '__main__':
    unittest.main(verbosity=2)
