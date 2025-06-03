import optuna
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, cast, TypeVar, overload
from fnn_model import FeedforwardNeuralNetwork
from features import FeatureExtractor
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from datetime import datetime
import time

# Definir tipos para conversión de valores
T = TypeVar('T')
ConvertibleToFloat = Union[float, int, str, np.number, np.ndarray]

logger = logging.getLogger(__name__)

class OptunaOptimizer:
    """
    Clase para optimizar hiperparámetros de modelos usando Optuna.
    """
    def __init__(self, 
                 study_name: str = "soccer_model_optimization",
                 n_trials: int = 50, 
                 timeout: Optional[int] = None, 
                 save_dir: str = "models/optuna"):
        """
        Inicializa el optimizador.
        
        Args:
            study_name: Nombre del estudio de Optuna
            n_trials: Número de intentos de optimización
            timeout: Tiempo límite para la optimización en segundos (opcional)
            save_dir: Directorio para guardar resultados
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.timeout = timeout
        self.save_dir = save_dir
        
        # Crear directorio si no existe
        os.makedirs(save_dir, exist_ok=True)
        
        # Generar timestamp para identificar la ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def optimize_fnn(self, 
                     X_train: np.ndarray, 
                     y_train: np.ndarray, 
                     X_val: np.ndarray, 
                     y_val: np.ndarray,
                     early_stopping_rounds: int = 10) -> Dict[str, Any]:
        """
        Optimiza un modelo de red neuronal feedforward.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Targets de entrenamiento
            X_val: Features de validación
            y_val: Targets de validación
            early_stopping_rounds: Número de épocas sin mejora para detener entrenamiento
            
        Returns:
            Diccionario con los mejores hiperparámetros
        """
        def objective(trial) -> float:
            # Definir parámetros a optimizar
            params = {
                'input_dim': X_train.shape[1],
                'hidden_layers': trial.suggest_int('hidden_layers', 1, 3),
                'neurons': trial.suggest_int('neurons', 16, 128, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_int('batch_size', 16, 128, log=True),
                'activation': trial.suggest_categorical('activation', ['relu', 'elu', 'selu']),
                'l2_regularization': trial.suggest_float('l2_regularization', 1e-5, 1e-2, log=True)
            }
            
            # Configurar modelo
            model = self._create_custom_fnn(params)
            
            # Configurar callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=early_stopping_rounds,
                    restore_best_weights=True
                )
            ]
            
            # Entrenar modelo - casteando verbose a str como espera TensorFlow
            history = model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,  # Máximo número de épocas
                batch_size=params['batch_size'],
                callbacks=callbacks,
                verbose=cast(str, "0")  # Corregido: Usar str como espera TensorFlow
            )
            
            # Evaluar modelo - casteando verbose a str como espera TensorFlow
            val_loss = model.model.evaluate(X_val, y_val, verbose=cast(str, "0"))
            
            # Asegurarse de que val_loss es un float (podría ser un array)
            if isinstance(val_loss, (list, np.ndarray)):
                return float(val_loss[0])
            # Asegurar que cualquier tipo se convierte a float adecuadamente
            try:
                return float(cast(ConvertibleToFloat, val_loss))
            except (TypeError, ValueError):
                logger.warning(f"No se pudo convertir {val_loss} a float, retornando valor por defecto")
                return float('inf')  # Valor por defecto en caso de error
        
        # Crear estudio de Optuna
        storage_name = f"sqlite:///{self.save_dir}/{self.study_name}_{self.timestamp}.db"
        study = optuna.create_study(
            study_name=f"{self.study_name}_{self.timestamp}",
            direction="minimize",
            storage=storage_name
        )
        
        # Optimizar
        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout, show_progress_bar=True)
            
            # Guardar resultados
            best_params = study.best_params
            best_value = study.best_value
            
            # Guardar el estudio
            joblib.dump(study, f"{self.save_dir}/study_{self.study_name}_{self.timestamp}.pkl")
            
            # Crear el mejor modelo con los parámetros óptimos
            best_params['input_dim'] = X_train.shape[1]
            best_model = self._create_custom_fnn(best_params)
            
            # Entrenar el mejor modelo
            best_model.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=best_params['batch_size'],
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=early_stopping_rounds,
                        restore_best_weights=True
                    )
                ],
                verbose=cast(str, "1")  # Usar string como espera TensorFlow
            )
            
            # Guardar el mejor modelo
            model_save_path = f"{self.save_dir}/best_fnn_model_{self.timestamp}"
            try:
                # Guardar modelo usando Keras directamente
                best_model.model.save(model_save_path)
                logger.info(f"Modelo guardado en {model_save_path}")
            except Exception as save_error:
                logger.error(f"Error al guardar el modelo: {save_error}")
                # Intentar guardar solo los pesos como alternativa
                try:
                    weights_path = f"{self.save_dir}/best_model_weights_{self.timestamp}.h5"
                    best_model.model.save_weights(weights_path)
                    logger.info(f"Se guardaron los pesos del modelo en {weights_path}")
                except Exception as weights_error:
                    logger.error(f"También falló al guardar los pesos: {weights_error}")
            
            # Guardar también los parámetros en formato JSON
            import json
            with open(f"{self.save_dir}/best_params_{self.timestamp}.json", 'w') as f:
                json.dump({
                    "best_params": best_params,
                    "best_value": best_value,
                    "datetime": self.timestamp
                }, f, indent=4)
                
            logger.info(f"Optimización completada. Mejor valor: {best_value}")
            logger.info(f"Mejores parámetros: {best_params}")
            
            return {
                "best_params": best_params,
                "best_value": best_value,
                "model_path": f"{self.save_dir}/best_fnn_model_{self.timestamp}",
                "study_path": f"{self.save_dir}/study_{self.study_name}_{self.timestamp}.pkl"
            }
            
        except Exception as e:
            logger.error(f"Error en optimización: {e}")
            # Devolver un diccionario con valores compatibles con el tipo de retorno esperado
            return {
                "best_params": {},
                "best_value": float('inf'),
                "model_path": "",
                "study_path": ""
            }
    
    def optimize_ensemble_weights(self,
                                 models: List[Callable[[np.ndarray], np.ndarray]],
                                 model_names: List[str],
                                 X_val: np.ndarray,
                                 y_val: np.ndarray) -> Dict[str, float]:
        """
        Optimiza los pesos para un ensemble de modelos.
        
        Args:
            models: Lista de funciones de predicción de los modelos
            model_names: Nombres de los modelos
            X_val: Features de validación
            y_val: Targets de validación
            
        Returns:
            Diccionario con los pesos óptimos para cada modelo
        """
        def objective(trial) -> float:
            # Generar pesos aleatorios para cada modelo
            weights = [trial.suggest_float(f"weight_{name}", 0.0, 1.0) for name in model_names]
            
            # Normalizar pesos para que sumen 1
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Obtener predicciones de cada modelo
            predictions = []
            for model_func in models:
                pred = model_func(X_val)
                predictions.append(pred)
            
            # Calcular predicción ponderada
            weighted_pred = np.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred
            
            # Calcular error
            mse = np.mean((weighted_pred - y_val) ** 2)
            return float(mse)
        
        # Crear estudio
        storage_name = f"sqlite:///{self.save_dir}/ensemble_{self.study_name}_{self.timestamp}.db"
        study = optuna.create_study(
            study_name=f"ensemble_{self.study_name}_{self.timestamp}",
            direction="minimize",
            storage=storage_name
        )
        
        # Optimizar
        try:
            study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
            
            # Obtener los mejores pesos
            best_weights = [study.best_params[f"weight_{name}"] for name in model_names]
            best_weights = np.array(best_weights)
            best_weights = best_weights / np.sum(best_weights)  # Normalizar
            
            # Guardar resultados
            ensemble_weights = {}
            for i, name in enumerate(model_names):
                ensemble_weights[name] = float(best_weights[i])
            
            # Guardar en archivo
            import json
            with open(f"{self.save_dir}/ensemble_weights_{self.timestamp}.json", 'w') as f:
                json.dump({
                    "weights": ensemble_weights,
                    "best_value": float(study.best_value),
                    "datetime": self.timestamp
                }, f, indent=4)
            
            logger.info(f"Optimización de ensemble completada. Mejor MSE: {study.best_value}")
            logger.info(f"Mejores pesos: {ensemble_weights}")
            
            return ensemble_weights
            
        except Exception as e:
            logger.error(f"Error en optimización de ensemble: {e}")
            # Devolver un diccionario con valores flotantes en vez de string para cumplir con el tipo de retorno
            return {"error_value": 0.0}
    
    def _create_custom_fnn(self, params: Dict[str, Any]) -> FeedforwardNeuralNetwork:
        """
        Crea un modelo FNN con los parámetros especificados.
        
        Args:
            params: Diccionario de parámetros
            
        Returns:
            Instancia de FeedforwardNeuralNetwork
        """
        model = FeedforwardNeuralNetwork(
            input_dim=params['input_dim'],
            learning_rate=params['learning_rate']
        )
        
        # Crear modelo personalizado
        inputs = tf.keras.Input(shape=(params['input_dim'],))
        x = inputs
        
        # Añadir capas ocultas
        for _ in range(params['hidden_layers']):
            x = tf.keras.layers.Dense(
                params['neurons'],
                activation=params['activation'],
                kernel_regularizer=tf.keras.regularizers.l2(params['l2_regularization'])
            )(x)
            x = tf.keras.layers.Dropout(params['dropout_rate'])(x)
        
        # Capa de salida (2 valores: home_goals, away_goals)
        outputs = tf.keras.layers.Dense(2)(x)
        
        # Compilar modelo
        model.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model

def train_optimized_fnn(data_path: str = "data/preprocessed_features.csv") -> Optional[str]:
    """
    Entrena un modelo FNN optimizado con Optuna.
    
    Args:
        data_path: Ruta al CSV con datos preprocesados
        
    Returns:
        Ruta al modelo optimizado o None si hay error
    """
    try:
        # Cargar datos
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            logger.error(f"No se encontró el archivo {data_path}")
            return None
        
        # Separar features y targets
        X = df.drop(['home_goals', 'away_goals'], axis=1).values
        y = df[['home_goals', 'away_goals']].values
        
        # Escalar features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split train/val/test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
        
        # Iniciar optimización
        optimizer = OptunaOptimizer(n_trials=30, timeout=3600)  # 1 hora de timeout
        result = optimizer.optimize_fnn(X_train, y_train, X_val, y_val)
        
        # Guardar scaler
        joblib.dump(scaler, f"models/optuna/scaler_{optimizer.timestamp}.pkl")
        
        # Evaluar en test set
        try:
            # Crear una instancia y cargar el modelo manualmente
            best_model = FeedforwardNeuralNetwork(input_dim=X_test.shape[1])
            
            # Evitar asignar directamente a .model y usar try-except para manejar errores de tipo
            try:
                keras_model = tf.keras.models.load_model(result['model_path'])
                # Verificar que el modelo cargado sea compatible
                best_model.model = cast(tf.keras.Model, keras_model)
            except Exception as model_error:
                logger.error(f"Error cargando modelo: {model_error}")
                return None
            
            test_loss = best_model.model.evaluate(X_test, y_test, verbose=cast(str, "0"))
        except Exception as load_error:
            logger.error(f"Error cargando o evaluando modelo: {load_error}")
            test_loss = None
        
        logger.info(f"Test loss: {test_loss}")
        logger.info(f"Modelo optimizado guardado en: {result['model_path']}")
        
        return result['model_path']
        
    except Exception as e:
        logger.error(f"Error en entrenamiento optimizado: {e}")
        return None

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Verificar instalación de Optuna
    try:
        import optuna
        logger.info("Optuna instalado correctamente")
    except ImportError:
        logger.error("Optuna no está instalado. Instale con: pip install optuna")
        exit(1)
    
    # Entrenar modelo optimizado
    train_optimized_fnn()
