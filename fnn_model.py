import tensorflow as tf
import keras
from keras import Model, Sequential, utils
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input, Add, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.models import load_model
from keras.regularizers import l2
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Tuple, Union, ClassVar
import os
import math
import requests
import json
from datetime import datetime

logger = logging.getLogger(__name__)

def custom_poisson_loss(y_true, y_pred):
    """
    Custom Poisson loss function for soccer predictions.
    Calculates loss based on predicted vs actual expected goals.
    """
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred + 1e-7))

def poisson_loss_metric(y_true, y_pred):
    """
    Metric function for tracking Poisson loss during training.
    """
    return custom_poisson_loss(y_true, y_pred)

class FeedforwardNeuralNetwork:
    """Modelo de red neuronal feedforward para predicción de resultados de fútbol."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32, 16],
                 dropout_rate: float = 0.3, learning_rate: float = 0.001,
                 l2_reg: float = 0.001, use_leaky_relu: bool = False,
                 alpha_leaky: float = 0.1):
        """
        Inicializa el modelo de red neuronal feedforward.
        
        Args:
            input_dim: Dimensión de entrada (número de características)
            hidden_dims: Lista con número de neuronas en cada capa oculta
            learning_rate: Tasa de aprendizaje para el optimizador
            dropout_rate: Tasa de dropout para regularización
            l2_reg: Factor de regularización L2
            use_leaky_relu: Si True, usa LeakyReLU en lugar de ReLU
            alpha_leaky: Parámetro alpha para LeakyReLU
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.use_leaky_relu = use_leaky_relu
        self.alpha_leaky = alpha_leaky
        
        # Construir modelo
        self.model = self.build_model()
          # Compilar modelo con optimizer y custom loss functions
        self.compile_model()

    def build_model(self) -> Model:
        """
        Construye la arquitectura del modelo.
        
        Returns:
            Modelo compilado de Keras
        """
        # Capa de entrada
        inputs = Input(shape=(self.input_dim,))
        x = inputs

        # Capas ocultas
        for dim in self.hidden_dims:
            x = Dense(dim, kernel_regularizer=l2(self.l2_reg))(x)
            if self.use_leaky_relu:
                x = LeakyReLU(alpha=self.alpha_leaky)(x)
            else:
                x = Activation('relu')(x)
            x = Dropout(self.dropout_rate)(x)

        # Capa de salida (2 neuronas para home_xg y away_xg)
        outputs = Dense(2, activation='linear')(x)
          # Crear y devolver modelo
        model = Model(inputs, outputs)
        return model
        
    def compile_model(self, custom_objects: Optional[dict] = None) -> bool:
        """
        Compila el modelo con los parámetros especificados.
        Args:
            custom_objects: Objetos personalizados necesarios para la compilación
        Returns:
            True si la compilación fue exitosa, False en caso contrario
        """
        try:
            # Define custom objects if not provided
            if custom_objects is None:
                custom_objects = {
                    'custom_poisson_loss': custom_poisson_loss,
                    'poisson_loss_metric': poisson_loss_metric
                }
            
            with utils.custom_object_scope(custom_objects):
                self.model.compile(
                    optimizer='adam',
                    loss=custom_poisson_loss,
                    metrics=[poisson_loss_metric]
                )
                
            logger.info("Modelo compilado correctamente")
            return True
        except Exception as e:
            logger.error(f"Error compilando modelo: {e}")
            return False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             batch_size: int = 32, epochs: int = 100, 
             early_stop_patience: int = 10, reduce_lr_patience: int = 5,
             model_path: str = 'models/fnn_model.h5',
             tensorboard_log_dir: Optional[str] = 'logs/fit',
             custom_objects: Optional[dict] = None) -> Dict[str, Any]:
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Etiquetas de entrenamiento
            validation_data: Tupla de (X_val, y_val) para validación
            batch_size: Tamaño del batch para entrenamiento
            epochs: Número máximo de épocas de entrenamiento
            early_stop_patience: Paciencia para early stopping
            reduce_lr_patience: Paciencia para reducción de learning rate
            model_path: Ruta para guardar el mejor modelo
            tensorboard_log_dir: Directorio para logs de Tensorboard
            custom_objects: Objetos personalizados necesarios para la compilación
            
        Returns:
            Diccionario con el historial de entrenamiento
        """
        try:
            # Configurar early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=early_stop_patience,
                restore_best_weights=True,
                verbose=1
            )
            
            # Configurar model checkpoint
            checkpoint = ModelCheckpoint(
                model_path,
                monitor='val_loss' if validation_data else 'loss',
                save_best_only=True,
                verbose=1
            )
            
            # Configurar reduce LR on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
            
            # Configurar Tensorboard callback si se proporciona directorio
            callbacks = [early_stopping, checkpoint, reduce_lr]
            if tensorboard_log_dir:
                tensorboard_callback = TensorBoard(
                    log_dir=tensorboard_log_dir,
                    histogram_freq=1
                )
                callbacks.append(tensorboard_callback)
              # Recompilar el modelo con optimizador y loss functions configurados
            self.compile_model(custom_objects)
            
            # Entrenar modelo
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose="auto"
            )
            
            logger.info("Entrenamiento completado exitosamente")
            return history.history
            
        except Exception as e:
            logger.error(f"Error durante entrenamiento: {e}")
            return {"error": str(e)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo.
        
        Args:
            X: Datos de entrada (features)
            
        Returns:
            Array de predicciones [home_xg, away_xg] para cada muestra
        """
        try:
            return self.model.predict(X, verbose="auto")
        except Exception as e:
            logger.error(f"Error realizando predicciones: {e}")
            return np.array([])
