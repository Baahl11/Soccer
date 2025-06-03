"""
Time Series Cross Validation

Este módulo implementa validación cruzada con series temporales para evaluar
correctamente el rendimiento de los modelos de predicción de fútbol.

La validación cruzada tradicional selecciona datos aleatoriamente para entrenamiento
y prueba, lo cual no es apropiado para datos temporales porque puede causar
"data leakage" (filtración de datos del futuro al entrenamiento).

Esta implementación usa TimeSeriesSplit de scikit-learn para dividir los datos
respetando su orden cronológico.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from pathlib import Path
import json
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
from datetime import datetime
import os
from math import sqrt

# Configuración de logging
logger = logging.getLogger(__name__)

class TimeSeriesValidator:
    """
    Clase para validación cruzada con series temporales.
    
    Permite evaluar modelos de predicción respetando el orden temporal
    de los datos, evitando filtración de información futura.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        gap: int = 0,
        test_size: Optional[int] = None,
        max_train_size: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Inicializa el validador de series temporales.
        
        Args:
            n_splits: Número de divisiones (folds) para la validación cruzada
            gap: Número de muestras entre los conjuntos de entrenamiento y prueba
            test_size: Tamaño del conjunto de prueba en cada división
            max_train_size: Tamaño máximo del conjunto de entrenamiento
            verbose: Si es True, muestra información detallada durante la validación
        """
        self.n_splits = n_splits
        self.gap = gap
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.verbose = verbose
        self.cv_results_ = None
        
    def validate(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        dates: np.ndarray,
        model_factory: Callable,
        scaler: Optional[Any] = None,
        metrics: Optional[List[str]] = None,
        save_models: bool = False,
        output_dir: str = "models/cv_results"
    ) -> Dict[str, List[float]]:
        """
        Realiza validación cruzada con series temporales.
        
        Args:
            X: Matriz de características
            y: Vector de etiquetas/valores objetivo
            dates: Array de fechas correspondientes a cada muestra
            model_factory: Función que crea y devuelve un modelo entrenado
            scaler: Opcional, un scaler para normalizar datos (si es None, se usa StandardScaler)
            metrics: Lista de métricas a calcular ('mae', 'rmse', 'r2')
            save_models: Si es True, guarda los modelos entrenados en cada fold
            output_dir: Directorio donde guardar resultados y modelos
            
        Returns:
            Diccionario con resultados de validación para cada métrica
        """
        # Configurar métricas
        if metrics is None:
            metrics = ['mae', 'rmse', 'r2']
            
        # Ordenar datos por fecha si es necesario
        if dates is not None:
            sort_idx = np.argsort(dates)
            X = X[sort_idx]
            y = y[sort_idx]
            dates = dates[sort_idx]
        
        # Configurar TimeSeriesSplit
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            test_size=self.test_size,
            max_train_size=self.max_train_size
        )
        
        # Preparar resultados
        results = {metric: [] for metric in metrics}
        results['train_size'] = []
        results['test_size'] = []
        results['train_start'] = []
        results['train_end'] = []
        results['test_start'] = []
        results['test_end'] = []
        
        # Crear directorio para resultados
        if save_models:
            os.makedirs(output_dir, exist_ok=True)
        
        # Realizar validación cruzada
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            fold_name = f"Fold {fold+1}/{self.n_splits}"
            
            # Obtener conjuntos de entrenamiento y prueba
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Guardar fechas para análisis
            if dates is not None:
                train_dates = dates[train_idx]
                test_dates = dates[test_idx]
                results['train_start'].append(train_dates.min())
                results['train_end'].append(train_dates.max())
                results['test_start'].append(test_dates.min())
                results['test_end'].append(test_dates.max())
            
            # Registrar tamaños de conjuntos
            results['train_size'].append(len(X_train))
            results['test_size'].append(len(X_test))
            
            # Escalar datos si es necesario
            if scaler is None:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = scaler.transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            if self.verbose:
                logger.info(f"Entrenando modelo para {fold_name}...")
                logger.info(f"  - Datos de entrenamiento: {X_train.shape[0]} muestras")
                logger.info(f"  - Datos de prueba: {X_test.shape[0]} muestras")
                
                if dates is not None:
                    logger.info(f"  - Período de entrenamiento: {train_dates.min()} a {train_dates.max()}")
                    logger.info(f"  - Período de prueba: {test_dates.min()} a {test_dates.max()}")
            
            # Crear y entrenar modelo usando la factory function
            model = model_factory(X_train_scaled, y_train)
            
            # Realizar predicciones
            y_pred = model.predict(X_test_scaled)
            
            # Calcular métricas
            if 'mae' in metrics:
                mae = mean_absolute_error(y_test, y_pred)
                results['mae'].append(mae)
                
            if 'rmse' in metrics:
                rmse = sqrt(mean_squared_error(y_test, y_pred))
                results['rmse'].append(rmse)
                
            if 'r2' in metrics:
                r2 = r2_score(y_test, y_pred)
                results['r2'].append(r2)
                
            # Mostrar resultados
            if self.verbose:
                logger.info(f"Resultados para {fold_name}:")
                for metric in metrics:
                    logger.info(f"  - {metric.upper()}: {results[metric][-1]:.4f}")
            
            # Guardar modelo y scaler
            if save_models:
                model_path = os.path.join(output_dir, f"model_fold_{fold+1}.pkl")
                scaler_path = os.path.join(output_dir, f"scaler_fold_{fold+1}.pkl")
                
                joblib.dump(model, model_path)
                joblib.dump(scaler, scaler_path)
                
                # Guardar métricas del fold
                fold_results = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'test_size': len(X_test)                }
                
                if dates is not None:
                    fold_results.update({
                        'train_start': train_dates.min(),
                        'train_end': train_dates.max(),
                        'test_start': test_dates.min(),
                        'test_end': test_dates.max()
                    })
                
                for metric in metrics:
                    fold_results[metric] = results[metric][-1]  # Ya es un float
                
                results_path = os.path.join(output_dir, f"results_fold_{fold+1}.json")
                with open(results_path, 'w') as f:
                    json.dump(fold_results, f, indent=2, default=str)
        
        # Calcular promedios
        for metric in metrics:
            metric_avg = np.mean(results[metric])
            metric_std = np.std(results[metric])
            logger.info(f"Promedio {metric.upper()}: {metric_avg:.4f} ± {metric_std:.4f}")
        
        # Guardar resultados consolidados
        if save_models:
            summary = {
                'n_splits': self.n_splits,
                'gap': self.gap,
                'total_samples': len(X),
                'timestamp': datetime.now().isoformat()
            }
            
            for metric in metrics:
                summary[f'{metric}_avg'] = float(np.mean(results[metric]))
                summary[f'{metric}_std'] = float(np.std(results[metric]))
            
            summary_path = os.path.join(output_dir, "cv_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        # Guardar resultados en la instancia
        self.cv_results_ = results
        
        return results
    
    def plot_results(
        self,
        output_dir: str = "models/cv_results",
        filename: str = "cv_metrics.png"
    ) -> None:
        """
        Genera gráficas de los resultados de la validación cruzada.
        
        Args:
            output_dir: Directorio donde guardar las gráficas
            filename: Nombre del archivo para guardar la gráfica
        """
        if self.cv_results_ is None:
            raise ValueError("No hay resultados de validación para graficar. "
                            "Primero ejecute el método validate().")
        
        metrics = [k for k in self.cv_results_.keys() if k not in [
            'train_size', 'test_size', 'train_start', 'train_end', 'test_start', 'test_end'
        ]]
        
        n_metrics = len(metrics)
        
        # Crear figura        plt.figure(figsize=(5 * n_metrics, 4))
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, n_metrics, i)
            
            # Graficar métrica por fold
            values = self.cv_results_[metric]
            folds = range(1, len(values) + 1)
            plt.plot(folds, values, 'o-', label=f'{metric.upper()}')
            plt.axhline(y=float(np.mean(values)), color='r', linestyle='--', label=f'Promedio: {np.mean(values):.4f}')
            
            plt.title(f'{metric.upper()} por Fold')
            plt.xlabel('Fold')
            plt.ylabel(metric.upper())
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        # Guardar gráfica
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()
        
    def plot_validation_scheme(
        self,
        n_samples: int,
        output_dir: str = "models/cv_results",
        filename: str = "cv_scheme.png"
    ) -> None:
        """
        Genera una visualización del esquema de validación cruzada.
        
        Args:
            n_samples: Número total de muestras en el dataset
            output_dir: Directorio donde guardar las gráficas
            filename: Nombre del archivo para guardar la gráfica
        """
        tscv = TimeSeriesSplit(
            n_splits=self.n_splits,
            gap=self.gap,
            test_size=self.test_size,
            max_train_size=self.max_train_size
        )
        
        # Crear figura
        plt.figure(figsize=(10, self.n_splits * 0.5 + 1))
        
        for i, (train_idx, test_idx) in enumerate(tscv.split(np.zeros(n_samples))):
            # Calcular rangos para entrenamientos y pruebas
            train_start = min(train_idx)
            train_end = max(train_idx)
            test_start = min(test_idx)
            test_end = max(test_idx)
            
            # Graficar conjunto de entrenamiento
            plt.plot([train_start, train_end], [i, i], 'b-', linewidth=10, alpha=0.4, label='Train' if i == 0 else "")
            
            # Graficar conjunto de prueba
            plt.plot([test_start, test_end], [i, i], 'r-', linewidth=10, alpha=0.4, label='Test' if i == 0 else "")
            
            # Añadir separador
            if self.gap > 0:
                plt.axvspan(train_end + 0.5, test_start - 0.5, alpha=0.2, color='gray')
            
            # Añadir etiquetas
            plt.text(train_start + 0.1 * (train_end - train_start), i, f"Train {len(train_idx)}", 
                     ha='left', va='center', color='darkblue')
            plt.text(test_start + 0.1 * (test_end - test_start), i, f"Test {len(test_idx)}",
                     ha='left', va='center', color='darkred')
        
        plt.yticks(range(self.n_splits), [f'Fold {i+1}' for i in range(self.n_splits)])
        plt.xlabel('Índice de muestra')
        plt.title('Esquema de validación cruzada con series temporales')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        plt.tight_layout()
        
        # Guardar gráfica
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

def get_train_data_with_dates():
    """
    Obtiene datos de entrenamiento con fechas incluidas.
    Esta función extiende get_training_data para incluir fechas para ordenación temporal.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: X, y, dates
    """
    from train_model import get_training_data
    import pandas as pd
      # Obtener datos completos (no procesados para preservar fechas)
    api = __import__('data').FootballAPI()
    all_data = []
    ligas_ids = [39, 140, 135, 78, 61]  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
    
    # Definir temporadas específicas - últimos 3 años para garantizar que haya datos
    current_year = datetime.now().year
    temporadas = [current_year-2, current_year-1, current_year]
    
    logger.info(f"Usando temporadas específicas: {temporadas}")
    
    for liga_id in ligas_ids:
        for temporada in temporadas:
            logger.info(f"Obteniendo datos de liga {liga_id} temporada {temporada}...")
            try:
                liga_data = api.get_historical_data(liga_id, temporada)
                if liga_data is not None and not liga_data.empty:
                    logger.info(f"Obtenidos {len(liga_data)} partidos para liga {liga_id}, temporada {temporada}")
                    all_data.append(liga_data)
            except Exception as e:
                logger.warning(f"Error obteniendo datos para liga {liga_id} temporada {temporada}: {e}")
    
    if not all_data:
        # Como alternativa, intentar cargar datos desde archivo local para pruebas
        try:
            from pathlib import Path
            data_path = Path("data/sample_matches.csv")
            if data_path.exists():
                logger.info("Usando datos de muestra desde archivo local")
                sample_data = pd.read_csv(data_path)
                all_data = [sample_data]
            else:
                logger.error(f"No se encontró archivo de datos de muestra en {data_path}")
                raise ValueError("No se pudieron obtener datos históricos")
        except Exception as e:
            logger.error(f"Error al cargar datos de muestra: {e}")
            raise ValueError("No se pudieron obtener datos históricos")
    
    # Combinar datos y ordenar por fecha
    data = pd.concat(all_data, axis=0, ignore_index=True)
    
    # Convertir fechas a datetime
    data['date'] = pd.to_datetime(data['date'])
    
    # Ordenar por fecha
    data = data.sort_values('date')
    
    # Extraer fechas como array de timestamps numéricos
    dates = np.array(data['date'].astype(np.int64) // 10**9)
    
    # Obtener X, y del preprocesamiento estándar
    from train_model import prepare_features, handle_missing_values
    
    data = handle_missing_values(data)
    X, y = prepare_features(data)
    
    return X, y, dates

def train_model_with_timeseries_cv(
    n_splits: int = 5, 
    gap: int = 0,
    save_models: bool = True,
    output_dir: str = "models/ts_cv_results"
):
    """
    Entrena modelos utilizando validación cruzada con series temporales.
    
    Args:
        n_splits: Número de divisiones para validación cruzada
        gap: Número de muestras a omitir entre entrenamiento y prueba
        save_models: Si es True, guarda modelos y resultados
        output_dir: Directorio para guardar resultados
        
    Returns:
        dict: Resultados de la validación cruzada
    """
    try:
        logger.info("Iniciando entrenamiento con validación cruzada de series temporales...")
        
        # Obtener datos con fechas
        X, y, dates = get_train_data_with_dates()
        logger.info(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Función para crear y entrenar modelo
        def model_factory(X_train, y_train):
            from fnn_model_fixed import FeedforwardNeuralNetworkFixed
            
            model = FeedforwardNeuralNetworkFixed(input_dim=X_train.shape[1])
            model.train(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                verbose=0  # Evitar excesivos logs durante CV
            )
            return model
        
        # Crear validador
        validator = TimeSeriesValidator(
            n_splits=n_splits,
            gap=gap,
            verbose=True
        )
        
        # Mostrar esquema de validación
        validator.plot_validation_scheme(
            n_samples=len(X),
            output_dir=output_dir
        )
        
        # Realizar validación cruzada
        results = validator.validate(
            X=X,
            y=y,
            dates=dates,
            model_factory=model_factory,
            metrics=['mae', 'rmse', 'r2'],
            save_models=save_models,
            output_dir=output_dir
        )
        
        # Generar gráficas de resultados
        validator.plot_results(output_dir=output_dir)
        
        logger.info("Validación cruzada con series temporales completada exitosamente")
        return results
        
    except Exception as e:
        logger.error(f"Error en validación cruzada con series temporales: {e}")
        raise

if __name__ == "__main__":
    # Configuración de logging para ejecución directa
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ejecutar validación cruzada
    train_model_with_timeseries_cv(n_splits=5, gap=10)
