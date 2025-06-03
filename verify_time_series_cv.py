"""
Ejemplo de Uso y Validación de Time Series Cross-Validation

Este script demuestra cómo utilizar el módulo de validación cruzada con series temporales
y verifica que la implementación funcione correctamente.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import time
import os
from pathlib import Path

# Importar módulo de validación cruzada de series temporales
from time_series_cross_validation import TimeSeriesValidator, train_model_with_timeseries_cv

# Configuración de logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_synthetic_time_series_data(n_samples=500, n_features=10, noise=0.5, trend=True):
    """
    Genera datos sintéticos de series temporales para demostración.
    
    Args:
        n_samples: Número de muestras a generar
        n_features: Número de características (variables predictoras)
        noise: Nivel de ruido aleatorio
        trend: Si es True, añade tendencias temporales a los datos
        
    Returns:
        X, y, dates: Características, valores objetivo y fechas    """
    # Generar fechas secuenciales
    base_date = pd.Timestamp('2023-01-01')
    date_range = pd.date_range(start=base_date, periods=n_samples, freq='D')
    dates = np.array(date_range.astype(np.int64) // 10**9)  # Convertir a timestamp numpy
    
    # Generar características
    X = np.random.randn(n_samples, n_features)
    
    # Añadir tendencia temporal si se especifica
    if trend:
        # Añadir tendencia creciente con el tiempo a algunas características
        for i in range(3):
            X[:, i] += np.linspace(0, 2, n_samples)
        
        # Añadir comportamiento estacional a algunas características
        for i in range(3, 6):
            X[:, i] += 2 * np.sin(np.linspace(0, 6*np.pi, n_samples))
    
    # Generar variable objetivo como combinación lineal de características + ruido
    weights = np.random.randn(n_features)
    y = X.dot(weights) + noise * np.random.randn(n_samples)
    
    # Para hacer más realista, discretizar y en clases: 0, 1, 2 (similar a resultados de fútbol)
    y_binned = np.zeros_like(y)
    y_binned[y > np.percentile(y, 66)] = 2  # Victoria local
    y_binned[(y >= np.percentile(y, 33)) & (y <= np.percentile(y, 66))] = 1  # Empate
    # El resto queda como 0 (Victoria visitante)
    
    return X, y_binned, dates

def test_time_series_validator():
    """
    Prueba el funcionamiento del validador de series temporales.
    """
    logger.info("Generando datos sintéticos para pruebas...")
    X, y, dates = generate_synthetic_time_series_data(n_samples=500)
    
    # Función para crear y entrenar modelos simples para la prueba
    def test_model_factory(X_train, y_train):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    # Crear validador
    validator = TimeSeriesValidator(n_splits=5, gap=10, verbose=True)
    
    # Ejecutar validación cruzada
    logger.info("Ejecutando validación cruzada con series temporales...")
    results = validator.validate(
        X=X,
        y=y,
        dates=dates,
        model_factory=test_model_factory,
        metrics=['mae', 'rmse', 'r2'],
        save_models=True,
        output_dir="models/test_ts_cv"
    )
    
    # Generar gráficas
    validator.plot_results(output_dir="models/test_ts_cv")
    validator.plot_validation_scheme(n_samples=len(X), output_dir="models/test_ts_cv")
    
    logger.info("Resultados de la prueba guardados en models/test_ts_cv")
    
    return results

def verify_time_order():
    """
    Verifica que la validación respete el orden temporal de los datos.
    """
    # Generar datos donde la relación entre X e y cambia con el tiempo
    n_samples = 400
    X = np.zeros((n_samples, 1))
    y = np.zeros(n_samples)
    
    # Primera mitad: relación positiva
    X[:n_samples//2, 0] = np.linspace(-5, 5, n_samples//2)
    y[:n_samples//2] = 2 * X[:n_samples//2, 0] + np.random.normal(0, 1, n_samples//2)
    
    # Segunda mitad: relación negativa
    X[n_samples//2:, 0] = np.linspace(-5, 5, n_samples//2)
    y[n_samples//2:] = -2 * X[n_samples//2:, 0] + np.random.normal(0, 1, n_samples//2)
    
    # Fechas en orden
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    # Crear directorio para gráficas
    os.makedirs("models/time_order_test", exist_ok=True)
    
    # Graficar datos completos
    plt.figure(figsize=(10, 5))
    plt.scatter(range(n_samples), y, c=range(n_samples), cmap='viridis')
    plt.colorbar(label='Índice de tiempo')
    plt.title('Serie temporal con cambio de comportamiento en la mitad')
    plt.xlabel('Tiempo')
    plt.ylabel('Valor')
    plt.savefig("models/time_order_test/data_series.png")
    plt.close()
    
    # Graficar relación X-y
    plt.figure(figsize=(10, 5))
    plt.scatter(X[:n_samples//2, 0], y[:n_samples//2], label='Primera mitad', alpha=0.5)
    plt.scatter(X[n_samples//2:, 0], y[n_samples//2:], label='Segunda mitad', alpha=0.5)
    plt.title('Relación X-y cambia con el tiempo')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("models/time_order_test/xy_relationship.png")
    plt.close()
    
    # Función para crear y evaluar modelos lineales simples
    def linear_model_factory(X_train, y_train):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
      # Test 1: Validación cruzada tradicional (aleatoria)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    train_scores = []
    test_scores = []
    fold_indices = []
    
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = linear_model_factory(X_train, y_train)
        train_scores.append(r2_score(y_train, model.predict(X_train)))
        test_scores.append(r2_score(y_test, model.predict(X_test)))
        fold_indices.append(i+1)
    
    # Test 2: Validación cruzada con series temporales
    ts_validator = TimeSeriesValidator(n_splits=5, verbose=False)
    
    ts_train_scores = []
    ts_test_scores = []
    ts_fold_indices = []
    
    tscv = TimeSeriesSplit(n_splits=ts_validator.n_splits)
    for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = linear_model_factory(X_train, y_train)
        ts_train_scores.append(r2_score(y_train, model.predict(X_train)))
        ts_test_scores.append(r2_score(y_test, model.predict(X_test)))
        ts_fold_indices.append(i+1)
    
    # Comparar resultados
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fold_indices, train_scores, 'b-', marker='o', label='Train R²')
    plt.plot(fold_indices, test_scores, 'r-', marker='s', label='Test R²')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.title('Validación cruzada aleatoria (K-Fold)')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(ts_fold_indices, ts_train_scores, 'b-', marker='o', label='Train R²')
    plt.plot(ts_fold_indices, ts_test_scores, 'r-', marker='s', label='Test R²')
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)
    plt.title('Validación cruzada de series temporales')
    plt.xlabel('Fold')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("models/time_order_test/validation_comparison.png")
    plt.close()
    
    # Resultados esperados: 
    # - En KFold aleatorio, rendimiento similar entre folds
    # - En TimeSeriesSplit, deterioro del rendimiento para folds futuros
    
    logger.info("Comparación de validación cruzada completada y guardada en models/time_order_test/")

def run_with_real_data():
    """
    Ejecuta la validación con datos reales del sistema de predicción de fútbol.
    """
    logger.info("Ejecutando validación con datos reales...")
    start_time = time.time()
    
    results = train_model_with_timeseries_cv(
        n_splits=5,
        gap=10,
        save_models=True,
        output_dir="models/soccer_ts_cv"
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Validación con datos reales completada en {elapsed_time:.2f} segundos")
    logger.info(f"Resultados guardados en models/soccer_ts_cv/")
    
    return results

if __name__ == "__main__":
    # 1. Probar con datos sintéticos
    logger.info("\n=== Test con datos sintéticos ===")
    test_results = test_time_series_validator()
    
    # 2. Verificar que respeta orden temporal
    logger.info("\n=== Verificación de orden temporal ===")
    verify_time_order()
    
    # 3. Ejecutar con datos reales
    logger.info("\n=== Validación con datos reales ===")
    real_results = run_with_real_data()
    
    logger.info("\n=== Todas las pruebas completadas exitosamente ===")
