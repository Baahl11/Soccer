"""
Entrenamiento de modelos de conjunto (RF y XGBoost) para predicción de córners.

Este script toma los datos recopilados de la API de fútbol y entrena
modelos de Random Forest y XGBoost para la predicción de córners,
los cuales serán utilizados por el VotingEnsembleCornersModel.
"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import xgboost as xgb

# Configuración
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Asegurar que los directorios existen
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    Path(dir_path).mkdir(parents=True, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(RESULTS_DIR, 'corner_model_training.log')
)
logger = logging.getLogger(__name__)

def load_training_data() -> pd.DataFrame:
    """
    Carga los datos de entrenamiento más recientes.
    
    Returns:
        DataFrame con los datos de entrenamiento
    """
    # Encontrar el archivo de datos más reciente
    files = glob.glob(os.path.join(DATA_DIR, 'corners_training_data_*.csv'))
    if not files:
        raise FileNotFoundError(f"No training data files found in {DATA_DIR}")
    
    latest_file = max(files, key=os.path.getctime)
    logger.info(f"Loading training data from {latest_file}")
    
    # Cargar el archivo
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} training examples")
    
    return df

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Prepara las características para el entrenamiento del modelo.
    
    Args:
        df: DataFrame con datos brutos
        
    Returns:
        Tuple de (X, y, feature_names)
    """
    # Lista de características a utilizar
    feature_cols = [
        'home_avg_corners_for', 'home_avg_corners_against',
        'away_avg_corners_for', 'away_avg_corners_against',
        'home_form_score', 'away_form_score',
    ]
    
    # Añadir estadísticas adicionales si están disponibles
    potential_features = [
        'possession', 'shots_on_goal', 'shots_off_goal', 'total_shots',
        'blocked_shots', 'shots_inside_box', 'shots_outside_box',
        'fouls', 'dangerous_attacks', 'goal_attempts', 'crosses',
        'corners_percentage', 'ball_recovery', 'aerials_won',
        'passes_to_final_third', 'offensive_duels_won', 'tackles',
        'interceptions', 'defensive_duels_won', 'pressing_duels',
        'formation', 'field_width', 'field_length', 'weather',
        'head_to_head_avg_corners', 'referee_avg_corners'
    ]
    
    for feat in potential_features:
        home_feat = f'home_{feat}'
        away_feat = f'away_{feat}'
        if home_feat in df.columns and away_feat in df.columns:
            feature_cols.append(home_feat)
            feature_cols.append(away_feat)
            
    # Add contextual features if available
    contextual_features = [
        'minutes_played', 'stadium_size', 'pitch_condition',
        'attendance_percentage', 'time_of_day', 'days_since_last_match',
        'is_derby', 'season_stage', 'competition_stage'
    ]
    
    for feat in contextual_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    # Añadir ID de liga como característica categórica
    if 'league_id' in df.columns:
        feature_cols.append('league_id')
    
    # Verificar que las columnas existen
    valid_cols = [col for col in feature_cols if col in df.columns]
    if len(valid_cols) < 6:  # Mínimo de características necesarias
        logger.warning(f"Insufficient features available. Found only {valid_cols}")
        raise ValueError("Insufficient features for training")
    
    # Crear matrices de característica y objetivo
    X = df[valid_cols].copy()
    
    # Manejar valores faltantes
    X.fillna(X.mean(), inplace=True)
    
    # Objetivos: total, home y away corners
    y = df[['total_corners', 'home_corners', 'away_corners']].copy()
    
    logger.info(f"Prepared features: {valid_cols}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    return X, y, valid_cols

def train_random_forest_model(X: pd.DataFrame, y: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """
    Entrena un modelo RandomForest para la predicción de córners.
    
    Args:
        X: Matriz de características
        y: Valores objetivo
        feature_names: Nombres de las características
        
    Returns:
        Diccionario con el modelo entrenado y métricas
    """
    logger.info("Training Random Forest model with cross-validation...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Definir parámetros para búsqueda
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Crear modelo base
    base_rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Realizar búsqueda de hiperparámetros con validación cruzada
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Starting grid search with cross-validation...")
    grid_search.fit(X_train, y_train['total_corners'])
    
    # Obtener mejor modelo
    best_rf = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluar modelo
    cv_scores = cross_val_score(
        best_rf, X_train, y_train['total_corners'],
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    logger.info(f"Cross-validation RMSE: {cv_rmse:.3f}")
    
    # Evaluar en conjunto de prueba
    y_pred = best_rf.predict(X_test)
    mse = mean_squared_error(y_test['total_corners'], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test['total_corners'], y_pred)
    r2 = r2_score(y_test['total_corners'], y_pred)
    
    logger.info(f"Random Forest metrics - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    
    # Calcular importancia de características usando SHAP values
    from shap import TreeExplainer
    explainer = TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)
    
    # Importancia de características promedio absoluto SHAP
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(0)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    logger.info(f"Top 5 features (SHAP): {feature_importance.head(5).to_dict()}")
    
    # Generar visualización de importancia de características
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Random Forest - Feature Importance (SHAP values)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'rf_feature_importance.png'))
    
    return {
        'model': best_rf,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse
        },
        'feature_importance': feature_importance,
        'best_params': grid_search.best_params_
    }

def train_xgboost_model(X: pd.DataFrame, y: pd.DataFrame, feature_names: List[str]) -> Dict[str, Any]:
    """
    Entrena el modelo XGBoost para predicción de córners.
    
    Args:
        X: Features de entrenamiento
        y: Variable objetivo
        feature_names: Nombres de las características
        
    Returns:
        Diccionario con el modelo entrenado y sus métricas
    """
    logger.info("Training XGBoost model with cross-validation...")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Definir parámetros para búsqueda
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1]
    }
    
    # Crear modelo base
    base_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        tree_method='auto'
    )
    
    # Realizar búsqueda de hiperparámetros con validación cruzada
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_xgb,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    logger.info("Starting grid search with cross-validation...")
    grid_search.fit(X_train, y_train['total_corners'])
    
    # Obtener mejor modelo
    best_xgb = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluar modelo con validación cruzada
    cv_scores = cross_val_score(
        best_xgb, X_train, y_train['total_corners'],
        cv=5, scoring='neg_mean_squared_error'
    )
    cv_rmse = np.sqrt(-cv_scores.mean())
    logger.info(f"Cross-validation RMSE: {cv_rmse:.3f}")
    
    # Evaluar en conjunto de prueba
    y_pred = best_xgb.predict(X_test)
    mse = mean_squared_error(y_test['total_corners'], y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test['total_corners'], y_pred)
    r2 = r2_score(y_test['total_corners'], y_pred)
    
    logger.info(f"XGBoost metrics - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
    
    # Calcular importancia de características usando SHAP values
    from shap import TreeExplainer
    explainer = TreeExplainer(best_xgb)
    shap_values = explainer.shap_values(X_test)
    
    # Importancia de características promedio absoluto SHAP
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(shap_values).mean(0)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    logger.info(f"Top 5 features (SHAP): {feature_importance.head(5).to_dict()}")
    
    # Generar visualización de importancia de características
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('XGBoost - Feature Importance (SHAP values)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'xgb_feature_importance.png'))
    
    return {
        'model': best_xgb,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse
        },
        'feature_importance': feature_importance,
        'best_params': grid_search.best_params_
    }

def compare_models(rf_results: Dict[str, Any], xgb_results: Dict[str, Any]) -> None:
    """
    Compara los modelos entrenados y genera visualizaciones.
    
    Args:
        rf_results: Resultados del modelo Random Forest
        xgb_results: Resultados del modelo XGBoost
    """
    # Comparar métricas
    metrics_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost'],
        'RMSE': [rf_results['metrics']['rmse'], xgb_results['metrics']['rmse']],
        'MAE': [rf_results['metrics']['mae'], xgb_results['metrics']['mae']],
        'R²': [rf_results['metrics']['r2'], xgb_results['metrics']['r2']]
    })
    
    logger.info(f"Model comparison:\n{metrics_df.to_string()}")
    
    # Visualizar comparación
    plt.figure(figsize=(12, 8))
    
    # RMSE
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='RMSE', data=metrics_df)
    plt.title('RMSE Comparison')
    
    # MAE
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='MAE', data=metrics_df)
    plt.title('MAE Comparison')
    
    # R²
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='R²', data=metrics_df)
    plt.title('R² Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
    
    # Comparar Feature Importance
    plt.figure(figsize=(12, 10))
    
    # Combinar importancia de características
    rf_importance = rf_results['feature_importance'].head(10).copy()
    rf_importance['Model'] = 'Random Forest'
    
    xgb_importance = xgb_results['feature_importance'].head(10).copy()
    xgb_importance['Model'] = 'XGBoost'
    
    combined_importance = pd.concat([rf_importance, xgb_importance])
    
    # Graficar
    sns.barplot(x='Importance', y='Feature', hue='Model', data=combined_importance)
    plt.title('Feature Importance Comparison')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance_comparison.png'))

def save_models(rf_model, xgb_model) -> None:
    """
    Guarda los modelos entrenados en disco.
    
    Args:
        rf_model: Modelo Random Forest entrenado
        xgb_model: Modelo XGBoost entrenado
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Guardar con timestamp y como modelos activos
    joblib.dump(rf_model, os.path.join(MODELS_DIR, f'random_forest_corners_{timestamp}.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, f'xgboost_corners_{timestamp}.pkl'))
    
    # Guardar como modelos activos (para uso por VotingEnsembleCornersModel)
    joblib.dump(rf_model, os.path.join(MODELS_DIR, 'random_forest_corners.pkl'))
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgboost_corners.pkl'))
    
    logger.info(f"Saved models to {MODELS_DIR}")

def main():
    """
    Función principal para entrenar los modelos.
    """
    try:
        logger.info("Starting corner prediction model training...")
        
        # Cargar datos
        df = load_training_data()
        
        # Preparar características
        X, y, feature_names = prepare_features(df)
        
        # Entrenar modelos
        rf_results = train_random_forest_model(X, y, feature_names)
        xgb_results = train_xgboost_model(X, y, feature_names)
        
        # Comparar modelos
        compare_models(rf_results, xgb_results)
        
        # Guardar modelos
        save_models(rf_results['model'], xgb_results['model'])
        
        logger.info("Model training completed successfully")
        
        # Imprimir resumen
        print("\n===== Corner Prediction Model Training Summary =====")
        print(f"Data points used: {len(X)}")
        print(f"Features used: {len(feature_names)}")
        print("\nRandom Forest Results:")
        print(f"  RMSE: {rf_results['metrics']['rmse']:.3f}")
        print(f"  MAE: {rf_results['metrics']['mae']:.3f}")
        print(f"  R²: {rf_results['metrics']['r2']:.3f}")
        print("\nXGBoost Results:")
        print(f"  RMSE: {xgb_results['metrics']['rmse']:.3f}")
        print(f"  MAE: {xgb_results['metrics']['mae']:.3f}")
        print(f"  R²: {xgb_results['metrics']['r2']:.3f}")
        print("\nModels saved to:", MODELS_DIR)
        print(f"Results and visualizations saved to: {RESULTS_DIR}")
        print("==========================================\n")
        
    except Exception as e:
        logger.exception(f"Error in model training: {e}")
        print(f"An error occurred during model training: {e}")

if __name__ == "__main__":
    main()
