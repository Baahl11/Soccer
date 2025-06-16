"""
Train models for the voting ensemble corner predictions.

This module implements training functions for the Random Forest and XGBoost
models that are part of the voting ensemble, based on academic research showing
that this combination achieves the highest accuracy for soccer predictions.

References:
- "Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques"
  Journal of Big Data, 2024
"""

import numpy as np
import pandas as pd
import logging
import json
import gc
import glob
from typing import Generator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from typing import Dict, Any, Tuple, List, Optional
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
from corner_data_collector import FormationDataCollector
from time_series_cross_validation import TimeSeriesValidator
import psutil

logger = logging.getLogger(__name__)

def load_corner_training_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load training data for corner predictions.
    
    Args:
        filepath: Optional path to data file. If None, uses latest sample data.
        
    Returns:
        DataFrame with training data
    """
    try:
        # If filepath provided, use it, otherwise find most recent sample data
        if filepath is None:
            files = glob.glob("data/corners_training_data_SAMPLE_*.csv")
            if files:
                filepath = max(files, key=os.path.getctime)
            else:
                filepath = "data/corner_training_data.csv"
            
        if not os.path.exists(filepath):
            logger.error(f"Training data file not found: {filepath}")
            return pd.DataFrame()  # Return empty dataframe
            
        data = pd.read_csv(filepath)
        logger.info(f"Loaded {len(data)} samples from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading corner training data: {e}")
        return pd.DataFrame()

def load_corner_json_data(league_ids: List[int] = [39, 140, 135, 78, 61]) -> pd.DataFrame:
    """
    Carga datos de corners de archivos JSON para múltiples ligas.
    
    Args:
        league_ids: Lista de IDs de ligas a incluir
        
    Returns:
        DataFrame combinado con todos los datos
    """
    all_data = []
    
    try:
        # Buscar todos los archivos JSON de corners
        import glob
        json_files = glob.glob("corner_data_*_2023_*.json")
        
        for file in json_files:
            try:
                # Extraer league_id del nombre del archivo
                league_id = int(file.split('_')[2])
                if league_id in league_ids:
                    with open(file, 'r', encoding='utf-8') as f:
                        import json
                        data = json.load(f)
                        
                        # Verificar que hay datos y que son válidos
                        if isinstance(data, list) and len(data) > 0:
                            df = pd.DataFrame(data)
                            
                            # Verificar que tenemos las columnas necesarias
                            required_cols = {'fixture_id', 'home_team_id', 'away_team_id', 
                                           'total_corners', 'home_formation', 'away_formation', 
                                           'tactical_indices'}
                            
                            if all(col in df.columns for col in required_cols):
                                # Filtrar filas con datos completos
                                df = df.dropna(subset=['total_corners', 'home_formation', 'away_formation'])
                                if len(df) > 0:
                                    all_data.append(df)
                                    logger.info(f"Loaded {len(df)} valid samples from {file}")
                            else:
                                logger.warning(f"Missing required columns in {file}")
                        else:
                            logger.warning(f"No valid data in {file}")
            except Exception as e:
                logger.warning(f"Error processing file {file}: {str(e)}")
                continue
        
        if not all_data:
            logger.error("No valid corner data found")
            return pd.DataFrame()
        
        # Combinar todos los DataFrames
        combined_data = pd.concat(all_data, ignore_index=True)
        logger.info(f"Combined dataset has {len(combined_data)} valid samples from {len(league_ids)} leagues")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error loading JSON corner data: {str(e)}")
        return pd.DataFrame()

def process_formation_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa y enriquece el dataset con features de formación.
    
    Args:
        data: DataFrame con datos base
        
    Returns:
        DataFrame enriquecido con features de formación
    """
    formation_collector = FormationDataCollector()
    
    # Convertir formaciones a índices numéricos
    data['home_formation_id'] = data['home_formation'].apply(
        lambda x: int(str(formation_collector.format_formation(x)).replace('-', ''))
    )
    data['away_formation_id'] = data['away_formation'].apply(
        lambda x: int(str(formation_collector.format_formation(x)).replace('-', ''))
    )
    
    # Calcular índices tácticos para local
    tactical_home = []
    for _, row in data.iterrows():
        features = formation_collector.get_formation_features(row['home_formation'])
        tactical_home.append(features)
    
    # Convertir a DataFrame y añadir prefijo 'home_'
    tactical_home_df = pd.DataFrame(tactical_home).add_prefix('home_')
    
    # Calcular índices tácticos para visitante
    tactical_away = []
    for _, row in data.iterrows():
        features = formation_collector.get_formation_features(row['away_formation'])
        tactical_away.append(features)
    
    # Convertir a DataFrame y añadir prefijo 'away_'
    tactical_away_df = pd.DataFrame(tactical_away).add_prefix('away_')
    
    # Combinar features
    data = pd.concat([data, tactical_home_df, tactical_away_df], axis=1)
    
    # Calcular ventaja táctica
    data['formation_advantage'] = [
        formation_collector.calculate_matchup_advantage(home, away)
        for home, away in zip(data['home_formation'], data['away_formation'])
    ]
    
    return data

def preprocess_corner_data(data: pd.DataFrame, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocesa datos para el modelo de predicción de corners.
    
    Args:
        data: DataFrame con datos crudos
        batch_size: Tamaño de los lotes para procesamiento
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        logger.info("Starting batch processing...")
        
        # Features básicas que deberían estar disponibles
        base_features = [
            'home_team_id', 'away_team_id',
            'home_avg_corners_for', 'home_avg_corners_against',
            'away_avg_corners_for', 'away_avg_corners_against',
            'home_form_score', 'away_form_score',
            'home_total_shots', 'away_total_shots',
            'home_ball_possession', 'away_ball_possession',
            'league_id'
        ]
        
        # Verificar columnas y manejar valores faltantes
        for col in base_features:
            if col not in data.columns:
                logger.warning(f"Columna {col} no encontrada, añadiendo con valores por defecto")
                if 'form_score' in col:
                    data[col] = 50  # valor medio para form_score
                elif 'possession' in col:
                    data[col] = 50  # valor medio para posesión
                else:
                    data[col] = 0
        
        # Features adicionales si están disponibles
        additional_features = ['is_derby']
        for col in additional_features:
            if col in data.columns:
                base_features.append(col)
                
        X = data[base_features].values
        y = data['total_corners'].values
        
        # Split train/test asegurando arrays numpy
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Log dimensiones
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Features used: {base_features}")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Error en preprocesamiento: {e}")
        # Retornar arrays vacíos en caso de error
        empty = np.array([])
        return empty, empty, empty, empty

def train_random_forest_corners_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    tune_hyperparams: bool = False,
    batch_size: int = 1000
) -> RandomForestRegressor:
    """
    Entrena un modelo Random Forest para predicción de corners con CV.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Valores objetivo
        params: Parámetros de hiperoptimización
        tune_hyperparams: Si se debe realizar búsqueda de hiperparámetros
        
    Returns:
        Modelo entrenado con los mejores parámetros
    """
    if tune_hyperparams:
        # Expanded parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 15, 20, 25],
            'min_samples_split': [2, 4, 6, 8],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': [0.7, 0.8, 0.9]
        }
        
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5,  # 5-fold cross-validation
            scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
            refit='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        # Log best results
        logger.info(f"Best Random Forest CV score: {-grid_search.best_score_:.4f} MSE")
        logger.info(f"Best parameters:\n{grid_search.best_params_}")
        
        # Store cross-validation results
        cv_results = pd.DataFrame(grid_search.cv_results_)
        os.makedirs('results', exist_ok=True)
        cv_results.to_csv('results/rf_cv_results.csv', index=False)
        
        # Use best parameters
        params = grid_search.best_params_
        model = grid_search.best_estimator_
    else:
        if params is None:
            params = {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 4,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'max_samples': 0.8
            }
        model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
    
    return model

def train_xgboost_corners_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    tune_hyperparams: bool = False
) -> xgb.XGBRegressor:
    """
    Entrena un modelo XGBoost para predicción de corners.
    
    Args:
        X_train: Features de entrenamiento
        y_train: Valores objetivo
        params: Parámetros de hiperoptimización
        tune_hyperparams: Si se debe realizar búsqueda de hiperparámetros
        
    Returns:
        Modelo entrenado
    """
    if tune_hyperparams:
        # Parámetros para búsqueda
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5]
        }
        
        model = xgb.XGBRegressor(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        
        # Usar los mejores parámetros encontrados
        params = grid_search.best_params_
        logger.info(f"Best XGBoost parameters: {params}")
    
    if params is None:
        # Hiperparámetros por defecto optimizados para features tácticas
        params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3
        }
    
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    
    return model

def evaluate_corners_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model for logging
        
    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"{model_name} evaluation: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")
        return metrics
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return {'mse': 0, 'rmse': 0, 'mae': 0, 'r2': 0}

def save_model(model: Any, model_name: str) -> bool:
    """
    Guarda un modelo entrenado usando joblib para mejor compatibilidad.
    
    Args:
        model: Modelo entrenado (RandomForest o XGBoost)
        model_name: Nombre para guardar el modelo
        
    Returns:
        bool: True si se guardó correctamente
    """
    try:
        if not os.path.exists('models'):
            os.makedirs('models')
            
        if isinstance(model, xgb.XGBRegressor):
            # Para XGBoost, usar formato nativo
            model_path = f'models/{model_name}.json'
            model.save_model(model_path)
        else:
            # Para otros modelos (RandomForest), usar joblib
            model_path = f'models/{model_name}.joblib'
            joblib.dump(model, model_path)
            
        logger.info(f"Modelo guardado exitosamente en {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error guardando modelo {model_name}: {e}")
        return False

def plot_model_comparison(metrics_rf: Dict[str, float], metrics_xgb: Dict[str, float], metrics_ensemble: Dict[str, float]) -> None:
    """
    Plot a comparison of model metrics.
    
    Args:
        metrics_rf: Metrics for Random Forest model
        metrics_xgb: Metrics for XGBoost model
        metrics_ensemble: Metrics for ensemble model
    """
    try:
        plt.figure(figsize=(10, 6))
        
        metrics = ['rmse', 'mae']
        rf_vals = [metrics_rf[m] for m in metrics]
        xgb_vals = [metrics_xgb[m] for m in metrics]
        ensemble_vals = [metrics_ensemble[m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.25
        
        plt.bar(x - width, rf_vals, width, label='Random Forest')
        plt.bar(x, xgb_vals, width, label='XGBoost')
        plt.bar(x + width, ensemble_vals, width, label='Ensemble')
        
        plt.xlabel('Metrics')
        plt.ylabel('Value (lower is better)')
        plt.title('Model Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        
        # Save figure
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/corners_model_comparison.png')
        logger.info("Model comparison plot saved to results/corners_model_comparison.png")
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")

def analyze_feature_importance(model, feature_names: List[str], model_name: str) -> None:
    """
    Analiza y visualiza la importancia de las features para un modelo.
    """
    try:
        # Obtener importancia de features
        importance = model.feature_importances_
        
        # Ordenar features por importancia
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        
        # Crear visualización
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importance)), sorted_importance)
        plt.xticks(range(len(importance)), sorted_features, rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()
        
        # Guardar gráfico
        os.makedirs('results', exist_ok=True)
        plt.savefig(f'results/{model_name.lower()}_feature_importance.png')
        
        # Imprimir resultados
        logger.info(f"\n{model_name} Feature Importance:")
        for feat, imp in zip(sorted_features, sorted_importance):
            logger.info(f"{feat}: {imp:.4f}")
            
    except Exception as e:
        logger.error(f"Error analyzing feature importance: {str(e)}")
        
    finally:
        plt.close()

def train_voting_ensemble_models(tune_hyperparams: bool = False) -> Dict[str, Any]:
    """
    Main function to train both Random Forest and XGBoost models
    for the voting ensemble.
    
    Args:
        tune_hyperparams: Whether to perform hyperparameter tuning
        
    Returns:
        Dictionary with training results
    """
    try:
        # Load and preprocess data
        data = load_corner_training_data()
        if data.empty:
            logger.error("No training data available")
            return {'success': False, 'error': 'No training data available'}
            
        X_train, X_test, y_train, y_test = preprocess_corner_data(data)
        if len(X_train) == 0:
            logger.error("Preprocessing failed")
            return {'success': False, 'error': 'Preprocessing failed'}
        
        # Train Random Forest
        logger.info("Training Random Forest model")
        rf_model = train_random_forest_corners_model(
            X_train, 
            y_train, 
            tune_hyperparams=tune_hyperparams
        )
        
        # Train XGBoost
        logger.info("Training XGBoost model")
        xgb_model = train_xgboost_corners_model(
            X_train,
            y_train,
            tune_hyperparams=tune_hyperparams
        )
        
        # Evaluate models
        metrics_rf = evaluate_corners_model(rf_model, X_test, y_test, "Random Forest")
        metrics_xgb = evaluate_corners_model(xgb_model, X_test, y_test, "XGBoost")
        
        # Save models
        save_model(rf_model, "random_forest_corners")
        save_model(xgb_model, "xgboost_corners")
        
        # Simple ensemble prediction (weighted average)
        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_ensemble = (y_pred_rf * 0.55) + (y_pred_xgb * 0.45)  # Weights from research paper
        
        # Calculate ensemble metrics
        mse_ensemble = mean_squared_error(y_test, y_pred_ensemble)
        rmse_ensemble = np.sqrt(mse_ensemble)
        mae_ensemble = mean_absolute_error(y_test, y_pred_ensemble)
        r2_ensemble = r2_score(y_test, y_pred_ensemble)
        
        metrics_ensemble = {
            'mse': mse_ensemble,
            'rmse': rmse_ensemble,
            'mae': mae_ensemble,
            'r2': r2_ensemble
        }
        
        logger.info(f"Ensemble evaluation: RMSE={rmse_ensemble:.3f}, MAE={mae_ensemble:.3f}, R²={r2_ensemble:.3f}")
        
        # Plot comparison of models
        plot_model_comparison(metrics_rf, metrics_xgb, metrics_ensemble)
        
        # Analyze and plot feature importance
        feature_names = pd.DataFrame(data).drop(['total_corners'], axis=1).columns.tolist()
        analyze_feature_importance(rf_model, feature_names, "Random Forest")
        analyze_feature_importance(xgb_model, feature_names, "XGBoost")
        
        # Generate result summary
        result = {
            'success': True,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'rf_metrics': metrics_rf,
            'xgb_metrics': metrics_xgb,
            'ensemble_metrics': metrics_ensemble,
            'rf_feature_importance': dict(zip(range(X_train.shape[1]), rf_model.feature_importances_)),
            'xgb_feature_importance': dict(zip(range(X_train.shape[1]), xgb_model.feature_importances_))
        }
        
        # Save results
        os.makedirs('results', exist_ok=True)
        with open('results/ensemble_training_results.json', 'w') as f:
            import json
            json.dump(result, f, indent=2)
        
        return result
    except Exception as e:
        logger.error(f"Error training ensemble models: {e}")
        return {'success': False, 'error': str(e)}

def train_voting_ensemble(league_ids: List[int] = [39, 140, 135, 78, 61], model_dir: str = "models") -> Dict[str, Any]:
    """
    Entrena el ensemble de modelos para predicción de corners.
    
    Args:
        league_ids: Lista de IDs de ligas a incluir
        model_dir: Directorio para guardar modelos
        
    Returns:
        Dict con métricas de rendimiento
    """
    try:
        # Cargar datos de todas las ligas
        logger.info("Loading corner data from multiple leagues...")
        data = load_corner_json_data(league_ids)
        
        if data.empty:
            raise ValueError("No se pudieron cargar los datos de corners")
        
        # Preprocesar datos
        logger.info("Preprocessing corner data...")
        X_train, X_test, y_train, y_test = preprocess_corner_data(data)
        
        # Lista de nombres de features
        feature_names = [
            'home_team_id', 'away_team_id', 'home_formation_id', 'away_formation_id',
            'home_wing_attack', 'home_high_press', 'home_possession',
            'away_wing_attack', 'away_high_press', 'away_possession',
            'formation_advantage'
        ]
        
        # Entrenar Random Forest
        logger.info("\nTraining Random Forest model...")
        rf_model = train_random_forest_corners_model(X_train, y_train)
        analyze_feature_importance(rf_model, feature_names, "Random Forest")
        
        # Entrenar XGBoost
        logger.info("\nTraining XGBoost model...")
        xgb_model = train_xgboost_corners_model(X_train, y_train)
        analyze_feature_importance(xgb_model, feature_names, "XGBoost")
        
        # Evaluar modelos
        logger.info("\nEvaluating modelos...")
        metrics_rf = evaluate_corners_model(rf_model, X_test, y_test, "Random Forest")
        metrics_xgb = evaluate_corners_model(xgb_model, X_test, y_test, "XGBoost")
        
        # Calcular predicciones del ensemble
        y_pred_rf = rf_model.predict(X_test)
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_ensemble = (y_pred_rf + y_pred_xgb) / 2
        
        metrics_ensemble = {
            'mae': mean_absolute_error(y_test, y_pred_ensemble),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_ensemble)),
            'r2': r2_score(y_test, y_pred_ensemble)
        }
        
        logger.info("\nEnsemble Metrics:")
        logger.info(f"MAE: {metrics_ensemble['mae']:.3f}")
        logger.info(f"RMSE: {metrics_ensemble['rmse']:.3f}")
        logger.info(f"R2: {metrics_ensemble['r2']:.3f}")
        
        # Guardar modelos
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rf_path = os.path.join(model_dir, f"rf_corners_{timestamp}.joblib")
        xgb_path = os.path.join(model_dir, f"xgb_corners_{timestamp}.joblib")
        
        joblib.dump(rf_model, rf_path)
        joblib.dump(xgb_model, xgb_path)
        
        logger.info(f"\nModels saved to {model_dir}")
        
        return {
            'random_forest': metrics_rf,
            'xgboost': metrics_xgb,
            'ensemble': metrics_ensemble,
            'rf_path': rf_path,
            'xgb_path': xgb_path
        }
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {e}")
        raise

def cross_validate_ensemble(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict[str, Any]:
    """
    Realiza validación cruzada del ensemble de modelos.
    
    Args:
        X: Features de entrenamiento
        y: Target variable
        n_splits: Número de particiones para CV
        
    Returns:
        Dict con métricas de validación cruzada
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = {
        'rf_rmse': [], 'rf_mae': [], 'rf_r2': [],
        'xgb_rmse': [], 'xgb_mae': [], 'xgb_r2': [],
        'ensemble_rmse': [], 'ensemble_mae': [], 'ensemble_r2': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Entrenar modelos en esta partición
        rf_model = train_random_forest_corners_model(X_train_fold, y_train_fold)
        xgb_model = train_xgboost_corners_model(X_train_fold, y_train_fold)
        
        # Predicciones
        y_pred_rf = rf_model.predict(X_val_fold)
        y_pred_xgb = xgb_model.predict(X_val_fold)
        y_pred_ensemble = 0.55 * y_pred_rf + 0.45 * y_pred_xgb
        
        # Calcular métricas
        for model_name, y_pred in [
            ('rf', y_pred_rf),
            ('xgb', y_pred_xgb),
            ('ensemble', y_pred_ensemble)
        ]:
            rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
            mae = mean_absolute_error(y_val_fold, y_pred)
            r2 = r2_score(y_val_fold, y_pred)
            
            cv_scores[f'{model_name}_rmse'].append(rmse)
            cv_scores[f'{model_name}_mae'].append(mae)
            cv_scores[f'{model_name}_r2'].append(r2)
            
        logger.info(f"Fold {fold + 1} completado")
    
    # Calcular promedios
    cv_results = {}
    for metric, scores in cv_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        cv_results[metric] = {'mean': mean_score, 'std': std_score}
    
    return cv_results

def train_ensemble_with_temporal_cv(
    n_splits: int = 5,
    gap: int = 10,
    tune_hyperparams: bool = False,
    output_dir: str = "models/ts_cv_ensemble"
) -> Dict[str, Any]:
    """
    Entrena y evalúa el ensemble usando validación cruzada temporal.
    
    Args:
        n_splits: Número de divisiones temporales
        gap: Número de muestras a omitir entre conjuntos de entrenamiento y prueba
        tune_hyperparams: Si se deben optimizar hiperparámetros
        output_dir: Directorio para guardar modelos y resultados
        
    Returns:
        Dict con resultados de validación
    """
    try:
        # Cargar datos con fechas incluidas
        logger.info("Loading and preparing data...")
        data = load_corner_training_data()
        data['date'] = pd.to_datetime(data['date'])
        dates = np.array(data['date'].astype(np.int64) // 10**9)
        
        # Preparar features y target
        feature_columns = [
            'home_team_id', 'away_team_id', 'home_formation_id', 'away_formation_id',
            'home_wing_attack', 'home_high_press', 'home_possession',
            'away_wing_attack', 'away_high_press', 'away_possession',
            'formation_advantage'
        ]
        X = data[feature_columns].values
        y = data['total_corners'].values        # Función factory para crear modelo ensemble
        def ensemble_factory(X_train, y_train):
            # Entrenar RF de forma incremental
            rf = train_random_forest_corners_model(
                X_train, y_train,
                tune_hyperparams=tune_hyperparams,
                batch_size=1000
            )
            rf = train_model_incrementally(rf, X_train, y_train)
            
            # Entrenar XGBoost de forma incremental
            xgb_model = train_xgboost_corners_model(
                X_train, y_train,
                tune_hyperparams=tune_hyperparams
            )
            xgb_model = train_model_incrementally(xgb_model, X_train, y_train)
              # Crear y retornar ensemble dinámico
            return DynamicWeightEnsemble(rf, xgb_model, window_size=10)

        # Crear y configurar validador temporal
        validator = TimeSeriesValidator(
            n_splits=n_splits,
            gap=gap,
            verbose=True
        )
        
        # Ejecutar validación cruzada
        results = validator.validate(
            X=X,
            y=y,
            dates=dates,
            model_factory=ensemble_factory,
            metrics=['mae', 'rmse', 'r2'],
            save_models=True,
            output_dir=output_dir
        )
        
        # Analizar estabilidad temporal
        stability_results = {
            'rmse_stability': evaluate_temporal_stability(results, 'rmse'),
            'mae_stability': evaluate_temporal_stability(results, 'mae'),
            'r2_stability': evaluate_temporal_stability(results, 'r2')
        }
        
        # Añadir resultados de estabilidad
        results['stability_analysis'] = stability_results
        
        # Generar visualizaciones
        validator.plot_results(output_dir=output_dir)
        validator.plot_validation_scheme(
            n_samples=len(X),
            output_dir=output_dir
        )
        
        # Guardar resultados completos
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'temporal_cv_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in temporal cross-validation: {e}")
        raise

def evaluate_temporal_stability(results: Dict[str, Any], metric: str = 'rmse') -> Dict[str, float]:
    """
    Evalúa la estabilidad temporal del modelo analizando la variación
    de las métricas a lo largo del tiempo.
    
    Args:
        results: Diccionario con resultados de validación temporal
        metric: Métrica a analizar ('rmse', 'mae', 'r2')
        
    Returns:
        Dict con métricas de estabilidad
    """
    try:
        # Extraer valores de la métrica para cada fold temporal
        values = np.array([fold[metric] for fold in results.get('fold_metrics', [])])
        
        if len(values) < 2:
            logger.warning("Insufficient data for stability analysis")
            return {}
            
        # Calcular métricas de estabilidad
        stability_metrics = {
            'temporal_variance': np.var(values),  # Varianza entre folds
            'temporal_trend': np.polyfit(np.arange(len(values)), values, 1)[0],  # Tendencia lineal
            'max_deviation': np.max(np.abs(values - np.mean(values))),  # Máxima desviación
            'stability_score': 1 / (1 + np.std(values))  # Score de estabilidad (más alto = más estable)
        }
        
        # Análisis de degradación
        stability_metrics['degradation_rate'] = (values[-1] - values[0]) / len(values)
        
        # Logging de resultados
        logger.info(f"\nTemporal Stability Analysis for {metric}:")
        for name, value in stability_metrics.items():
            logger.info(f"{name}: {value:.4f}")
            
        return stability_metrics
        
    except Exception as e:
        logger.error(f"Error in stability analysis: {e}")
        return {}

class DynamicWeightEnsemble:
    """
    Ensemble con pesos dinámicos que se ajustan según el rendimiento reciente.
    """
    def __init__(self, rf_model, xgb_model, window_size=10):
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.window_size = window_size
        self.rf_errors = []
        self.xgb_errors = []
        self.rf_weight = 0.55  # Peso inicial RF
        self.xgb_weight = 0.45  # Peso inicial XGB
        
    def _update_weights(self):
        """Actualiza los pesos basados en el error promedio reciente."""
        if len(self.rf_errors) < self.window_size:
            return
            
        # Calcular error promedio en la ventana reciente
        rf_mean_error = np.mean(self.rf_errors[-self.window_size:])
        xgb_mean_error = np.mean(self.xgb_errors[-self.window_size:])
        
        # Evitar división por cero
        total_error = rf_mean_error + xgb_mean_error
        if total_error == 0:
            return
            
        # Actualizar pesos inversamente proporcional al error
        self.rf_weight = xgb_mean_error / total_error
        self.xgb_weight = rf_mean_error / total_error
        
        # Normalizar pesos
        total_weight = self.rf_weight + self.xgb_weight
        self.rf_weight /= total_weight
        self.xgb_weight /= total_weight
        
        # Limpiar memoria histórica si excede el tamaño de ventana
        if len(self.rf_errors) > self.window_size * 2:
            self.rf_errors = self.rf_errors[-self.window_size:]
            self.xgb_errors = self.xgb_errors[-self.window_size:]
            gc.collect()  # Forzar liberación de memoria
        
        logger.info(f"Updated weights - RF: {self.rf_weight:.3f}, XGB: {self.xgb_weight:.3f}")
        
    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Realiza predicción con el ensemble y opcionalmente actualiza los pesos.
        
        Args:
            X: Features para predicción
            y_true: Valores reales para actualizar pesos (opcional)
            
        Returns:
            Array con predicciones
        """
        # Procesar en lotes si el input es grande
        if len(X) > 1000:
            predictions = []
            for i in range(0, len(X), 1000):
                batch = X[i:i+1000]
                rf_pred_batch = self.rf_model.predict(batch)
                xgb_pred_batch = self.xgb_model.predict(batch)
                pred_batch = self.rf_weight * rf_pred_batch + self.xgb_weight * xgb_pred_batch
                predictions.append(pred_batch)
                del rf_pred_batch, xgb_pred_batch, pred_batch
                gc.collect()
            return np.concatenate(predictions)
            
        # Para conjuntos pequeños, procesar todo junto
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        predictions = self.rf_weight * rf_pred + self.xgb_weight * xgb_pred
        
        # Si tenemos valores reales, actualizar errores y pesos
        if y_true is not None:
            self.rf_errors.extend(np.abs(rf_pred - y_true))
            self.xgb_errors.extend(np.abs(xgb_pred - y_true))
            self._update_weights()
            
            # Limpiar variables temporales
            del rf_pred, xgb_pred
            gc.collect()
        
        return predictions
        
    def get_current_weights(self) -> Dict[str, float]:
        """Retorna los pesos actuales del ensemble."""
        return {
            'random_forest': self.rf_weight,
            'xgboost': self.xgb_weight
        }

def plot_weight_evolution(results: Dict[str, Any], output_dir: str) -> None:
    """
    Visualiza la evolución de los pesos del ensemble a lo largo del tiempo.
    
    Args:
        results: Diccionario con resultados que incluye pesos
        output_dir: Directorio para guardar la visualización
    """
    try:
        # Extraer pesos del ensemble a lo largo del tiempo
        rf_weights = [fold.get('weights', {}).get('random_forest', 0.55) 
                     for fold in results.get('fold_metrics', [])]
        xgb_weights = [fold.get('weights', {}).get('xgboost', 0.45) 
                      for fold in results.get('fold_metrics', [])]
        
        # Crear visualización
        plt.figure(figsize=(10, 6))
        x = range(len(rf_weights))
        
        plt.plot(x, rf_weights, 'b-', label='Random Forest', marker='o')
        plt.plot(x, xgb_weights, 'r-', label='XGBoost', marker='s')
        
        plt.xlabel('Fold Temporal')
        plt.ylabel('Peso del Modelo')
        plt.title('Evolución de Pesos del Ensemble')
        plt.grid(True)
        plt.legend()
        
        # Guardar gráfico
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'weight_evolution.png'))
        logger.info(f"Weight evolution plot saved to {output_dir}/weight_evolution.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error plotting weight evolution: {e}")

def process_in_batches(data: pd.DataFrame, batch_size: int = 1000) -> Generator[pd.DataFrame, None, None]:
    """
    Procesa el DataFrame en lotes para optimizar memoria.
    
    Args:
        data: DataFrame a procesar
        batch_size: Tamaño de cada lote
        
    Yields:
        Generator que produce lotes del DataFrame
    """
    num_batches = len(data) // batch_size + (1 if len(data) % batch_size != 0 else 0)
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(data))
        
        # Obtener lote
        batch = data.iloc[start_idx:end_idx].copy()
        
        yield batch
        
        # Limpiar memoria explícitamente
        del batch
        gc.collect()

def monitor_memory_usage(label: str = '', detailed: bool = False) -> Dict[str, float]:
    """
    Monitorea el uso de memoria actual con detalles opcionales.
    
    Args:
        label: Etiqueta para identificar el punto de monitoreo
        detailed: Si se debe incluir información detallada de memoria
        
    Returns:
        Dict con métricas de memoria
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = {
            'rss': memory_info.rss / 1024 / 1024,  # Memoria RSS en MB
            'vms': memory_info.vms / 1024 / 1024   # Memoria virtual en MB
        }
        
        if detailed:
            memory_maps = process.memory_maps()
            metrics.update({
                'shared': sum(m.shared for m in memory_maps) / 1024 / 1024,
                'private': sum(m.private for m in memory_maps) / 1024 / 1024,
                'peak': process.memory_info().peak_wset / 1024 / 1024
            })
            
            # Información del recolector de basura
            gc.collect()  # Forzar recolección
            metrics['gc_objects'] = len(gc.get_objects())
            metrics['gc_generations'] = [count for count in gc.get_count()]
        
        if label:
            logger.info(f"Memory usage at {label}: {metrics['rss']:.2f} MB RSS")
            
        return metrics
        
    except Exception as e:
        logger.error(f"Error monitoring memory: {e}")
        return {'error': str(e)}

if __name__ == '__main__':
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar validación cruzada temporal
    logger.info("Starting temporal cross-validation for ensemble model...")
    results = train_ensemble_with_temporal_cv(
        n_splits=5,  # 5 divisiones temporales
        gap=10,      # 10 partidos de separación 
        tune_hyperparams=True  # Optimizar hiperparámetros
    )
    
    # Mostrar resultados
    logger.info("\nResults:")
    for metric, values in results.items():
        if metric not in ['train_size', 'test_size', 'train_start', 'train_end', 'test_start', 'test_end']:
            mean_val = np.mean(values)
            std_val = np.std(values)
            logger.info(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

def train_model_incrementally(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 1000,
    warm_start: bool = True,
    monitor_memory: bool = True
) -> Any:
    """
    Entrena un modelo de forma incremental por lotes para optimizar memoria.
    
    Args:
        model: Modelo a entrenar (RF o XGBoost)
        X: Features de entrenamiento
        y: Target variable
        batch_size: Tamaño de los lotes
        warm_start: Si se debe usar warm start para entrenamiento incremental
        monitor_memory: Si se debe monitorear el uso de memoria
        
    Returns:
        Modelo entrenado
    """
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size + (1 if n_samples % batch_size > 0 else 0)
    
    logger.info(f"Training model incrementally with {n_batches} batches...")
    
    if monitor_memory:
        initial_memory = monitor_memory_usage("Initial state", detailed=True)
    
    try:
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            # Obtener lote actual
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]
            
            # Entrenar en el lote actual
            if isinstance(model, RandomForestRegressor):
                if i == 0 or not warm_start:
                    model.fit(X_batch, y_batch)
                else:
                    # Añadir árboles adicionales
                    additional_estimators = model.n_estimators
                    model.n_estimators += additional_estimators
                    model.fit(X_batch, y_batch)
            else:
                # Para XGBoost, usar xgb.train con actualización incremental
                if i == 0:
                    model.fit(X_batch, y_batch)
                else:
                    model.fit(X_batch, y_batch, xgb_model=model)
            
            # Monitorear memoria y limpiar
            if monitor_memory:
                batch_memory = monitor_memory_usage(f"After batch {i+1}/{n_batches}")
                
                # Si el uso de memoria es alto, forzar limpieza
                if batch_memory['rss'] > initial_memory['rss'] * 1.5:
                    logger.warning("High memory usage detected, forcing cleanup...")
                    del X_batch, y_batch
                    gc.collect()
                    
            # Logging de progreso
            logger.info(f"Batch {i+1}/{n_batches} completed")
        
        return model
        
    except Exception as e:
        logger.error(f"Error in incremental training: {e}")
        raise
    
    finally:
        # Limpieza final
        gc.collect()
        if monitor_memory:
            final_memory = monitor_memory_usage("Final state", detailed=True)
            logger.info(f"Memory change: {final_memory['rss'] - initial_memory['rss']:.2f} MB")
