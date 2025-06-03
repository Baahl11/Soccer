"""
Script para entrenar los modelos de predicción de corners.
"""

import json
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
from datetime import datetime
import glob
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

class CornerModelTrainer:
    """Clase para entrenar modelos de predicción de corners"""
    
    def __init__(self):
        self.features = [
            'home_avg_corners_for', 'home_avg_corners_against',
            'away_avg_corners_for', 'away_avg_corners_against',
            'home_form_score', 'away_form_score',
            'home_total_shots', 'away_total_shots',
            'league_id', 'home_elo', 'away_elo',
            'elo_diff', 'elo_win_probability'
        ]
        self.target = 'total_corners'
        self.models_dir = 'models'
        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

    def load_corner_data(self) -> pd.DataFrame:
        """Carga y prepara datos históricos de corners"""
        all_data = []
        
        # Cargar todos los archivos JSON de datos de corners
        json_pattern = os.path.join(os.getcwd(), 'corner_data_*.json')
        json_files = glob.glob(json_pattern)
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
            except Exception as e:
                logger.error(f"Error cargando {file_path}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_data)
        
        # Limpiar y preparar datos
        df = self._prepare_data(df)
        
        return df

    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara y limpia los datos para entrenamiento"""
        # Convertir a tipos numéricos
        for feature in self.features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            
        # Eliminar filas con valores faltantes
        df = df.dropna(subset=self.features + [self.target])
        
        # Eliminar outliers
        df = self._remove_outliers(df)
        
        return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina outliers usando IQR"""
        for col in self.features + [self.target]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[col] < (Q1 - 1.5 * IQR)) | 
                     (df[col] > (Q3 + 1.5 * IQR)))]
        return df

    def train_random_forest(self, X_train, y_train) -> RandomForestRegressor:
        """Entrena modelo Random Forest"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def train_xgboost(self, X_train, y_train) -> xgb.XGBRegressor:
        """Entrena modelo XGBoost"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test) -> dict:
        """Evalúa el rendimiento del modelo"""
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        return {
            'rmse': rmse,
            'r2': r2
        }

    def train_and_save_models(self):
        """Entrena y guarda ambos modelos"""
        # Cargar datos
        df = self.load_corner_data()
        logger.info(f"Datos cargados: {len(df)} registros")
        
        # Dividir en train/test
        X = df[self.features]
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar Random Forest
        logger.info("Entrenando Random Forest...")
        rf_model = self.train_random_forest(X_train, y_train)
        rf_metrics = self.evaluate_model(rf_model, X_test, y_test)
        logger.info(f"Random Forest metrics: {rf_metrics}")
        
        # Guardar Random Forest
        rf_path = os.path.join(self.models_dir, 'random_forest_corners.joblib')
        joblib.dump(rf_model, rf_path)
        logger.info(f"Random Forest guardado en {rf_path}")
        
        # Entrenar XGBoost
        logger.info("Entrenando XGBoost...")
        xgb_model = self.train_xgboost(X_train, y_train)
        xgb_metrics = self.evaluate_model(xgb_model, X_test, y_test)
        logger.info(f"XGBoost metrics: {xgb_metrics}")
        
        # Guardar XGBoost
        xgb_path = os.path.join(self.models_dir, 'xgboost_corners.json')
        xgb_model.save_model(xgb_path)
        logger.info(f"XGBoost guardado en {xgb_path}")
        
        # Guardar métricas
        metrics = {
            'random_forest': rf_metrics,
            'xgboost': xgb_metrics,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        metrics_path = os.path.join(self.models_dir, 'corner_models_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Métricas guardadas en {metrics_path}")

if __name__ == '__main__':
    trainer = CornerModelTrainer()
    trainer.train_and_save_models()
