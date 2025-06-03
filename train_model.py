# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from fnn_model import FeedforwardNeuralNetwork
import logging
from data import FootballAPI
from typing import Tuple, List
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_training_data(
    ligas_ids=[39, 140, 135, 78, 61], 
    temporadas=None, 
    limit_seasons=2, 
    shuffle=True,
    nan_strategy='mean',
    min_matches_per_league=100
):
    """
    Obtiene y prepara datos de entrenamiento de múltiples ligas y temporadas.
    
    Args:
        ligas_ids: IDs de las ligas a incluir (Premier League, La Liga, Serie A, Bundesliga, Ligue 1 por defecto)
        temporadas: Temporadas específicas o None para usar las últimas 3
        limit_seasons: Limitar a las últimas N temporadas por liga para reducir tiempo de procesamiento
        shuffle: Si se deben mezclar los datos
        nan_strategy: Estrategia para manejar valores NaN ('mean', 'median', 'most_frequent', 'constant')
        min_matches_per_league: Número mínimo de partidos que debe tener una liga para ser incluida
        
    Returns:
        DataFrame con datos de entrenamiento preparados
    """
    api = FootballAPI()
    all_data = []
    
    # Si no se especifican temporadas, usar las últimas 3
    if temporadas is None:
        current_year = datetime.now().year
        temporadas = list(range(current_year-2, current_year+1))
    
    logger.info(f"Usando temporadas: {temporadas}")
    
    for liga_id in ligas_ids:
        logger.info(f"\nProcesando lote para liga {liga_id}...")
        liga_data = []
        
        for temporada in sorted(temporadas, reverse=True)[:limit_seasons]:
            try:
                logger.info(f"Obteniendo datos para liga {liga_id} temporada {temporada}")
                
                # Reducir solicitudes API para temporadas antiguas si no es la actual
                if temporada < max(temporadas) - 1:
                    logger.info(f"Omitiendo temporada {temporada} para reducir solicitudes API")
                    continue
                
                # Obtener datos históricos
                df = api.get_historical_data(liga_id, temporada)
                
                if not df.empty and len(df) > 0:
                    logger.info(f"Obtenidos {len(df)} partidos para liga {liga_id}, temporada {temporada}")
                    liga_data.append(df)
                else:
                    logger.warning(f"No se encontraron datos para liga {liga_id} temporada {temporada}")
            except Exception as e:
                logger.error(f"Error obteniendo datos para liga {liga_id} temporada {temporada}: {e}")
        
        # Combinar datos de todas las temporadas para esta liga
        if liga_data:
            df_liga = pd.concat(liga_data, ignore_index=True)
            if len(df_liga) >= min_matches_per_league:
                all_data.append(df_liga)
                logger.info(f"Total para liga {liga_id}: {len(df_liga)} partidos")
            else:
                logger.warning(f"Liga {liga_id} descartada por tener menos de {min_matches_per_league} partidos ({len(df_liga)})")
        else:
            logger.error(f"Error procesando lote: No se encontraron datos para liga {liga_id}")
    
    # Combinar datos de todas las ligas
    if not all_data:
        raise ValueError("No se pudieron obtener datos de entrenamiento")
    
    data = pd.concat(all_data, ignore_index=True)
    logger.info(f"\nTotal de datos combinados: {len(data)} partidos")
    
    # Manejar valores NaN según la estrategia especificada
    data = handle_missing_values(data, strategy=nan_strategy)
    
    # Preparar características y etiquetas para entrenamiento
    X, y = prepare_features(data)
    
    # Mezclar datos si se solicita
    if shuffle:
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    return X, y

def handle_missing_values(data, strategy='mean'):
    """
    Maneja valores faltantes en los datos usando diferentes estrategias.
    
    Args:
        data: DataFrame con los datos
        strategy: Estrategia para imputar valores faltantes 
                  ('mean', 'median', 'most_frequent', 'constant')
    
    Returns:
        DataFrame con valores NaN tratados
    """
    # Verificar si hay valores NaN
    if data.isnull().sum().sum() > 0:
        logger.warning(f"Se detectaron {data.isnull().sum().sum()} valores NaN en los datos")
        
        # Columnas numéricas que queremos imputar
        stat_columns = [col for col in data.columns if any(x in col for x in [
            'shots', 'goals', 'fouls', 'corner', 'offside', 'possession', 
            'yellow', 'red', 'saves', 'passes'
        ])]
        
        # Extraer columnas de estadísticas de equipos local y visitante
        home_stats_cols = [col for col in stat_columns if 'home_stats' in col]
        away_stats_cols = [col for col in stat_columns if 'away_stats' in col]
        
        # Imputación para estadísticas de equipos local y visitante por separado
        for cols_group in [home_stats_cols, away_stats_cols]:
            if strategy == 'mean':
                values = data[cols_group].mean()
            elif strategy == 'median':
                values = data[cols_group].median()
            elif strategy == 'most_frequent':
                values = data[cols_group].mode().iloc[0]
            elif strategy == 'constant':
                values = pd.Series(0, index=cols_group)
            else:
                raise ValueError(f"Estrategia de imputación no válida: {strategy}")
            
            # Aplicar imputación
            for col in cols_group:
                data[col].fillna(values[col], inplace=True)
        
        # Verificar si quedan valores NaN
        remaining_nans = data.isnull().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"Quedan {remaining_nans} valores NaN después de la imputación")
            # Llenar cualquier otro valor NaN con 0
            data.fillna(0, inplace=True)
    
    return data

def prepare_features(data):
    """
    Prepara características y etiquetas para entrenamiento.
    
    Args:
        data: DataFrame con los datos
    
    Returns:
        X: Características para entrenamiento
        y: Etiquetas (resultado del partido)
    """
    # Normalizar estadísticas por partido
    for prefix in ['home_stats', 'away_stats']:
        # Normalizar posesión de balón
        if f'{prefix}.ball_possession' in data.columns:
            data[f'{prefix}.ball_possession'] = data[f'{prefix}.ball_possession'] / 100.0
        
        # Normalizar precisión de pases
        if f'{prefix}.passes_percentage' in data.columns:
            data[f'{prefix}.passes_percentage'] = data[f'{prefix}.passes_percentage'] / 100.0
    
    # Extraer características adicionales
    data['total_goals'] = data['home_goals'] + data['away_goals']
    data['goal_difference'] = data['home_goals'] - data['away_goals']
    
    # Crear etiquetas: 0 = victoria visitante, 1 = empate, 2 = victoria local
    data['result'] = np.select(
        [data['home_goals'] > data['away_goals'], 
         data['home_goals'] == data['away_goals']],
        [2, 1],
        default=0
    )
    
    # Seleccionar características para el modelo
    feature_columns = [col for col in data.columns if any(x in col for x in [
        'shots', 'goals', 'fouls', 'corner', 'offside', 'possession', 
        'yellow', 'red', 'saves', 'passes'
    ])]
    
    # Añadir características adicionales importantes de la API
    if 'fixture_id' in data.columns and 'date' in data.columns:
        # Extraer características temporales
        data['match_day'] = pd.to_datetime(data['date']).dt.dayofweek
        data['match_hour'] = pd.to_datetime(data['date']).dt.hour
        
        # Añadir estas columnas a las características si existen
        feature_columns.extend(['match_day', 'match_hour'])
    
    X = data[feature_columns].values
    y = data['result'].values
    
    return X, y

def train_and_save_model(force_refresh: bool = False):
    try:
        # Obtener datos de entrenamiento
        X, y = get_training_data()
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Crear y entrenar modelo
        logger.info("\nEntrenando modelo...")
        model = FeedforwardNeuralNetwork(input_dim=X.shape[1])
        history = model.train(
            X_train_scaled, y_train,
            X_val=X_test_scaled,
            y_val=y_test,
            epochs=100
        )
        
        # Evaluar modelo
        train_predictions = model.predict(X_train_scaled)
        test_predictions = model.predict(X_test_scaled)
        
        # Calcular error medio absoluto
        train_mae = np.mean(np.abs(train_predictions - y_train))
        test_mae = np.mean(np.abs(test_predictions - y_test))
        logger.info(f"Train MAE: {train_mae:.3f}, Test MAE: {test_mae:.3f}")
        
        # Guardar modelo y scaler
        Path('models').mkdir(exist_ok=True)
        joblib.dump(model, 'models/nb_model.pkl')
        joblib.dump(scaler, 'models/scaler.pkl')
        
        # Guardar métricas y configuración del entrenamiento
        training_info = {
            'train_mae': float(train_mae),
            'test_mae': float(test_mae),
            'num_samples': len(X),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('models/training_info.json', 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("Modelo entrenado y guardado exitosamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en entrenamiento del modelo: {e}")
        return False

if __name__ == "__main__":
    train_and_save_model()
