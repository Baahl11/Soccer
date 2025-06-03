"""
Script para reentrenar y guardar el modelo de predicción en formato .h5 utilizando datos reales de la API Football
"""
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fnn_model import FeedforwardNeuralNetwork
import joblib
import requests
import json
from tqdm import tqdm
import time
import dotenv

# Cargar variables de entorno
dotenv.load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes para la API - Usar la API key del archivo .env
FOOTBALL_API_KEY = os.environ.get("API_FOOTBALL_KEY") or os.environ.get("API_KEY")
FOOTBALL_API_HOST = os.environ.get("API_HOST", "api-football-v1.p.rapidapi.com")
FOOTBALL_API_BASE_URL = os.environ.get("API_BASE_URL", "https://api-football-v1.p.rapidapi.com/v3")

# Corregido para manejar caso donde FOOTBALL_API_KEY es None
if FOOTBALL_API_KEY:
    # Si existe la API key, mostrarla parcialmente por seguridad
    logger.info(f"API Key configurada: {FOOTBALL_API_KEY[:5]}...{FOOTBALL_API_KEY[-5:]}")
else:
    logger.warning("API Key no disponible. Verifica tu archivo .env")

logger.info(f"API Host: {FOOTBALL_API_HOST}")
logger.info(f"API Base URL: {FOOTBALL_API_BASE_URL}")

def fetch_historical_matches(league_id, season):
    """
    Obtiene datos históricos de partidos reales para un campeonato y temporada específicos
    
    Args:
        league_id: ID del campeonato (ej: 39 para Premier League)
        season: Temporada en formato YYYY (ej: 2023)
    
    Returns:
        DataFrame con los datos históricos
    """
    logger.info(f"Obteniendo datos históricos para league_id={league_id}, season={season}")
    
    headers = {
        'x-rapidapi-host': FOOTBALL_API_HOST,
        'x-rapidapi-key': FOOTBALL_API_KEY
    }
    
    # Obtener todas las jornadas para esta liga y temporada
    url = f"{FOOTBALL_API_BASE_URL}/fixtures/rounds"
    params = {
        'league': league_id,
        'season': season
    }
    
    try:
        logger.info(f"Consultando API: {url}")
        rounds_response = requests.get(url, headers=headers, params=params)
        rounds_data = rounds_response.json()
        
        if rounds_data.get('errors'):
            logger.error(f"Error obteniendo rondas: {rounds_data['errors']}")
            return None
        
        rounds = rounds_data.get('response', [])
        logger.info(f"Se encontraron {len(rounds)} rondas para esta liga y temporada")
        
        # Lista para almacenar todos los partidos
        all_matches = []
        
        # Para cada ronda, obtenemos los partidos
        for round_name in tqdm(rounds, desc="Obteniendo partidos por ronda"):
            url = f"{FOOTBALL_API_BASE_URL}/fixtures"
            params = {
                'league': league_id,
                'season': season,
                'round': round_name,
                'status': 'FT'  # Solo partidos finalizados
            }
            
            fixtures_response = requests.get(url, headers=headers, params=params)
            fixtures_data = fixtures_response.json()
            
            if fixtures_data.get('errors'):
                logger.warning(f"Error obteniendo partidos para la ronda {round_name}: {fixtures_data['errors']}")
                continue
                
            fixtures = fixtures_data.get('response', [])
            logger.debug(f"Se encontraron {len(fixtures)} partidos en la ronda {round_name}")
            
            # Agregamos a la lista de partidos
            all_matches.extend(fixtures)
            
            # Respetamos los límites de la API
            time.sleep(0.5)
        
        # Procesamos los datos para crear un DataFrame
        processed_data = []
        
        for match in tqdm(all_matches, desc="Procesando datos de partidos"):
            # Extraemos datos básicos del partido
            fixture_id = match['fixture']['id']
            home_team_id = match['teams']['home']['id']
            home_team_name = match['teams']['home']['name']
            away_team_id = match['teams']['away']['id']
            away_team_name = match['teams']['away']['name']
            home_goals = match['goals']['home']
            away_goals = match['goals']['away']
            match_date = match['fixture']['date']
            
            # Obtenemos estadísticas avanzadas del partido
            url = f"{FOOTBALL_API_BASE_URL}/fixtures/statistics"
            params = {
                'fixture': fixture_id
            }
            
            stats_response = requests.get(url, headers=headers, params=params)
            stats_data = stats_response.json()
            
            # Inicializamos variables para estadísticas
            home_possession = 50
            away_possession = 50
            home_shots = 0
            away_shots = 0
            home_shots_on_target = 0
            away_shots_on_target = 0
            home_corners = 0
            away_corners = 0
            
            if not stats_data.get('errors') and len(stats_data.get('response', [])) > 0:
                for team_stats in stats_data['response']:
                    team_id = team_stats['team']['id']
                    stats = team_stats['statistics']
                    
                    for stat in stats:
                        if team_id == home_team_id:
                            if stat['type'] == 'Ball Possession':
                                home_possession = int(stat['value'].replace('%', '')) if stat['value'] else 50
                            elif stat['type'] == 'Total Shots':
                                home_shots = int(stat['value']) if stat['value'] else 0
                            elif stat['type'] == 'Shots on Goal':
                                home_shots_on_target = int(stat['value']) if stat['value'] else 0
                            elif stat['type'] == 'Corner Kicks':
                                home_corners = int(stat['value']) if stat['value'] else 0
                        else:
                            if stat['type'] == 'Ball Possession':
                                away_possession = int(stat['value'].replace('%', '')) if stat['value'] else 50
                            elif stat['type'] == 'Total Shots':
                                away_shots = int(stat['value']) if stat['value'] else 0
                            elif stat['type'] == 'Shots on Goal':
                                away_shots_on_target = int(stat['value']) if stat['value'] else 0
                            elif stat['type'] == 'Corner Kicks':
                                away_corners = int(stat['value']) if stat['value'] else 0
            
            # Construimos un diccionario con todos los datos
            match_data = {
                'match_id': fixture_id,
                'date': match_date,
                'home_team_id': home_team_id,
                'home_team_name': home_team_name,
                'away_team_id': away_team_id,
                'away_team_name': away_team_name,
                'home_goals': home_goals,
                'away_goals': away_goals,
                'home_possession': home_possession,
                'away_possession': away_possession,
                'home_shots': home_shots,
                'away_shots': away_shots,
                'home_shots_on_target': home_shots_on_target,
                'away_shots_on_target': away_shots_on_target,
                'home_corners': home_corners,
                'away_corners': away_corners
            }
            
            processed_data.append(match_data)
            
            # Respetamos los límites de la API
            time.sleep(0.5)
        
        # Creamos un DataFrame con todos los datos
        df = pd.DataFrame(processed_data)
        
        # Guardamos los datos en CSV para uso futuro
        os.makedirs('data', exist_ok=True)
        csv_file = f'data/historical_matches_league_{league_id}_season_{season}.csv'
        df.to_csv(csv_file, index=False)
        logger.info(f"Datos históricos guardados en {csv_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error al obtener datos históricos: {e}")
        return None

def calculate_team_form(df, team_id, num_matches=5):
    """
    Calcula estadísticas de forma para un equipo basado en sus últimos partidos
    
    Args:
        df: DataFrame con los partidos históricos
        team_id: ID del equipo
        num_matches: Número de partidos para calcular la forma
    
    Returns:
        Diccionario con estadísticas de forma
    """
    # Filtramos los partidos del equipo (como local o visitante)
    home_matches = df[df['home_team_id'] == team_id].copy()
    away_matches = df[df['away_team_id'] == team_id].copy()
    
    # Añadimos campos para facilitar el análisis
    home_matches['is_home'] = True
    home_matches['team_goals'] = home_matches['home_goals']
    home_matches['opponent_goals'] = home_matches['away_goals']
    
    away_matches['is_home'] = False
    away_matches['team_goals'] = away_matches['away_goals']
    away_matches['opponent_goals'] = away_matches['home_goals']
    
    # Combinamos todos los partidos del equipo
    team_matches = pd.concat([home_matches, away_matches])
    
    # Ordenamos por fecha
    team_matches['date'] = pd.to_datetime(team_matches['date'])
    team_matches = team_matches.sort_values('date')
    
    # Tomamos los últimos num_matches partidos
    recent_matches = team_matches.tail(num_matches)
    
    if len(recent_matches) == 0:
        return {
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'win_percentage': 0,
            'draw_percentage': 0,
            'loss_percentage': 0,
            'clean_sheets_ratio': 0,
            'form_trend': 0
        }
    
    # Calculamos estadísticas
    total_matches = len(recent_matches)
    goals_scored = recent_matches['team_goals'].sum()
    goals_conceded = recent_matches['opponent_goals'].sum()
    
    wins = sum((recent_matches['is_home'] & (recent_matches['home_goals'] > recent_matches['away_goals'])) | 
               (~recent_matches['is_home'] & (recent_matches['away_goals'] > recent_matches['home_goals'])))
    
    draws = sum(recent_matches['home_goals'] == recent_matches['away_goals'])
    
    losses = total_matches - wins - draws
    
    clean_sheets = sum(recent_matches['opponent_goals'] == 0)
    
    # Calculamos tendencia de forma (ponderando partidos más recientes)
    weights = np.linspace(0.5, 1.0, total_matches)
    
    recent_results = []
    for _, match in recent_matches.iterrows():
        if match['is_home']:
            if match['home_goals'] > match['away_goals']:
                recent_results.append(3)  # Victoria
            elif match['home_goals'] == match['away_goals']:
                recent_results.append(1)  # Empate
            else:
                recent_results.append(0)  # Derrota
        else:
            if match['away_goals'] > match['home_goals']:
                recent_results.append(3)  # Victoria
            elif match['away_goals'] == match['home_goals']:
                recent_results.append(1)  # Empate
            else:
                recent_results.append(0)  # Derrota
    
    # Calculamos tendencia ponderada
    if recent_results:
        weighted_form = np.average(recent_results, weights=weights) / 3.0  # Normalizado a [0,1]
    else:
        weighted_form = 0.5
    
    return {
        'avg_goals_scored': goals_scored / total_matches,
        'avg_goals_conceded': goals_conceded / total_matches,
        'win_percentage': wins / total_matches * 100,
        'draw_percentage': draws / total_matches * 100,
        'loss_percentage': losses / total_matches * 100,
        'clean_sheets_ratio': clean_sheets / total_matches,
        'form_trend': weighted_form - 0.5  # Centrado en 0 (-0.5 a 0.5)
    }

def get_h2h_stats(df, team1_id, team2_id, num_matches=10):
    """
    Obtiene estadísticas de enfrentamientos directos entre dos equipos
    
    Args:
        df: DataFrame con los partidos históricos
        team1_id: ID del primer equipo
        team2_id: ID del segundo equipo
        num_matches: Número máximo de partidos a considerar
    
    Returns:
        Diccionario con estadísticas h2h
    """
    # Filtramos partidos donde se enfrentan estos equipos
    h2h_matches = df[
        ((df['home_team_id'] == team1_id) & (df['away_team_id'] == team2_id)) |
        ((df['home_team_id'] == team2_id) & (df['away_team_id'] == team1_id))
    ]
    
    # Ordenamos por fecha
    h2h_matches['date'] = pd.to_datetime(h2h_matches['date'])
    h2h_matches = h2h_matches.sort_values('date', ascending=False)
    
    # Tomamos los últimos num_matches partidos
    recent_h2h = h2h_matches.head(num_matches)
    
    if len(recent_h2h) == 0:
        return {
            'matches_played': 0,
            'team1_wins': 0,
            'team2_wins': 0,
            'draws': 0,
            'avg_goals_team1': 0,
            'avg_goals_team2': 0,
            'total_goals_avg': 0
        }
    
    # Calculamos estadísticas
    total_matches = len(recent_h2h)
    
    team1_wins = 0
    team2_wins = 0
    draws = 0
    team1_goals = 0
    team2_goals = 0
    
    for _, match in recent_h2h.iterrows():
        if match['home_team_id'] == team1_id:
            if match['home_goals'] > match['away_goals']:
                team1_wins += 1
            elif match['home_goals'] < match['away_goals']:
                team2_wins += 1
            else:
                draws += 1
            
            team1_goals += match['home_goals']
            team2_goals += match['away_goals']
        else:
            if match['home_goals'] > match['away_goals']:
                team2_wins += 1
            elif match['home_goals'] < match['away_goals']:
                team1_wins += 1
            else:
                draws += 1
            
            team1_goals += match['away_goals']
            team2_goals += match['home_goals']
    
    return {
        'matches_played': total_matches,
        'team1_wins': team1_wins,
        'team2_wins': team2_wins,
        'draws': draws,
        'avg_goals_team1': team1_goals / total_matches,
        'avg_goals_team2': team2_goals / total_matches,
        'total_goals_avg': (team1_goals + team2_goals) / total_matches
    }

def prepare_training_data(df):
    """
    Prepara los datos para entrenamiento a partir del DataFrame de partidos históricos
    
    Args:
        df: DataFrame con los partidos históricos
    
    Returns:
        X: Características para el modelo
        y: Targets (goles)
    """
    logger.info("Preparando datos para entrenamiento...")
    
    # Ordenamos por fecha
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Lista para almacenar las características y targets
    X_data = []
    y_data = []
    
    # Para cada partido, calculamos características basadas en el historial previo
    for i, match in tqdm(df.iterrows(), total=len(df), desc="Procesando partidos"):
        # Solo usamos partidos donde tenemos un historial previo
        past_matches = df[df['date'] < match['date']].copy()
        
        if len(past_matches) < 10:  # Necesitamos al menos 10 partidos de historial
            continue
        
        home_team_id = match['home_team_id']
        away_team_id = match['away_team_id']
        
        # Calculamos forma de los equipos
        home_form = calculate_team_form(past_matches, home_team_id)
        away_form = calculate_team_form(past_matches, away_team_id)
        
        # Calculamos estadísticas h2h
        h2h_stats = get_h2h_stats(past_matches, home_team_id, away_team_id)
        
        # Creamos vector de características (14 características en total)
        features = [
            home_form['avg_goals_scored'],
            home_form['avg_goals_conceded'],
            home_form['win_percentage'] / 100.0,  # Normalizado a [0,1]
            home_form['clean_sheets_ratio'],
            
            away_form['avg_goals_scored'],
            away_form['avg_goals_conceded'],
            away_form['win_percentage'] / 100.0,  # Normalizado a [0,1]
            away_form['clean_sheets_ratio'],
            
            (home_form['avg_goals_scored'] + away_form['avg_goals_conceded']) / 2.0,  # Expectativa gol local
            (away_form['avg_goals_scored'] + home_form['avg_goals_conceded']) / 2.0,  # Expectativa gol visitante
            
            h2h_stats['team1_wins'] / max(1, h2h_stats['matches_played']),  # Ratio victorias h2h
            h2h_stats['total_goals_avg'],
            
            home_form['form_trend'] + 0.5,  # Normalizado a [0,1]
            away_form['form_trend'] + 0.5   # Normalizado a [0,1]
        ]
        
        # Target: [goles_local, goles_visitante]
        target = [match['home_goals'], match['away_goals']]
        
        X_data.append(features)
        y_data.append(target)
    
    X = np.array(X_data, dtype=np.float32)
    y = np.array(y_data, dtype=np.float32)
    
    logger.info(f"Datos preparados: {X.shape[0]} muestras con {X.shape[1]} características")
    
    return X, y

def generate_synthetic_data(n_samples=5000):
    """
    Genera datos sintéticos para entrenamiento si no hay datos reales disponibles.
    """
    logger.info(f"Generando {n_samples} muestras de datos sintéticos para entrenamiento...")
    
    # Crear características para el modelo
    # X contiene 14 características como espera nuestro modelo
    X = np.random.normal(0, 1, size=(n_samples, 14))
    
    # Características básicas: estadísticas de equipos
    X[:, 0] = np.random.poisson(1.5, n_samples)  # home_goals_per_match
    X[:, 1] = np.random.poisson(1.2, n_samples)  # home_goals_conceded
    X[:, 2] = np.random.beta(5, 3, n_samples)    # home_win_percentage
    X[:, 3] = np.random.binomial(5, 0.3, n_samples) / 5.0  # home_clean_sheets
    
    X[:, 4] = np.random.poisson(1.2, n_samples)  # away_goals_per_match
    X[:, 5] = np.random.poisson(1.5, n_samples)  # away_goals_conceded
    X[:, 6] = np.random.beta(3, 5, n_samples)    # away_win_percentage
    X[:, 7] = np.random.binomial(5, 0.2, n_samples) / 5.0  # away_clean_sheets
    
    # Características combinadas
    X[:, 8] = (X[:, 0] + X[:, 5]) / 2.0  # Expectativa de gol local
    X[:, 9] = (X[:, 4] + X[:, 1]) / 2.0  # Expectativa de gol visitante
    X[:, 10] = np.random.beta(2, 2, n_samples)  # h2h_ratio
    X[:, 11] = np.random.normal(2.5, 0.5, n_samples)  # h2h_avg_goals
    X[:, 12] = np.random.beta(2, 2, n_samples)  # form_trend_home
    X[:, 13] = np.random.beta(2, 2, n_samples)  # form_trend_away
    
    # Crear targets
    # Y contiene [home_goals, away_goals]
    y = np.zeros((n_samples, 2))
    
    # Factor para añadir algo de ruido para que no sea una relación perfecta
    noise_factor = 0.3
    
    # Calcular targets basados en las características
    # home_goals basado en expectativa y un poco de ruido
    y[:, 0] = np.maximum(0, X[:, 8] * (1 + np.random.normal(0, noise_factor, n_samples)))
    
    # away_goals basado en expectativa y un poco de ruido
    y[:, 1] = np.maximum(0, X[:, 9] * (1 + np.random.normal(0, noise_factor, n_samples)))
    
    logger.info("Datos sintéticos generados exitosamente.")
    return X, y

def main():
    """
    Función principal para reentrenar el modelo y guardarlo.
    """
    # Asegurar que el directorio models existe
    os.makedirs('models', exist_ok=True)
    
    # Ligas principales para obtener datos
    # ID 39: Premier League (Inglaterra)
    # ID 140: La Liga (España)
    # ID 135: Serie A (Italia)
    # ID 78: Bundesliga (Alemania)
    # ID 61: Ligue 1 (Francia)
    leagues_to_fetch = [
        {'id': 39, 'season': 2023},  # Premier League actual
        {'id': 39, 'season': 2022},  # Premier League temporada anterior
        {'id': 140, 'season': 2023}, # La Liga actual
        {'id': 135, 'season': 2023}, # Serie A actual
        {'id': 78, 'season': 2023},  # Bundesliga actual
        {'id': 61, 'season': 2023},  # Ligue 1 actual
    ]
    
    all_data_frames = []
    
    # Intentamos cargar datos ya descargados primero
    for league in leagues_to_fetch:
        csv_file = f'data/historical_matches_league_{league["id"]}_season_{league["season"]}.csv'
        if os.path.exists(csv_file):
            logger.info(f"Cargando datos existentes de {csv_file}")
            df = pd.read_csv(csv_file)
            all_data_frames.append(df)
    
    # Si no hay datos suficientes, obtenemos desde la API
    if len(all_data_frames) < 2:  # Necesitamos al menos datos de 2 ligas
        for league in tqdm(leagues_to_fetch, desc="Obteniendo datos de ligas"):
            df = fetch_historical_matches(league['id'], league['season'])
            if df is not None and len(df) > 0:
                all_data_frames.append(df)
    
    # Combinamos todos los DataFrames
    if all_data_frames:
        logger.info(f"Combinando datos de {len(all_data_frames)} fuentes")
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        
        # Eliminamos duplicados si los hay
        combined_df = combined_df.drop_duplicates(subset=['match_id'])
        
        logger.info(f"Dataset combinado: {len(combined_df)} partidos")
        
        # Preparamos los datos para entrenamiento
        X, y = prepare_training_data(combined_df)
        
        # Guardamos el dataset combinado para referencia
        combined_df.to_csv('data/historical_matches.csv', index=False)
    else:
        # Si no podemos obtener datos reales, usamos datos sintéticos
        logger.warning("No se pudieron obtener datos reales. Generando datos sintéticos.")
        X, y = generate_synthetic_data(5000)
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Escalar características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Guardar el scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    logger.info("Scaler guardado en models/scaler.pkl")
    
    # Crear y entrenar modelo
    logger.info("Creando modelo de red neuronal...")
    model = FeedforwardNeuralNetwork(
        input_dim=14,
        hidden_dims=[128, 64, 32],
        learning_rate=0.001,
        dropout_rate=0.3
    )
    
    logger.info("Entrenando modelo...")
    history = model.train(
        X_train_scaled, y_train,
        X_val=X_test_scaled, y_val=y_test,
        epochs=100,
        batch_size=32,
        model_save_path='models/best_fnn_model.h5'
    )
    
    # Guardar el modelo final
    logger.info("Guardando modelo final...")
    model.model.save('models/fnn_model.h5')
    
    # Guardar también en formato .pkl para compatibilidad
    logger.info("Guardando modelo en formato .pkl para compatibilidad...")
    joblib.dump(model, 'models/fnn_model.pkl')
    
    logger.info("Entrenamiento y guardado completados")
    
    # Evaluar modelo
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    
    # Calcular métricas de error simple
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    logger.info(f"Métricas finales:")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  R²: {r2:.4f}")
    
    # Guardar métricas para referencia
    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_features": 14,
        "model_architecture": [14, 128, 64, 32, 2],
        "samples_count": X.shape[0]
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("Proceso completado exitosamente")

if __name__ == "__main__":
    main()
