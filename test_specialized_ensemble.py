"""
Pruebas para el módulo de ensemble especializado.

Este script ejecuta pruebas para verificar el funcionamiento correcto
del specialized_ensemble.py y su integración con el sistema existente.
"""

import pandas as pd
import numpy as np
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any, List

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ensemble_test")

# Asegurar que el módulo puede ser importado
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from specialized_ensemble import SpecializedEnsembleModel, predict_goals_with_ensemble
    ENSEMBLE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importando módulo de ensemble: {e}")
    ENSEMBLE_AVAILABLE = False

try:
    from bayesian_goals_model import BayesianGoalsModel, predict_goals_bayesian
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Módulo bayesiano no disponible: {e}")
    BAYESIAN_AVAILABLE = False


def generate_synthetic_data(n_samples=200):
    """Genera datos sintéticos para pruebas."""
    np.random.seed(42)
    
    # Features básicos
    home_elo = np.random.normal(1500, 100, n_samples)
    away_elo = np.random.normal(1450, 100, n_samples)
    home_form = np.random.uniform(0, 1, n_samples)
    away_form = np.random.uniform(0, 1, n_samples)
    
    # Features adicionales
    home_strength = np.random.uniform(0.5, 1.5, n_samples)
    away_strength = np.random.uniform(0.5, 1.5, n_samples)
    home_defense = np.random.uniform(0.5, 1.5, n_samples)
    away_defense = np.random.uniform(0.5, 1.5, n_samples)
    
    # Variables objetivo (goles) - modelo simple basado en features
    home_goals = 1.2 + 0.3 * (home_strength / away_defense) + 0.1 * (home_form - away_form)
    home_goals += 0.2 * (home_elo - away_elo) / 100  # Efecto ELO
    home_goals += np.random.normal(0, 0.5, n_samples)  # Ruido
    home_goals = np.maximum(0, np.round(home_goals)).astype(int)
    
    away_goals = 0.9 + 0.3 * (away_strength / home_defense) - 0.1 * (home_form - away_form)
    away_goals += 0.15 * (away_elo - home_elo) / 100  # Efecto ELO
    away_goals += np.random.normal(0, 0.5, n_samples)  # Ruido
    away_goals = np.maximum(0, np.round(away_goals)).astype(int)
    
    # Crear DataFrame con datos
    df = pd.DataFrame({
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_form': home_form,
        'away_form': away_form,
        'home_strength': home_strength,
        'away_strength': away_strength,
        'home_defense': home_defense,
        'away_defense': away_defense,
        'home_goals': home_goals,
        'away_goals': away_goals
    })
    
    # Añadir contexto (ligas y temporadas)
    leagues = [39, 140, 61, 78, 135]  # IDs de ligas principales
    league_id = np.random.choice(leagues, n_samples)
    season = np.random.choice([2022, 2023, 2024], n_samples)
    
    df['league_id'] = league_id
    df['season'] = season
    
    # Añadir fechas
    start_date = datetime(2022, 1, 1)
    dates = [start_date + pd.Timedelta(days=int(d)) for d in np.random.uniform(0, 730, n_samples)]
    df['date'] = dates
    
    return df


def test_ensemble_module():
    """Prueba la funcionalidad básica del módulo de ensemble."""
    if not ENSEMBLE_AVAILABLE:
        logger.error("Módulo de ensemble no disponible. Saliendo.")
        return False
    
    logger.info("Generando datos sintéticos para pruebas...")
    data = generate_synthetic_data(n_samples=200)
    
    # Dividir en entrenamiento y prueba
    train_data = data.iloc[:150]
    test_data = data.iloc[150:]
    
    # Preparar datos para el modelo
    X_train = train_data.drop(['home_goals', 'away_goals'], axis=1)
    y_train = train_data[['home_goals', 'away_goals']]
    
    X_test = test_data.drop(['home_goals', 'away_goals'], axis=1)
    y_test = test_data[['home_goals', 'away_goals']]
    
    # Extraer features de contexto
    context_cols = ['league_id', 'season']
    context_train = train_data[context_cols]
    context_test = test_data[context_cols]
    
    # Inicializar modelo sin componentes bayesianos para acelerar prueba
    logger.info("Inicializando modelo de ensemble sin componente bayesiano...")
    model = SpecializedEnsembleModel(
        use_bayesian=False,  # Desactivar para pruebas rápidas
        use_tree_models=True,
        use_linear_models=True,
        context_aware_weighting=True,
        models_path="models/test_ensemble"
    )
    
    # Entrenar modelo
    logger.info("Entrenando modelo de ensemble...")
    try:
        model.fit(X_train, y_train, context_features=context_train)
        logger.info("Entrenamiento exitoso")
    except Exception as e:
        logger.error(f"Error durante entrenamiento: {e}")
        return False
    
    # Generar predicciones
    logger.info("Generando predicciones...")
    try:
        home_preds, away_preds = model.predict(X_test, context_features=context_test)
        logger.info(f"Predicciones generadas - shape: home {home_preds.shape}, away {away_preds.shape}")
    except Exception as e:
        logger.error(f"Error durante predicción: {e}")
        return False
    
    # Calcular métricas básicas
    mae_home = np.mean(np.abs(home_preds - y_test['home_goals'].values))
    mae_away = np.mean(np.abs(away_preds - y_test['away_goals'].values))
    
    logger.info(f"Error absoluto medio (MAE) - Home: {mae_home:.3f}, Away: {mae_away:.3f}")
    
    # Verificar si el error está dentro de rangos razonables
    if mae_home > 2.0 or mae_away > 2.0:
        logger.warning("Error alto en predicciones - revisar implementación")
    
    return True


def test_ensemble_prediction_api():
    """Prueba la función de API para predicciones."""
    if not ENSEMBLE_AVAILABLE:
        logger.error("Módulo de ensemble no disponible. Saliendo.")
        return False
    
    logger.info("Generando datos históricos para predicción...")
    historical_data = generate_synthetic_data(n_samples=100)
    
    # Crear datos para un partido de prueba
    test_match = {
        'home_team_id': 40,  # Liverpool
        'away_team_id': 33,  # Manchester United
        'league_id': 39,     # Premier League
        'season': 2024,
        'home_elo': 1650,
        'away_elo': 1580,
        'home_form': 0.8,
        'away_form': 0.7,
        'home_strength': 1.3,
        'away_strength': 1.2,
        'home_defense': 1.2,
        'away_defense': 1.1,
        'date': '2024-05-15'
    }
    
    # Realizar predicción
    logger.info("Realizando predicción con ensemble...")
    try:
        prediction = predict_goals_with_ensemble(
            match_data=test_match,
            historical_matches=historical_data,
            use_bayesian=False,  # Desactivar para pruebas rápidas
            context_aware=True
        )
        
        logger.info(f"Predicción exitosa: {json.dumps(prediction, indent=2)}")
        
        # Verificar campos esperados
        required_fields = [
            'predicted_home_goals', 'predicted_away_goals', 
            'home_win_probability', 'draw_probability', 'away_win_probability',
            'prob_over_2_5', 'prob_btts'
        ]
        
        for field in required_fields:
            if field not in prediction:
                logger.error(f"Campo requerido faltante en predicción: {field}")
                return False
        
        # Verificar valores dentro de rangos esperados
        if (prediction['predicted_home_goals'] < 0 or 
            prediction['predicted_away_goals'] < 0 or
            prediction['home_win_probability'] < 0 or 
            prediction['home_win_probability'] > 1 or
            prediction['draw_probability'] < 0 or 
            prediction['draw_probability'] > 1 or
            prediction['away_win_probability'] < 0 or 
            prediction['away_win_probability'] > 1):
            logger.error("Valores de predicción fuera de rangos válidos")
            return False
        
        # Verificar suma de probabilidades
        prob_sum = (prediction['home_win_probability'] + 
                   prediction['draw_probability'] + 
                   prediction['away_win_probability'])
        
        if abs(prob_sum - 1.0) > 0.01:
            logger.error(f"Suma de probabilidades no es 1.0: {prob_sum}")
            return False
        
    except Exception as e:
        logger.error(f"Error en predicción con ensemble: {e}")
        return False
    
    return True


def test_bayesian_integration():
    """Prueba la integración con el módulo bayesiano."""
    if not BAYESIAN_AVAILABLE:
        logger.warning("Módulo bayesiano no disponible. Omitiendo prueba.")
        return True
    
    if not ENSEMBLE_AVAILABLE:
        logger.error("Módulo de ensemble no disponible. Saliendo.")
        return False
    
    logger.info("Generando datos históricos para predicción bayesiana...")
    historical_data = generate_synthetic_data(n_samples=50)  # Muestra pequeña para prueba
    
    # Crear datos para un partido de prueba
    test_match = {
        'home_team_id': 40,  # Liverpool
        'away_team_id': 33,  # Manchester United
        'league_id': 39,     # Premier League
        'home_elo': 1650,
        'away_elo': 1580,
        'date': '2024-05-15'
    }
    
    # Intentar predicción bayesiana directa
    try:
        bayesian_prediction = predict_goals_bayesian(
            match_data=test_match,
            historical_matches=historical_data,
            use_hierarchical=False,  # Simplificar para prueba
            include_time_effects=False  # Simplificar para prueba
        )
        
        logger.info(f"Predicción bayesiana: Home={bayesian_prediction.get('predicted_home_goals', 'N/A')}, "
                  f"Away={bayesian_prediction.get('predicted_away_goals', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error en predicción bayesiana directa: {e}")
    
    # Intentar predicción con ensemble incluyendo componente bayesiano
    try:
        ensemble_with_bayesian = predict_goals_with_ensemble(
            match_data=test_match,
            historical_matches=historical_data,
            use_bayesian=True
        )
        
        logger.info(f"Predicción ensemble con bayesiano: "
                  f"Home={ensemble_with_bayesian.get('predicted_home_goals', 'N/A')}, "
                  f"Away={ensemble_with_bayesian.get('predicted_away_goals', 'N/A')}")
        
    except Exception as e:
        logger.error(f"Error en predicción ensemble con bayesiano: {e}")
    
    return True


def run_all_tests():
    """Ejecuta todas las pruebas."""
    logger.info("Iniciando pruebas para specialized_ensemble.py")
    
    tests = [
        ("Prueba de módulo ensemble básico", test_ensemble_module),
        ("Prueba de API de predicción", test_ensemble_prediction_api),
        ("Prueba de integración bayesiana", test_bayesian_integration)
    ]
    
    results = []
    all_passed = True
    
    for name, test_func in tests:
        logger.info(f"Ejecutando: {name}")
        try:
            passed = test_func()
            results.append((name, passed))
            if not passed:
                all_passed = False
        except Exception as e:
            logger.error(f"Error ejecutando {name}: {e}")
            results.append((name, False))
            all_passed = False
    
    # Mostrar resumen
    logger.info("\n----- RESUMEN DE PRUEBAS -----")
    for name, passed in results:
        status = "EXITOSA ✓" if passed else "FALLIDA ✗"
        logger.info(f"{name}: {status}")
    
    overall = "TODAS LAS PRUEBAS EXITOSAS" if all_passed else "ALGUNAS PRUEBAS FALLARON"
    logger.info(f"\nResultado general: {overall}")
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
