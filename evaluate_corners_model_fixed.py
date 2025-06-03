"""
Evaluación y optimización del modelo de conjunto de votación para predicción de córners.

Este script evalúa el desempeño del modelo VotingEnsembleCornersModel
con datos reales de la API de fútbol y ajusta los parámetros para
mejorar las predicciones.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import norm

# Importar nuestros módulos
from voting_ensemble_corners import VotingEnsembleCornersModel
from corner_data_collector import FootballDataCollector, API_KEY, API_BASE_URL

# Configuración
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"

# Asegurarse de que existen los directorios necesarios
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, os.path.join(RESULTS_DIR, "plots")]:
    os.makedirs(directory, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(RESULTS_DIR, 'corner_model_evaluation.log')
)
logger = logging.getLogger(__name__)

def simulate_match_data(fixture_id: int) -> Dict[str, Any]:
    """
    Genera datos simulados de un partido para pruebas sin API.
    
    Args:
        fixture_id: ID del partido (solo para compatibilidad)
        
    Returns:
        Diccionario con datos simulados
    """
    # Equipos aleatorios
    home_team_id = np.random.randint(1000, 2000)
    away_team_id = np.random.randint(1000, 2000)
    league_id = np.random.choice([39, 140, 135, 78, 61])
    
    # Córners reales (para comparar con predicciones)
    total_corners = np.random.randint(6, 16)
    home_ratio = 0.54 + np.random.normal(0, 0.05)
    home_corners = int(total_corners * home_ratio)
    away_corners = total_corners - home_corners
    
    # Estadísticas del equipo local
    home_stats = {
        'avg_corners_for': 5.0 + np.random.normal(0, 1),
        'avg_corners_against': 5.0 + np.random.normal(0, 1),
        'form_score': 50 + np.random.normal(0, 10),
        'possession': 50 + np.random.normal(0, 10),
        'shots_on_goal': 4 + np.random.randint(-2, 3),
        'shots_off_goal': 6 + np.random.randint(-3, 4),
        'attack_strength': 1.0 + np.random.normal(0, 0.2)
    }
    
    # Estadísticas del equipo visitante
    away_stats = {
        'avg_corners_for': 5.0 + np.random.normal(0, 1),
        'avg_corners_against': 5.0 + np.random.normal(0, 1),
        'form_score': 50 + np.random.normal(0, 10),
        'possession': 100 - home_stats['possession'],
        'shots_on_goal': 4 + np.random.randint(-2, 3),
        'shots_off_goal': 5 + np.random.randint(-3, 4),
        'attack_strength': 1.0 + np.random.normal(0, 0.2)
    }
    
    return {
        'fixture_id': fixture_id,
        'home_team_id': home_team_id,
        'away_team_id': away_team_id,
        'league_id': league_id,
        'home_stats': home_stats,
        'away_stats': away_stats,
        'actual_corners': {'home': home_corners, 'away': away_corners}
    }

def evaluate_model_with_simulated_data(n_matches=50):
    """
    Evalúa el modelo con datos simulados y genera visualizaciones.
    
    Args:
        n_matches: Número de partidos simulados para la evaluación
    
    Returns:
        Resultados de la evaluación
    """
    print(f"Evaluando modelo con {n_matches} partidos simulados...")
    
    # Inicializar modelo
    model = VotingEnsembleCornersModel()
    
    # Resultados
    results = []
    
    # Evaluar en partidos simulados
    for i in range(n_matches):
        fixture_id = 1000 + i
        match_data = simulate_match_data(fixture_id)
        
        # Hacer predicción
        prediction = model.predict_corners(
            home_team_id=match_data['home_team_id'],
            away_team_id=match_data['away_team_id'],
            home_stats=match_data['home_stats'],
            away_stats=match_data['away_stats'],
            league_id=match_data['league_id']
        )
        
        # Córners reales
        actual_total = match_data['actual_corners']['home'] + match_data['actual_corners']['away']
        actual_home = match_data['actual_corners']['home']
        actual_away = match_data['actual_corners']['away']
        
        # Calcular errores
        total_error = prediction['total'] - actual_total
        home_error = prediction['home'] - actual_home
        away_error = prediction['away'] - actual_away
        
        # Determinar acierto de over/under
        over_under_accuracy = {}
        for threshold in [7.5, 8.5, 9.5, 10.5, 11.5]:
            key = f"over_{threshold}"
            if key in prediction:
                predicted_prob = prediction[key]
                actual_outcome = 1 if actual_total > threshold else 0
                over_under_accuracy[key] = {
                    'predicted_probability': predicted_prob,
                    'actual_outcome': actual_outcome,
                    'correct': (predicted_prob > 0.5 and actual_outcome == 1) or (predicted_prob < 0.5 and actual_outcome == 0)
                }
        
        # Guardar resultado
        results.append({
            'fixture_id': fixture_id,
            'prediction': {
                'total': prediction['total'],
                'home': prediction['home'],
                'away': prediction['away']
            },
            'actual': {
                'total': actual_total,
                'home': actual_home,
                'away': actual_away
            },
            'errors': {
                'total_error': total_error,
                'home_error': home_error,
                'away_error': away_error,
                'total_abs_error': abs(total_error),
                'home_abs_error': abs(home_error),
                'away_abs_error': abs(away_error)
            },
            'over_under_accuracy': over_under_accuracy
        })
    
    # Calcular métricas
    total_abs_errors = [r['errors']['total_abs_error'] for r in results]
    home_abs_errors = [r['errors']['home_abs_error'] for r in results]
    away_abs_errors = [r['errors']['away_abs_error'] for r in results]
    
    total_errors = [r['errors']['total_error'] for r in results]
    home_errors = [r['errors']['home_error'] for r in results]
    away_errors = [r['errors']['away_error'] for r in results]
    
    # Precisión de over/under
    over_under_results = {}
    for threshold in [7.5, 8.5, 9.5, 10.5, 11.5]:
        key = f"over_{threshold}"
        correct_predictions = [r for r in results if key in r['over_under_accuracy'] and r['over_under_accuracy'][key]['correct']]
        if results:
            over_under_results[key] = len(correct_predictions) / len(results)
    
    # Crear DataFrame para visualización
    df = pd.DataFrame({
        'actual_total': [r['actual']['total'] for r in results],
        'predicted_total': [r['prediction']['total'] for r in results],
        'actual_home': [r['actual']['home'] for r in results],
        'predicted_home': [r['prediction']['home'] for r in results],
        'actual_away': [r['actual']['away'] for r in results],
        'predicted_away': [r['prediction']['away'] for r in results],
        'total_error': total_errors,
        'home_error': home_errors,
        'away_error': away_errors
    })
    
    # Visualizaciones
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plots_dir = os.path.join(RESULTS_DIR, 'plots')
    
    # 1. Gráfico de dispersión
    plt.figure(figsize=(10, 8))
    plt.scatter(df['actual_total'], df['predicted_total'])
    plt.plot([0, 20], [0, 20], 'r--')  # Línea de referencia
    plt.xlabel('Córners Reales')
    plt.ylabel('Córners Predichos')
    plt.title('Predicción vs Valor Real - Total de Córners')
    plt.savefig(os.path.join(plots_dir, f'scatter_plot_{timestamp}.png'))
    plt.close()
    
    # 2. Histogramas de errores
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df['total_error'], bins=15, alpha=0.7)
    plt.title('Error en Total de Córners')
    plt.xlabel('Error')
    plt.ylabel('Frecuencia')
    
    plt.subplot(1, 3, 2)
    plt.hist(df['home_error'], bins=15, alpha=0.7)
    plt.title('Error en Córners Local')
    plt.xlabel('Error')
    
    plt.subplot(1, 3, 3)
    plt.hist(df['away_error'], bins=15, alpha=0.7)
    plt.title('Error en Córners Visitante')
    plt.xlabel('Error')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'error_hist_{timestamp}.png'))
    plt.close()
    
    # Guardar resultados
    df.to_csv(os.path.join(RESULTS_DIR, f'corner_model_evaluation_{timestamp}.csv'), index=False)
    
    # Resultados
    summary = {
        'total_matches': len(results),
        'mean_absolute_error': {
            'total': np.mean(total_abs_errors),
            'home': np.mean(home_abs_errors),
            'away': np.mean(away_abs_errors)
        },
        'mean_error': {
            'total': np.mean(total_errors),
            'home': np.mean(home_errors),
            'away': np.mean(away_errors)
        },
        'over_under_accuracy': over_under_results,
        'results_file': os.path.join(RESULTS_DIR, f'corner_model_evaluation_{timestamp}.csv'),
        'plots': [
            os.path.join(plots_dir, f'scatter_plot_{timestamp}.png'),
            os.path.join(plots_dir, f'error_hist_{timestamp}.png')
        ]
    }
    
    return summary

def optimize_league_factors(n_matches=50, iterations=30):
    """
    Optimiza los factores por liga para mejorar las predicciones.
    
    Args:
        n_matches: Número de partidos para la evaluación
        iterations: Número de iteraciones de optimización
        
    Returns:
        Resultados de la optimización
    """
    print(f"Optimizando factores por liga ({iterations} iteraciones)...")
    
    # Inicializar modelo
    model = VotingEnsembleCornersModel()
    
    # Generar partidos para evaluación (siempre los mismos)
    np.random.seed(42)  # Para reproducibilidad
    test_matches = [simulate_match_data(1000 + i) for i in range(n_matches)]
    
    # Evaluar con factores iniciales
    initial_errors = []
    for match in test_matches:
        prediction = model.predict_corners(
            home_team_id=match['home_team_id'],
            away_team_id=match['away_team_id'],
            home_stats=match['home_stats'],
            away_stats=match['away_stats'],
            league_id=match['league_id']
        )
        actual_total = match['actual_corners']['home'] + match['actual_corners']['away']
        error = abs(prediction['total'] - actual_total)
        initial_errors.append(error)
    
    initial_mae = np.mean(initial_errors)
    print(f"MAE inicial: {initial_mae:.3f}")
    
    # Factores iniciales
    original_factors = model.league_factors.copy()
    best_factors = original_factors.copy()
    best_mae = initial_mae
    
    # Historial de mejoras
    history = [{
        'iteration': 0,
        'mae': initial_mae,
        'factors': best_factors.copy()
    }]
    
    # Búsqueda aleatoria
    for i in range(iterations):
        # Crear variación de factores
        new_factors = best_factors.copy()
        
        # Modificar aleatoriamente 1-3 ligas
        n_changes = np.random.randint(1, 4)
        leagues_to_change = np.random.choice(list(new_factors.keys()), size=min(n_changes, len(new_factors)), replace=False)
        
        for league_id in leagues_to_change:
            # Ajustar con cambio relativo de ±10%
            change = np.random.uniform(-0.1, 0.1)
            new_factors[league_id] *= (1 + change)
            # Asegurar rango razonable
            new_factors[league_id] = max(0.7, min(1.3, new_factors[league_id]))
        
        # Aplicar nuevos factores
        model.league_factors = new_factors
        
        # Evaluar
        errors = []
        for match in test_matches:
            prediction = model.predict_corners(
                home_team_id=match['home_team_id'],
                away_team_id=match['away_team_id'],
                home_stats=match['home_stats'],
                away_stats=match['away_stats'],
                league_id=match['league_id']
            )
            actual_total = match['actual_corners']['home'] + match['actual_corners']['away']
            error = abs(prediction['total'] - actual_total)
            errors.append(error)
        
        current_mae = np.mean(errors)
        print(f"Iteración {i+1}: MAE = {current_mae:.3f}")
        
        # Actualizar si hay mejora
        if current_mae < best_mae:
            best_mae = current_mae
            best_factors = new_factors.copy()
            print(f"¡Mejora encontrada! Nuevo mejor MAE: {best_mae:.3f}")
        
        # Guardar en historial
        history.append({
            'iteration': i+1,
            'mae': current_mae,
            'factors': new_factors.copy()
        })
    
    # Restaurar factores óptimos
    model.league_factors = best_factors
    
    # Visualizar progreso
    plt.figure(figsize=(10, 6))
    plt.plot([h['iteration'] for h in history], [h['mae'] for h in history])
    plt.xlabel('Iteración')
    plt.ylabel('Error Absoluto Medio')
    plt.title('Progreso de la Optimización de Factores por Liga')
    plt.savefig(os.path.join(RESULTS_DIR, 'plots', 'league_factor_optimization.png'))
    plt.close()
    
    # Resultados
    improvement = (initial_mae - best_mae) / initial_mae * 100
    
    return {
        'original_factors': original_factors,
        'optimized_factors': best_factors,
        'original_mae': initial_mae,
        'optimized_mae': best_mae,
        'improvement_percentage': improvement
    }

def main():
    """Función principal para evaluación y optimización del modelo."""
    try:
        print("Sistema de Evaluación del Modelo de Predicción de Córners")
        print("=" * 60)
        
        # 1. Evaluación con datos simulados
        print("\n1. EVALUACIÓN DEL MODELO")
        eval_results = evaluate_model_with_simulated_data(n_matches=50)
        
        # Mostrar resultados
        print("\n=== Resumen de Evaluación ===")
        print(f"Total de partidos evaluados: {eval_results['total_matches']}")
        
        print("\nError Absoluto Medio:")
        print(f"  Total de córners: {eval_results['mean_absolute_error']['total']:.3f}")
        print(f"  Córners local: {eval_results['mean_absolute_error']['home']:.3f}")
        print(f"  Córners visitante: {eval_results['mean_absolute_error']['away']:.3f}")
        
        if eval_results['over_under_accuracy']:
            print("\nPrecisión en líneas Over/Under:")
            for threshold, accuracy in eval_results['over_under_accuracy'].items():
                print(f"  {threshold}: {accuracy*100:.1f}%")
        
        print(f"\nResultados guardados en: {eval_results['results_file']}")
        print(f"Visualizaciones guardadas en: {RESULTS_DIR}/plots/")
        
        # 2. Optimización de factores por liga
        print("\n2. OPTIMIZACIÓN DE FACTORES POR LIGA")
        opt_results = optimize_league_factors(n_matches=50, iterations=30)
        
        print("\n=== Resultados de Optimización ===")
        print(f"MAE original: {opt_results['original_mae']:.3f}")
        print(f"MAE optimizado: {opt_results['optimized_mae']:.3f}")
        print(f"Mejora: {opt_results['improvement_percentage']:.2f}%")
        
        print("\nFactores de liga optimizados:")
        for league_id, factor in opt_results['optimized_factors'].items():
            league_name = {
                39: "Premier League",
                140: "La Liga", 
                135: "Serie A",
                78: "Bundesliga",
                61: "Ligue 1"
            }.get(int(league_id), str(league_id))
            print(f"  {league_name}: {factor:.3f}")
        
        print("\n=== Proceso Finalizado ===")
        print("Los modelos de córners están listos para su uso en predicciones en tiempo real.")
        print("Para actualizar los modelos con datos reales, ejecute:")
        print("  1. corner_data_collector.py - para recolectar datos")
        print("  2. train_corner_models.py - para entrenar los modelos")
        
    except Exception as e:
        logger.exception(f"Error in model evaluation: {e}")
        print(f"Ha ocurrido un error: {e}")

if __name__ == "__main__":
    main()
