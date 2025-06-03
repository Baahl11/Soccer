#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Football API Integration Example

Este script demuestra cómo utilizar la integración de la API de Fútbol
con el modelo de predicción FNN para obtener predicciones mejoradas
basadas en datos en tiempo real.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging
import json
from typing import Dict, Any

# Importar nuestros módulos
from fnn_model import FeedforwardNeuralNetwork, demonstrate_api_prediction
from config import API_KEY, MODEL_PATH

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para demostrar la integración con la API de fútbol."""
    
    # Verificar si tenemos una API key configurada
    api_key = API_KEY if API_KEY != "your-api-key-here" else os.environ.get("FOOTBALL_API_KEY")
    
    if not api_key:
        logger.error("No se encontró API key para la API de fútbol.")
        logger.info("Por favor configura la variable de entorno FOOTBALL_API_KEY o actualiza config.py")
        return
    
    # Datos de ejemplo para la demostración
    # Estos IDs corresponden a equipos y ligas en la API de fútbol
    # Deberás reemplazarlos con IDs válidos para tu caso de uso
    home_team_id = 33      # Barcelona (ejemplo)
    away_team_id = 541     # Real Madrid (ejemplo)
    league_id = 140        # La Liga (ejemplo)
    season = "2024"        # Temporada actual
    
    # Características de ejemplo para la predicción base
    # Estas deberían ser las características que tu modelo espera
    # (estadísticas de equipos, forma reciente, etc.)
    example_features = np.array([
        [0.65, 0.72, 12, 4, 3, 2, 1.8, 0.9, 2.1, 1.1, 0.6, 0.4, 5, 3]
    ])
    
    # Ruta al modelo (ajustar según tus necesidades)
    model_path = "models/fnn_model.h5"
    
    logger.info("Demostrando predicción con datos de API para Barcelona vs Real Madrid")
    
    try:
        # Realizar predicción enriquecida con API
        prediction = demonstrate_api_prediction(
            model_path=model_path,
            features=example_features,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            season=season,
            api_key=api_key
        )
        
        # Mostrar resultados
        print("\n===== PREDICCIÓN ENRIQUECIDA CON API =====")
        print(f"Partido: Barcelona vs Real Madrid")
        print(f"Temporada: {season}\n")
        
        print(f"Probabilidad victoria local: {prediction['home_win_probability']:.2%}")
        print(f"Probabilidad empate: {prediction['draw_probability']:.2%}")
        print(f"Probabilidad victoria visitante: {prediction['away_win_probability']:.2%}")
        
        print(f"\nGoles esperados local: {prediction['home_goals_expected']:.2f}")
        print(f"Goles esperados visitante: {prediction['away_goals_expected']:.2f}")
        
        print(f"\nResultado más probable: {prediction['most_likely_score']}")
        
        # Mostrar datos de la API que enriquecieron la predicción
        print("\n===== DATOS DE LA API UTILIZADOS =====")
        if 'api_data' in prediction:
            # Datos del equipo local
            home_team = prediction['api_data'].get('home_team', {})
            print("\nDatos equipo local:")
            print(f"  - Forma reciente: {home_team.get('form', 'N/A')}")
            print(f"  - Promedio goles marcados: {home_team.get('avg_goals_scored', 0):.2f}")
            print(f"  - Promedio goles concedidos: {home_team.get('avg_goals_conceded', 0):.2f}")
            print(f"  - Ventaja como local: {home_team.get('home_advantage', 1):.2f}")
            
            # Datos del equipo visitante
            away_team = prediction['api_data'].get('away_team', {})
            print("\nDatos equipo visitante:")
            print(f"  - Forma reciente: {away_team.get('form', 'N/A')}")
            print(f"  - Promedio goles marcados: {away_team.get('avg_goals_scored', 0):.2f}")
            print(f"  - Promedio goles concedidos: {away_team.get('avg_goals_conceded', 0):.2f}")
            print(f"  - Rendimiento como visitante: {away_team.get('away_performance', 1):.2f}")
            
            # Datos head-to-head
            h2h = prediction['api_data'].get('head_to_head', {})
            print("\nDatos head-to-head:")
            print(f"  - Partidos totales: {h2h.get('total_matches', 0)}")
            print(f"  - Victorias local: {h2h.get('home_wins', 0)}")
            print(f"  - Empates: {h2h.get('draws', 0)}")
            print(f"  - Victorias visitante: {h2h.get('away_wins', 0)}")
            print(f"  - Dominancia H2H: {h2h.get('h2h_dominance', 0):.2f}")
        else:
            print("No se encontraron datos de la API en la predicción.")
            
    except Exception as e:
        logger.error(f"Error al realizar predicción con API: {e}")
        import traceback
        traceback.print_exc()
        
    print("\nNota: Para usar este script con datos reales:")
    print("1. Asegúrate de tener una API key válida configurada")
    print("2. Reemplaza los IDs de equipos y liga con los correctos")
    print("3. Proporciona características reales para tu modelo base")
    print("4. Verifica que la ruta al modelo sea correcta")

if __name__ == "__main__":
    main()