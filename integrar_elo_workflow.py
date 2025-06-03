"""
Script para integrar el flujo de trabajo de ELO (elo_prediction_workflow.py) 
con el endpoint /api/upcoming_predictions de la API.

Este script muestra cómo ejecutar el flujo de trabajo ELO automáticamente
cuando se soliciten predicciones a través de la API.
"""

import requests
import webbrowser
import json
import time
from pathlib import Path

# Asegurarse de que el directorio de resultados existe
results_dir = Path("elo_prediction_results")
results_dir.mkdir(exist_ok=True)

# URL base para la API
API_BASE_URL = "http://127.0.0.1:5000"

def obtener_predicciones_con_visualizaciones(league_id=39, season=2025, limit=10, pretty=1):
    """
    Obtiene predicciones para próximos partidos e incluye visualizaciones generadas 
    por el flujo de trabajo ELO.
    
    Args:
        league_id: ID de la liga (default: 39 - Premier League)
        season: Temporada (default: 2025)
        limit: Número máximo de partidos a predecir (default: 10)
        pretty: Si se debe formatear la respuesta JSON (default: 1)
        
    Returns:
        Respuesta JSON con predicciones y URLs a visualizaciones
    """
    # Construir la URL con los parámetros
    url = f"{API_BASE_URL}/api/upcoming_predictions"
    params = {
        "league_id": league_id,
        "season": season,
        "limit": limit,
        "pretty": pretty,
        "visualizations": "true"  # Activar la generación de visualizaciones
    }
    
    print(f"Solicitando predicciones para la liga {league_id}, temporada {season}...")
    
    # Realizar la solicitud
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        print(f"Predicciones obtenidas correctamente. Estado: {response.status_code}")
        
        # Analizar la respuesta JSON
        data = response.json()
        
        # Mostrar detalles de las predicciones
        if "match_predictions" in data:
            predictions = data["match_predictions"]
            print(f"\nSe obtuvieron {len(predictions)} predicciones:")
            
            for i, pred in enumerate(predictions):
                home = pred.get("home_team", "Desconocido")
                away = pred.get("away_team", "Desconocido")
                date = pred.get("date", "Fecha desconocida")
                
                print(f"\n{i+1}. {home} vs {away} ({date})")
                
                # Mostrar si hay URLs de visualización disponibles
                if "visualization_url" in pred:
                    vis_url = f"{API_BASE_URL}{pred['visualization_url']}"
                    report_url = f"{API_BASE_URL}{pred['report_url']}"
                    
                    print(f"   - Visualización: {vis_url}")
                    print(f"   - Informe: {report_url}")
                    
                    # Opcionalmente, abrir la visualización en el navegador
                    if i == 0:  # Solo abrimos la primera para no saturar
                        webbrowser.open(vis_url)
                        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error al realizar la solicitud: {e}")
        return None

def main():
    # Ejecutar con parámetros predeterminados (Premier League)
    result = obtener_predicciones_con_visualizaciones()
    
    # Guardar el resultado en un archivo para referencia
    if result:
        with open("integrated_predictions.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("\nResultados guardados en 'integrated_predictions.json'")
    
    # Mostrar cómo acceder a visualizaciones desde otras ligas
    print("\nPara obtener predicciones de otras ligas, puede usar:")
    print("- La Liga (España): league_id=140")
    print("- Serie A (Italia): league_id=135")
    print("- Bundesliga (Alemania): league_id=78")
    print("- Ligue 1 (Francia): league_id=61")
    print("- Champions League: league_id=2")

if __name__ == "__main__":
    main()
