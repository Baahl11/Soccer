"""
Demo de integración de datos de mercado.

Este script demuestra las funcionalidades del módulo market_integration.py,
mostrando cómo se pueden utilizar los datos de mercado para mejorar las predicciones.
"""

import logging
import time
import json
from typing import Dict, Any, List
from market_integration import MarketDataIntegrator, integrate_market_features, create_market_monitor
from predictions import make_global_prediction
from data import get_fixture_data
from odds_analyzer import OddsAnalyzer

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_market_features(fixture_id: int):
    """
    Demuestra la extracción de características de mercado.
    
    Args:
        fixture_id: ID del partido
    """
    print(f"\n=== Demo de Características de Mercado para partido {fixture_id} ===\n")
    
    # Inicializar el integrador
    integrator = MarketDataIntegrator()
    
    # Extraer características
    market_features = integrator.extract_odds_features(fixture_id)
    
    print("Características extraídas del mercado:")
    for feature, value in market_features.items():
        print(f"  {feature}: {value}")
    
    # Simular integración con otras características
    dummy_features = {
        "home_form": 0.75,
        "away_form": 0.68,
        "home_goals_scored_avg": 1.5,
        "away_goals_scored_avg": 1.2
    }
    
    enriched_features = integrate_market_features(dummy_features, fixture_id)
    
    print("\nCaracterísticas combinadas para el modelo:")
    for feature, value in enriched_features.items():
        if feature in market_features:
            print(f"  {feature}: {value} (MERCADO)")
        else:
            print(f"  {feature}: {value}")

def demo_calibration(fixture_id: int):
    """
    Demuestra la calibración basada en movimientos del mercado.
    
    Args:
        fixture_id: ID del partido
    """
    print(f"\n=== Demo de Calibración con Mercado para partido {fixture_id} ===\n")
      # Obtener predicción base
    fixture_data = get_fixture_data(fixture_id)
    if not fixture_data:
        print(f"No se encontraron datos para el partido {fixture_id}")
        return
    
    base_prediction = make_global_prediction(fixture_id)
    print("Predicción original:")
    print(f"  Home: {base_prediction.get('home_team')} - {base_prediction.get('prob_home_win', 0):.3f}")
    print(f"  Draw: {base_prediction.get('prob_draw', 0):.3f}")
    print(f"  Away: {base_prediction.get('away_team')} - {base_prediction.get('prob_away_win', 0):.3f}")
    print(f"  Expected goals: {base_prediction.get('expected_goals', 0):.2f}")
    
    # Calibrar con movimientos
    integrator = MarketDataIntegrator()
    calibrated = integrator.calibrate_prediction_with_movements(base_prediction, fixture_id)
    
    print("\nCalibración estándar con odds:")
    print(f"  Home: {calibrated.get('home_team')} - {calibrated.get('prob_home_win', 0):.3f}")
    print(f"  Draw: {calibrated.get('prob_draw', 0):.3f}")
    print(f"  Away: {calibrated.get('away_team')} - {calibrated.get('prob_away_win', 0):.3f}")
    print(f"  Expected goals: {calibrated.get('expected_goals', 0):.2f}")
    
    if "market_calibration" in calibrated and "movement_adjustment" in calibrated["market_calibration"]:
        print("\nAjustes por movimientos de mercado:")
        adjustment = calibrated["market_calibration"]["movement_adjustment"]
        if adjustment.get("significant_movements", False):
            original = adjustment.get("original_calibrated", {})
            print(f"  Confianza del mercado: {adjustment.get('market_confidence', 0):.2f}")
            print(f"  Cambio en home: {calibrated.get('prob_home_win', 0) - original.get('prob_home_win', 0):.3f}")
            print(f"  Cambio en draw: {calibrated.get('prob_draw', 0) - original.get('prob_draw', 0):.3f}")
            print(f"  Cambio en away: {calibrated.get('prob_away_win', 0) - original.get('prob_away_win', 0):.3f}")
        else:
            print("  No se detectaron movimientos significativos.")

def demo_movement_monitor(fixture_ids: List[int]):
    """
    Demuestra el monitor de movimientos significativos.
    
    Args:
        fixture_ids: Lista de IDs de partidos a monitorear
    """
    print(f"\n=== Demo de Monitor de Movimientos para {len(fixture_ids)} partidos ===\n")
    
    # Crear monitor
    monitor_results = create_market_monitor(fixture_ids)
    
    if not monitor_results:
        print("No se detectaron movimientos significativos en ningún partido.")
    else:
        print(f"Se detectaron movimientos significativos en {len(monitor_results)} partidos:")
        
        for fixture_id, data in monitor_results.items():
            print(f"\nPartido {fixture_id}:")
            
            movements = data.get("movements", [])
            for movement in movements[:3]:  # Mostrar solo los 3 primeros movimientos
                market = movement.get("market", "Unknown")
                selection = movement.get("selection", "Unknown")
                change = movement.get("change", 0)
                trend = movement.get("trend", "Unknown")
                
                direction = "⬇️" if trend == "decreasing" else "⬆️"
                print(f"  {market} - {selection}: {direction} {abs(change)*100:.1f}%")
            
            implications = data.get("implications", [])
            if implications:
                print("  Implicaciones:")
                for imp in implications:
                    print(f"    - {imp.get('description', '')}")

def demo_full_integration(fixture_id: int):
    """
    Demuestra la integración completa con datos de mercado.
    
    Args:
        fixture_id: ID del partido
    """
    print(f"\n=== Demo de Integración Completa para partido {fixture_id} ===\n")
      # Obtener predicción base
    fixture_data = get_fixture_data(fixture_id)
    if not fixture_data:
        print(f"No se encontraron datos para el partido {fixture_id}")
        return
    
    # Obtener predicción base
    base_prediction = make_global_prediction(fixture_id)
    
    # Enriquecer con datos de mercado
    integrator = MarketDataIntegrator()
    enriched = integrator.enrich_prediction_with_market_data(base_prediction, fixture_id)
    
    # Mostrar diferencias clave
    print(f"Partido: {enriched.get('home_team')} vs {enriched.get('away_team')}")
    
    print("\nProbabilidades Originales vs Calibradas:")
    print(f"  Home: {base_prediction.get('prob_home_win', 0):.3f} -> {enriched.get('prob_home_win', 0):.3f}")
    print(f"  Draw: {base_prediction.get('prob_draw', 0):.3f} -> {enriched.get('prob_draw', 0):.3f}")
    print(f"  Away: {base_prediction.get('prob_away_win', 0):.3f} -> {enriched.get('prob_away_win', 0):.3f}")
    
    # Mostrar análisis de mercado
    market_data = enriched.get("market_data", {})
    if market_data:
        analysis = market_data.get("analysis", {})
        print("\nAnálisis de Mercado:")
        print(f"  Eficiencia: {analysis.get('efficiency', 0):.2%}")
        print(f"  Margen: {analysis.get('margin', 0):.2%}")
        print(f"  Confianza: {analysis.get('confidence', 0):.2f}")
        
        movements = market_data.get("movements", {})
        if movements.get("detected", False):
            print("\nMovimientos Detectados:")
            for market in movements.get("significant_markets", []):
                print(f"  - {market}")
    
    # Mostrar oportunidades de valor
    value_opps = enriched.get("value_opportunities", {})
    if value_opps:
        print("\nOportunidades de Valor:")
        for market, opportunities in value_opps.items():
            print(f"  {market.upper()}:")
            for outcome, data in opportunities.items():
                edge = data.get("edge", 0)
                if abs(edge) >= 2.0:
                    direction = "POSITIVO" if edge > 0 else "NEGATIVO"
                    print(f"    {outcome}: Edge {edge:.2f}% ({direction})")
                    print(f"      Nuestra prob: {data.get('our_prob', 0):.2f}")
                    print(f"      Odds: {data.get('market_odds', 0):.2f}")

if __name__ == "__main__":
    print("Iniciando Demo de Integración de Datos de Mercado...")
    
    # Usar un fixture_id de ejemplo (reemplazar con un ID válido)
    fixture_id = 1208825  # Ejemplo: Valladolid vs Alaves
    
    # Ejecutar demos individuales
    demo_market_features(fixture_id)
    demo_calibration(fixture_id)
    
    # Demo de monitor con varios partidos
    fixture_ids = [1208825, 1208826, 1208827, 1208828, 1208829]
    demo_movement_monitor(fixture_ids)
    
    # Demo de integración completa
    demo_full_integration(fixture_id)
    
    print("\nDemo completada.")
