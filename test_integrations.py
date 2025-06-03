"""
Test de integración para las mejoras de predicción de fútbol

Este script prueba todas las mejoras implementadas:
1. El validador de coherencia entre ratings Elo y predicciones de goles
2. El analizador táctico mejorado para cualquier liga
3. El sistema de obtención de clima preciso por ciudad
4. El validador de datos de partidos para detectar anomalías

Autor: Equipo de Desarrollo
Fecha: Mayo 25, 2025
"""

import logging
import json
from typing import Dict, Any, List
from datetime import datetime
import os
import sys
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("integration_test")

# Importar componentes
from prediction_coherence_validator import CoherenceValidator
from enhanced_tactical_analyzer import EnhancedTacticalAnalyzer
from weather_api import get_weather_forecast, get_precise_location_weather
from match_data_validator import MatchDataValidator, validate_matches
from elo_prediction_workflow import ELOEnhancedPredictionWorkflow
from prediction_integration import make_integrated_prediction

def test_coherence_validator() -> bool:
    """Prueba el validador de coherencia entre Elo y predicciones"""
    logger.info("=== Probando validador de coherencia ===")
    
    # Crear validador
    validator = CoherenceValidator()
    
    # Caso de prueba 1: Predicción coherente
    prediction1 = {
        'home_xg': 1.7,
        'away_xg': 1.2,
        'home_win_probability': 0.45,
        'away_win_probability': 0.25,
        'draw_probability': 0.30
    }
    
    elo_ratings1 = {
        'home': 1600,
        'away': 1500,
        'elo_diff': 100
    }
    
    is_coherent1 = validator.is_prediction_coherent_with_elo(prediction1, elo_ratings1)
    logger.info(f"Predicción coherente: {is_coherent1}")
    
    # Caso de prueba 2: Predicción incoherente
    prediction2 = {
        'home_xg': 0.8,
        'away_xg': 2.1,
        'home_win_probability': 0.55,
        'away_win_probability': 0.25,
        'draw_probability': 0.20
    }
    
    elo_ratings2 = {
        'home': 1650,
        'away': 1450,
        'elo_diff': 200
    }
    
    is_coherent2 = validator.is_prediction_coherent_with_elo(prediction2, elo_ratings2)
    logger.info(f"Predicción incoherente: {is_coherent2}")
    
    adjusted = None
    if not is_coherent2:
        adjusted = validator.validate_and_adjust_goal_predictions(prediction2, elo_ratings2)
        logger.info(f"Predicción ajustada: {adjusted}")
    
    return is_coherent1 and not is_coherent2 and adjusted is not None

def test_tactical_analyzer() -> bool:
    """Prueba el analizador táctico mejorado"""
    logger.info("=== Probando analizador táctico ===")
    
    # Crear analizador
    analyzer = EnhancedTacticalAnalyzer()
    
    # Probar con diferentes ligas
    leagues_to_test = [
        # Liga importante
        {"league_id": 39, "name": "Premier League", "team_id": 33, "team_name": "Manchester United"},
        # Liga secundaria europea
        {"league_id": 179, "name": "Superliga (Denmark)", "team_id": 400, "team_name": "Brondby"},
        # Liga latinoamericana
        {"league_id": 128, "name": "Peruvian Primera División", "team_id": 1147, "team_name": "Alianza Lima"},
        # Liga asiática
        {"league_id": 141, "name": "K-League 1", "team_id": 2756, "team_name": "FC Seoul"}
    ]
    
    success_count = 0
    
    for league in leagues_to_test:
        try:
            # Generar perfil táctico
            profile = analyzer.get_team_tactical_profile(
                team_id=league["team_id"],
                team_name=league["team_name"]
            )
            
            # Verificar contenido mínimo
            if (profile and 
                'possession_style' in profile and 
                'defensive_style' in profile and
                'offensive_style' in profile):
                success_count += 1
                
            logger.info(f"Perfil táctico para {league['name']} ({league['team_name']}):")
            logger.info(f"Posesión: {profile.get('possession_style', {}).get('name')}")
            logger.info(f"Defensa: {profile.get('defensive_style', {}).get('name')}")
            logger.info(f"Ataque: {profile.get('offensive_style', {}).get('name')}")
            
            # Simular otro equipo para matchup
            other_profile = analyzer.get_team_tactical_profile(
                team_id=league["team_id"] + 10,
                team_name="Opponent Team"
            )
            
            # Analizar enfrentamiento
            matchup = analyzer.analyze_tactical_matchup(profile, other_profile)
            logger.info(f"Análisis de enfrentamiento generado: {bool(matchup)}")
            
        except Exception as e:
            logger.error(f"Error con liga {league['name']}: {e}")
    
    return success_count == len(leagues_to_test)

def test_precise_weather() -> bool:
    """Prueba el sistema de clima preciso por ciudad"""
    logger.info("=== Probando sistema de clima preciso ===")
    
    cities_to_test = [
        {"city": "Manchester", "country": "England", "venue": "Old Trafford"},
        {"city": "Lima", "country": "Peru", "venue": None},
        {"city": "Copenhagen", "country": "Denmark", "venue": "Parken Stadium"},
        {"city": "Seoul", "country": "South Korea", "venue": "Seoul World Cup Stadium"}
    ]
    
    success_count = 0
    current_date = datetime.now()
    
    for city_info in cities_to_test:
        try:
            # Obtener clima genérico
            generic_weather = get_weather_forecast(
                city_info["city"], 
                city_info["country"], 
                current_date
            )
            
            # Obtener clima preciso
            precise_weather = get_precise_location_weather(
                city_info["city"],
                city_info["country"],
                current_date,
                city_info["venue"]
            )
            
            # Verificar contenido
            if (precise_weather and
                'city' in precise_weather and
                'temperature' in precise_weather):
                success_count += 1
            
            logger.info(f"Clima para {city_info['city']}, {city_info['country']}:")
            logger.info(f"Genérico: {generic_weather.get('condition', 'N/A')}, {generic_weather.get('temperature', 'N/A')}°C")
            logger.info(f"Preciso: {precise_weather.get('condition', 'N/A')}, {precise_weather.get('temperature', 'N/A')}°C")
            
            # Verificar campo específico de ubicación
            location_precision = precise_weather.get('coordinate_precision', 'N/A')
            logger.info(f"Precisión de ubicación: {location_precision}")
            
        except Exception as e:
            logger.error(f"Error con ciudad {city_info['city']}: {e}")
    
    return success_count == len(cities_to_test)

def test_match_validator() -> bool:
    """Prueba el validador de datos de partidos"""
    logger.info("=== Probando validador de datos de partidos ===")
    
    # Crear datos de prueba
    good_match = {
        'fixture_id': 1001,
        'home_team_id': 33,
        'away_team_id': 34,
        'home_team_name': 'Manchester United',
        'away_team_name': 'Newcastle',
        'league_id': 39,
        'match_date': datetime.now().isoformat(),
        'statistics': {
            'corners': {'home': 5, 'away': 4},
            'yellow_cards': {'home': 2, 'away': 3},
            'red_cards': {'home': 0, 'away': 0},
        },
        'home_goals': 2,
        'away_goals': 1,
        'winner': 'home',
        'elo_ratings': {
            'home': 1600,
            'away': 1520
        },
        'home_xg': 1.7,
        'away_xg': 1.1,
        'probabilities': {
            'home_win': 0.48,
            'away_win': 0.25,
            'draw': 0.27
        }
    }
    
    # Partido con problemas
    bad_match = {
        'fixture_id': 1002,
        'home_team_id': 40,
        'away_team_id': 41,
        'home_team_name': 'Liverpool',
        'away_team_name': 'Chelsea',
        'league_id': 39,
        'match_date': datetime.now().isoformat(),
        'statistics': {
            'corners': {'home': 35, 'away': 0},  # Valor anómalo de corners
            'yellow_cards': {'home': 10, 'away': 1},  # Demasiadas amarillas
            'red_cards': {'home': 0, 'away': 4},  # Demasiadas rojas
        },
        'home_goals': 0,
        'away_goals': 1,
        'winner': 'home',  # Inconsistente con los goles
        'elo_ratings': {
            'home': 1650,
            'away': 1510
        },
        'home_xg': 0.3,  # Inconsistente con Elo alto
        'away_xg': 2.5,
        'weather': {
            'city': 'Liverpool',
            'condition': 'Clear',
            'temperature': 22,  # Datos genéricos
            'wind_speed': 5
        }
    }
    
    # Ejecutar validación
    validator = MatchDataValidator()
    good_result = validator.validate_match_data(good_match)
    bad_result = validator.validate_match_data(bad_match)
    
    # Validar múltiples partidos
    multiple_result = validate_matches([good_match, bad_match])
    
    # Logs de resultados
    logger.info(f"Partido bueno es válido: {good_result['valid']}")
    logger.info(f"Partido bueno - errores: {len(good_result['errors'])}")
    
    logger.info(f"Partido malo es válido: {bad_result['valid']}")
    logger.info(f"Partido malo - errores: {len(bad_result['errors'])}")
    logger.info(f"Errores encontrados: {bad_result['errors']}")
    
    logger.info(f"Validación múltiple - partidos válidos: {multiple_result['valid_matches']}")
    logger.info(f"Validación múltiple - partidos inválidos: {multiple_result['invalid_matches']}")
    
    return (good_result['valid'] and 
            not bad_result['valid'] and 
            len(bad_result['errors']) >= 2 and
            multiple_result['invalid_matches'] == 1)

def test_integrated_workflow() -> bool:
    """Prueba el flujo completo de predicción con todas las mejoras"""
    logger.info("=== Probando flujo integrado ===")
    
    try:
        # Crear flujo de trabajo
        workflow = ELOEnhancedPredictionWorkflow()
        
        # Generar partidos de prueba
        fixtures = workflow.get_upcoming_matches(39, days_ahead=1)  # Premier League
        
        if not fixtures:
            logger.error("No se generaron partidos de prueba")
            return False
            
        logger.info(f"Generados {len(fixtures)} partidos para pruebas")
        
        # Ejecutar predicciones
        predictions = workflow.make_predictions_for_matches(fixtures[:2])  # Solo usar 2 para prueba
        
        if not predictions:
            logger.error("No se generaron predicciones")
            return False
            
        logger.info(f"Generadas {len(predictions)} predicciones")
        
        # Verificar componentes clave
        prediction = predictions[0]
        has_elo = 'elo_ratings' in prediction or 'advanced_elo_analysis' in prediction
        has_tactical = 'tactical_analysis' in prediction
        
        logger.info(f"Predicción con datos Elo: {has_elo}")
        logger.info(f"Predicción con análisis táctico: {has_tactical}")
        
        # Guardar ejemplo de predicción
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "example_prediction.json", "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2, default=str)
            
        return has_elo and has_tactical
        
    except Exception as e:
        logger.error(f"Error en flujo integrado: {e}")
        return False

def main():
    """Ejecuta todas las pruebas"""
    tests = {
        "Validador de coherencia": test_coherence_validator,
        "Analizador táctico": test_tactical_analyzer,
        "Clima preciso por ciudad": test_precise_weather,
        "Validador de datos de partidos": test_match_validator,
        "Flujo integrado": test_integrated_workflow
    }
    
    results = {}
    overall_success = True
    
    # Crear directorio para resultados
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    # Ejecutar pruebas
    for name, test_func in tests.items():
        logger.info(f"\n\nEjecutando prueba: {name}")
        try:
            success = test_func()
            results[name] = success
            if not success:
                overall_success = False
            logger.info(f"Resultado de '{name}': {'ÉXITO' if success else 'FALLO'}")
        except Exception as e:
            results[name] = False
            overall_success = False
            logger.error(f"Error en prueba '{name}': {e}")
    
    # Generar informe
    logger.info("\n\n=== RESULTADOS DE PRUEBAS ===")
    for name, success in results.items():
        status = "✓" if success else "✗"
        logger.info(f"{status} {name}")
    
    logger.info(f"\nResultado general: {'ÉXITO' if overall_success else 'FALLO'}")
    
    # Guardar resultados
    with open(output_dir / "test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "overall_success": overall_success,
            "tests": results
        }, f, indent=2)
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())
