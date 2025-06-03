"""
Integración mejorada de análisis táctico y odds

Este módulo implementa la integración del analizador táctico mejorado y
el gestor de odds avanzado en el sistema de predicción.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import logging
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import os

# Importar los nuevos componentes mejorados
from tactical_analyzer_enhanced import TacticalAnalyzerEnhanced
from odds_manager import OddsManager

logger = logging.getLogger(__name__)

# Inicializar componentes
tactical_analyzer = TacticalAnalyzerEnhanced()
odds_manager = OddsManager()

def enrich_prediction_with_tactical_odds(
    prediction: Dict[str, Any],
    home_team_id: Optional[int] = None,
    away_team_id: Optional[int] = None,
    fixture_id: Optional[int] = None,
    competition_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Enriquece la predicción con análisis táctico y odds mejoradas.
    
    Args:
        prediction: Predicción actual a enriquecer
        home_team_id: ID del equipo local (opcional si ya está en prediction)
        away_team_id: ID del equipo visitante (opcional si ya está en prediction)
        fixture_id: ID del partido (opcional si ya está en prediction)
        competition_id: ID de la competición (opcional para odds más específicas)
        
    Returns:
        Predicción enriquecida con información táctica y odds
    """
    try:
        # Extraer IDs de equipos si no se proporcionaron
        if not home_team_id:
            home_team_id = prediction.get('teams', {}).get('home', {}).get('id')
        
        if not away_team_id:
            away_team_id = prediction.get('teams', {}).get('away', {}).get('id')
            
        if not fixture_id:
            fixture_id = prediction.get('fixture_id')
        
        # Validar que tenemos los datos necesarios
        if not home_team_id or not away_team_id:
            logger.error("No se pudieron extraer los IDs de los equipos para el análisis táctico")
            prediction['tactical_analysis'] = {
                'error': "IDs de equipos no disponibles",
                'status': 'error'
            }
        else:
            # Obtener análisis táctico mejorado
            tactical_analysis = tactical_analyzer.get_tactical_analysis(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_name=prediction.get('teams', {}).get('home', {}).get('name', ''),
                away_name=prediction.get('teams', {}).get('away', {}).get('name', '')
            )
            
            # Incluir el análisis táctico en la predicción
            prediction['tactical_analysis'] = tactical_analysis
                
        # Obtener odds mejoradas (si se proporcionó fixture_id)
        if fixture_id:
            odds_data = odds_manager.get_odds_for_fixture(
                fixture_id=fixture_id,
                competition_id=competition_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                use_cache=True
            )
            
            # Incluir el análisis de odds en la predicción
            prediction['odds_analysis'] = odds_data
            
        return prediction
        
    except Exception as e:
        logger.error(f"Error en enrichment de predicción táctico/odds: {e}")
        
        # No perder el resto de la predicción por un error en el enriquecimiento
        if 'tactical_analysis' not in prediction:
            prediction['tactical_analysis'] = {
                'error': f"Error en análisis táctico: {str(e)}",
                'status': 'error'
            }
            
        if 'odds_analysis' not in prediction:
            prediction['odds_analysis'] = {
                'error': f"Error en análisis de odds: {str(e)}",
                'status': 'error'
            }
            
        return prediction


def update_prediction_format(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actualiza el formato de la predicción para mover tactical_analysis y odds_analysis
    al nivel principal y asegurarse de que cumplen con el esquema esperado.
    
    Args:
        prediction: Predicción a actualizar
        
    Returns:
        Predicción con formato actualizado
    """
    # Asegurarse de que la estructura básica está presente
    if 'teams' not in prediction:
        prediction['teams'] = {
            'home': {'id': 0, 'name': 'Desconocido'},
            'away': {'id': 0, 'name': 'Desconocido'}
        }
    
    # Mover tactical_analysis al nivel principal si está anidado
    if 'prediction_details' in prediction and 'tactical_analysis' in prediction['prediction_details']:
        prediction['tactical_analysis'] = prediction['prediction_details']['tactical_analysis']
        del prediction['prediction_details']['tactical_analysis']
    
    # Mover odds_analysis al nivel principal si está anidado
    if 'prediction_details' in prediction and 'odds_analysis' in prediction['prediction_details']:
        prediction['odds_analysis'] = prediction['prediction_details']['odds_analysis']
        del prediction['prediction_details']['odds_analysis']
    
    # Asegurar que tactical_analysis existe
    if 'tactical_analysis' not in prediction:
        prediction['tactical_analysis'] = {}
    
    # Asegurar que odds_analysis existe
    if 'odds_analysis' not in prediction:
        prediction['odds_analysis'] = {}
    
    return prediction


def process_predictions_batch(predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Procesa un lote de predicciones aplicando análisis táctico y odds.
    
    Args:
        predictions: Lista de predicciones a procesar
        
    Returns:
        Predicciones procesadas con análisis táctico y odds
    """
    processed_predictions = []
    
    for prediction in predictions:
        # Actualizar formato
        updated_prediction = update_prediction_format(prediction)
        
        # Enriquecer con análisis táctico y odds
        enriched_prediction = enrich_prediction_with_tactical_odds(updated_prediction)
        
        processed_predictions.append(enriched_prediction)
    
    return processed_predictions


def run_test_processing():
    """
    Ejecuta una prueba de procesamiento con datos de muestra.
    """
    # Cargar datos de ejemplo
    sample_path = Path(__file__).parent / "data" / "sample_predictions.json"
    
    try:
        with open(sample_path, 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
        
        # Procesar predicciones
        predictions = sample_data.get('predictions', [])
        processed = process_predictions_batch(predictions)
        
        # Guardar resultados procesados
        output_path = Path(__file__).parent / "data" / "processed_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'predictions': processed}, f, indent=2)
            
        logger.info(f"Procesamiento completado. Resultados guardados en: {output_path}")
        
        # Verificar calidad de los datos procesados
        tactical_generic_count = 0
        odds_simulated_count = 0
        
        for pred in processed:
            tactical = pred.get('tactical_analysis', {})
            odds = pred.get('odds_analysis', {})
            
            # Verificar si el análisis táctico es genérico (simplificado)
            home_style = tactical.get('tactical_style', {}).get('home', {})
            if home_style.get('possession') == 'medio' and home_style.get('pressing') == 'medio':
                tactical_generic_count += 1
                
            # Verificar si las odds son simuladas
            if odds.get('simulated', True):
                odds_simulated_count += 1
        
        # Mostrar estadísticas
        total = len(processed)
        if total > 0:
            tactical_generic_pct = (tactical_generic_count / total) * 100
            odds_simulated_pct = (odds_simulated_count / total) * 100
            
            logger.info(f"Calidad de datos:")
            logger.info(f"- Análisis tácticos genéricos: {tactical_generic_count}/{total} ({tactical_generic_pct:.1f}%)")
            logger.info(f"- Odds simuladas: {odds_simulated_count}/{total} ({odds_simulated_pct:.1f}%)")
        
    except Exception as e:
        logger.error(f"Error en procesamiento de prueba: {e}")


if __name__ == "__main__":
    # Configuración de logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar prueba
    run_test_processing()
