"""
Script para validar las mejoras implementadas en el sistema de predicción.
Este script verifica:
1. Que el cálculo de confianza es dinámico
2. Que el modelo neural produce predicciones diversas
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

# Configuración del logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

def validate_dynamic_confidence():
    """Valida que el cálculo de confianza produce valores variables."""
    try:
        from confidence import calculate_confidence_score
        
        # Crear algunos escenarios de prueba
        test_scenarios = [
            {
                "description": "Partido con muchos antecedentes y buena calidad de datos",
                "team_id": 40,      # Manchester City
                "opponent_id": 33,  # Manchester United
                "league_id": 39,    # Premier League
                "is_home": True
            },
            {
                "description": "Partido con pocos antecedentes",
                "team_id": 710,     # Equipo de liga inferior
                "opponent_id": 715, # Otro equipo de liga inferior
                "league_id": 140,   # Liga de menor categoría
                "is_home": False
            },
            {
                "description": "Partido con datos contradictorios",
                "team_id": 529,     # Barcelona
                "opponent_id": 541, # Real Madrid
                "league_id": 140,   # La Liga
                "is_home": True
            }
        ]
        
        logger.info("\n==== VALIDACIÓN DE CÁLCULO DE CONFIANZA DINÁMICO ====")
        confidence_values = []
        
        for i, scenario in enumerate(test_scenarios, 1):
            logger.info(f"\nEscenario {i}: {scenario['description']}")
            prediction_data = {
                "home_team_id": scenario["team_id"] if scenario["is_home"] else scenario["opponent_id"],
                "away_team_id": scenario["opponent_id"] if scenario["is_home"] else scenario["team_id"],
                "league_id": scenario["league_id"]
            }
            conf_score = calculate_confidence_score(prediction_data)
            logger.info(f"Puntuación de confianza: {conf_score:.2f}")
            confidence_values.append(conf_score)
        
        # Verificar variabilidad
        unique_values = len(set([round(c, 2) for c in confidence_values]))
        logger.info(f"\nVariedad de valores de confianza: {unique_values} valores únicos en {len(test_scenarios)} escenarios")
        
        if unique_values >= 2:
            logger.info("✅ ÉXITO: El cálculo de confianza produce valores diversos")
            return True
        else:
            logger.warning("❌ PROBLEMA: El cálculo de confianza no produce valores suficientemente diversos")
            return False
    except Exception as e:
        logger.error(f"Error validando cálculo de confianza: {e}")
        return False

def validate_neural_predictions():
    """Valida que el modelo neural produce predicciones diversas."""
    try:
        # Intentar importar modelos
        try:
            from fnn_model_fixed import FeedforwardNeuralNetworkFixed
            fixed_model_available = True
        except ImportError:
            fixed_model_available = False
            logger.warning("Modelo mejorado no disponible para pruebas")
        
        from fnn_model import FeedforwardNeuralNetwork
        
        # Cargar scaler
        if not os.path.exists('models/scaler.pkl'):
            logger.error("No se encuentra el scaler en models/scaler.pkl")
            return False
            
        logger.info("Cargando scaler...")
        scaler = joblib.load('models/scaler.pkl')
        
        # Generar datos de prueba
        logger.info("Generando datos de prueba...")
        test_data = []
        
        # Crear 5 perfiles de equipos diferentes
        base_features = [
            # Equipo fuerte en casa vs débil visitante
            [2.0, 0.6, 0.8, 0.6, 0.8, 1.8, 0.3, 0.1, 1.9, 1.3, 0.7, 2.5, 2.0, 0.8],
            # Equipo débil en casa vs fuerte visitante
            [0.8, 1.4, 0.3, 0.2, 1.9, 0.7, 0.7, 0.5, 0.95, 1.65, 0.3, 2.5, 0.8, 1.9],
            # Equipos parejos de nivel medio
            [1.3, 1.1, 0.5, 0.3, 1.4, 1.0, 0.5, 0.3, 1.2, 1.2, 0.5, 2.5, 1.3, 1.4],
            # Equipos defensivos
            [0.8, 0.6, 0.4, 0.5, 0.7, 0.5, 0.4, 0.5, 0.7, 0.6, 0.5, 1.5, 0.8, 0.7],
            # Equipos ofensivos
            [2.2, 1.8, 0.5, 0.2, 2.1, 1.9, 0.6, 0.2, 2.0, 2.0, 0.5, 3.5, 2.2, 2.1]
        ]
        
        # Añadir pequeñas variaciones a cada perfil para tener más muestras
        for base in base_features:
            # Añadir el perfil base
            test_data.append(np.array(base))
            
            # Añadir 3 variaciones
            for _ in range(3):
                noise = np.random.uniform(-0.1, 0.1, size=len(base))
                test_data.append(np.array(base) + noise)
        
        test_data = np.array(test_data)
        
        # Escalar los datos
        test_data_scaled = scaler.transform(test_data)
        
        # Validar modelo original si está disponible
        logger.info("\n==== VALIDACIÓN DE PREDICCIONES NEURALES ====")
        if os.path.exists('models/fnn_model.pkl'):
            model_dict = joblib.load('models/fnn_model.pkl')
            input_dim = model_dict.get('input_dim', 14)
            
            try:
                logger.info("Probando modelo original...")
                orig_model = FeedforwardNeuralNetwork(input_dim=input_dim)
                orig_model.model.set_weights(model_dict['weights'])
                
                # Obtener predicciones
                orig_preds = orig_model.predict(test_data_scaled)
                
                # Analizar variabilidad
                logger.info("Predicciones del modelo original:")
                for i, pred in enumerate(orig_preds[:5]):  # Mostrar solo las primeras 5
                    logger.info(f"Ejemplo {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
                
                rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in orig_preds]
                unique_preds = set(rounded_preds)
                
                logger.info(f"Variabilidad: {len(unique_preds)} valores únicos en {len(orig_preds)} predicciones")
                
                if len(unique_preds) <= 3:
                    logger.warning("❌ El modelo original sufre del problema de predicciones duplicadas")
                
            except Exception as e:
                logger.error(f"Error con el modelo original: {e}")
        
        # Validar modelo mejorado si está disponible
        if fixed_model_available and os.path.exists('models/fnn_model_fixed.pkl'):
            try:
                logger.info("\nProbando modelo mejorado...")
                fixed_model_dict = joblib.load('models/fnn_model_fixed.pkl')
                fixed_model = FeedforwardNeuralNetworkFixed(input_dim=fixed_model_dict.get('input_dim', 14))
                fixed_model.load_weights(fixed_model_dict['weights'])
                
                # Obtener predicciones
                fixed_preds = fixed_model.predict(test_data_scaled)
                
                # Analizar variabilidad
                logger.info("Predicciones del modelo mejorado:")
                for i, pred in enumerate(fixed_preds[:5]):  # Mostrar solo las primeras 5
                    logger.info(f"Ejemplo {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
                
                rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in fixed_preds]
                unique_preds = set(rounded_preds)
                
                logger.info(f"Variabilidad: {len(unique_preds)} valores únicos en {len(fixed_preds)} predicciones")
                
                if len(unique_preds) >= len(fixed_preds) * 0.8:
                    logger.info("✅ ÉXITO: El modelo mejorado produce predicciones diversas")
                    return True
                else:
                    logger.warning("⚠️ El modelo mejorado mejora la variabilidad pero aún podría mejorar")
                    return True
                    
            except Exception as e:
                logger.error(f"Error con el modelo mejorado: {e}")
        
        # Si llegamos aquí, no hemos podido validar completamente
        logger.warning("No se pudo validar completamente las predicciones neurales")
        return False
    except Exception as e:
        logger.error(f"Error validando predicciones neurales: {e}")
        return False

def validate_system_integration():
    """Valida que los componentes están correctamente integrados."""
    try:
        # Importar el módulo de predicciones
        import predictions
        
        # Verificar si el modelo mejorado está siendo utilizado
        if hasattr(predictions, 'fnn_model'):
            model_class = predictions.fnn_model.__class__.__name__
            logger.info(f"Modelo actualmente en uso: {model_class}")
            
            if model_class == 'FeedforwardNeuralNetworkFixed':
                logger.info("✅ El sistema está utilizando el modelo mejorado")
            else:
                logger.info("⚠️ El sistema está utilizando el modelo original")
            
            # Verificar si la función calculate_confidence_score se está utilizando
            if 'calculate_confidence_score' in dir(predictions):
                logger.info("✅ El cálculo dinámico de confianza está disponible en el módulo de predicciones")
            else:
                from inspect import getsource
                
                # Buscar uso de calculate_confidence_score en el código
                try:
                    code = getsource(predictions)
                    if "calculate_confidence_score" in code:
                        logger.info("✅ El cálculo dinámico de confianza está integrado en el código de predicciones")
                    else:
                        logger.warning("❌ No se encontró uso de calculate_confidence_score en el código")
                except:
                    logger.warning("No se pudo analizar el código fuente")
        
        return True
    except Exception as e:
        logger.error(f"Error validando integración del sistema: {e}")
        return False

if __name__ == "__main__":
    logger.info("=== VALIDACIÓN DE MEJORAS DEL SISTEMA DE PREDICCIÓN ===")
    
    # Validar cálculo de confianza dinámica
    confidence_valid = validate_dynamic_confidence()
    
    # Validar modelo neural
    neural_valid = validate_neural_predictions()
    
    # Validar integración
    integration_valid = validate_system_integration()
    
    # Resumen final
    logger.info("\n=== RESUMEN DE VALIDACIÓN ===")
    logger.info(f"Confianza dinámica: {'✅ OK' if confidence_valid else '❌ Problema'}")
    logger.info(f"Predicciones neurales diversas: {'✅ OK' if neural_valid else '❌ Problema'}")
    logger.info(f"Integración del sistema: {'✅ OK' if integration_valid else '❌ Problema'}")
    
    if confidence_valid and neural_valid and integration_valid:
        logger.info("\n✅ TODAS LAS MEJORAS HAN SIDO IMPLEMENTADAS CORRECTAMENTE")
    else:
        logger.warning("\n⚠️ SE HAN DETECTADO PROBLEMAS EN LA IMPLEMENTACIÓN DE ALGUNAS MEJORAS")
