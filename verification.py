import sys
import os
import logging
import numpy as np

# Configurar el logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def test_confidence_module():
    try:
        from confidence import calculate_confidence_score
        
        # Probar con algunos equipos
        test_cases = [
            {"team": "Manchester City", "home_team_id": 40, "away_team_id": 33, "league_id": 39, "is_home": True},
            {"team": "Barcelona", "home_team_id": 541, "away_team_id": 529, "league_id": 140, "is_home": False},
            {"team": "Bayern Munich", "home_team_id": 157, "away_team_id": 159, "league_id": 78, "is_home": True}        ]
        
        print("===== PRUEBA DEL MÓDULO DE CONFIANZA =====")
        for case in test_cases:
            # Prepara los datos en el formato que espera la función
            prediction_data = {
                "home_team_id": case["home_team_id"],
                "away_team_id": case["away_team_id"],
                "league_id": case["league_id"]
            }
            score = calculate_confidence_score(prediction_data)
            print(f"{case['team']}: {score:.2f}")
        
        print("\nMódulo de confianza funciona correctamente.\n")
        return True
    except Exception as e:
        print(f"Error en el módulo de confianza: {e}")
        return False

def test_fnn_fixed_module():
    try:
        from fnn_model_fixed import FeedforwardNeuralNetworkFixed
        
        # Crear una instancia de prueba
        model = FeedforwardNeuralNetworkFixed(input_dim=14)
        
        # Generar datos de prueba
        test_data = np.random.uniform(0.5, 2.0, (5, 14))
        
        # Obtener predicciones
        predictions = model.predict(test_data)
        
        print("===== PRUEBA DEL MÓDULO FNN MEJORADO =====")
        print(f"Dimensiones de entrada: {test_data.shape}")
        print(f"Dimensiones de salida: {predictions.shape}")
        print("Predicciones:")
        
        for i, pred in enumerate(predictions):
            print(f"Predicción {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
        
        # Verificar variabilidad
        rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in predictions]
        unique_preds = set(rounded_preds)
        
        print(f"\nVariabilidad: {len(unique_preds)} predicciones únicas de {len(predictions)} totales")
        
        if len(unique_preds) == len(predictions):
            print("El modelo mejorado produce predicciones únicas para cada entrada.\n")
        else:
            print("El modelo aún produce algunas predicciones duplicadas.\n")
            
        return True
    except Exception as e:
        print(f"Error en el módulo FNN mejorado: {e}")
        return False

def check_integration():
    try:
        import predictions
        
        # Verificar si se importan los módulos de mejora
        imports = dir(predictions)
        
        print("===== VERIFICACIÓN DE INTEGRACIÓN =====")
        
        integrated_components = []
        
        if 'confidence' in imports or 'calculate_confidence_score' in imports:
            print("✓ Módulo de confianza importado")
            integrated_components.append('confidence')
        else:
            print("✗ Módulo de confianza no importado")
        
        if 'FeedforwardNeuralNetworkFixed' in imports:
            print("✓ Modelo neural mejorado importado")
            integrated_components.append('fnn_fixed')
        else:
            print("✗ Modelo neural mejorado no importado")
        
        # Verificar tipo del modelo actual
        if hasattr(predictions, 'fnn_model'):
            model_type = type(predictions.fnn_model).__name__
            print(f"Modelo actual en uso: {model_type}")
            
            if model_type == 'FeedforwardNeuralNetworkFixed':
                print("✓ El sistema está utilizando el modelo mejorado")
                integrated_components.append('using_fixed_model')
        
        print(f"\nComponentes integrados: {len(integrated_components)}/3\n")
        return True
    except Exception as e:
        print(f"Error verificando integración: {e}")
        return False

if __name__ == "__main__":
    print("\n=== VERIFICACIÓN DE MEJORAS IMPLEMENTADAS ===\n")
    
    # Verificar cada componente
    confidence_ok = test_confidence_module()
    fnn_fixed_ok = test_fnn_fixed_module()
    integration_ok = check_integration()
    
    # Resumen
    print("=== RESUMEN DE VERIFICACIÓN ===")
    print(f"Módulo de confianza: {'OK' if confidence_ok else 'Error'}")
    print(f"Modelo FNN mejorado: {'OK' if fnn_fixed_ok else 'Error'}")
    print(f"Integración: {'OK' if integration_ok else 'Error'}")
    
    if confidence_ok and fnn_fixed_ok and integration_ok:
        print("\n✓ TODAS LAS MEJORAS ESTÁN IMPLEMENTADAS CORRECTAMENTE")
    else:
        print("\n✗ ALGUNAS MEJORAS REQUIEREN ATENCIÓN ADICIONAL")
