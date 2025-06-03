"""
Script muy simplificado para verificar el modelo corregido.
"""
import os
import sys
import numpy as np
import joblib
import json
import logging

# Configurar logging simple
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixed_model():
    """Prueba directa del modelo mejorado con datos sintéticos"""
    print("Probando modelo fnn_model_fixed.py directamente...")
    
    try:
        # Importar modelo
        from fnn_model_fixed import FeedforwardNeuralNetworkFixed
        
        # Cargar modelo
        model_path = os.path.join('models', 'fnn_model_fixed.pkl')
        model_dict = joblib.load(model_path)
        fixed_model = FeedforwardNeuralNetworkFixed(
            input_dim=model_dict['input_dim'],
            hidden_dims=[32, 16, 8]
        )
        fixed_model.load_weights(model_dict['weights'])
        
        # Cargar scaler
        scaler_path = os.path.join('models', 'scaler.pkl')
        scaler = joblib.load(scaler_path)
        
        # Crear datos de prueba (5 copias del mismo vector)
        base_vector = [1.2, 1.1, 0.6, 0.4, 1.5, 1.2, 0.5, 0.3, 1.3, 1.4, 0.4, 2.5, 1.4, 1.3]
        test_data = np.array([base_vector] * 5)
        test_data_scaled = scaler.transform(test_data)
        
        # Generar predicciones
        predictions = fixed_model.predict(test_data_scaled)
        
        # Mostrar resultados
        print("\nPredicciones generadas:")
        home_goals = []
        away_goals = []
        
        for i, pred in enumerate(predictions):
            h = pred[0]
            a = pred[1]
            home_goals.append(h)
            away_goals.append(a)
            print(f"Predicción #{i+1}: Home={h:.2f}, Away={a:.2f}, Total={h+a:.2f}")
        
        # Verificar variabilidad
        home_std = np.std(home_goals)
        away_std = np.std(away_goals)
        
        print(f"\nVariabilidad (desviación estándar):")
        print(f"- Home goals: {home_std:.4f}")
        print(f"- Away goals: {away_std:.4f}")
        
        if home_std > 0.05 and away_std > 0.05:
            print("\n✓ El modelo muestra buena variabilidad")
            return True
        else:
            print("\n✗ El modelo muestra baja variabilidad")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_fixed_model()
