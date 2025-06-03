import numpy as np
import joblib
import os
import sys

# Configuración para mostrar la salida inmediatamente
import functools
print = functools.partial(print, flush=True)

def diagnostic_test():
    print("Diagnóstico de modelo FNN")
    print("=========================")
    
    # 1. Verificar existencia de archivos
    print("\n1. Verificación de archivos:")
    model_path = 'models/fnn_model.pkl'
    scaler_path = 'models/scaler.pkl'
    
    try:
        if os.path.exists(model_path):
            print(f"✓ El archivo {model_path} existe")
            file_size = os.path.getsize(model_path) / 1024
            print(f"  Tamaño: {file_size:.2f} KB")
        else:
            print(f"✗ El archivo {model_path} NO existe")
        
        if os.path.exists(scaler_path):
            print(f"✓ El archivo {scaler_path} existe")
            file_size = os.path.getsize(scaler_path) / 1024
            print(f"  Tamaño: {file_size:.2f} KB")
        else:
            print(f"✗ El archivo {scaler_path} NO existe")
    except Exception as e:
        print(f"Error verificando archivos: {e}")
    
    # 2. Intentar cargar el scaler
    print("\n2. Carga de scaler:")
    try:
        scaler = joblib.load(scaler_path)
        print(f"✓ Scaler cargado correctamente")
        print(f"  Número de features: {scaler.n_features_in_}")
    except Exception as e:
        print(f"✗ Error cargando scaler: {e}")
    
    # 3. Intentar cargar el modelo
    print("\n3. Carga de modelo:")
    try:
        model_dict = joblib.load(model_path)
        
        if isinstance(model_dict, dict):
            print(f"✓ Archivo de modelo cargado como diccionario")
            
            print(f"  Claves disponibles: {model_dict.keys()}")
            if 'input_dim' in model_dict:
                print(f"  Dimensión de entrada: {model_dict['input_dim']}")
            if 'weights' in model_dict:
                print(f"  Pesos disponibles: {len(model_dict['weights'])} capas")
            
            # Más información sobre los pesos
            if 'weights' in model_dict:
                weights = model_dict['weights']
                for i, w in enumerate(weights):
                    print(f"  Capa {i+1}: {w.shape}")
        else:
            print(f"✗ El archivo no contiene un diccionario, sino: {type(model_dict)}")
    except Exception as e:
        print(f"✗ Error cargando modelo: {e}")

if __name__ == "__main__":
    diagnostic_test()
