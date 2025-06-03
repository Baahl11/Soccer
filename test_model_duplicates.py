import joblib
import numpy as np
from fnn_model import FeedforwardNeuralNetwork

# Cargar el scaler
print("Cargando scaler...")
scaler = joblib.load('models/scaler.pkl')
print(f'Dimensión del scaler: {scaler.n_features_in_}')

# Cargar el modelo desde .pkl
print("Cargando modelo neural...")
model_dict = joblib.load('models/fnn_model.pkl')
fnn_model = FeedforwardNeuralNetwork(input_dim=model_dict['input_dim'])
fnn_model.model.set_weights(model_dict['weights'])

# Crear datos de prueba
print("Datos de prueba fijos para diferentes equipos:")
print("-----------------------------------------------")

# Datos para 5 casos diferentes
test_data = [
    # Equipo fuerte en casa vs débil visitante
    [2.0, 0.6, 0.8, 0.6, 0.8, 1.8, 0.3, 0.1, 1.9, 1.3, 0.7, 2.5, 2.0, 0.8],
    
    # Equipo débil en casa vs fuerte visitante
    [0.8, 1.4, 0.3, 0.2, 1.9, 0.7, 0.7, 0.5, 0.95, 1.65, 0.3, 2.5, 0.8, 1.9],
    
    # Equipos parejos de nivel medio
    [1.3, 1.1, 0.5, 0.3, 1.4, 1.0, 0.5, 0.3, 1.2, 1.2, 0.5, 2.5, 1.3, 1.4],
    
    # Equipos del mismo nivel muy defensivos
    [0.8, 0.6, 0.4, 0.5, 0.7, 0.5, 0.4, 0.5, 0.7, 0.6, 0.5, 1.5, 0.8, 0.7],
    
    # Equipos del mismo nivel muy ofensivos
    [2.2, 1.8, 0.5, 0.2, 2.1, 1.9, 0.6, 0.2, 2.0, 2.0, 0.5, 3.5, 2.2, 2.1],
]

# Escalar datos y hacer predicciones
scaled_data = scaler.transform(np.array(test_data))
predictions = fnn_model.predict(scaled_data)

print("\nResultados:")
team_descriptions = [
    "Equipo fuerte en casa vs débil visitante",
    "Equipo débil en casa vs fuerte visitante",
    "Equipos parejos de nivel medio",
    "Equipos del mismo nivel muy defensivos",
    "Equipos del mismo nivel muy ofensivos"
]

for i, (desc, pred) in enumerate(zip(team_descriptions, predictions)):
    print(f"{i+1}. {desc}:")
    print(f"   Home xG: {pred[0]:.3f}, Away xG: {pred[1]:.3f}, Total: {pred[0] + pred[1]:.3f}")

# Verificar duplicados
rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in predictions]
unique_preds = set(rounded_preds)

print("\nAnálisis de duplicados:")
print(f"Número de predicciones únicas: {len(unique_preds)} de {len(predictions)}")

if len(unique_preds) < len(predictions):
    print("PROBLEMA DETECTADO: Existen predicciones duplicadas.")
    
    # Encontrar y mostrar los duplicados
    from collections import Counter
    dup_counter = Counter(rounded_preds)
    
    print("\nDetalles de duplicados:")
    for pred, count in dup_counter.items():
        if count > 1:
            print(f"Predicción {pred} aparece {count} veces")
else:
    print("El modelo produce predicciones distintas para diferentes entradas.")
