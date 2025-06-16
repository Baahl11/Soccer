import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_data_in_batches(batch_size=5, n_features=14, total_samples=100):
    """Genera datos de prueba en lotes pequeños para evitar problemas de memoria."""
    for start_idx in range(0, total_samples, batch_size):
        current_batch_size = min(batch_size, total_samples - start_idx)
        batch = np.random.rand(current_batch_size, n_features)
        yield batch
        # Forzar liberación de memoria
        del batch
        gc.collect()

try:
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    
    # Crear y ajustar el scaler
    logger.info("Creando y ajustando el scaler...")
    scaler = StandardScaler()
    
    # Procesar datos en lotes pequeños
    logger.info("Procesando datos en lotes...")
    for i, batch in enumerate(create_data_in_batches(batch_size=5)):
        logger.info(f"Procesando lote {i+1}...")
        if i == 0:
            # Para el primer lote, hacemos fit
            scaler.partial_fit(batch)
        else:
            # Para los siguientes lotes, actualizamos el fit
            scaler.partial_fit(batch)
        
        # Forzar liberación de memoria después de cada lote
        gc.collect()

    # Crear directorio models si no existe
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Directorio creado: {models_dir}")

    # Guardar el scaler
    logger.info(f"Guardando scaler en {scaler_path}...")
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler guardado exitosamente")

    # Limpieza final de memoria
    del scaler
    gc.collect()

except Exception as e:
    logger.error(f"Error durante la ejecución: {e}", exc_info=True)
