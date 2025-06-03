"""
Script para migrar modelos existentes al nuevo formato del optimizador mejorado.
Este script:
1. Detecta modelos entrenados con el optimizador anterior
2. Convierte la configuración al nuevo formato con tipos correctos
3. Actualiza los metadatos para compatibilidad futura
"""

import os
import sys
import logging
import json
import shutil
from datetime import datetime
import glob

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("migrate_optimizer_models.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def migrate_optimizer_models():
    """
    Migra los modelos entrenados con el optimizador anterior al nuevo formato.
    """
    logger.info("Iniciando migración de modelos de optimizador...")
    
    # Directorios de búsqueda
    search_dirs = [
        'models/optuna',
        'models'
    ]
    
    # Contador de archivos procesados
    models_found = 0
    models_migrated = 0
    
    # Crear directorio de backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = f"backup/optimizer_migration_{timestamp}"
    os.makedirs(backup_dir, exist_ok=True)
    logger.info(f"Directorio de backup creado: {backup_dir}")
    
    # Buscar archivos json de parámetros
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            logger.info(f"Directorio {search_dir} no existe, saltando...")
            continue
            
        # Buscar archivos JSON de parámetros
        param_files = glob.glob(f"{search_dir}/**/best_params_*.json", recursive=True)
        param_files.extend(glob.glob(f"{search_dir}/**/ensemble_weights_*.json", recursive=True))
        
        logger.info(f"Encontrados {len(param_files)} archivos de parámetros en {search_dir}")
        models_found += len(param_files)
        
        for param_file in param_files:
            try:
                # Leer archivo
                with open(param_file, 'r') as f:
                    data = json.load(f)
                
                # Hacer backup
                backup_file = os.path.join(backup_dir, os.path.basename(param_file))
                shutil.copy2(param_file, backup_file)
                
                # Verificar si necesita migración (strings como valores numéricos)
                needs_migration = False
                
                # Verificar parámetros
                if "best_params" in data:
                    for key, value in data["best_params"].items():
                        if isinstance(value, str) and key not in ["activation"]:
                            needs_migration = True
                            # Intentar convertir a número
                            try:
                                if value.isdigit() or (value.replace('.', '', 1).isdigit() and value.count('.') <= 1):
                                    if '.' in value:
                                        data["best_params"][key] = float(value)
                                    else:
                                        data["best_params"][key] = int(value)
                            except ValueError:
                                pass
                
                # Verificar verbose (común para strings like "0" o "1")
                if "verbose" in data:
                    if isinstance(data["verbose"], str):
                        needs_migration = True
                        try:
                            data["verbose"] = int(data["verbose"])
                        except ValueError:
                            data["verbose"] = 0
                
                # Verificar weights para ensemble
                if "weights" in data:
                    for key, value in data["weights"].items():
                        if isinstance(value, str):
                            needs_migration = True
                            try:
                                data["weights"][key] = float(value)
                            except ValueError:
                                data["weights"][key] = 0.0
                
                # Guardar versión actualizada si necesita migración
                if needs_migration:
                    # Añadir metadatos de migración
                    data["migrated"] = True
                    data["migration_date"] = datetime.now().isoformat()
                    data["migration_version"] = "hyperparameter_optimizer_improved.py"
                    
                    with open(param_file, 'w') as f:
                        json.dump(data, f, indent=4)
                    
                    logger.info(f"Migrado archivo: {param_file}")
                    models_migrated += 1
                else:
                    logger.info(f"Archivo ya en formato correcto, no requiere migración: {param_file}")
            
            except Exception as e:
                logger.error(f"Error procesando {param_file}: {e}")
    
    # Resumen
    logger.info(f"Migración completada. Modelos encontrados: {models_found}, migrados: {models_migrated}")
    
    return models_found, models_migrated

if __name__ == "__main__":
    found, migrated = migrate_optimizer_models()
    
    if migrated > 0:
        print(f"\n✅ Se migraron correctamente {migrated} de {found} modelos encontrados")
    elif found > 0:
        print(f"\n✅ Se encontraron {found} modelos pero ninguno requería migración")
    else:
        print("\n❓ No se encontraron modelos para migrar")
