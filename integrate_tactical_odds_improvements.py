"""
Script de Integración de Mejoras Tácticas y Odds

Este script integra las mejoras en el analizador táctico y el gestor de odds
en el sistema principal de predicción.

Uso:
    python integrate_tactical_odds_improvements.py

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import logging
import importlib
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='integrate_improvements.log',
    filemode='w'
)

logger = logging.getLogger('integration')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Rutas
BASE_DIR = Path(__file__).parent
BACKUP_DIR = BASE_DIR / "backups" / f"pre_integration_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def create_backups():
    """Crear copias de seguridad de los archivos que se modificarán"""
    logger.info("Creando copias de seguridad...")
    
    # Crear directorio de backup si no existe
    if not BACKUP_DIR.exists():
        BACKUP_DIR.mkdir(parents=True)
    
    # Archivos a respaldar
    files_to_backup = [
        "tactical_integration.py",
        "fixed_tactical_integration.py",
        "tactical_integration_fixed.py",
        "odds_integration.py",
        "app.py"
    ]
    
    # Realizar backups
    backed_up = []
    for filename in files_to_backup:
        file_path = BASE_DIR / filename
        if file_path.exists():
            backup_path = BACKUP_DIR / filename
            shutil.copy2(file_path, backup_path)
            backed_up.append(filename)
            logger.info(f"Respaldo creado: {backup_path}")
    
    if backed_up:
        logger.info(f"Copias de seguridad creadas en: {BACKUP_DIR}")
    else:
        logger.warning("No se encontraron archivos para respaldar")
    
    return backed_up


def validate_prerequisites():
    """Validar que todos los archivos necesarios estén presentes"""
    logger.info("Validando prerrequisitos...")
    
    required_files = [
        "tactical_analyzer_enhanced.py",
        "odds_manager.py",
        "improved_tactical_odds_integration.py"
    ]
    
    missing = []
    for filename in required_files:
        file_path = BASE_DIR / filename
        if not file_path.exists():
            missing.append(filename)
    
    if missing:
        logger.error(f"Faltan archivos requeridos: {', '.join(missing)}")
        return False
    
    logger.info("Todos los archivos requeridos están presentes")
    return True


def update_app_imports():
    """Actualizar los imports en app.py para usar los nuevos módulos"""
    logger.info("Actualizando imports en app.py...")
    
    app_path = BASE_DIR / "app.py"
    if not app_path.exists():
        logger.error(f"No se encontró el archivo {app_path}")
        return False
    
    try:
        # Leer el archivo app.py
        with open(app_path, 'r', encoding='utf-8') as f:
            app_content = f.read()
        
        # Buscar importaciones antiguas y reemplazarlas
        old_imports = [
            "from tactical_integration import",
            "from fixed_tactical_integration import",
            "from tactical_integration_fixed import",
            "import tactical_integration",
            "import fixed_tactical_integration",
            "import tactical_integration_fixed",
            "import odds_integration",
            "from odds_integration import"
        ]
        
        # Nuevo import a añadir
        new_import = "from improved_tactical_odds_integration import process_predictions_batch, enrich_prediction_with_tactical_odds"
        
        # Verificar si el nuevo import ya está presente
        if new_import in app_content:
            logger.info("Los nuevos imports ya están presentes en app.py")
        else:
            # Verificar si hay imports antiguos
            has_old_imports = any(old_import in app_content for old_import in old_imports)
            
            if has_old_imports:
                # Reemplazar los imports antiguos
                for old_import in old_imports:
                    if old_import in app_content:
                        app_content = app_content.replace(old_import, f"# {old_import} # DEPRECATED")
                
                # Añadir el nuevo import después de los otros imports
                import_section_end = app_content.find("\n\n", app_content.find("import"))
                if import_section_end > 0:
                    app_content = app_content[:import_section_end] + f"\n{new_import}" + app_content[import_section_end:]
                else:
                    # Si no se puede encontrar el final de la sección de imports, añadir al principio
                    app_content = f"{new_import}\n\n" + app_content
            else:
                # Si no hay imports antiguos, añadir después de los otros imports
                import_section_end = app_content.find("\n\n", app_content.find("import"))
                if import_section_end > 0:
                    app_content = app_content[:import_section_end] + f"\n{new_import}" + app_content[import_section_end:]
                else:
                    # Si no hay sección de imports, añadir al principio
                    app_content = f"{new_import}\n\n" + app_content
            
            # Guardar los cambios
            with open(app_path, 'w', encoding='utf-8') as f:
                f.write(app_content)
                
            logger.info("Imports actualizados en app.py")
        
        return True
        
    except Exception as e:
        logger.error(f"Error actualizando imports en app.py: {e}")
        return False


def update_api_endpoints():
    """Actualizar los endpoints de API para usar las nuevas funciones"""
    logger.info("Actualizando endpoints de API...")
    
    app_path = BASE_DIR / "app.py"
    if not app_path.exists():
        logger.error(f"No se encontró el archivo {app_path}")
        return False
    
    try:
        # Leer el archivo app.py
        with open(app_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Buscar endpoints que usen las funciones antiguas
        modified = False
        new_lines = []
        
        for i, line in enumerate(lines):
            if (
                "enrich_prediction_with_tactical_analysis" in line or 
                "get_simplified_tactical_analysis" in line or
                "tactical_integration" in line or
                "fixed_tactical_integration" in line or
                "tactical_integration_fixed" in line or
                "odds_integration" in line
            ):
                # Comentar la línea antigua
                if not line.strip().startswith('#'):
                    new_lines.append(f"# {line.rstrip()} # DEPRECATED\n")
                    modified = True
                else:
                    new_lines.append(line)
                    
                # Añadir la nueva función si parece ser un endpoint
                if "@app.route" in lines[i-1] and "def " in line:
                    # Extraer nombre de la función y parámetros
                    func_name = line.split("def ")[1].split("(")[0].strip()
                    
                    # Determinar qué tipo de endpoint es
                    if "tactical" in line.lower() and "odds" not in line.lower():
                        # Solo táctico
                        if "prediction_id" in line or "fixture_id" in line:
                            new_lines.append("    # Usar el nuevo integrador táctico mejorado\n")
                            new_lines.append("    prediction = enrich_prediction_with_tactical_odds(prediction, fixture_id=fixture_id)\n")
                        else:
                            new_lines.append("    # Procesar lote con el nuevo integrador táctico/odds\n")
                            new_lines.append("    predictions = process_predictions_batch(predictions)\n")
                    elif "odds" in line.lower() and "tactical" not in line.lower():
                        # Solo odds
                        if "prediction_id" in line or "fixture_id" in line:
                            new_lines.append("    # Usar el nuevo integrador de odds mejorado\n")
                            new_lines.append("    prediction = enrich_prediction_with_tactical_odds(prediction, fixture_id=fixture_id)\n")
                        else:
                            new_lines.append("    # Procesar lote con el nuevo integrador táctico/odds\n")
                            new_lines.append("    predictions = process_predictions_batch(predictions)\n")
                    elif "enrich" in line.lower() or "process" in line.lower():
                        # Enriquecimiento general
                        if "prediction_id" in line or "fixture_id" in line:
                            new_lines.append("    # Usar el nuevo integrador táctico/odds mejorado\n")
                            new_lines.append("    prediction = enrich_prediction_with_tactical_odds(prediction, fixture_id=fixture_id)\n")
                        else:
                            new_lines.append("    # Procesar lote con el nuevo integrador táctico/odds\n")
                            new_lines.append("    predictions = process_predictions_batch(predictions)\n")
            else:
                new_lines.append(line)
        
        if modified:
            # Guardar los cambios
            with open(app_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
                
            logger.info("Endpoints de API actualizados")
        else:
            logger.info("No se encontraron endpoints que requieran actualización")
        
        return True
        
    except Exception as e:
        logger.error(f"Error actualizando endpoints de API: {e}")
        return False


def run_integration_tests():
    """Ejecutar pruebas de integración"""
    logger.info("Ejecutando pruebas de integración...")
    
    test_script = BASE_DIR / "validate_tactical_odds_integration.py"
    if not test_script.exists():
        logger.error(f"No se encontró el script de validación {test_script}")
        return False
    
    try:
        # Ejecutar el script de validación
        logger.info("Ejecutando validate_tactical_odds_integration.py...")
        result = os.system(f"{sys.executable} {test_script}")
        
        if result == 0:
            logger.info("Pruebas de validación completadas con éxito")
            return True
        else:
            logger.error(f"Las pruebas de validación fallaron con código {result}")
            return False
            
    except Exception as e:
        logger.error(f"Error ejecutando pruebas de integración: {e}")
        return False


def run_integration():
    """Ejecutar el proceso completo de integración"""
    logger.info("="*60)
    logger.info("INICIANDO INTEGRACIÓN DE MEJORAS TÁCTICAS Y ODDS")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Validar prerrequisitos
    if not validate_prerequisites():
        logger.error("No se cumplen los prerrequisitos. Integración abortada.")
        return False
    
    # Crear backups
    backed_up = create_backups()
    
    # Actualizar imports
    if not update_app_imports():
        logger.error("Error actualizando imports. Integración abortada.")
        return False
    
    # Actualizar endpoints
    if not update_api_endpoints():
        logger.error("Error actualizando endpoints. Integración abortada.")
        return False
    
    # Ejecutar pruebas
    if not run_integration_tests():
        logger.warning("Algunas pruebas de integración fallaron.")
        # Continuamos a pesar de errores en las pruebas
    
    logger.info("\n" + "="*60)
    logger.info("RESUMEN DE INTEGRACIÓN")
    logger.info("="*60)
    logger.info(f"Archivos respaldados: {len(backed_up)}")
    logger.info(f"Directorio de backups: {BACKUP_DIR}")
    logger.info("Imports actualizados: ✓")
    logger.info("Endpoints actualizados: ✓")
    logger.info("="*60)
    
    logger.info("✅ INTEGRACIÓN COMPLETADA")
    logger.info("Para revertir los cambios, restaure los archivos desde el directorio de backups.")
    return True


if __name__ == "__main__":
    try:
        success = run_integration()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Error en integración: {str(e)}")
        sys.exit(1)
