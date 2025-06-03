"""
Script para corregir la estructura JSON del endpoint /api/upcoming_predictions
Asegura que tactical_analysis y odds_analysis aparezcan en el nivel principal de las predicciones.
"""

import json
import logging
import sys
from fixed_tactical_integration import create_default_tactical_analysis

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_json_structure(json_file_path):
    """
    Modifica el archivo JSON para asegurar que tactical_analysis y odds_analysis estén en el nivel principal
    """
    try:
        # Leer el archivo JSON actual
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if "match_predictions" not in data:
            logger.error(f"No se encontró 'match_predictions' en {json_file_path}")
            return False
        
        predictions = data["match_predictions"]
        modified = False
        
        for i, pred in enumerate(predictions):
            # Comprobar si ya existe tactical_analysis
            if "tactical_analysis" not in pred:
                # Si está en additional_data, moverlo al nivel principal
                if "additional_data" in pred and "tactical_analysis" in pred["additional_data"]:
                    pred["tactical_analysis"] = pred["additional_data"]["tactical_analysis"]
                    del pred["additional_data"]["tactical_analysis"]
                    logger.info(f"Predicción {i+1}: tactical_analysis movido desde additional_data")
                    modified = True
                else:
                    # Si no está disponible, crear uno predeterminado
                    home_team_id = pred.get("home_team_id")
                    away_team_id = pred.get("away_team_id")
                    
                    if home_team_id and away_team_id:
                        try:
                            pred["tactical_analysis"] = create_default_tactical_analysis(home_team_id, away_team_id)
                            logger.info(f"Predicción {i+1}: tactical_analysis generado por defecto")
                            modified = True
                        except Exception as e:
                            logger.warning(f"Error al crear tactical_analysis para predicción {i+1}: {e}")
                            # Estructura mínima
                            pred["tactical_analysis"] = {
                                "style_comparison": "No hay datos tácticos disponibles",
                                "key_advantages": [],
                                "suggested_approach": "No se pudo generar un análisis táctico"
                            }
                            modified = True
                    else:
                        logger.warning(f"Predicción {i+1}: No se pueden generar datos tácticos (faltan IDs de equipo)")
                        # Estructura mínima incluso sin IDs
                        pred["tactical_analysis"] = {
                            "style_comparison": "Datos insuficientes para análisis táctico",
                            "key_advantages": []
                        }
                        modified = True
            
            # Comprobar si ya existe odds_analysis
            if "odds_analysis" not in pred:
                # Si está en additional_data, moverlo al nivel principal
                if "additional_data" in pred and "odds_analysis" in pred["additional_data"]:
                    pred["odds_analysis"] = pred["additional_data"]["odds_analysis"]
                    del pred["additional_data"]["odds_analysis"]
                    logger.info(f"Predicción {i+1}: odds_analysis movido desde additional_data")
                    modified = True
                else:
                    # Si no existe, agregar uno mínimo
                    pred["odds_analysis"] = {
                        "market_analysis": {
                            "efficiency": 0.0,
                            "margin": 1.0
                        }
                    }
                    logger.info(f"Predicción {i+1}: odds_analysis agregado por defecto")
                    modified = True
        
        if modified:
            # Guardar el archivo JSON actualizado
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Archivo {json_file_path} actualizado correctamente")
            return True
        else:
            logger.info(f"No fue necesario modificar {json_file_path}")
            return False
            
    except Exception as e:
        logger.error(f"Error al procesar el archivo {json_file_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = "api_response.json"
        
    success = fix_json_structure(json_file)
    
    if success:
        print("✓ La estructura JSON se ha corregido correctamente")
    else:
        print("✗ Ha habido problemas al corregir la estructura JSON")
