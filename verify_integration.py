#!/usr/bin/env python3
"""
Script para verificar que el análisis táctico y las odds estén correctamente integrados
en las estructuras JSON de las predicciones.
"""

import json
import requests
from pprint import pprint
import sys
from tactical_integration import get_simplified_tactical_analysis
from fixed_tactical_integration import get_simplified_tactical_analysis as fixed_get_simplified_tactical_analysis

def verify_modules():
    """
    Verifica que los módulos estén correctamente importados
    y proporcionan datos en la estructura esperada.
    """
    print("Verificando módulos de análisis táctico...")
    
    # Test para el módulo original (puede faltar algunas campos)
    try:
        print("\nUsando tactical_integration.py:")
        tactical_analysis = get_simplified_tactical_analysis(1, 2)
        fields = tactical_analysis.keys()
        print(f"- Campos disponibles: {', '.join(fields)}")
        print(f"- Total de campos: {len(fields)}")
        
        if 'analysis_methods' in tactical_analysis:
            methods = tactical_analysis['analysis_methods'].keys()
            print(f"- Métodos de análisis: {', '.join(methods)}")
        else:
            print("- No se encontró campo 'analysis_methods'")
    except Exception as e:
        print(f"Error usando tactical_integration.py: {e}")
    
    # Test para el módulo corregido
    try:
        print("\nUsando fixed_tactical_integration.py:")
        fixed_tactical_analysis = fixed_get_simplified_tactical_analysis(1, 2)
        fields = fixed_tactical_analysis.keys()
        print(f"- Campos disponibles: {', '.join(fields)}")
        print(f"- Total de campos: {len(fields)}")
        
        # Verificación específica de campos requeridos
        required_fields = [
            'tactical_style', 'key_battles', 'strengths', 'weaknesses',
            'tactical_recommendation', 'expected_formations'
        ]
        
        missing = [f for f in required_fields if f not in fields]
        if missing:
            print(f"- ADVERTENCIA: Campos faltantes: {', '.join(missing)}")
        else:
            print("- Todos los campos requeridos están presentes")
    except Exception as e:
        print(f"Error usando fixed_tactical_integration.py: {e}")

def verify_api_integration(base_url="http://localhost:5000"):
    """
    Verifica que la API esté integrando correctamente el análisis táctico y las odds
    en la estructura JSON de las predicciones.
    """
    print("\nVerificando integración en la API...")
    
    # Llamar a la API con include_additional_data=true para asegurar que obtenemos todos los datos
    url = f"{base_url}/api/upcoming_predictions?league_id=71&season=2024&include_additional_data=true"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Generar excepción para códigos de estado de error
        
        data = response.json()
        
        if 'match_predictions' not in data:
            print("ERROR: No se encontraron predicciones en la respuesta")
            return False
            
        match_predictions = data['match_predictions']
        
        if not match_predictions:
            print("ERROR: Lista de predicciones vacía")
            return False
            
        success = True
        
        # Verificar al menos una predicción con análisis táctico y odds
        for i, prediction in enumerate(match_predictions):
            print(f"\nVerificando predicción {i+1}/{len(match_predictions)} - {prediction.get('home_team')} vs {prediction.get('away_team')}")
            
            # Verificar análisis táctico en el nivel principal
            if 'tactical_analysis' not in prediction:
                print(f"ERROR: Falta tactical_analysis en la predicción {i+1}")
                success = False
            else:
                print("✅ tactical_analysis encontrado en nivel principal")
                
                # Verificar estructura de análisis táctico
                ta = prediction['tactical_analysis']
                if not isinstance(ta, dict):
                    print(f"ERROR: tactical_analysis no es un diccionario en la predicción {i+1}")
                    success = False
                else:
                    # Verificar claves esperadas en tactical_analysis
                    expected_keys = ['style_comparison', 'key_advantages', 'suggested_approach', 'tactical_style', 'matchup_analysis']
                    missing_keys = [key for key in expected_keys if key not in ta]
                    
                    if missing_keys:
                        print(f"ADVERTENCIA: Faltan claves esperadas en tactical_analysis: {', '.join(missing_keys)}")
                    else:
                        print("✅ Todas las claves esperadas encontradas en tactical_analysis")
            
            # Verificar odds_analysis en el nivel principal
            if 'odds_analysis' not in prediction:
                print(f"ERROR: Falta odds_analysis en la predicción {i+1}")
                success = False
            else:
                print("✅ odds_analysis encontrado en nivel principal")
            
            # Verificar si odds_analysis está incorrectamente en additional_data
            if 'additional_data' in prediction and 'odds_analysis' in prediction.get('additional_data', {}):
                print(f"ERROR: odds_analysis encontrado en additional_data en la predicción {i+1} (debería estar solo en el nivel principal)")
                success = False
                
            # Mostrar muestra de la estructura para verificación manual
            print("\nEstructura de ejemplo de la predicción:")
            print(f"Claves en el nivel principal: {list(prediction.keys())}")
            
            # Solo mostrar estructura detallada para la primera predicción para evitar salida excesiva
            if i == 0:
                if 'tactical_analysis' in prediction:
                    print("\nEstructura de Análisis Táctico:")
                    pprint({k: type(v).__name__ for k, v in prediction['tactical_analysis'].items()})
                
                if 'odds_analysis' in prediction:
                    print("\nEstructura de Análisis de Odds:")
                    pprint({k: type(v).__name__ for k, v in prediction['odds_analysis'].items()})
                
        if success:
            print("\n✅ ÉXITO: Todas las verificaciones pasaron. Tanto tactical_analysis como odds_analysis están en el nivel principal.")
        else:
            print("\n❌ FALLO: Algunas verificaciones fallaron. Ver errores arriba.")
            
        return success
        
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Fallo en la petición a la API: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Error inesperado: {e}")
        return False

def save_sample_json(base_url="http://localhost:5000"):
    """
    Guarda una muestra del JSON de respuesta para referencia futura
    y documentación de la estructura.
    """
    print("\nGuardando muestra de JSON para documentación...")
    
    url = f"{base_url}/api/upcoming_predictions?league_id=71&season=2024&include_additional_data=true&limit=1"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        if 'match_predictions' not in data or not data['match_predictions']:
            print("ERROR: No hay predicciones disponibles para guardar")
            return False
            
        # Guardar solo la primera predicción como muestra
        sample = {
            "match_predictions": [data['match_predictions'][0]]
        }
        
        # Guardar el JSON formateado
        with open('sample_prediction_structure.json', 'w', encoding='utf-8') as f:
            json.dump(sample, f, indent=2, ensure_ascii=False)
            
        print(f"✅ Muestra guardada en 'sample_prediction_structure.json'")
        return True
        
    except Exception as e:
        print(f"ERROR: No se pudo guardar la muestra: {e}")
        return False

if __name__ == "__main__":
    print("Iniciando verificación de integración de análisis táctico y odds")
    
    # Verificar módulos primero
    verify_modules()
    
    # Verificar integración en API si el servidor está en ejecución
    try:
        api_ok = verify_api_integration()
        if api_ok:
            print("\n✅ ÉXITO: Todas las verificaciones de API pasaron!")
            
            # Si la API está funcionando, guardar una muestra del JSON
            save_sample_json()
            
            sys.exit(0)
        else:
            print("\n⚠️ ADVERTENCIA: Algunas verificaciones fallaron. Revisar errores arriba.")
            sys.exit(1)
    except Exception as e:
        print(f"\nError al verificar API. ¿El servidor está en ejecución? Error: {e}")
        print("\n✅ Verificación de módulos completada. Ahora el endpoint utilizará la versión corregida.")
