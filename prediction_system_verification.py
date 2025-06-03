#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Verificación de Integración del Sistema de Predicción con Odds API Optimizada

Este script verifica que el sistema principal de predicción esté correctamente
integrado con la solución optimizada de la API de odds.

Autor: Equipo de Desarrollo
Fecha: Mayo 23, 2025
"""

import sys
import json
import logging
from datetime import datetime
import os
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='prediction_odds_integration.log',
    filemode='w'
)

logger = logging.getLogger('prediction_odds_integration')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

def banner(text):
    """Mostrar banner de texto"""
    print("\n" + "=" * 70)
    print(f" {text} ".center(70, "="))
    print("=" * 70 + "\n")

def check_config():
    """Verificar que la configuración está correctamente integrada"""
    banner("VERIFICACIÓN DE CONFIGURACIÓN")
    
    try:
        import config
        
        # Verificar que existen las configuraciones requeridas
        required_configs = [
            "API_KEY", 
            "API_BASE_URL", 
            "ODDS_ENDPOINTS", 
            "ODDS_BOOKMAKERS_PRIORITY"
        ]
        
        problems = []
        
        # Verificar existencia de configuraciones
        for conf in required_configs:
            if hasattr(config, conf):
                print(f"✅ Configuración encontrada: {conf}")
            else:
                print(f"❌ Configuración faltante: {conf}")
                problems.append(f"Falta la configuración {conf}")
          # Verificar que no existen configuraciones obsoletas
        obsolete_configs = [
            "ODDS_API_KEY", 
            "ODDS_API_URL", 
            "ODDS_API_PROVIDER"
        ]
        
        for conf in obsolete_configs:
            if hasattr(config, conf):
                print(f"❌ Configuración obsoleta encontrada: {conf}")
                problems.append(f"Configuración obsoleta {conf} debe ser eliminada")
            else:
                print(f"✅ No se encontró configuración obsoleta: {conf}")
        
        # Verificar que se usan las nuevas configuraciones
        new_configs = [
            "API_KEY",
            "API_BASE_URL",
            "ODDS_ENDPOINTS"
        ]
        
        for conf in new_configs:
            if hasattr(config, conf):
                print(f"✅ Nueva configuración encontrada: {conf}")
            else:
                print(f"❌ Nueva configuración faltante: {conf}")
                problems.append(f"Falta la nueva configuración {conf}")
                
        # Verificar ODDS_ENDPOINTS
        if hasattr(config, "ODDS_ENDPOINTS"):
            required_endpoints = ["pre_match", "live", "bookmakers", "bets"]
            for endpoint in required_endpoints:
                if endpoint in config.ODDS_ENDPOINTS:
                    print(f"✅ Endpoint encontrado: {endpoint}")
                else:
                    print(f"❌ Endpoint faltante: {endpoint}")
                    problems.append(f"Falta el endpoint {endpoint} en ODDS_ENDPOINTS")
        
        # Verificar cache config
        if hasattr(config, "CACHE_CONFIG") and "odds" in config.CACHE_CONFIG:
            print(f"✅ Configuración de caché para odds encontrada")
        else:
            print(f"❌ Falta configuración de caché para odds")
            problems.append("Falta configuración de caché para odds en CACHE_CONFIG")
        
        return len(problems) == 0, problems
            
    except ImportError as e:
        print(f"❌ Error importando módulo de configuración: {e}")
        return False, [f"Error de importación: {e}"]

def check_optimize_odds_integration():
    """Verificar que el módulo optimize_odds_integration está funcionando"""
    banner("VERIFICACIÓN DE MÓDULO DE ODDS")
    
    try:
        import optimize_odds_integration
        
        required_functions = [
            "get_fixture_odds", 
            "normalize_odds_data", 
            "setup_cache", 
            "clear_expired_cache"
        ]
        
        problems = []
        
        for func in required_functions:
            if hasattr(optimize_odds_integration, func):
                print(f"✅ Función encontrada: {func}")
            else:
                print(f"❌ Función faltante: {func}")
                problems.append(f"Falta la función {func} en optimize_odds_integration")
        
        # Verificar que el directorio de caché existe
        cache_dir = optimize_odds_integration.CACHE_DIR
        if cache_dir.exists():
            print(f"✅ Directorio de caché encontrado: {cache_dir}")
        else:
            print(f"❌ Directorio de caché no encontrado: {cache_dir}")
            problems.append(f"El directorio de caché {cache_dir} no existe")
            
        return len(problems) == 0, problems
            
    except ImportError as e:
        print(f"❌ Error importando módulo de optimización: {e}")
        return False, [f"Error de importación: {e}"]

def test_api_integration():
    """Probar la integración con la API"""
    banner("PRUEBA DE INTEGRACIÓN DE API")
    
    try:
        import optimize_odds_integration
        import config
        
        # Primero configuramos el caché
        optimize_odds_integration.setup_cache()
        
        # Intentamos obtener odds para algunos fixtures recientes
        # Estos IDs deberían reemplazarse con fixtures relevantes para su sistema
        test_fixtures = [1208383, 1208384, 1208385]
        
        problems = []
        real_data_count = 0
        
        for fixture_id in test_fixtures:
            print(f"📊 Obteniendo odds para fixture {fixture_id}...")
            odds_data = optimize_odds_integration.get_fixture_odds(
                fixture_id, 
                use_cache=True, 
                force_refresh=False
            )
            
            if odds_data:
                print(f"✅ Datos obtenidos para fixture {fixture_id}")
                
                # Verificar si son datos simulados
                if odds_data.get("simulated", True):
                    print(f"⚠️ Los datos para fixture {fixture_id} son simulados")
                    problems.append(f"Datos simulados para fixture {fixture_id}")
                else:
                    print(f"✅ Datos REALES para fixture {fixture_id}")
                    real_data_count += 1
                    
                # Verificar estructura básica
                required_keys = ["fixture_id", "bookmakers"]
                for key in required_keys:
                    if key not in odds_data:
                        print(f"❌ Datos incompletos, falta: {key}")
                        problems.append(f"Estructura de datos incorrecta, falta {key}")
            else:
                print(f"❌ Error obteniendo datos para fixture {fixture_id}")
                problems.append(f"No se pudieron obtener datos para fixture {fixture_id}")
        
        # Calcular porcentaje de datos reales
        if test_fixtures:
            real_data_percentage = (real_data_count / len(test_fixtures)) * 100
            print(f"\n📈 Porcentaje de datos reales: {real_data_percentage:.1f}%")
            
            if real_data_percentage < 50:
                problems.append(f"Porcentaje de datos reales muy bajo: {real_data_percentage:.1f}%")
        
        return len(problems) == 0, problems
            
    except Exception as e:
        print(f"❌ Error en prueba de integración: {e}")
        return False, [f"Error de integración: {e}"]

def test_prediction_integration():
    """Verificar integración con el sistema de predicción"""
    banner("INTEGRACIÓN CON SISTEMA DE PREDICCIÓN")
    
    try:
        # Intentar importar el sistema de predicción
        print("⚠️ Esta prueba requiere que el sistema de predicción esté implementado")
        print("⚠️ Verificando si podemos importar el módulo de predicción...")
        
        prediction_module_exists = False
        prediction_imports_odds = False
        
        # Verificar si existe el archivo de predicciones
        if Path("predictions.py").exists():
            prediction_module_exists = True
            print("✅ Módulo de predicción encontrado")
            
            # Leer el contenido para buscar importaciones
            with open("predictions.py", "r", encoding="utf-8") as f:
                content = f.read()
                
            if "import optimize_odds_integration" in content or "from optimize_odds_integration" in content:
                prediction_imports_odds = True
                print("✅ El sistema de predicción importa el módulo optimizado")
            else:
                print("❌ El sistema de predicción no importa el módulo optimizado")
        else:
            print("❌ No se encontró el módulo de predicción (predictions.py)")
        
        problems = []
        if not prediction_module_exists:
            problems.append("No se encontró el módulo de predicción")
        
        if prediction_module_exists and not prediction_imports_odds:
            problems.append("El sistema de predicción no importa el módulo optimizado")
        
        return len(problems) == 0, problems
    except Exception as e:
        print(f"❌ Error verificando integración de predicción: {e}")
        return False, [f"Error de verificación: {e}"]

def generate_report(results):
    """Generar reporte de integración"""
    banner("REPORTE DE INTEGRACIÓN")
    
    success = all(result[0] for result in results)
    
    print(f"Fecha de verificación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Estado general: {'✅ ÉXITO' if success else '❌ FALLOS ENCONTRADOS'}")
    print("\nResumen por componente:")
    
    for i, (component, (status, problems)) in enumerate(results.items()):
        print(f"\n{i+1}. {component}: {'✅ OK' if status else '❌ PROBLEMAS'}")
        
        if not status:
            print("   Problemas encontrados:")
            for j, problem in enumerate(problems):
                print(f"   {j+1}. {problem}")
    
    if not success:
        print("\n⚠️ ACCIONES RECOMENDADAS:")
        print("1. Revisar y corregir todos los problemas listados")
        print("2. Verificar que todas las dependencias están instaladas")
        print("3. Asegurar que los archivos están en las ubicaciones correctas")
        print("4. Ejecutar este script nuevamente después de las correcciones")
    else:
        print("\n✅ INTEGRACIÓN EXITOSA")
        print("El sistema de predicción está correctamente integrado con el módulo optimizado de odds.")
        print("Recomendaciones para el futuro:")
        print("1. Monitorear el porcentaje de datos reales vs simulados")
        print("2. Realizar esta verificación después de actualizaciones")
        print("3. Considerar ajustes de rendimiento según el uso real")
    
    # Guardar reporte en archivo
    report_file = f"integration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "components": {
            component: {
                "status": status,
                "problems": problems
            } for component, (status, problems) in results.items()
        }
    }
    
    try:
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nReporte guardado en: {report_file}")
    except Exception as e:
        print(f"\n❌ Error guardando reporte: {e}")

def main():
    """Función principal"""
    banner("VERIFICACIÓN DE INTEGRACIÓN DE SISTEMA DE PREDICCIÓN")
    print("Verificando la correcta integración del sistema optimizado de odds...")
    
    # Ejecutar todas las verificaciones
    results = {
        "Configuración": check_config(),
        "Módulo de Odds": check_optimize_odds_integration(),
        "Integración de API": test_api_integration(),
        "Sistema de Predicción": test_prediction_integration()
    }
    
    # Generar reporte
    generate_report(results)

if __name__ == "__main__":
    main()
