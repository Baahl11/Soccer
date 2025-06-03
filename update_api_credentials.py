"""
Actualización de Credenciales API

Este script verifica y actualiza las credenciales para la API de fútbol,
incluyendo las credenciales específicas para la API de odds.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import os
import sys
import json
import logging
from pathlib import Path
import requests
from datetime import datetime

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='api_credentials.log',
    filemode='w'
)

logger = logging.getLogger('credentials')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

def check_current_credentials():
    """
    Verifica las credenciales actuales en los archivos de configuración
    
    Returns:
        Dict con estado de las credenciales
    """
    config_file = Path("config.py")
    env_file = Path(".env")
    
    credentials = {
        "config_exists": config_file.exists(),
        "env_exists": env_file.exists(),
        "api_key_in_config": False,
        "api_key_in_env": False,
        "api_key": None
    }
    
    # Verificar .env
    if credentials["env_exists"]:
        try:
            with open(env_file, 'r') as f:
                env_content = f.read()
                
            for line in env_content.splitlines():
                if line.startswith("FOOTBALL_API_KEY="):
                    credentials["api_key_in_env"] = True
                    api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if api_key and api_key != "your-api-key-here":
                        credentials["api_key"] = api_key
                    break
        except Exception as e:
            logger.error(f"Error leyendo .env: {str(e)}")
    
    # Verificar config.py
    if credentials["config_exists"]:
        try:
            sys.path.insert(0, ".")
            import config
            
            if hasattr(config, "API_KEY"):
                credentials["api_key_in_config"] = True
                
                if config.API_KEY != "your-api-key-here" and config.API_KEY:
                    if not credentials["api_key"]:
                        credentials["api_key"] = config.API_KEY
        except Exception as e:
            logger.error(f"Error importando config.py: {str(e)}")
    
    return credentials

def test_api_key(api_key, base_url="https://v3.football.api-sports.io", test_odds=True):
    """
    Prueba una clave de API para verificar validez
    
    Args:
        api_key: Clave a probar
        base_url: URL base de la API
        test_odds: Si es True, prueba también el endpoint de odds
        
    Returns:
        Tuple (válido, mensaje)
    """
    try:
        logger.info(f"Probando clave API: ...{api_key[-4:] if len(api_key) > 4 else '****'}")
        
        # Endpoint de estado
        endpoint = f"{base_url}/status"
        headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": base_url.replace("https://", "")
        }
        
        response = requests.get(endpoint, headers=headers, timeout=10)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                if "errors" in data and data["errors"]:
                    logger.warning(f"API respondió con errores: {data['errors']}")
                    return False, f"API respondió con errores: {data['errors']}"
                
                # Verificar límites de uso
                if "response" in data:
                    response_data = data["response"]
                    
                    if isinstance(response_data, dict):
                        subscription = response_data.get("subscription", {})
                        
                        plan = subscription.get("plan", "Desconocido")
                        remaining = subscription.get("requests", {}).get("remaining", 0)
                        
                        logger.info(f"API operativa. Plan: {plan}, Solicitudes restantes: {remaining}")
                        return True, f"API operativa. Plan: {plan}, Solicitudes restantes: {remaining}"
                
                logger.info("API operativa, pero no se pudo extraer información de la suscripción")
                return True, "API operativa, pero no se pudo extraer información de la suscripción"
                
            except json.JSONDecodeError:
                logger.warning(f"Respuesta no es JSON válido")
                return False, "Respuesta no es JSON válido"
        else:
            logger.error(f"Error en API: {response.status_code}")
            return False, f"Error {response.status_code}: La API no responde correctamente"
    except Exception as e:
        logger.error(f"Excepción conectando con API: {str(e)}")
        return False, f"Error de conexión: {str(e)}"

def update_credentials(api_key, update_env=True, update_config=True):
    """
    Actualiza las credenciales en los archivos de configuración
    
    Args:
        api_key: Nueva clave de API
        update_env: Si es True, actualizar .env
        update_config: Si es True, actualizar config.py
        
    Returns:
        Tuple (éxito, mensaje)
    """
    success = True
    messages = []
    
    # Variable de entorno
    env_var_name = "FOOTBALL_API_KEY"
    
    # Actualizar .env
    if update_env:
        env_file = Path(".env")
        
        try:
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_content = f.read()
                
                # Verificar si ya existe la variable
                api_key_line = None
                new_content = []
                
                for line in env_content.splitlines():
                    if line.startswith(f"{env_var_name}="):
                        api_key_line = f'{env_var_name}="{api_key}"'
                        new_content.append(api_key_line)
                    else:
                        new_content.append(line)
                
                # Si no existe la variable, añadirla
                if not api_key_line:
                    new_content.append(f'{env_var_name}="{api_key}"')
                
                # Escribir archivo
                with open(env_file, 'w') as f:
                    f.write("\n".join(new_content))
                
                messages.append(f"Actualizada {env_var_name} en {env_file}")
                logger.info(f"Actualizada {env_var_name} en {env_file}")
            else:
                # Crear archivo .env
                with open(env_file, 'w') as f:
                    f.write(f'{env_var_name}="{api_key}"\n')
                
                messages.append(f"Creado archivo {env_file} con {env_var_name}")
                logger.info(f"Creado archivo {env_file} con {env_var_name}")
        except Exception as e:
            success = False
            error_msg = f"Error actualizando .env: {str(e)}"
            messages.append(error_msg)
            logger.error(error_msg)
    
    # Actualizar config.py
    if update_config:
        config_file = Path("config.py")
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_content = f.read()
                
                # Buscar la línea con API_KEY
                import re
                pattern = r'API_KEY\s*=\s*os\.getenv\("FOOTBALL_API_KEY",\s*"[^"]*"\)'
                
                if re.search(pattern, config_content):
                    # Reemplazar usando regex
                    new_config = re.sub(
                        pattern,
                        f'API_KEY = os.getenv("FOOTBALL_API_KEY", "{api_key}")',
                        config_content
                    )
                    
                    with open(config_file, 'w') as f:
                        f.write(new_config)
                    
                    messages.append(f"Actualizada API key en {config_file}")
                    logger.info(f"Actualizada API key en {config_file}")
                else:
                    messages.append(f"No se pudo encontrar API_KEY en {config_file}")
                    logger.warning(f"No se pudo encontrar API_KEY en {config_file}")
            else:
                messages.append(f"El archivo {config_file} no existe")
                logger.warning(f"El archivo {config_file} no existe")
        except Exception as e:
            success = False
            error_msg = f"Error actualizando config.py: {str(e)}"
            messages.append(error_msg)
            logger.error(error_msg)
    
    return success, messages

def display_api_info():
    """
    Muestra información sobre la API según la documentación
    """
    print("\n" + "="*60)
    print("INFORMACIÓN DE LA API DE FOOTBALL")
    print("="*60)
    print("\nEndpoints principales:")
    print("- /status        - Estado y límites de la API")
    print("- /leagues       - Información de ligas")
    print("- /fixtures      - Partidos")
    print("- /teams         - Información de equipos")
    
    print("\nEndpoints de odds (misma API):")
    print("- /odds          - Odds pre-partido")
    print("- /odds/live     - Odds en tiempo real")
    print("- /odds/bets     - Tipos de apuestas disponibles")
    print("- /odds/bookmakers - Lista de casas de apuestas")
    
    print("\nActualización de odds:")
    print("- Pre-partido: Actualización cada 3 horas")
    print("- En vivo: Actualización cada 5-60 segundos")
    
    print("\nCobertura de odds:")
    print("- Pre-partido: Entre 1 y 14 días antes del partido")
    print("- Historial: 7 días")
    
    print("\nRecomendaciones de uso:")
    print("- Pre-partido: 1 llamada cada 3 horas")
    print("- En vivo: Depende del plan (cuidado con límites de API)")
    print("="*60)



def main():
    """Función principal"""
    logger.info("="*60)
    logger.info("VERIFICACIÓN Y ACTUALIZACIÓN DE CREDENCIALES API")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    # Verificar credenciales actuales
    credentials = check_current_credentials()
    
    logger.info(f"Archivo config.py: {'Encontrado' if credentials['config_exists'] else 'No encontrado'}")
    logger.info(f"Archivo .env: {'Encontrado' if credentials['env_exists'] else 'No encontrado'}")
    logger.info(f"API key en config.py: {'Sí' if credentials['api_key_in_config'] else 'No'}")
    logger.info(f"API key en .env: {'Sí' if credentials['api_key_in_env'] else 'No'}")
    
    # Verificar API key actual
    if credentials["api_key"]:
        logger.info(f"API key actual: ...{credentials['api_key'][-4:] if len(credentials['api_key']) > 4 else '****'}")
        
        # Probar API key actual
        valid, message = test_api_key(credentials["api_key"])
        
        if valid:
            logger.info(f"La API key actual es válida: {message}")
            print("\n✅ La API key actual es válida y funcional.")
            
            # Verificar también el acceso a los datos de odds con la API key existente
            print("\nVerificando acceso a datos de odds...")
            try:
                # Probar un endpoint de odds
                base_url = "https://v3.football.api-sports.io"
                odds_endpoint = f"{base_url}/odds"
                odds_headers = {
                    "x-rapidapi-key": credentials["api_key"],
                    "x-rapidapi-host": base_url.replace("https://", "")
                }
                odds_params = {"league": "39", "season": "2023", "bet": "1"}
                odds_response = requests.get(odds_endpoint, headers=odds_headers, params=odds_params, timeout=15)
                
                if odds_response.status_code == 200:
                    print("✅ Acceso a datos de odds confirmado")
                    logger.info("Acceso a datos de odds confirmado")
                    
                    # Mostrar información sobre la API
                    display_api_info()
                    return
                else:
                    print(f"⚠️ No se pudo acceder a los datos de odds: Error {odds_response.status_code}")
                    logger.warning(f"No se pudo acceder a los datos de odds: Error {odds_response.status_code}")
                    
                    # Solo pedimos una nueva API key si la actual no puede acceder a odds
                    print("\nSe necesita una nueva API key para acceder a los datos de odds.")
            except Exception as e:
                print(f"⚠️ Error verificando acceso a odds: {str(e)}")
                logger.error(f"Error verificando acceso a odds: {str(e)}")
                
                # Solo pedimos una nueva API key si la actual no puede acceder a odds
                print("\nSe necesita una nueva API key para acceder a los datos de odds.")
            else:            logger.warning(f"La API key actual no es válida: {message}")
            print("\n⚠️ La API key actual no es válida o ha expirado.")
            
            # Solo pedimos una nueva API key si la actual no funciona
            print("\nSe necesita una nueva API key para acceder a todos los endpoints.")
    else:
        logger.warning("No se encontró API key o tiene un valor predeterminado")
        print("\n⚠️ No se encontró una API key válida en la configuración.")
        
        # Solo pedimos una nueva API key si no hay ninguna configurada
        print("\nSe necesita configurar una API key para acceder a los endpoints.")
    
    # Preguntamos al usuario si desea introducir una nueva API key
    resp = input("¿Desea introducir una nueva API key? (s/n): ").lower()
    if resp != 's':
        print("\nOperación cancelada. Se mantendrá la configuración actual.")
        print("Para ejecutar la optimización de odds, utilice:")
        print("python optimize_odds_integration.py")
        return
        
    print("\nPor favor, ingrese una nueva API key para la API de fútbol:")
    new_api_key = input("> ").strip()
    
    if not new_api_key:
        logger.warning("Usuario no proporcionó una nueva API key")
        print("❌ No se proporcionó una API key. Operación cancelada.")
        return
        
    # Probar nueva API key
    print("\nVerificando la nueva API key...")
    valid, message = test_api_key(new_api_key)
    
    if valid:
        logger.info(f"La nueva API key es válida: {message}")
        print(f"\n✅ API key válida: {message}")
        
        # Actualizar credenciales
        print("\nActualizando archivos de configuración...")
        success, messages = update_credentials(new_api_key)
        
        for msg in messages:
            print(f"- {msg}")
        
        if success:
            print("\n✅ Credenciales actualizadas con éxito.")
            
            # Mostrar información sobre la API
            display_api_info()
        else:
            print("\n⚠️ Hubo algunos problemas al actualizar las credenciales.")
    else:
        logger.error(f"La nueva API key no es válida: {message}")
        print(f"\n❌ La API key no es válida: {message}")
        print("Operación cancelada.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperación cancelada por el usuario.")
    except Exception as e:
        logger.error(f"Error inesperado: {str(e)}")
        print(f"\n❌ Error inesperado: {str(e)}")
        print("Verifique el archivo de log para más detalles.")
