"""
Optimización de los endpoints de odds

Este script actualiza la configuración de la API de odds y mejora la integración
con el sistema de predicción existente.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
import requests

# Cargar configuraciones
import config

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='odds_optimization.log',
    filemode='w'
)

logger = logging.getLogger('odds_optimization')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Constantes
API_KEY = config.API_KEY  # Usamos la misma API key para toda la API
API_BASE_URL = config.API_BASE_URL  # Usamos la misma URL base
CACHE_DIR = Path("data/odds_cache")
CACHE_DURATION = config.CACHE_CONFIG.get("odds", timedelta(minutes=15))

def setup_cache():
    """Configurar directorio de caché"""
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio de caché creado: {CACHE_DIR}")
    return CACHE_DIR

def clear_expired_cache():
    """Limpiar archivos de caché expirados"""
    count = 0
    now = datetime.now()
    
    for cache_file in CACHE_DIR.glob("odds_*.json"):
        try:
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if now - file_time > CACHE_DURATION:
                cache_file.unlink()
                count += 1
        except Exception as e:
            logger.warning(f"Error limpiando caché {cache_file.name}: {str(e)}")
    
    logger.info(f"Limpiados {count} archivos de caché expirados")
    return count

def get_fixture_odds(fixture_id, use_cache=True, force_refresh=False, bookmaker_id=None, bet_id=None):
    """
    Obtener odds para un partido específico
    
    Args:
        fixture_id: ID del partido
        use_cache: Si es True, buscar en caché primero
        force_refresh: Si es True, ignorar caché y refrescar datos
        bookmaker_id: ID del bookmaker específico (opcional)
        bet_id: ID del tipo de apuesta (opcional, 1=Match Winner)
        
    Returns:
        Datos de odds del partido
    """
    cache_file = CACHE_DIR / f"odds_{fixture_id}.json"
    
    # Verificar caché si está habilitado y no se fuerza refresco
    if use_cache and not force_refresh and cache_file.exists():
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        if datetime.now() - file_time <= CACHE_DURATION:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                logger.info(f"Usando odds en caché para partido {fixture_id}")
                return cached_data
            except Exception as e:
                logger.warning(f"Error leyendo caché para partido {fixture_id}: {str(e)}")
    
    # Solicitar a la API
    logger.info(f"Solicitando odds para partido {fixture_id}")
    
    endpoint = f"{API_BASE_URL}/odds"
    headers = {
        "x-rapidapi-key": API_KEY,
        "x-rapidapi-host": API_BASE_URL.replace("https://", "")
    }
    
    # Parámetros de la solicitud
    params = {
        "fixture": fixture_id
    }
    
    # Añadir parámetros opcionales si están presentes
    if bookmaker_id:
        params["bookmaker"] = bookmaker_id
    
    if bet_id:
        params["bet"] = bet_id
    
    try:
        response = requests.get(endpoint, headers=headers, params=params, timeout=15)
        
        if response.status_code == 200:
            try:
                data = response.json()
                
                # Verificar si hay datos de odds reales
                if data.get("response") and len(data["response"]) > 0:
                    # Procesar y normalizar los datos de la API
                    normalized_data = normalize_odds_data(data["response"][0])
                    
                    # Guardar en caché
                    if use_cache:
                        try:
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                json.dump(normalized_data, f, indent=2)
                        except Exception as e:
                            logger.warning(f"Error guardando caché para partido {fixture_id}: {str(e)}")
                    
                    logger.info(f"Odds obtenidas con éxito para partido {fixture_id}")
                    return normalized_data
                else:
                    logger.warning(f"No hay datos de odds disponibles para partido {fixture_id}")
            except json.JSONDecodeError:
                logger.warning(f"Respuesta no es JSON válido: {response.text[:200]}...")
        else:
            logger.error(f"Error en API: {response.status_code} - {response.text[:200]}...")
    except Exception as e:
        logger.error(f"Excepción conectando con API: {str(e)}")
    
    # Si llegamos aquí, generamos odds simuladas
    logger.warning(f"Generando odds simuladas para partido {fixture_id}")
    return generate_simulated_odds(fixture_id)

def normalize_odds_data(api_odds_data):
    """
    Normalizar datos de odds del formato de la API al formato interno
    
    Args:
        api_odds_data: Datos de odds de la API
        
    Returns:
        Datos normalizados al formato interno
    """
    try:
        # Extraer información básica
        fixture_id = api_odds_data.get("fixture", {}).get("id", 0)
        bookmakers = api_odds_data.get("bookmakers", [])
        
        # Preparar estructura normalizada
        normalized = {
            "simulated": len(bookmakers) == 0,  # Si no hay bookmakers, es simulado
            "source": "API Football" if len(bookmakers) > 0 else "Datos simulados",
            "timestamp": datetime.now().isoformat(),
            "fixture_id": fixture_id
        }
        
        # Si hay bookmakers, procesar los datos reales
        if bookmakers:
            # Extraer probabilidades implícitas de Match Winner
            main_bookmaker = next((b for b in bookmakers if b.get("id") in [1, 6, 8, 2]), None) or bookmakers[0]
            
            match_winner_bets = next((bet for bet in main_bookmaker.get("bets", []) if bet.get("id") == 1), None)
            
            if match_winner_bets and "values" in match_winner_bets:
                values = match_winner_bets["values"]
                home_odd = next((v.get("odd", "0") for v in values if v.get("value") == "Home"), "0")
                draw_odd = next((v.get("odd", "0") for v in values if v.get("value") == "Draw"), "0")
                away_odd = next((v.get("odd", "0") for v in values if v.get("value") == "Away"), "0")
                
                # Convertir a float y calcular probabilidades implícitas
                try:
                    home_odd = float(home_odd)
                    draw_odd = float(draw_odd)
                    away_odd = float(away_odd)
                    
                    total_probability = (1/home_odd) + (1/draw_odd) + (1/away_odd)
                    
                    if total_probability > 0:
                        home_prob = (1/home_odd) / total_probability
                        draw_prob = (1/draw_odd) / total_probability
                        away_prob = (1/away_odd) / total_probability
                    else:
                        home_prob = 0.45
                        draw_prob = 0.28
                        away_prob = 0.27
                    
                    # Análisis de valor
                    value_opps = []
                    if home_odd >= 1.5:
                        value_opps.append({
                            "market": "Match Winner",
                            "selection": "Home",
                            "market_odds": home_odd,
                            "fair_odds": 1/home_prob,
                            "recommendation": get_recommendation(home_odd, 1/home_prob),
                            "confidence": "Media",
                            "value": round((home_odd * home_prob) - 1, 2)
                        })
                    
                    if away_odd >= 1.5:
                        value_opps.append({
                            "market": "Match Winner",
                            "selection": "Away",
                            "market_odds": away_odd,
                            "fair_odds": 1/away_prob,
                            "recommendation": get_recommendation(away_odd, 1/away_prob),
                            "confidence": "Media",
                            "value": round((away_odd * away_prob) - 1, 2)
                        })
                    
                    # Añadir análisis de mercado y valor
                    normalized["market_sentiment"] = {
                        "description": "Basado en datos de mercado",
                        "implied_probabilities": {
                            "home_win": round(home_prob, 2),
                            "draw": round(draw_prob, 2),
                            "away_win": round(away_prob, 2)
                        }
                    }
                    
                    normalized["value_opportunities"] = value_opps
                    
                except (ValueError, ZeroDivisionError) as e:
                    logger.warning(f"Error procesando odds para partido {fixture_id}: {str(e)}")
                    # Usar valores por defecto
                    normalized.update(generate_simulated_odds(fixture_id))
            else:
                normalized.update(generate_simulated_odds(fixture_id))
        else:
            normalized.update(generate_simulated_odds(fixture_id))
            
        return normalized
    
    except Exception as e:
        logger.error(f"Error normalizando datos de odds: {str(e)}")
        return generate_simulated_odds(0)

def get_recommendation(market_odds, fair_odds):
    """
    Determinar recomendación basada en valor
    
    Args:
        market_odds: Odds del mercado
        fair_odds: Odds calculados como justos
        
    Returns:
        Recomendación (Value, Avoid, Neutral)
    """
    if market_odds >= fair_odds * 1.05:
        return "Value"
    elif market_odds <= fair_odds * 0.95:
        return "Avoid"
    else:
        return "Neutral"

def generate_simulated_odds(fixture_id):
    """
    Genera odds simuladas cuando no hay datos reales
    
    Args:
        fixture_id: ID del partido
        
    Returns:
        Datos de odds simulados
    """
    import random
    
    # En una implementación real, podríamos usar datos históricos o modelos para generar odds más realistas
    home_prob = random.uniform(0.35, 0.50)
    draw_prob = random.uniform(0.20, 0.30)
    away_prob = 1 - home_prob - draw_prob
    
    home_odd = round(1/home_prob, 2)
    draw_odd = round(1/draw_prob, 2)
    away_odd = round(1/away_prob, 2)
    
    return {
        "simulated": True,
        "source": "Datos simulados",
        "timestamp": datetime.now().isoformat(),
        "fixture_id": fixture_id,
        "market_sentiment": {
            "description": "Sin datos de mercado disponibles",
            "implied_probabilities": {
                "home_win": round(home_prob, 2),
                "draw": round(draw_prob, 2),
                "away_win": round(away_prob, 2)
            }
        },
        "value_opportunities": [
            {
                "market": "Match Winner",
                "selection": "Home" if home_prob > 0.4 else "Draw" if draw_prob > 0.28 else "Away",
                "market_odds": home_odd if home_prob > 0.4 else draw_odd if draw_prob > 0.28 else away_odd,
                "fair_odds": round(home_odd * random.uniform(0.95, 1.05), 2),
                "recommendation": "Neutral",
                "confidence": "Baja",
                "value": round(random.uniform(-0.02, 0.02), 2)
            }
        ]
    }

def update_config():
    """Actualizar configuración con información de la API de odds"""
    logger.info("Actualizando configuración...")
    
    # Asegurarse de que la configuración tiene los valores necesarios
    config_path = Path("config.py")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        changes_made = False
            
        # Verificar si ya existe la configuración específica de endpoints de odds
        if "ODDS_ENDPOINTS" not in content:
            # Añadimos información sobre endpoints de odds en la configuración
            odds_config = """
# Configuración de Odds (usa la misma API)
ODDS_ENDPOINTS = {
    "pre_match": "/odds",
    "live": "/odds/live", 
    "bookmakers": "/odds/bookmakers",
    "bets": "/odds/bets"
}
"""
            # Añadir al final del archivo
            with open(config_path, 'a', encoding='utf-8') as f:
                f.write(odds_config)
            
            logger.info("Añadida configuración de endpoints de odds en config.py")
            changes_made = True
        
        # Verificar si están los bookmakers prioritarios
        if "ODDS_BOOKMAKERS_PRIORITY" not in content:
            # Añadir la nueva configuración
            new_config = """
# Odds API Constants
ODDS_BOOKMAKERS_PRIORITY = [1, 6, 8, 2]  # Bookmakers prioritarios por ID
ODDS_DEFAULT_MARKETS = [1, 2, 3]  # Match Winner, Home/Away, Over/Under
"""
            
            # Añadir al final del archivo
            with open(config_path, 'a', encoding='utf-8') as f:
                f.write(new_config)
            
            logger.info("Configuración de odds ampliada en config.py")
            changes_made = True
        
        if changes_made:
            logger.info("Configuración actualizada con éxito")
        else:
            logger.info("La configuración ya estaba correctamente configurada")
            
        return True
            
    except Exception as e:
        logger.error(f"Error actualizando configuración: {str(e)}")
        return False

def test_odds_integration(fixture_ids=None):
    """
    Probar la integración de odds con el sistema
    
    Args:
        fixture_ids: Lista de IDs de partidos para probar
        
    Returns:
        Resultados de las pruebas
    """
    # Si no se proporcionan IDs, intentamos obtener partidos con odds disponibles
    if fixture_ids is None:
        # Intentar obtener partidos con odds directamente
        try:
            logger.info("Buscando partidos con odds disponibles...")
            
            # Intentar con múltiples ligas populares para aumentar posibilidades de encontrar odds
            leagues_to_try = ["39", "140", "78", "135", "61", "2", "3", "71"]  # Principales ligas europeas + Champions
            
            fixture_ids = []
            
            # Configuración de API
            odds_endpoint = f"{API_BASE_URL}/odds"
            odds_headers = {
                "x-rapidapi-key": API_KEY,
                "x-rapidapi-host": API_BASE_URL.replace("https://", "")
            }
            
            # Probamos diferentes bookmakers
            bookmakers_to_try = ["8", "6", "1", "2"]  # 888sport, Bwin, 10bet, 1xBet
            seasons_to_try = ["2023", "2024"]  # Temporada actual y siguiente
            
            # Bucle de búsqueda
            for league_id in leagues_to_try:
                if fixture_ids:
                    break
                    
                logger.info(f"Buscando en liga {league_id}...")
                
                for bookmaker_id in bookmakers_to_try:
                    if fixture_ids:
                        break
                        
                    for season in seasons_to_try:
                        if fixture_ids:
                            break
                            
                        # Parámetros de búsqueda
                        odds_params = {
                            "league": league_id,
                            "season": season,
                            "bookmaker": bookmaker_id,
                            "bet": "1"  # Match Winner
                        }
                        
                        try:
                            # Solicitud a la API
                            odds_response = requests.get(
                                odds_endpoint,
                                headers=odds_headers,
                                params=odds_params,
                                timeout=15
                            )
                            
                            # Procesar respuesta
                            if odds_response.status_code == 200:
                                data = odds_response.json()
                                if data.get("response") and len(data["response"]) > 0:
                                    # Filtrar partidos con bookmakers (datos reales)
                                    fixtures_with_odds = []
                                    
                                    for odds_data in data["response"][:10]:
                                        if odds_data.get("bookmakers", []):
                                            fixture_id = odds_data.get("fixture", {}).get("id")
                                            if fixture_id:
                                                fixtures_with_odds.append(fixture_id)
                                    
                                    # Si encontramos partidos con odds
                                    if fixtures_with_odds:
                                        fixture_ids = fixtures_with_odds[:3]
                                        logger.info(f"Encontrados {len(fixture_ids)} partidos con odds disponibles en liga {league_id}, temporada {season}")
                                        break
                                        
                        except Exception as e:
                            logger.warning(f"Error con bookmaker {bookmaker_id}, temporada {season}: {str(e)}")
                            continue
                
            if fixture_ids:
                logger.info("Se encontraron partidos con odds reales disponibles")
            else:
                logger.warning("No se encontraron partidos con odds disponibles en ninguna liga")
                
        except Exception as e:
            logger.warning(f"Error obteniendo partidos con odds: {str(e)}")
    
    # Si todavía no tenemos fixture_ids, intentar con partidos programados
    if not fixture_ids:
        try:
            logger.info("Buscando próximos partidos...")
            fixtures_endpoint = f"{API_BASE_URL}/fixtures"
            fixtures_params = {
                "league": "39",  # Premier League
                "season": "2023",
                "next": "5"  # Próximos 5 partidos
            }
            fixtures_headers = {
                "x-rapidapi-key": API_KEY,
                "x-rapidapi-host": API_BASE_URL.replace("https://", "")
            }
            
            fixtures_response = requests.get(
                fixtures_endpoint, 
                headers=fixtures_headers, 
                params=fixtures_params, 
                timeout=15
            )
            
            if fixtures_response.status_code == 200:
                data = fixtures_response.json()
                if data.get("response") and len(data["response"]) > 0:
                    # Usar los primeros 3 partidos encontrados
                    fixture_ids = [
                        fix.get("fixture", {}).get("id") 
                        for fix in data["response"][:3]
                    ]
                    logger.info(f"Obtenidos {len(fixture_ids)} partidos programados para probar")
        except Exception as e:
            logger.warning(f"Error obteniendo partidos programados: {str(e)}")
    
    # Si todavía no tenemos fixture_ids, usar valores más actuales basados en documentación
    if not fixture_ids:
        # IDs de partidos actuales de ligas principales
        fixture_ids = [1128079, 1128080, 1128081]  # IDs actualizados de 2023/2024
        logger.info("Usando IDs de partidos actuales predeterminados")
    
    results = {}
    simulated_count = 0
    
    for fixture_id in fixture_ids:
        logger.info(f"Probando odds para partido {fixture_id}...")
        
        # Obtener odds con caché
        start_time = datetime.now()
        data = get_fixture_odds(fixture_id, use_cache=True)
        cached_time = (datetime.now() - start_time).total_seconds()
        
        # Verificar si son simuladas
        is_simulated = data.get("simulated", True)
        if is_simulated:
            simulated_count += 1
        
        # Guardar resultados
        results[fixture_id] = {
            "simulated": is_simulated,
            "source": data.get("source", "Desconocido"),
            "time_cached": cached_time,
            "home_win_prob": data.get("market_sentiment", {}).get("implied_probabilities", {}).get("home_win", 0),
            "has_value_opps": len(data.get("value_opportunities", [])) > 0
        }
    
    # Estadísticas generales
    total = len(fixture_ids)
    simulated_pct = (simulated_count / total) * 100 if total > 0 else 0

    logger.info(f"Resultados de pruebas de odds:")
    logger.info(f"- Total de partidos: {total}")
    logger.info(f"- Odds simuladas: {simulated_count} ({simulated_pct:.1f}%)")
    
    for fixture_id, result in results.items():
        logger.info(f"- Partido {fixture_id}: {'Simulado' if result['simulated'] else 'Real'} | "
                   f"Fuente: {result['source']} | "
                   f"Prob. victoria local: {result['home_win_prob']}")
    
    return results

def main():
    """Ejecutar optimización de odds"""
    logger.info("="*60)
    logger.info("INICIANDO OPTIMIZACIÓN DE ODDS")
    logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"API URL: {API_BASE_URL}")
    logger.info(f"API Key: ...{API_KEY[-4:] if len(API_KEY) > 4 else '****'}")
    logger.info("="*60)
    
    # Configurar caché
    setup_cache()
    
    # Actualizar configuración
    update_config()
    
    # Limpiar caché
    clear_expired_cache()
    
    # Probar integración
    test_odds_integration()
    
    logger.info("\n" + "="*60)
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info("="*60)

if __name__ == "__main__":
    try:
        main()
        print("\nOptimización completada. Revise el archivo de log para detalles.")
    except Exception as e:
        logger.error(f"Error en optimización: {str(e)}")
        print(f"\nError durante la optimización: {str(e)}")
        print("Revise el archivo de log para más detalles.")
