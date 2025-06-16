import requests
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from data import get_cache_manager

logger = logging.getLogger(__name__)

class FootballAPI:
    BASE_URL = "https://v3.football.api-sports.io"
    API_KEY = os.getenv("API_FOOTBALL_KEY")
    RATE_LIMIT = 30  # requests per minute
    
    def __init__(self):
        self.session = requests.Session()
        api_key = self.API_KEY
        
        if not api_key:
            raise ValueError("API_FOOTBALL_KEY no está configurada en las variables de entorno")
            
        self.session.headers = {
            'x-rapidapi-host': 'v3.football.api-sports.io',
            'x-rapidapi-key': api_key
        }
        self.cache = get_cache_manager()
        self.last_request_time = time.time()
        self.request_times = []  # Track last minute of requests
    
    def _respect_rate_limit(self):
        """Ensure we respect the requests per minute limit with safety margin."""
        # Usar un límite más conservador (80% del límite real)
        effective_limit = int(self.RATE_LIMIT * 0.8)
        min_delay = 60.0 / effective_limit
        current_time = time.time()
        
        # Limpiar solicitudes antiguas (más de 60 segundos)
        self.request_times = [t for t in self.request_times if current_time - t < 60]
        
        # Si estamos cerca del límite, esperar más tiempo
        if len(self.request_times) >= (effective_limit - 2):
            wait_time = 60 - (current_time - self.request_times[0]) + 1
            if wait_time > 0:
                logger.info(f"Approaching rate limit, waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)
                current_time = time.time()
                self.request_times = []
        
        # Asegurar delay mínimo entre solicitudes
        if self.request_times:
            time_since_last = current_time - self.request_times[-1]
            if time_since_last < min_delay:
                time.sleep(min_delay - time_since_last)
        
        # Registrar esta solicitud
        self.request_times.append(time.time())
        self.last_request_time = self.request_times[-1]

    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, 
                     max_age_hours: int = 24, cache_bypass: bool = False) -> Dict[str, Any]:
        """Hace una petición a la API usando el caché"""
        params = params or {}
        max_retries = 3
        retry_delay = 2  # segundos base entre reintentos
        
        # Intentar obtener del caché primero
        if not cache_bypass and self.cache:
            cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
            cached_data = self.cache.get_data(cache_key)
            if cached_data is not None:
                logger.info(f"Usando datos en caché para {endpoint}")
                return cached_data
        
        for attempt in range(max_retries):
            try:
                # Respetar el rate limit
                self._respect_rate_limit()
                
                # Hacer la petición
                response = self.session.get(f"{self.BASE_URL}/{endpoint}", params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    # Verificar que la respuesta tenga datos válidos
                    if not data or not isinstance(data.get('response'), (list, dict)):
                        raise ValueError("Invalid API response format")
                    
                    # Guardar en caché solo si los datos son válidos
                    if self.cache:
                        cache_key = f"{endpoint}_{hash(frozenset(params.items()))}"
                        self.cache.set_data(cache_key, data)
                    return data
                    
                elif response.status_code == 429:  # Rate limit excedido
                    retry_after = int(response.headers.get('Retry-After', retry_delay * (attempt + 1)))
                    logger.warning(f"Rate limit excedido. Esperando {retry_after} segundos...")
                    time.sleep(retry_after)
                    continue
                    
                elif response.status_code == 403:  # API key inválida
                    logger.error("API key inválida o expirada")
                    raise ValueError("Invalid API key")
                    
                else:
                    logger.error(f"Error en petición a {endpoint}: {response.status_code}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))
                        continue
                    raise ValueError(f"API request failed with status {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error en intento {attempt + 1} para {endpoint}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
                
        logger.error(f"Todos los intentos fallaron para {endpoint}")
        raise ValueError(f"All retries failed for {endpoint}")

    def get_fixtures(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get fixtures based on parameters."""
        data = self._make_request('fixtures', params)
        return data.get('response', [])
