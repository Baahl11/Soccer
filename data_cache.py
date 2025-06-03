"""
Módulo para manejar el caché de datos de la API de fútbol.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class DataCache:
    """Clase para manejar el caché de datos de la API."""
    
    def __init__(self, cache_dir: str = 'cache', ttl_days: int = 7):
        """
        Inicializar el caché.
        
        Args:
            cache_dir: Directorio para almacenar los archivos de caché
            ttl_days: Tiempo de vida del caché en días
        """
        self.cache_dir = Path(cache_dir)
        self.ttl = timedelta(days=ttl_days)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Genera una clave única para los parámetros de la petición."""
        # Ordenar params para asegurar consistencia
        sorted_params = json.dumps(params, sort_keys=True)
        key = f"{endpoint}_{sorted_params}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Obtiene la ruta del archivo de caché para una clave."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Obtiene datos del caché si existen y son válidos.
        
        Returns:
            Datos del caché o None si no existen o están expirados
        """
        key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
            
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                
            # Verificar TTL
            cached_time = datetime.fromisoformat(cached_data['cached_at'])
            if datetime.now() - cached_time > self.ttl:
                return None
                
            return cached_data['data']
        except Exception:
            return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Dict[str, Any]):
        """
        Guarda datos en el caché.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            data: Datos a guardar
        """
        key = self._get_cache_key(endpoint, params)
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
        except Exception:
            # Si hay error al escribir el caché, lo ignoramos
            pass
