"""
Módulo de caché para datos de odds

Este módulo implementa un sistema de caché avanzado para datos de odds,
con soporte para:
- Caché por fixture_id
- Gestión de expiración basada en configuración
- Operaciones por lotes
- Estadísticas de uso
- Pre-carga de datos

Autor: Equipo de Desarrollo
Fecha: Mayo 23, 2025
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Set

# Configuración de logging
logger = logging.getLogger('odds_cache')

class OddsCache:
    """Sistema de caché para datos de odds."""
    
    def __init__(self, cache_dir: Union[str, Path], expiry_minutes: Optional[int] = None):
        """
        Inicializa el sistema de caché.
        
        Args:
            cache_dir: Directorio para almacenar los datos de caché
            expiry_minutes: Tiempo de expiración de la caché en minutos (opcional, por defecto usa config.CACHE_CONFIG)
        """
        self.cache_dir = Path(cache_dir)
        
        # Si no se especifica tiempo de expiración, usar el de la configuración
        if expiry_minutes is None:
            try:
                from config import CACHE_CONFIG
                # Convertir de timedelta a minutos
                expiry_td = CACHE_CONFIG.get("odds", timedelta(minutes=15))
                self.expiry_minutes = expiry_td.total_seconds() // 60
            except (ImportError, AttributeError):
                # Usar valor predeterminado si no se puede obtener de config
                self.expiry_minutes = 15
                logger.warning("No se pudo cargar CACHE_CONFIG, usando expiración predeterminada de 15 minutos")
        else:
            self.expiry_minutes = expiry_minutes
        
        # Crear directorio de caché si no existe
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Caché inicializada en {self.cache_dir} con expiración de {self.expiry_minutes} minutos")
        
        # Estadísticas de caché
        self.stats = {
            "hits": 0,
            "misses": 0,
            "expirations": 0,
            "writes": 0,
            "errors": 0,
            "batch_operations": 0,
            "cleanup_operations": 0,
            "items_cleaned": 0,
            "cache_size_bytes": 0
        }
        
        # Actualizar tamaño de caché al inicializar
        self._update_cache_size()
    
    def _update_cache_size(self) -> int:
        """Actualiza y devuelve el tamaño actual de la caché en bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("odds_*.json"):
            total_size += cache_file.stat().st_size
        
        self.stats["cache_size_bytes"] = total_size
        return total_size
    
    def _get_cache_file(self, fixture_id: int) -> Path:
        """Devuelve la ruta al archivo de caché para un partido."""
        return self.cache_dir / f"odds_{fixture_id}.json"
    
    def get(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Obtiene datos de odds de la caché si existen y no han expirado."""
        cache_file = self._get_cache_file(fixture_id)
        
        if not cache_file.exists():
            self.stats["misses"] += 1
            return None
        
        try:
            # Verificar si el archivo ha expirado
            file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - file_time > timedelta(minutes=self.expiry_minutes):
                logger.info(f"Caché expirada para partido {fixture_id}")
                self.stats["expirations"] += 1
                return None
            
            # Cargar datos de caché
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Datos de odds cargados de caché para partido {fixture_id}")
                self.stats["hits"] += 1
                return data
            
        except Exception as e:
            logger.warning(f"Error leyendo caché para partido {fixture_id}: {str(e)}")
            self.stats["errors"] += 1
            return None
    
    def set(self, fixture_id: int, odds_data: Dict[str, Any]) -> bool:
        """Guarda datos de odds en la caché."""
        cache_file = self._get_cache_file(fixture_id)
        
        try:
            # Guardar datos en caché
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(odds_data, f, indent=2)
                
            logger.info(f"Datos de odds guardados en caché para partido {fixture_id}")
            self.stats["writes"] += 1
            
            # Actualizar tamaño de caché
            self._update_cache_size()
            return True
            
        except Exception as e:
            logger.error(f"Error guardando caché para partido {fixture_id}: {str(e)}")
            self.stats["errors"] += 1
            return False
            
    def invalidate(self, fixture_id: int) -> bool:
        """Invalida la caché para un partido específico."""
        cache_file = self._get_cache_file(fixture_id)
        
        if cache_file.exists():
            try:
                cache_file.unlink()
                logger.info(f"Caché invalidada para partido {fixture_id}")
                
                # Actualizar tamaño de caché
                self._update_cache_size()
                return True
            except Exception as e:
                logger.error(f"Error invalidando caché para partido {fixture_id}: {str(e)}")
                self.stats["errors"] += 1
                return False
        return False
    
    def get_batch(self, fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """
        Obtiene datos de odds en lote para múltiples partidos.
        
        Args:
            fixture_ids: Lista de IDs de partidos
            
        Returns:
            Diccionario con {fixture_id: datos_odds} para los partidos encontrados en caché
        """
        self.stats["batch_operations"] += 1
        result = {}
        
        for fixture_id in fixture_ids:
            data = self.get(fixture_id)
            if data:
                result[fixture_id] = data
                
        logger.info(f"Operación por lotes: {len(result)}/{len(fixture_ids)} partidos encontrados en caché")
        return result
    
    def set_batch(self, odds_data_batch: Dict[int, Dict[str, Any]]) -> int:
        """
        Guarda datos de odds en lote para múltiples partidos.
        
        Args:
            odds_data_batch: Diccionario con {fixture_id: datos_odds}
            
        Returns:
            Número de partidos guardados correctamente
        """
        self.stats["batch_operations"] += 1
        success_count = 0
        
        for fixture_id, odds_data in odds_data_batch.items():
            if self.set(fixture_id, odds_data):
                success_count += 1
                
        logger.info(f"Operación por lotes: {success_count}/{len(odds_data_batch)} partidos guardados en caché")
        return success_count
    
    def clean_expired(self) -> int:
        """
        Limpia entradas de caché expiradas.
        
        Returns:
            Número de archivos de caché eliminados
        """
        self.stats["cleanup_operations"] += 1
        count = 0
        now = datetime.now()
        
        for cache_file in self.cache_dir.glob("odds_*.json"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if now - file_time > timedelta(minutes=self.expiry_minutes):
                    cache_file.unlink()
                    count += 1
            except Exception as e:
                logger.warning(f"Error limpiando caché {cache_file.name}: {str(e)}")
                self.stats["errors"] += 1
        
        # Actualizar estadísticas
        self.stats["items_cleaned"] += count
        self._update_cache_size()
        
        logger.info(f"Limpiados {count} archivos de caché expirados")
        return count
    
    def get_all_fixture_ids(self) -> Set[int]:
        """
        Obtiene el conjunto de todos los IDs de partidos en caché.
        
        Returns:
            Set con todos los fixture_ids en caché
        """
        fixture_ids = set()
        
        for cache_file in self.cache_dir.glob("odds_*.json"):
            try:
                # Extraer fixture_id del nombre de archivo (formato: odds_XXXX.json)
                filename = cache_file.name
                if filename.startswith("odds_") and filename.endswith(".json"):
                    fixture_id_str = filename[5:-5]  # Extraer parte entre "odds_" y ".json"
                    fixture_id = int(fixture_id_str)
                    fixture_ids.add(fixture_id)
            except (ValueError, IndexError):
                continue
                
        return fixture_ids
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de uso de la caché.
        
        Returns:
            Diccionario con estadísticas de uso
        """
        # Actualizar tamaño antes de devolver estadísticas
        self._update_cache_size()
          # Calcular métricas adicionales
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_ratio = self.stats["hits"] / max(total_requests, 1)
        
        # Añadir métricas calculadas - create a new dictionary with all values combined
        stats = {
            **self.stats,
            "total_requests": total_requests,
            "hit_ratio": hit_ratio,
            "cache_size_mb": self.stats["cache_size_bytes"] / (1024 * 1024),
            "timestamp": datetime.now().isoformat()
        }
        
        return stats
