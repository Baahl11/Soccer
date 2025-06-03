from typing import Dict, Any, Optional
import json
import logging
from flask_caching import Cache
from flask import Flask

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, app: Flask):
        """Initialize cache manager with Flask app"""
        self.cache = Cache(app, config={
            'CACHE_TYPE': 'simple',  # Use simple in-memory cache
            'CACHE_DEFAULT_TIMEOUT': 3600  # Default timeout of 1 hour
        })

    def set_data(self, key: str, data: Dict[str, Any], expire: int = 3600) -> bool:
        """Store data in cache with expiration"""
        try:
            self.cache.set(key, json.dumps(data), timeout=expire)
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
            
    def get_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve data from cache"""
        try:
            data = self.cache.get(key)
            return json.loads(data) if data else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def delete_data(self, key: str) -> bool:
        """Delete data from cache"""
        try:
            self.cache.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False