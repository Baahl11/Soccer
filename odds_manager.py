"""
OddsManager - Sistema de gestión de odds con múltiples proveedores

Este módulo implementa un sistema avanzado para obtener y gestionar odds de fútbol,
con soporte para múltiples proveedores, fallback automático y caché.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import requests
import logging
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import random  # Solo para simulación durante desarrollo
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from odds_cache import OddsCache  # Importar nuevo sistema de caché

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='odds_manager.log',
    filemode='a'
)

logger = logging.getLogger('odds_manager')

class OddsProviderConfig:
    """Configuración para un proveedor de odds."""
    
    def __init__(self, 
                 name: str, 
                 api_url: str,
                 api_key: str, 
                 endpoint_path: str = "/odds",
                 rate_limit: int = 100, 
                 timeout: int = 10,
                 retry_delay: int = 5,
                 max_retries: int = 3,
                 priority: int = 1):
        """
        Inicializa la configuración del proveedor.
        
        Args:
            name: Nombre del proveedor
            api_url: URL base de la API
            api_key: Clave de API para autenticación
            endpoint_path: Ruta del endpoint de odds
            rate_limit: Límite de solicitudes por minuto
            timeout: Tiempo de espera para solicitudes (segundos)
            retry_delay: Tiempo de espera entre reintentos (segundos)
            max_retries: Número máximo de reintentos
            priority: Prioridad del proveedor (1 es la más alta)
        """
        self.name = name
        self.api_url = api_url
        self.api_key = api_key
        self.endpoint_path = endpoint_path
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.priority = priority
        
    @property
    def full_url(self) -> str:
        """Devuelve la URL completa del endpoint de odds."""
        return f"{self.api_url.rstrip('/')}{self.endpoint_path}"
    
    def __repr__(self) -> str:
        """Representación de string de la configuración."""
        return f"OddsProviderConfig(name={self.name}, priority={self.priority})"


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
            "errors": 0
        }
    
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
                return True
            except Exception as e:
                logger.error(f"Error invalidando caché para partido {fixture_id}: {str(e)}")
                return False
        
        return True  # No hay caché para invalidar
    
    def clear_expired(self) -> int:
        """Limpia los archivos de caché que han expirado."""
        cleared_count = 0
        
        for cache_file in self.cache_dir.glob("odds_*.json"):
            try:
                file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                if datetime.now() - file_time > timedelta(minutes=self.expiry_minutes):
                    cache_file.unlink()
                    cleared_count += 1
            except Exception as e:
                logger.warning(f"Error al limpiar caché {cache_file.name}: {str(e)}")
        
        if cleared_count > 0:
            logger.info(f"Limpiados {cleared_count} archivos de caché expirados")
            
        return cleared_count


class OddsManager:
    """Gestor de odds con soporte para múltiples proveedores y fallback."""
    
    def __init__(self, cache_dir: Union[str, Path] = None):
        """
        Inicializa el gestor de odds.
        
        Args:
            cache_dir: Directorio para caché. Si es None, se usa ./odds_cache
        """
        self.providers: List[OddsProviderConfig] = []
        
        # Inicializar caché si se especifica directorio
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "odds_cache"
            
        self.cache = OddsCache(cache_dir)
        
        # Estadísticas de uso
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "api_success": 0,
            "api_failures": 0,
            "simulated_odds": 0
        }
        
        logger.info("OddsManager inicializado")
    
    def add_provider(self, provider: OddsProviderConfig) -> None:
        """Añade un proveedor de odds al gestor."""
        self.providers.append(provider)
        # Ordenar por prioridad (1 es la más alta)
        self.providers.sort(key=lambda p: p.priority)
        logger.info(f"Proveedor añadido: {provider.name} (prioridad {provider.priority})")
    
    def get_odds(self, fixture_id: int, force_refresh: bool = False) -> Tuple[Dict[str, Any], bool]:
        """
        Obtiene odds para un partido, con sistema de proveedores en cascada.
        
        Args:
            fixture_id: ID del partido
            force_refresh: Si es True, ignora la caché y refresca los datos
            
        Returns:
            Tuple con (datos de odds, es_simulado)
        """
        self.stats["total_requests"] += 1
        
        # 1. Intentar obtener de caché si no se fuerza refresco
        if not force_refresh:
            cached_odds = self.cache.get(fixture_id)
            if cached_odds:
                self.stats["cache_hits"] += 1
                return cached_odds, cached_odds.get("simulated", True)
        
        # 2. Intentar cada proveedor en orden de prioridad
        for provider in self.providers:
            try:
                # Intentar obtener odds del proveedor
                odds_data = self._fetch_from_provider(provider, fixture_id)
                
                # Si se obtuvieron datos, guardar en caché y devolver
                if odds_data:
                    odds_data["simulated"] = False
                    odds_data["source"] = f"Proveedor: {provider.name}"
                    odds_data["timestamp"] = datetime.now().isoformat()
                    
                    # Guardar en caché
                    self.cache.set(fixture_id, odds_data)
                    
                    self.stats["api_success"] += 1
                    return odds_data, False
                    
            except Exception as e:
                logger.warning(f"Error obteniendo odds de {provider.name} para partido {fixture_id}: {str(e)}")
        
        # 3. Si todos los proveedores fallan, usar datos simulados
        self.stats["api_failures"] += 1
        self.stats["simulated_odds"] += 1
        
        simulated_odds = self._generate_simulated_odds(fixture_id)
        
        # No guardamos odds simuladas en caché para evitar perpetuar datos falsos
        
        logger.warning(f"Generando odds simuladas para partido {fixture_id} después de agotar proveedores")
        return simulated_odds, True
    
    def _fetch_from_provider(self, provider: OddsProviderConfig, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Obtiene odds de un proveedor específico con sistema de reintentos.
        
        Args:
            provider: Configuración del proveedor
            fixture_id: ID del partido
            
        Returns:
            Datos de odds o None si falla
        """
        endpoint = f"{provider.full_url}/fixture/{fixture_id}"
        headers = {
            "X-API-KEY": provider.api_key,
            "Content-Type": "application/json"
        }
        
        # Sistema de reintentos con backoff exponencial
        for attempt in range(1, provider.max_retries + 1):
            try:
                logger.info(f"Intento {attempt}/{provider.max_retries} obteniendo odds de {provider.name} para partido {fixture_id}")
                
                response = requests.get(
                    endpoint, 
                    headers=headers, 
                    timeout=provider.timeout
                )
                
                # Si la solicitud es exitosa, parsear y devolver datos
                if response.status_code == 200:
                    try:
                        data = response.json()
                        return self._normalize_odds_data(data, provider.name)
                    except json.JSONDecodeError:
                        logger.warning(f"Respuesta de {provider.name} no es JSON válido")
                        
                # Si hay rate limiting, esperar y reintentar
                elif response.status_code in (429, 503):  # Too Many Requests o Service Unavailable
                    wait_time = provider.retry_delay * (2 ** (attempt - 1))  # Backoff exponencial
                    logger.warning(f"Rate limiting detectado en {provider.name}, esperando {wait_time}s antes de reintentar")
                    time.sleep(wait_time)
                    continue
                    
                # Si hay error de autenticación, no seguir intentando
                elif response.status_code in (401, 403):  # Unauthorized o Forbidden
                    logger.error(f"Error de autenticación con {provider.name}: {response.status_code}")
                    break
                    
                # Otros errores
                else:
                    logger.warning(f"Error obteniendo odds de {provider.name}: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout obteniendo odds de {provider.name}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Error de conexión con {provider.name}: {str(e)}")
                
            # Esperar antes de reintentar, solo si no es el último intento
            if attempt < provider.max_retries:
                wait_time = provider.retry_delay * (2 ** (attempt - 1))
                time.sleep(wait_time)
        
        return None
    
    def _normalize_odds_data(self, data: Dict[str, Any], provider_name: str) -> Dict[str, Any]:
        """
        Normaliza los datos de odds a un formato estándar.
        Cada proveedor puede tener un formato diferente, así que aquí se adapta.
        """
        # Aquí implementaríamos la lógica específica según el proveedor
        # Por ahora, un ejemplo simplificado
        
        normalized = {
            "simulated": False,
            "source": f"Proveedor: {provider_name}",
            "timestamp": datetime.now().isoformat()
        }
        
        # Intentar extraer datos según el formato del proveedor
        try:
            # Si el proveedor ya usa nuestro formato
            if "market_sentiment" in data and "value_opportunities" in data:
                normalized.update(data)
                return normalized
                
            # Formato del proveedor 1
            if "odds" in data and "bookmakers" in data["odds"]:
                normalized["market_sentiment"] = {
                    "description": self._generate_sentiment_description(data),
                    "implied_probabilities": self._calculate_implied_probabilities(data)
                }
                normalized["value_opportunities"] = self._extract_value_opportunities(data)
                return normalized
                
            # Formato del proveedor 2
            if "markets" in data:
                # Implementar conversión específica
                pass
                
            # Si no coincide con ningún formato conocido
            return self._generate_simulated_odds(data.get("fixture_id", 0))
            
        except Exception as e:
            logger.warning(f"Error normalizando datos de {provider_name}: {str(e)}")
            return self._generate_simulated_odds(data.get("fixture_id", 0))
    
    def _generate_sentiment_description(self, data: Dict[str, Any]) -> str:
        """Genera una descripción del sentimiento del mercado basada en los datos."""
        # Lógica para generar descripción
        # Esta sería una implementación real que analiza las probabilidades implícitas
        return "Basado en datos del mercado"
    
    def _calculate_implied_probabilities(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calcula las probabilidades implícitas basadas en las odds del mercado."""
        # Implementación real que convierte odds a probabilidades
        # Por ahora devolvemos valores de ejemplo
        return {
            "home_win": 0.45,
            "draw": 0.28, 
            "away_win": 0.27
        }
    
    def _extract_value_opportunities(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extrae oportunidades de valor basadas en las odds del mercado."""
        # Implementación real que compara nuestras probabilidades con las del mercado
        # Por ahora devolvemos valores de ejemplo
        return [
            {
                "market": "Match Winner",
                "selection": "Home",
                "market_odds": 2.20,
                "fair_odds": 2.22,
                "recommendation": "Value",
                "confidence": "Media",
                "value": 0.01
            }
        ]
    
    def _generate_simulated_odds(self, fixture_id: int) -> Dict[str, Any]:
        """Genera odds simuladas cuando no se pueden obtener datos reales."""
        # Nota: Esta función debería implementarse con un modelo más sofisticado
        # Por ahora generamos valores aleatorios para demostración
        
        # En una implementación real, usaríamos datos históricos o modelos estadísticos
        
        home_prob = random.uniform(0.35, 0.50)
        draw_prob = random.uniform(0.20, 0.30)
        away_prob = 1 - home_prob - draw_prob
        
        return {
            "simulated": True,
            "source": "Datos simulados",
            "timestamp": datetime.now().isoformat(),
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
                    "market_odds": round(1/home_prob, 2),
                    "fair_odds": round(1/home_prob * random.uniform(0.95, 1.05), 2),
                    "recommendation": "Neutral",
                    "confidence": "Baja",
                    "value": round(random.uniform(-0.02, 0.02), 2)                }
            ]
        }
        
    def get_stats(self) -> Dict[str, int]:
        """Devuelve estadísticas de uso del gestor de odds."""
        return self.stats
    
    def clear_cache(self) -> int:
        """Limpia los archivos de caché expirados."""
        return self.cache.clear_expired()
        
    def _setup_default_providers(self):
        """Configura proveedores por defecto si no se han configurado manualmente."""
        logger.info("Configurando proveedores de odds por defecto")
        
        # Verificar si ya hay proveedores configurados
        if self.providers:
            return
          # Intentar cargar desde variables de entorno o config.py
        try:
            import os
            from config import API_KEY, API_BASE_URL
            
            # Proveedor principal desde configuración
            self.add_provider(OddsProviderConfig(
                name="API-Football",
                api_url=API_BASE_URL,
                api_key=API_KEY,
                priority=1
            ))
            
        except (ImportError, AttributeError):
            logger.warning("No se pudo cargar configuración de odds desde config.py")
            
            # Si no se pudo cargar desde config, usar valores genéricos
            api_key = os.environ.get('API_FOOTBALL_KEY', 'key_no_encontrada')
            
            self.add_provider(OddsProviderConfig(
                name="OddsProvider1",
                api_url="https://api.oddsapi.com/v1",
                api_key=api_key,
                priority=1
            ))
        
        # Añadir un segundo proveedor de respaldo
        self.add_provider(OddsProviderConfig(
            name="OddsProvider2",
            api_url="https://api.odds-backup.com/v2",
            api_key=os.environ.get('ODDS_BACKUP_API_KEY', 'key_no_encontrada'),
            priority=2
        ))
        
        logger.info(f"Configurados {len(self.providers)} proveedores por defecto")
    
    def get_odds_for_fixture(
        self, 
        fixture_id: int, 
        competition_id: Optional[int] = None,
        home_team_id: Optional[int] = None,
        away_team_id: Optional[int] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Obtiene odds para un partido específico.
        Este es el método principal para usar desde otros componentes del sistema.
        
        Args:
            fixture_id: ID del partido
            competition_id: ID de la competición (opcional para priorizar proveedores)
            home_team_id: ID del equipo local (opcional para enriquecer datos simulados)
            away_team_id: ID del equipo visitante (opcional para enriquecer datos simulados)
            use_cache: Si es True, consultar primero la caché
            
        Returns:
            Datos de odds (se indica si son simulados en el campo "simulated")
        """
        # Verificar si hay proveedores configurados
        if not self.providers:
            # Si no hay proveedores, configuramos algunos por defecto
            self._setup_default_providers()
        
        # Obtener odds con el método interno
        odds_data, is_simulated = self.get_odds(fixture_id, force_refresh=not use_cache)
        
        # Si se recibieron IDs de equipos, mejorar datos simulados
        if is_simulated and home_team_id and away_team_id:
            # En una implementación real, aquí usaríamos los IDs de equipo
            # para generar odds más realistas basadas en datos históricos
            pass
        
        return odds_data


# Función de inicialización para uso directo en el sistema
def initialize_odds_manager() -> OddsManager:
    """Inicializa el gestor de odds con los proveedores configurados."""
    manager = OddsManager()
    
    # Cargar configuración (en producción se cargaría de config.py o .env)
    try:
        # Proveedor principal
        manager.add_provider(OddsProviderConfig(
            name="OddsPro API",
            api_url="https://api.oddspro.com/v1",
            api_key=os.environ.get("ODDS_PRO_API_KEY", "api_key_not_set"),
            priority=1
        ))
        
        # Proveedor secundario (fallback)
        manager.add_provider(OddsProviderConfig(
            name="BetData API",
            api_url="https://api.betdata.io/v2",
            api_key=os.environ.get("BETDATA_API_KEY", "api_key_not_set"),
            priority=2
        ))
        
        # Proveedor terciario (último recurso)
        manager.add_provider(OddsProviderConfig(
            name="OddsMarket API",
            api_url="https://api.oddsmarket.com/v1",
            api_key=os.environ.get("ODDSMARKET_API_KEY", "api_key_not_set"),
            priority=3
        ))
        
    except Exception as e:
        logger.error(f"Error inicializando proveedores de odds: {str(e)}")
    
    return manager


# Ejemplo de uso
if __name__ == "__main__":
    # Inicializar gestor
    odds_manager = initialize_odds_manager()
    
    # Ejemplo de obtención de odds
    fixture_id = 123456
    odds_data, is_simulated = odds_manager.get_odds(fixture_id)
    
    print(f"Odds para partido {fixture_id}:")
    print(f"Simuladas: {is_simulated}")
    print(f"Fuente: {odds_data.get('source', 'Desconocida')}")
    
    if "market_sentiment" in odds_data:
        sentiment = odds_data["market_sentiment"]
        print(f"Sentimiento: {sentiment.get('description', 'No disponible')}")
        
        if "implied_probabilities" in sentiment:
            probs = sentiment["implied_probabilities"]
            print(f"Victoria local: {probs.get('home_win', 0) * 100:.1f}%")
            print(f"Empate: {probs.get('draw', 0) * 100:.1f}%")
            print(f"Victoria visitante: {probs.get('away_win', 0) * 100:.1f}%")
    
    # Estadísticas de uso
    print("\nEstadísticas:")
    stats = odds_manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
