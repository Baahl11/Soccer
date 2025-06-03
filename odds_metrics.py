"""
Módulo de métricas y monitoreo para la integración de odds API

Este módulo proporciona herramientas para:
- Monitorizar la salud de la integración con la API de odds
- Recopilar estadísticas sobre uso y eficiencia
- Analizar patrones de uso y calidad de datos 
- Alertar sobre problemas críticos

Autor: Equipo de Desarrollo
Fecha: Mayo 24, 2025
"""

import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuración de logging
logger = logging.getLogger('odds_metrics')

class OddsAPIMetrics:
    """Sistema para recopilar y analizar métricas de la integración de odds API"""
    
    def __init__(self, metrics_dir: Path = None):
        """
        Inicializar el sistema de métricas.
        
        Args:
            metrics_dir: Directorio para almacenar métricas
        """
        self.metrics_dir = metrics_dir or Path("data") / "metrics" / "odds"
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "simulated_count": 0,
            "real_data_count": 0,
            "response_times_ms": [],
            "bookmaker_coverage": {},
            "market_coverage": {},
        }
        
    def record_api_call(self, endpoint: str, params: Dict[str, Any], 
                      success: bool, response_time_ms: float,
                      simulated: bool = False) -> None:
        """
        Registrar una llamada a la API.
        
        Args:
            endpoint: Endpoint de la API llamado
            params: Parámetros de la solicitud
            success: Si la llamada fue exitosa
            response_time_ms: Tiempo de respuesta en milisegundos
            simulated: Si los datos devueltos fueron simulados
        """
        self.current_metrics["api_calls"] += 1
        self.current_metrics["response_times_ms"].append(response_time_ms)
        
        if not success:
            self.current_metrics["errors"] += 1
        
        if simulated:
            self.current_metrics["simulated_count"] += 1
        else:
            self.current_metrics["real_data_count"] += 1
            
        # Guardar detalles específicos por endpoint
        endpoint_key = endpoint.replace("/", "_").strip("_")
        if endpoint_key not in self.current_metrics:
            self.current_metrics[endpoint_key] = {
                "calls": 0,
                "errors": 0,
                "simulated": 0,
                "real": 0,
                "response_times_ms": []
            }
            
        self.current_metrics[endpoint_key]["calls"] += 1
        self.current_metrics[endpoint_key]["response_times_ms"].append(response_time_ms)
        
        if not success:
            self.current_metrics[endpoint_key]["errors"] += 1
            
        if simulated:
            self.current_metrics[endpoint_key]["simulated"] += 1
        else:
            self.current_metrics[endpoint_key]["real"] += 1
    
    def record_cache_event(self, hit: bool, fixture_id: int) -> None:
        """
        Registrar un evento de caché.
        
        Args:
            hit: True si fue un acierto, False si fue una falta
            fixture_id: ID del partido
        """
        if hit:
            self.current_metrics["cache_hits"] += 1
        else:
            self.current_metrics["cache_misses"] += 1
    
    def record_bookmaker_coverage(self, bookmaker_id: int, bookmaker_name: str) -> None:
        """
        Registrar cobertura de un bookmaker.
        
        Args:
            bookmaker_id: ID del bookmaker
            bookmaker_name: Nombre del bookmaker
        """
        if str(bookmaker_id) not in self.current_metrics["bookmaker_coverage"]:
            self.current_metrics["bookmaker_coverage"][str(bookmaker_id)] = {
                "name": bookmaker_name,
                "count": 0
            }
        
        self.current_metrics["bookmaker_coverage"][str(bookmaker_id)]["count"] += 1
    
    def record_market_coverage(self, market_id: int, market_name: str) -> None:
        """
        Registrar cobertura de un mercado.
        
        Args:
            market_id: ID del mercado
            market_name: Nombre del mercado
        """
        if str(market_id) not in self.current_metrics["market_coverage"]:
            self.current_metrics["market_coverage"][str(market_id)] = {
                "name": market_name,
                "count": 0
            }
        
        self.current_metrics["market_coverage"][str(market_id)]["count"] += 1
    
    def save_current_metrics(self) -> None:
        """Guardar las métricas actuales en un archivo."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = self.metrics_dir / f"odds_metrics_{timestamp}.json"
            
            # Calcular métricas derivadas
            total_data = (self.current_metrics["simulated_count"] + 
                         self.current_metrics["real_data_count"])
            
            if total_data > 0:
                self.current_metrics["simulated_pct"] = (
                    self.current_metrics["simulated_count"] / total_data * 100
                )
            else:
                self.current_metrics["simulated_pct"] = 0
                
            total_cache = (self.current_metrics["cache_hits"] + 
                          self.current_metrics["cache_misses"])
                          
            if total_cache > 0:
                self.current_metrics["cache_hit_rate"] = (
                    self.current_metrics["cache_hits"] / total_cache * 100
                )
            else:
                self.current_metrics["cache_hit_rate"] = 0
                
            if self.current_metrics["api_calls"] > 0:
                self.current_metrics["error_rate"] = (
                    self.current_metrics["errors"] / self.current_metrics["api_calls"] * 100
                )
            else:
                self.current_metrics["error_rate"] = 0
                
            if self.current_metrics["response_times_ms"]:
                self.current_metrics["avg_response_time_ms"] = (
                    sum(self.current_metrics["response_times_ms"]) / 
                    len(self.current_metrics["response_times_ms"])
                )
            else:
                self.current_metrics["avg_response_time_ms"] = 0
            
            # Guardar en archivo
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.current_metrics, f, indent=2)
                
            logger.info(f"Métricas guardadas en {metrics_file}")
            
        except Exception as e:
            logger.error(f"Error guardando métricas: {e}")
    
    def reset_metrics(self) -> None:
        """Reiniciar las métricas actuales."""
        self.save_current_metrics()
        self.current_metrics = {
            "timestamp": datetime.now().isoformat(),
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "simulated_count": 0,
            "real_data_count": 0,
            "response_times_ms": [],
            "bookmaker_coverage": {},
            "market_coverage": {},
        }
    
    def get_daily_metrics(self) -> Dict[str, Any]:
        """
        Obtener métricas agregadas del día actual.
        
        Returns:
            Métricas diarias agregadas
        """
        today = datetime.now().strftime("%Y%m%d")
        daily_metrics = {
            "date": today,
            "api_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "simulated_count": 0,
            "real_data_count": 0,
            "response_times_ms": [],
            "endpoints": {}
        }
        
        try:
            # Buscar archivos de métricas del día actual
            for metrics_file in self.metrics_dir.glob(f"odds_metrics_{today}_*.json"):
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                        
                    # Agregar métricas
                    daily_metrics["api_calls"] += metrics.get("api_calls", 0)
                    daily_metrics["cache_hits"] += metrics.get("cache_hits", 0)
                    daily_metrics["cache_misses"] += metrics.get("cache_misses", 0)
                    daily_metrics["errors"] += metrics.get("errors", 0)
                    daily_metrics["simulated_count"] += metrics.get("simulated_count", 0)
                    daily_metrics["real_data_count"] += metrics.get("real_data_count", 0)
                    daily_metrics["response_times_ms"].extend(metrics.get("response_times_ms", []))
                    
                    # Agregar datos por endpoint
                    for key, value in metrics.items():
                        if key.startswith("odds") or key.startswith("fixtures"):
                            if key not in daily_metrics["endpoints"]:
                                daily_metrics["endpoints"][key] = {
                                    "calls": 0,
                                    "errors": 0,
                                    "simulated": 0,
                                    "real": 0,
                                    "response_times_ms": []
                                }
                            
                            endpoint_metrics = daily_metrics["endpoints"][key]
                            endpoint_metrics["calls"] += value.get("calls", 0)
                            endpoint_metrics["errors"] += value.get("errors", 0)
                            endpoint_metrics["simulated"] += value.get("simulated", 0)
                            endpoint_metrics["real"] += value.get("real", 0)
                            endpoint_metrics["response_times_ms"].extend(value.get("response_times_ms", []))
                            
                except Exception as e:
                    logger.warning(f"Error procesando archivo {metrics_file}: {e}")
                    continue
                    
            # Calcular métricas agregadas
            # Tasa de aciertos de caché
            total_cache = daily_metrics["cache_hits"] + daily_metrics["cache_misses"]
            if total_cache > 0:
                daily_metrics["cache_hit_rate"] = (
                    daily_metrics["cache_hits"] / total_cache * 100
                )
            else:
                daily_metrics["cache_hit_rate"] = 0
                
            # Tasa de error de API
            if daily_metrics["api_calls"] > 0:
                daily_metrics["error_rate"] = (
                    daily_metrics["errors"] / daily_metrics["api_calls"] * 100
                )
            else:
                daily_metrics["error_rate"] = 0
                
            # Datos simulados vs reales
            total_data = daily_metrics["simulated_count"] + daily_metrics["real_data_count"]
            if total_data > 0:
                daily_metrics["simulated_pct"] = (
                    daily_metrics["simulated_count"] / total_data * 100
                )
            else:
                daily_metrics["simulated_pct"] = 0
                
            # Tiempo de respuesta promedio
            if daily_metrics["response_times_ms"]:
                daily_metrics["avg_response_time_ms"] = (
                    sum(daily_metrics["response_times_ms"]) / 
                    len(daily_metrics["response_times_ms"])
                )
            else:
                daily_metrics["avg_response_time_ms"] = 0
                
            return daily_metrics
            
        except Exception as e:
            logger.error(f"Error obteniendo métricas diarias: {e}")
            return daily_metrics
    
    def generate_health_report(self, thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Generar un informe de salud basado en métricas y umbrales.
        
        Args:
            thresholds: Diccionario con umbrales para alertas
            
        Returns:
            Informe de salud con alertas
        """
        daily_metrics = self.get_daily_metrics()
        
        # Inicializar informe
        report = {
            "timestamp": datetime.now().isoformat(),
            "status": "healthy",  # healthy, warning, critical
            "metrics": daily_metrics,
            "alerts": []
        }
        
        # Verificar umbral de datos simulados
        simulated_pct = daily_metrics.get("simulated_pct", 0)
        if simulated_pct > thresholds.get("simulated_data_pct", 10.0):
            report["alerts"].append({
                "type": "warning" if simulated_pct < 50 else "critical",
                "metric": "simulated_data_pct",
                "value": simulated_pct,
                "threshold": thresholds.get("simulated_data_pct"),
                "message": f"Alto porcentaje de datos simulados: {simulated_pct:.1f}%"
            })
        
        # Verificar umbral de tasa de error API
        error_rate = daily_metrics.get("error_rate", 0)
        if error_rate > thresholds.get("api_error_rate", 5.0):
            report["alerts"].append({
                "type": "warning" if error_rate < 15 else "critical",
                "metric": "api_error_rate",
                "value": error_rate,
                "threshold": thresholds.get("api_error_rate"),
                "message": f"Alta tasa de error en API: {error_rate:.1f}%"
            })
        
        # Verificar umbral de tiempo de respuesta
        avg_response_time = daily_metrics.get("avg_response_time_ms", 0)
        if avg_response_time > thresholds.get("response_time_ms", 1000):
            report["alerts"].append({
                "type": "warning",
                "metric": "response_time_ms",
                "value": avg_response_time,
                "threshold": thresholds.get("response_time_ms"),
                "message": f"Tiempo de respuesta alto: {avg_response_time:.1f}ms"
            })
        
        # Verificar umbral de tasa de aciertos de caché
        cache_hit_rate = daily_metrics.get("cache_hit_rate", 0)
        min_cache_hit_rate = thresholds.get("cache_hit_rate_min", 85.0)
        if cache_hit_rate < min_cache_hit_rate and daily_metrics.get("api_calls", 0) > 10:
            report["alerts"].append({
                "type": "warning",
                "metric": "cache_hit_rate",
                "value": cache_hit_rate,
                "threshold": min_cache_hit_rate,
                "message": f"Baja tasa de aciertos de caché: {cache_hit_rate:.1f}%"
            })
        
        # Determinar el estado general
        critical_alerts = sum(1 for alert in report["alerts"] if alert["type"] == "critical")
        warning_alerts = sum(1 for alert in report["alerts"] if alert["type"] == "warning")
        
        if critical_alerts > 0:
            report["status"] = "critical"
        elif warning_alerts > 0:
            report["status"] = "warning"
        
        return report
    
    def create_visualization(self, days: int = 7) -> Path:
        """
        Crear visualización de métricas históricas.
        
        Args:
            days: Número de días para incluir en la visualización
            
        Returns:
            Ruta al archivo de imagen generado
        """
        try:
            # Obtener datos de los últimos días
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            dates = []
            simulated_pcts = []
            error_rates = []
            response_times = []
            cache_hit_rates = []
            
            # Recopilar datos para cada día
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y%m%d")
                
                # Métricas para este día
                daily_data = {
                    "api_calls": 0,
                    "errors": 0,
                    "simulated_count": 0,
                    "real_data_count": 0,
                    "response_times_ms": [],
                    "cache_hits": 0,
                    "cache_misses": 0
                }
                
                # Buscar archivos de métricas para este día
                metrics_files = list(self.metrics_dir.glob(f"odds_metrics_{date_str}_*.json"))
                
                for metrics_file in metrics_files:
                    try:
                        with open(metrics_file, 'r', encoding='utf-8') as f:
                            metrics = json.load(f)
                            
                        # Agregar métricas
                        daily_data["api_calls"] += metrics.get("api_calls", 0)
                        daily_data["errors"] += metrics.get("errors", 0)
                        daily_data["simulated_count"] += metrics.get("simulated_count", 0)
                        daily_data["real_data_count"] += metrics.get("real_data_count", 0)
                        daily_data["response_times_ms"].extend(metrics.get("response_times_ms", []))
                        daily_data["cache_hits"] += metrics.get("cache_hits", 0)
                        daily_data["cache_misses"] += metrics.get("cache_misses", 0)
                        
                    except Exception as e:
                        logger.warning(f"Error procesando archivo {metrics_file}: {e}")
                
                # Calcular métricas para este día
                # Datos simulados vs reales
                total_data = daily_data["simulated_count"] + daily_data["real_data_count"]
                simulated_pct = 0
                if total_data > 0:
                    simulated_pct = (daily_data["simulated_count"] / total_data * 100)
                
                # Tasa de error API
                error_rate = 0
                if daily_data["api_calls"] > 0:
                    error_rate = (daily_data["errors"] / daily_data["api_calls"] * 100)
                
                # Tiempo de respuesta promedio
                avg_response_time = 0
                if daily_data["response_times_ms"]:
                    avg_response_time = (sum(daily_data["response_times_ms"]) / 
                                      len(daily_data["response_times_ms"]))
                
                # Tasa de aciertos de caché
                cache_hit_rate = 0
                total_cache = daily_data["cache_hits"] + daily_data["cache_misses"]
                if total_cache > 0:
                    cache_hit_rate = (daily_data["cache_hits"] / total_cache * 100)
                
                # Agregar a listas
                dates.append(current_date.strftime("%d/%m"))
                simulated_pcts.append(simulated_pct)
                error_rates.append(error_rate)
                response_times.append(avg_response_time)
                cache_hit_rates.append(cache_hit_rate)
                
                # Avanzar al siguiente día
                current_date += timedelta(days=1)
            
            # Crear gráfico
            plt.figure(figsize=(15, 10))
            
            # Subplot para porcentaje de datos simulados
            plt.subplot(2, 2, 1)
            plt.plot(dates, simulated_pcts, marker='o', color='blue')
            plt.axhline(y=10, color='r', linestyle='--')  # Umbral típico del 10%
            plt.title('Porcentaje de Datos Simulados')
            plt.ylabel('Porcentaje')
            plt.grid(True)
            
            # Subplot para tasa de error API
            plt.subplot(2, 2, 2)
            plt.plot(dates, error_rates, marker='o', color='red')
            plt.axhline(y=5, color='r', linestyle='--')  # Umbral típico del 5%
            plt.title('Tasa de Error API')
            plt.ylabel('Porcentaje')
            plt.grid(True)
            
            # Subplot para tiempo de respuesta promedio
            plt.subplot(2, 2, 3)
            plt.plot(dates, response_times, marker='o', color='green')
            plt.axhline(y=1000, color='r', linestyle='--')  # Umbral típico de 1000ms
            plt.title('Tiempo de Respuesta Promedio')
            plt.ylabel('Milisegundos')
            plt.grid(True)
            
            # Subplot para tasa de aciertos de caché
            plt.subplot(2, 2, 4)
            plt.plot(dates, cache_hit_rates, marker='o', color='purple')
            plt.axhline(y=85, color='r', linestyle='--')  # Umbral típico del 85%
            plt.title('Tasa de Aciertos de Caché')
            plt.ylabel('Porcentaje')
            plt.grid(True)
            
            # Ajustar layout y guardar
            plt.tight_layout()
            
            # Guardar gráfico
            reports_dir = Path("data") / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            current_date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_path = reports_dir / f"odds_metrics_graph_{current_date_str}.png"
            
            plt.savefig(graph_path)
            plt.close()
            
            logger.info(f"Gráfico guardado en {graph_path}")
            return graph_path
            
        except Exception as e:
            logger.error(f"Error creando visualización: {e}")
            return None

# Funciones de conveniencia para usar desde otros scripts

def record_api_call(endpoint: str, params: Dict[str, Any], 
                  success: bool, response_time_ms: float,
                  simulated: bool = False) -> None:
    """
    Registra una llamada a la API de odds.
    
    Args:
        endpoint: Endpoint de la API utilizado
        params: Parámetros de la solicitud
        success: Si la llamada fue exitosa
        response_time_ms: Tiempo de respuesta en milisegundos
        simulated: Si se devolvieron datos simulados
    """
    try:
        metrics = OddsAPIMetrics()
        metrics.record_api_call(endpoint, params, success, response_time_ms, simulated)
        metrics.save_current_metrics()
    except Exception as e:
        logger.error(f"Error registrando llamada a API: {e}")

def record_cache_event(hit: bool, fixture_id: int) -> None:
    """
    Registrar un evento de caché.
    
    Args:
        hit: Si fue un acierto de caché
        fixture_id: ID del partido
    """
    try:
        metrics = OddsAPIMetrics()
        metrics.record_cache_event(hit, fixture_id)
        metrics.save_current_metrics()
    except Exception as e:
        logger.error(f"Error registrando evento de caché: {e}")

def generate_health_report(thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Generar informe de salud del sistema de integración de odds.
    
    Args:
        thresholds: Umbrales para alertas (opcional)
        
    Returns:
        Informe de salud
    """
    try:
        # Usar umbrales predeterminados si no se proporcionan
        if thresholds is None:
            from monitor_odds_integration import MONITOR_CONFIG
            thresholds = MONITOR_CONFIG.get("thresholds", {})
            
        metrics = OddsAPIMetrics()
        return metrics.generate_health_report(thresholds)
    except Exception as e:
        logger.error(f"Error generando informe de salud: {e}")
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "unknown",
            "error": str(e),
            "alerts": [
                {
                    "type": "critical",
                    "message": f"Error generando informe de salud: {e}"
                }
            ]
        }

def create_metrics_visualization(days: int = 7) -> Optional[Path]:
    """
    Crear visualización de métricas históricas.
    
    Args:
        days: Número de días para incluir
        
    Returns:
        Ruta al archivo de imagen generado
    """
    try:
        metrics = OddsAPIMetrics()
        return metrics.create_visualization(days)
    except Exception as e:
        logger.error(f"Error creando visualización: {e}")
        return None
