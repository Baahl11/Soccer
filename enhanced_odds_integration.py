# filepath: c:\Users\gm_me\Soccer\enhanced_odds_integration.py
"""
Enhanced Odds Integration - Fase 2: Sistema de Múltiples Proveedores

Este módulo actualiza el sistema de odds existente para integrar múltiples
proveedores gratuitos con fallback automático, manteniendo compatibilidad
con el sistema actual.

Mejoras implementadas:
- Sistema de múltiples proveedores gratuitos
- Fallback automático entre APIs
- Mejor manejo de errores
- Caché mejorado por proveedor
- Monitoreo de performance por API

Autor: Sistema de Predicción Soccer - Fase 2
Fecha: Mayo 28, 2025
"""

import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

class EnhancedOddsIntegration:
    """Integración mejorada de odds con múltiples proveedores"""
    
    def __init__(self):
        self.multi_provider_manager = None
        self.fallback_to_original = True
        self.performance_metrics = {
            "multi_provider_requests": 0,
            "original_system_requests": 0,
            "multi_provider_success": 0,
            "original_system_success": 0,
            "total_simulated": 0,
            "total_real": 0
        }
        
        # Intentar inicializar el sistema de múltiples proveedores
        self._initialize_multi_provider()
    
    def _initialize_multi_provider(self):
        """Inicializa el sistema de múltiples proveedores"""
        try:
            from multi_provider_odds_manager import get_multi_provider_odds_manager
            self.multi_provider_manager = get_multi_provider_odds_manager()
            logger.info("✅ Sistema de múltiples proveedores inicializado correctamente")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo inicializar sistema de múltiples proveedores: {str(e)}")
            logger.info("📡 Se usará el sistema original como único proveedor")
    
    def get_fixture_odds_enhanced(self, fixture_id: int, use_cache: bool = True, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Obtiene odds usando el sistema mejorado con múltiples proveedores
        
        Args:
            fixture_id: ID del partido
            use_cache: Usar caché si está disponible
            force_refresh: Forzar actualización ignorando caché
            
        Returns:
            Datos de odds con metadatos mejorados
        """
        logger.info(f"🔍 Solicitando odds mejoradas para fixture {fixture_id}")
        
        # 1. Intentar con sistema de múltiples proveedores primero
        if self.multi_provider_manager:
            try:
                self.performance_metrics["multi_provider_requests"] += 1
                logger.info("🔄 Intentando con sistema de múltiples proveedores...")
                odds_data, provider_name = self.multi_provider_manager.get_odds(fixture_id, force_refresh)
                
                if odds_data:
                    if not odds_data.get("simulated", True):
                        self.performance_metrics["multi_provider_success"] += 1
                        self.performance_metrics["total_real"] += 1
                        logger.info(f"✅ Odds reales obtenidas de {provider_name}")
                    else:
                        self.performance_metrics["total_simulated"] += 1
                        logger.info(f"⚠️ Sistema de múltiples proveedores devolvió datos simulados de {provider_name}")
                    
                    # Enriquecer con metadatos del sistema mejorado (real o simulado)
                    enhanced_odds = self._enrich_odds_data(odds_data, provider_name, "multi_provider")
                    return enhanced_odds
                else:
                    logger.warning(f"❌ Sistema de múltiples proveedores no devolvió datos para {fixture_id}")
                    
            except Exception as e:
                logger.warning(f"❌ Error en sistema de múltiples proveedores: {str(e)}")
        
        # 2. Fallback al sistema original si múltiples proveedores falla
        if self.fallback_to_original:
            try:
                self.performance_metrics["original_system_requests"] += 1
                
                logger.info("🔄 Fallback al sistema original...")
                
                # Importar el sistema original
                from optimize_odds_integration import get_fixture_odds as get_fixture_odds_original
                
                original_odds = get_fixture_odds_original(
                    fixture_id=fixture_id,
                    use_cache=use_cache,
                    force_refresh=force_refresh
                )
                
                if original_odds:
                    if not original_odds.get("simulated", True):
                        self.performance_metrics["original_system_success"] += 1
                        self.performance_metrics["total_real"] += 1
                        logger.info("✅ Odds reales obtenidas del sistema original")
                    else:
                        self.performance_metrics["total_simulated"] += 1
                        logger.info("⚠️ Sistema original también devolvió datos simulados")
                    
                    # Enriquecer con metadatos
                    enhanced_odds = self._enrich_odds_data(original_odds, "API-Football-Original", "original_system")
                    return enhanced_odds
                    
            except Exception as e:
                logger.error(f"❌ Error en sistema original: {str(e)}")
        
        # 3. Último recurso: generar datos simulados básicos
        logger.warning(f"🆘 Generando odds simuladas de último recurso para fixture {fixture_id}")
        self.performance_metrics["total_simulated"] += 1
        
        return self._generate_emergency_odds(fixture_id)
    
    def _enrich_odds_data(self, odds_data: Dict[str, Any], provider_name: str, system_type: str) -> Dict[str, Any]:
        """Enriquece los datos de odds con metadatos del sistema mejorado"""
        
        # Clonar datos originales para no modificarlos
        enriched = odds_data.copy() if odds_data else {}
        
        # Añadir metadatos del sistema mejorado
        enriched.update({
            "enhanced_system": {
                "version": "2.0-multi-provider",
                "provider_used": provider_name,
                "system_type": system_type,
                "timestamp": datetime.now().isoformat(),
                "performance_tracking": True
            }
        })
        
        # Marcar si es simulado o real
        is_simulated = enriched.get("simulated", True)
        enriched["data_quality"] = "simulated" if is_simulated else "real_market_data"
        
        # Añadir información de confiabilidad
        if not is_simulated:
            enriched["reliability"] = {
                "source": provider_name,
                "data_freshness": "live" if system_type == "multi_provider" else "cached",
                "provider_type": "multi_api" if system_type == "multi_provider" else "single_api"
            }
        
        return enriched
    
    def _generate_emergency_odds(self, fixture_id: int) -> Dict[str, Any]:
        """Genera odds simuladas de emergencia cuando todo falla"""
        import random
        
        return {
            "fixture_id": fixture_id,
            "simulated": True,
            "source": "Sistema de Emergencia",
            "match_winner": {
                "home": round(random.uniform(1.9, 2.8), 2),
                "draw": round(random.uniform(3.1, 3.4), 2),  
                "away": round(random.uniform(1.9, 2.8), 2)
            },
            "timestamp": datetime.now().isoformat(),
            "enhanced_system": {
                "version": "2.0-multi-provider",
                "provider_used": "Emergency-Fallback",
                "system_type": "emergency",
                "timestamp": datetime.now().isoformat(),
                "note": "Datos de último recurso generados internamente"
            },
            "data_quality": "emergency_simulated"
        }
    
    def get_system_performance(self) -> Dict[str, Any]:
        """Obtiene métricas de performance del sistema mejorado"""
        total_requests = (self.performance_metrics["multi_provider_requests"] + 
                         self.performance_metrics["original_system_requests"])
        
        if total_requests == 0:
            return {"error": "No hay datos de performance disponibles"}
        
        multi_success_rate = (self.performance_metrics["multi_provider_success"] / 
                             max(1, self.performance_metrics["multi_provider_requests"])) * 100
        
        original_success_rate = (self.performance_metrics["original_system_success"] / 
                               max(1, self.performance_metrics["original_system_requests"])) * 100
        
        total_real_data_rate = (self.performance_metrics["total_real"] / 
                               max(1, total_requests)) * 100
        
        return {
            "total_requests": total_requests,
            "multi_provider": {
                "requests": self.performance_metrics["multi_provider_requests"],
                "success": self.performance_metrics["multi_provider_success"],
                "success_rate": f"{multi_success_rate:.1f}%"
            },
            "original_system": {
                "requests": self.performance_metrics["original_system_requests"],
                "success": self.performance_metrics["original_system_success"],
                "success_rate": f"{original_success_rate:.1f}%"
            },
            "overall": {
                "real_data": self.performance_metrics["total_real"],
                "simulated_data": self.performance_metrics["total_simulated"],
                "real_data_rate": f"{total_real_data_rate:.1f}%"
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_provider_health(self) -> Dict[str, Any]:
        """Obtiene estado de salud de todos los proveedores"""
        health_data = {
            "system_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "providers": {}
        }
        
        # Estado del sistema de múltiples proveedores
        if self.multi_provider_manager:
            try:
                provider_stats = self.multi_provider_manager.get_provider_stats()
                health_data["multi_provider_system"] = {
                    "status": "available",
                    "providers": provider_stats.get("providers", {}),
                    "total_providers": len(provider_stats.get("providers", {}))
                }
            except Exception as e:
                health_data["multi_provider_system"] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            health_data["multi_provider_system"] = {
                "status": "not_available",
                "reason": "Multi-provider system not initialized"
            }
        
        # Estado del sistema original
        health_data["original_system"] = {
            "status": "available" if self.fallback_to_original else "disabled",
            "provider": "API-Football-Original"
        }
        
        return health_data

# Instancia global del sistema mejorado
_enhanced_odds_integration = None

def get_enhanced_odds_integration() -> EnhancedOddsIntegration:
    """Obtiene la instancia global del sistema mejorado"""
    global _enhanced_odds_integration
    if _enhanced_odds_integration is None:
        _enhanced_odds_integration = EnhancedOddsIntegration()
    return _enhanced_odds_integration

def get_fixture_odds_enhanced(fixture_id: int, use_cache: bool = True, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """
    Función principal para obtener odds con el sistema mejorado
    
    Esta función reemplaza o complementa get_fixture_odds() original,
    proporcionando múltiples proveedores y mejor manejo de errores.
    
    Args:
        fixture_id: ID del partido
        use_cache: Usar caché si está disponible
        force_refresh: Forzar actualización ignorando caché
        
    Returns:
        Datos de odds mejorados con metadatos adicionales
    """
    enhanced_system = get_enhanced_odds_integration()
    return enhanced_system.get_fixture_odds_enhanced(fixture_id, use_cache, force_refresh)

def get_odds_system_performance() -> Dict[str, Any]:
    """Obtiene métricas de performance del sistema de odds mejorado"""
    enhanced_system = get_enhanced_odds_integration()
    return enhanced_system.get_system_performance()

def get_odds_providers_health() -> Dict[str, Any]:
    """Obtiene estado de salud de todos los proveedores de odds"""
    enhanced_system = get_enhanced_odds_integration()
    return enhanced_system.get_provider_health()

if __name__ == "__main__":
    # Test del sistema mejorado
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Probando sistema de odds mejorado (Fase 2)...")
    
    # Test fixture conocido
    test_fixture_id = 1208393
    
    print(f"\n📊 Obteniendo odds para fixture {test_fixture_id}...")
    
    # Probar sistema mejorado
    enhanced_odds = get_fixture_odds_enhanced(test_fixture_id)
    
    if enhanced_odds:
        print(f"\n✅ Resultado obtenido:")
        print(f"Proveedor: {enhanced_odds.get('enhanced_system', {}).get('provider_used', 'Unknown')}")
        print(f"Sistema: {enhanced_odds.get('enhanced_system', {}).get('system_type', 'Unknown')}")
        print(f"¿Simulado?: {enhanced_odds.get('simulated', 'No especificado')}")
        print(f"Calidad: {enhanced_odds.get('data_quality', 'Unknown')}")
    else:
        print("❌ No se pudieron obtener odds")
    
    # Mostrar métricas de performance
    print(f"\n📈 Métricas de Performance:")
    performance = get_odds_system_performance()
    for key, value in performance.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue}")
        else:
            print(f"  {key}: {value}")
    
    # Mostrar estado de proveedores
    print(f"\n🏥 Estado de Proveedores:")
    health = get_odds_providers_health()
    print(f"Sistema general: {health.get('system_status')}")
    if 'multi_provider_system' in health:
        mp_status = health['multi_provider_system'].get('status')
        print(f"Sistema múltiples proveedores: {mp_status}")
        if mp_status == "available":
            providers_count = health['multi_provider_system'].get('total_providers', 0)
            print(f"  Proveedores disponibles: {providers_count}")
