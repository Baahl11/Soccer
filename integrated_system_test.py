"""
Pruebas Integradas del Sistema de Odds API

Este script ejecuta pruebas completas de la integración de odds API con el sistema
de predicción, verificando que todos los componentes trabajen correctamente juntos.

Autor: Equipo de Desarrollo
Fecha: Mayo 25, 2025
"""

import logging
import json
import time
import sys
import random
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='integrated_odds_testing.log',
    filemode='w'
)

logger = logging.getLogger('integrated_testing')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Importar módulos del sistema
sys.path.append('.')
import config
import optimize_odds_integration
from odds_normalizer import OddsNormalizer, normalize_odds
from odds_cache import OddsCache
from odds_metrics import OddsAPIMetrics

class IntegratedSystemTest:
    """Clase para ejecutar pruebas integradas del sistema de odds"""
    
    def __init__(self):
        """Inicializar pruebas integradas"""
        self.metrics = OddsAPIMetrics()
        self.cache = OddsCache()
        self.normalizer = OddsNormalizer()
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "summary": {}
        }
        
    def run_all_tests(self):
        """Ejecutar todas las pruebas integradas"""
        logger.info("="*60)
        logger.info("INICIANDO PRUEBAS INTEGRADAS DEL SISTEMA DE ODDS")
        logger.info(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*60)
        
        # Ejecutar cada prueba y registrar resultados
        try:
            # Pruebas iniciales
            self.test_results["tests"]["config"] = self.test_config()
            self.test_results["tests"]["api_connection"] = self.test_api_connection()
            
            # Pruebas de integración
            self.test_results["tests"]["real_data"] = self.test_real_data()
            self.test_results["tests"]["cache_system"] = self.test_cache_system()
            self.test_results["tests"]["normalizer"] = self.test_normalizer()
            self.test_results["tests"]["metrics"] = self.test_metrics_system()
            
            # Pruebas avanzadas
            self.test_results["tests"]["load_test"] = self.test_high_load()
            self.test_results["tests"]["error_recovery"] = self.test_error_recovery()
            
            # Generar resumen
            self.generate_summary()
            
            # Guardar resultados
            self.save_results()
            
            # Generar gráficos
            self.generate_performance_charts()
            
            logger.info("\n" + "="*60)
            logger.info("PRUEBAS INTEGRADAS COMPLETADAS")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Error en pruebas integradas: {str(e)}")
            return False
    
    def test_config(self):
        """Probar que la configuración está correcta"""
        logger.info("Probando configuración...")
        
        results = {
            "passed": True,
            "details": {}
        }
        
        # Verificar configuraciones necesarias
        required_configs = [
            "API_KEY", "API_BASE_URL", "ODDS_ENDPOINTS", 
            "ODDS_BOOKMAKERS_PRIORITY", "CACHE_CONFIG"
        ]
        
        for cfg in required_configs:
            if hasattr(config, cfg):
                results["details"][cfg] = "Presente"
            else:
                results["details"][cfg] = "Ausente"
                results["passed"] = False
        
        # Verificar que no hay configuraciones obsoletas
        deprecated_configs = ["ODDS_API_KEY", "ODDS_API_URL", "ODDS_API_PROVIDER"]
        
        for cfg in deprecated_configs:
            if hasattr(config, cfg):
                results["details"][f"Obsoleto_{cfg}"] = "Presente (debe eliminarse)"
                results["passed"] = False
            else:
                results["details"][f"Obsoleto_{cfg}"] = "Ausente (correcto)"
                
        if results["passed"]:
            logger.info("✅ Configuración correcta")
        else:
            logger.warning("⚠️ Problemas en configuración")
            
        return results
    
    def test_api_connection(self):
        """Probar la conexión con la API de odds"""
        logger.info("Probando conexión con API...")
        
        results = {
            "passed": False,
            "details": {},
            "endpoints_tested": 0,
            "endpoints_passed": 0
        }
        
        # Probar conexión a endpoints principales
        endpoints_to_test = config.ODDS_ENDPOINTS
        
        for endpoint_name, endpoint_path in endpoints_to_test.items():
            try:
                logger.info(f"Probando endpoint {endpoint_name}...")
                
                # Usar parámetros básicos
                params = {}
                if endpoint_name == "pre_match" or endpoint_name == "live":
                    params = {"league": "39", "season": "2023"}
                
                # Registrar inicio
                start_time = time.time()
                
                # Intentar obtener datos
                from optimize_odds_integration import API_KEY, API_BASE_URL
                import requests
                
                endpoint_url = f"{API_BASE_URL}{endpoint_path}"
                headers = {
                    "x-rapidapi-key": API_KEY,
                    "x-rapidapi-host": API_BASE_URL.replace("https://", "")
                }
                
                response = requests.get(
                    endpoint_url, 
                    headers=headers,
                    params=params,
                    timeout=15
                )
                
                # Registrar tiempo
                elapsed_time = (time.time() - start_time) * 1000  # en ms
                
                # Evaluar respuesta
                results["details"][endpoint_name] = {
                    "status_code": response.status_code,
                    "time_ms": elapsed_time,
                    "passed": False
                }
                
                if response.status_code == 200:
                    data = response.json()
                    if "response" in data:
                        results["details"][endpoint_name]["data_count"] = len(data["response"])
                        results["details"][endpoint_name]["passed"] = True
                        results["endpoints_passed"] += 1
                        logger.info(f"✅ Endpoint {endpoint_name} accesible")
                    else:
                        results["details"][endpoint_name]["error"] = "No data in response"
                        logger.warning(f"⚠️ Endpoint {endpoint_name} no retorna datos")
                else:
                    results["details"][endpoint_name]["error"] = response.text[:100]
                    logger.warning(f"⚠️ Endpoint {endpoint_name} error {response.status_code}")
                    
                results["endpoints_tested"] += 1
                
                # Registrar métricas
                self.metrics.record_api_call(
                    endpoint=endpoint_path,
                    params=params,
                    success=results["details"][endpoint_name]["passed"],
                    response_time_ms=elapsed_time,
                    simulated=False
                )
                
            except Exception as e:
                results["details"][endpoint_name] = {
                    "error": str(e),
                    "passed": False
                }
                results["endpoints_tested"] += 1
                logger.error(f"❌ Error probando endpoint {endpoint_name}: {e}")
        
        # Evaluar resultado general
        if results["endpoints_tested"] > 0:
            success_rate = (results["endpoints_passed"] / results["endpoints_tested"]) * 100
            results["success_rate"] = success_rate
            results["passed"] = success_rate >= 50  # Al menos la mitad de endpoints funcionan
            
        if results["passed"]:
            logger.info(f"✅ Conexión API exitosa ({results['success_rate']:.1f}%)")
        else:
            logger.warning(f"⚠️ Conexión API limitada ({results.get('success_rate', 0):.1f}%)")
            
        return results
    
    def test_real_data(self):
        """Probar obtención de datos reales (no simulados)"""
        logger.info("Probando obtención de datos reales...")
        
        results = {
            "passed": False,
            "fixtures_tested": 0,
            "fixtures_real": 0,
            "details": {}
        }
        
        # Obtener resultados de prueba
        try:
            # Llamar a la función de prueba del sistema de odds
            odds_results = optimize_odds_integration.test_odds_integration()
            
            # Analizar resultados
            results["fixtures_tested"] = len(odds_results)
            
            for fixture_id, fixture_data in odds_results.items():
                is_simulated = fixture_data.get("simulated", True)
                results["details"][fixture_id] = {
                    "simulated": is_simulated,
                    "source": fixture_data.get("source", "Desconocido")
                }
                
                if not is_simulated:
                    results["fixtures_real"] += 1
            
            # Calcular tasa de datos reales
            if results["fixtures_tested"] > 0:
                real_data_rate = (results["fixtures_real"] / results["fixtures_tested"]) * 100
                results["real_data_rate"] = real_data_rate
                results["passed"] = real_data_rate > 0  # Al menos algunos datos reales
            
            if results["passed"]:
                logger.info(f"✅ Datos reales obtenidos ({results.get('real_data_rate', 0):.1f}%)")
            else:
                logger.warning("⚠️ No se obtuvieron datos reales")
            
        except Exception as e:
            logger.error(f"❌ Error probando datos reales: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def test_cache_system(self):
        """Probar sistema de caché"""
        logger.info("Probando sistema de caché...")
        
        results = {
            "passed": False,
            "details": {}
        }
        
        try:
            # Configurar y limpiar caché
            self.cache.setup()
            self.cache.clear_expired()
            
            # Realizar una primera solicitud (miss de caché)
            fixture_id = 1128079  # Usar ID consistente para pruebas
            
            start_time = time.time()
            first_call = optimize_odds_integration.get_fixture_odds(
                fixture_id=fixture_id,
                use_cache=True,
                force_refresh=False
            )
            first_call_time = (time.time() - start_time) * 1000  # ms
            
            # Verificar resultado
            results["details"]["first_call"] = {
                "time_ms": first_call_time,
                "cache_hit": False  # Primera llamada siempre es miss
            }
            
            # Pequeña pausa
            time.sleep(0.5)
            
            # Segunda llamada (debería ser hit de caché)
            start_time = time.time()
            second_call = optimize_odds_integration.get_fixture_odds(
                fixture_id=fixture_id,
                use_cache=True,
                force_refresh=False
            )
            second_call_time = (time.time() - start_time) * 1000  # ms
            
            results["details"]["second_call"] = {
                "time_ms": second_call_time,
                "cache_hit": second_call_time < first_call_time * 0.5  # Asumimos hit si es 50% más rápido
            }
            
            # Verificar que los datos son iguales
            same_data = (first_call == second_call)
            results["details"]["data_consistency"] = same_data
            
            # Probar refresco forzado
            start_time = time.time()
            force_call = optimize_odds_integration.get_fixture_odds(
                fixture_id=fixture_id,
                use_cache=True,
                force_refresh=True
            )
            force_call_time = (time.time() - start_time) * 1000  # ms
            
            results["details"]["force_refresh"] = {
                "time_ms": force_call_time,
                "cache_bypassed": force_call_time > second_call_time * 1.5  # Debería ser más lento
            }
            
            # Evaluar resultado general
            results["passed"] = (
                results["details"]["second_call"]["cache_hit"] and
                results["details"]["data_consistency"] and
                results["details"]["force_refresh"]["cache_bypassed"]
            )
            
            if results["passed"]:
                logger.info("✅ Sistema de caché funciona correctamente")
            else:
                logger.warning("⚠️ Sistema de caché tiene problemas")
            
            # Registrar métricas
            self.metrics.record_cache_event(hit=False, fixture_id=fixture_id)  # Primera llamada (miss)
            self.metrics.record_cache_event(hit=True, fixture_id=fixture_id)   # Segunda llamada (hit)
            
        except Exception as e:
            logger.error(f"❌ Error probando sistema de caché: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def test_normalizer(self):
        """Probar normalizador de datos"""
        logger.info("Probando normalización de datos...")
        
        results = {
            "passed": False,
            "details": {}
        }
        
        try:
            # Obtener datos de ejemplo
            fixture_id = 1128079  # ID consistente
            
            # Intentar obtener datos reales primero
            raw_data = optimize_odds_integration.get_fixture_odds(
                fixture_id=fixture_id,
                use_cache=False,
                force_refresh=True
            )
            
            is_simulated = raw_data.get("simulated", True)
            results["details"]["using_simulated"] = is_simulated
            
            # Probar normalización directa
            normalized = self.normalizer.normalize(raw_data)
            
            # Verificar estructura normalizada
            required_fields = [
                "fixture_id", "bookmakers", "odds", "best_odds", 
                "market_sentiment"
            ]
            
            all_fields_present = all(field in normalized for field in required_fields)
            results["details"]["structure_valid"] = all_fields_present
            
            # Verificar coherencia de probabilidades
            market_sentiment = normalized.get("market_sentiment", {})
            implied_probs = market_sentiment.get("implied_probabilities", {})
            
            if implied_probs:
                # Verificar que las probabilidades suman aproximadamente 1
                total_prob = (
                    implied_probs.get("home_win", 0) + 
                    implied_probs.get("draw", 0) + 
                    implied_probs.get("away_win", 0)
                )
                prob_valid = 0.95 <= total_prob <= 1.05  # Permitir pequeño margen de error
                results["details"]["probabilities_valid"] = prob_valid
            else:
                results["details"]["probabilities_valid"] = False
            
            # Evaluar resultado general
            results["passed"] = (
                all_fields_present and
                results["details"].get("probabilities_valid", False)
            )
            
            if results["passed"]:
                logger.info("✅ Normalización funciona correctamente")
            else:
                logger.warning("⚠️ Normalización tiene problemas")
                
        except Exception as e:
            logger.error(f"❌ Error probando normalización: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def test_metrics_system(self):
        """Probar sistema de métricas"""
        logger.info("Probando sistema de métricas...")
        
        results = {
            "passed": False,
            "details": {}
        }
        
        try:
            # El objeto metrics ya ha registrado eventos durante otras pruebas
            # Guardamos las métricas actuales
            self.metrics.save_current_metrics()
            
            # Verificar que las métricas actuales tienen valores
            has_api_calls = self.metrics.current_metrics["api_calls"] > 0
            has_cache_events = (
                self.metrics.current_metrics["cache_hits"] > 0 or
                self.metrics.current_metrics["cache_misses"] > 0
            )
            
            results["details"]["has_api_metrics"] = has_api_calls
            results["details"]["has_cache_metrics"] = has_cache_events
            
            # Intentar generar informe
            if hasattr(self.metrics, 'generate_health_report'):
                report = self.metrics.generate_health_report()
                results["details"]["report_generated"] = (report is not None)
            else:
                results["details"]["report_generated"] = False
                
            # Evaluar resultado
            results["passed"] = has_api_calls and has_cache_events
            
            if results["passed"]:
                logger.info("✅ Sistema de métricas funciona correctamente")
            else:
                logger.warning("⚠️ Sistema de métricas tiene problemas")
                
        except Exception as e:
            logger.error(f"❌ Error probando sistema de métricas: {str(e)}")
            results["error"] = str(e)
            
        return results
    
    def test_high_load(self):
        """Probar sistema bajo carga alta"""
        logger.info("Probando sistema bajo carga alta...")
        
        results = {
            "passed": False,
            "details": {
                "requests": [],
                "errors": 0,
                "avg_time_ms": 0
            }
        }
        
        try:
            # Preparar conjuntos de datos para prueba
            num_requests = 10  # Ajustable según necesidad
            fixture_ids = [
                1128079, 1128080, 1128081, 
                1127999, 1128000, 1128001,
                1128010, 1128020, 1128030, 1128040
            ]
            
            # Mezclar aleatoriamente
            random.shuffle(fixture_ids)
            
            # Limitar a número de solicitudes
            fixtures_to_test = fixture_ids[:num_requests]
            
            # Ejecutar solicitudes
            times = []
            for idx, fixture_id in enumerate(fixtures_to_test):
                try:
                    logger.info(f"Solicitud de carga {idx+1}/{num_requests} para partido {fixture_id}")
                    
                    start_time = time.time()
                    data = optimize_odds_integration.get_fixture_odds(
                        fixture_id=fixture_id,
                        use_cache=True,  # Usar caché para mejorar rendimiento
                        force_refresh=(idx % 3 == 0)  # Cada tercer solicitud fuerza refresco
                    )
                    elapsed_time = (time.time() - start_time) * 1000  # ms
                    
                    times.append(elapsed_time)
                    results["details"]["requests"].append({
                        "fixture_id": fixture_id,
                        "time_ms": elapsed_time,
                        "simulated": data.get("simulated", True),
                        "error": None
                    })
                    
                except Exception as e:
                    results["details"]["errors"] += 1
                    results["details"]["requests"].append({
                        "fixture_id": fixture_id,
                        "error": str(e)
                    })
            
            # Calcular tiempos promedio
            if times:
                results["details"]["avg_time_ms"] = sum(times) / len(times)
                results["details"]["max_time_ms"] = max(times)
                
            # Evaluar resultados
            success_rate = ((num_requests - results["details"]["errors"]) / num_requests) * 100
            results["details"]["success_rate"] = success_rate
            
            # Criterio de éxito: al menos 80% de solicitudes exitosas
            results["passed"] = success_rate >= 80
            
            if results["passed"]:
                logger.info(f"✅ Prueba de carga exitosa ({success_rate:.1f}%)")
            else:
                logger.warning(f"⚠️ Prueba de carga con problemas ({success_rate:.1f}%)")
            
        except Exception as e:
            logger.error(f"❌ Error en prueba de carga: {str(e)}")
            results["error"] = str(e)
            
        return results
    
    def test_error_recovery(self):
        """Probar recuperación ante errores"""
        logger.info("Probando recuperación ante errores...")
        
        results = {
            "passed": False,
            "details": {}
        }
        
        try:
            # 1. Probar con API key inválida temporalmente
            original_key = config.API_KEY
            config.API_KEY = "temporary_invalid_key"
            
            start_time = time.time()
            invalid_key_data = optimize_odds_integration.get_fixture_odds(1128079)
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            # Restaurar key correcta
            config.API_KEY = original_key
            
            # Verificar que se generaron datos simulados
            results["details"]["invalid_key"] = {
                "time_ms": elapsed_time,
                "simulated_generated": invalid_key_data.get("simulated", False),
                "error_handled": "error" not in invalid_key_data
            }
            
            # 2. Probar con ID de partido inválido
            start_time = time.time()
            invalid_id_data = optimize_odds_integration.get_fixture_odds(99999999)  # ID inválido
            elapsed_time = (time.time() - start_time) * 1000  # ms
            
            results["details"]["invalid_fixture"] = {
                "time_ms": elapsed_time,
                "simulated_generated": invalid_id_data.get("simulated", False),
                "error_handled": "error" not in invalid_id_data
            }
            
            # 3. Probar con caché corrupta
            # Primero guardamos algo en caché
            fixture_id = 1128079
            optimize_odds_integration.get_fixture_odds(fixture_id)
            
            # Corromper archivo de caché
            cache_file = optimize_odds_integration.CACHE_DIR / f"odds_{fixture_id}.json"
            if cache_file.exists():
                with open(cache_file, 'w', encoding='utf-8') as f:
                    f.write("{ corrupted json data")
                
                # Intentar leer la caché corrupta
                start_time = time.time()
                corrupt_cache_data = optimize_odds_integration.get_fixture_odds(fixture_id)
                elapsed_time = (time.time() - start_time) * 1000  # ms
                
                results["details"]["corrupt_cache"] = {
                    "time_ms": elapsed_time,
                    "error_handled": "error" not in corrupt_cache_data,
                    "recovered": corrupt_cache_data is not None
                }
            else:
                results["details"]["corrupt_cache"] = {
                    "error": "Cache file not found"
                }
            
            # Evaluar resultado general
            results["passed"] = (
                results["details"]["invalid_key"]["simulated_generated"] and
                results["details"]["invalid_fixture"]["simulated_generated"] and
                results["details"].get("corrupt_cache", {}).get("recovered", False)
            )
            
            if results["passed"]:
                logger.info("✅ Sistema se recupera correctamente de errores")
            else:
                logger.warning("⚠️ Sistema tiene problemas de recuperación ante errores")
            
        except Exception as e:
            logger.error(f"❌ Error probando recuperación de errores: {str(e)}")
            results["error"] = str(e)
            
        return results
    
    def generate_summary(self):
        """Generar resumen de resultados"""
        tests = self.test_results["tests"]
        
        # Calcular estadísticas generales
        total_tests = len(tests)
        passed_tests = sum(1 for t in tests.values() if t.get("passed", False))
        
        self.test_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
        }
        
        # Extraer datos específicos importantes
        if "real_data" in tests:
            self.test_results["summary"]["real_data_rate"] = tests["real_data"].get("real_data_rate", 0)
            
        if "api_connection" in tests:
            self.test_results["summary"]["api_success_rate"] = tests["api_connection"].get("success_rate", 0)
            
        if "high_load" in tests:
            self.test_results["summary"]["load_success_rate"] = tests["high_load"]["details"].get("success_rate", 0)
            self.test_results["summary"]["avg_response_time"] = tests["high_load"]["details"].get("avg_time_ms", 0)
        
    def save_results(self):
        """Guardar resultados de pruebas"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"integrated_test_results_{timestamp}.json"
        
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.test_results, f, indent=2)
                
            logger.info(f"Resultados guardados en {results_file}")
            
        except Exception as e:
            logger.error(f"Error guardando resultados: {e}")
    
    def generate_performance_charts(self):
        """Generar gráficos de rendimiento"""
        try:
            # Solo generar si hay datos suficientes
            if not self.test_results.get("tests"):
                return
                
            # Directorio para gráficos
            charts_dir = self.results_dir / "charts"
            charts_dir.mkdir(exist_ok=True)
            
            # Timestamp para nombres de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 1. Gráfico de tasa de éxito por prueba
            plt.figure(figsize=(10, 6))
            
            tests = []
            success_rates = []
            
            for test_name, test_data in self.test_results["tests"].items():
                tests.append(test_name)
                
                # Extraer tasa de éxito según el tipo de prueba
                if test_name == "real_data":
                    rate = test_data.get("real_data_rate", 0)
                elif test_name == "api_connection":
                    rate = test_data.get("success_rate", 0)
                elif test_name == "high_load":
                    rate = test_data["details"].get("success_rate", 0)
                else:
                    rate = 100 if test_data.get("passed", False) else 0
                    
                success_rates.append(rate)
            
            plt.bar(tests, success_rates)
            plt.title("Tasas de Éxito por Prueba")
            plt.ylabel("Tasa de Éxito (%)")
            plt.ylim(0, 105)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Guardar gráfico
            plt.savefig(charts_dir / f"success_rates_{timestamp}.png")
            
            # 2. Gráfico de tiempos de respuesta (si hay datos de carga)
            if "high_load" in self.test_results["tests"]:
                load_test = self.test_results["tests"]["high_load"]
                
                if "details" in load_test and "requests" in load_test["details"]:
                    # Extraer datos
                    fixture_ids = []
                    times = []
                    
                    for req in load_test["details"]["requests"]:
                        if "fixture_id" in req and "time_ms" in req and "error" not in req:
                            fixture_ids.append(str(req["fixture_id"]))
                            times.append(req["time_ms"])
                    
                    if fixture_ids and times:
                        plt.figure(figsize=(10, 6))
                        plt.bar(fixture_ids, times)
                        plt.title("Tiempos de Respuesta por Partido")
                        plt.ylabel("Tiempo (ms)")
                        plt.xlabel("ID de Partido")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Guardar gráfico
                        plt.savefig(charts_dir / f"response_times_{timestamp}.png")
            
            logger.info(f"Gráficos generados en {charts_dir}")
            
        except Exception as e:
            logger.error(f"Error generando gráficos: {e}")

def main():
    """Función principal"""
    try:
        # Ejecutar pruebas integradas
        tester = IntegratedSystemTest()
        success = tester.run_all_tests()
        
        if success:
            print("\nPruebas integradas completadas con éxito.")
            print(f"Tasa de éxito: {tester.test_results['summary']['success_rate']:.1f}%")
            
            if "real_data_rate" in tester.test_results["summary"]:
                print(f"Tasa de datos reales: {tester.test_results['summary']['real_data_rate']:.1f}%")
                
            if "avg_response_time" in tester.test_results["summary"]:
                print(f"Tiempo promedio de respuesta: {tester.test_results['summary']['avg_response_time']:.1f} ms")
        else:
            print("\nPruebas integradas completadas con errores.")
            
        print("Revise los archivos de log y resultados para más detalles.")
        return success
    except Exception as e:
        logger.error(f"Error ejecutando pruebas: {str(e)}")
        print(f"\nError durante las pruebas: {str(e)}")
        return False

if __name__ == "__main__":
    main()
