#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monitoreo Automatizado de Integración de Odds API

Este script está diseñado para ejecutarse periódicamente (diariamente) para
verificar la salud de la integración de la API de odds y enviar notificaciones
en caso de problemas.

Puede configurarse como una tarea cron en Linux:
0 9 * * * /usr/bin/python3 /ruta/a/monitor_odds_integration.py >> /var/log/odds_monitor.log 2>&1

O como una tarea programada en Windows.

Autor: Equipo de Desarrollo
Fecha: Mayo 23, 2025
"""

import os
import sys
import json
import logging
import smtplib
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuración de logging
log_file = Path("logs") / f"odds_monitor_{datetime.now().strftime('%Y%m%d')}.log"
log_file.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='a'
)

logger = logging.getLogger('odds_monitor')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console)

# Configuración de monitoreo
MONITOR_CONFIG = {
    # Umbrales de alertas
    "thresholds": {
        "simulated_data_pct": 10.0,    # Alerta si más del 10% de los datos son simulados
        "api_error_rate": 5.0,         # Alerta si la tasa de error de la API supera el 5%
        "response_time_ms": 1000,      # Alerta si el tiempo de respuesta supera 1000ms
        "cache_hit_rate_min": 85.0,    # Alerta si la tasa de aciertos de caché es menor al 85%
    },
    # Configuración de verificaciones
    "checks": {
        "fixtures_per_check": 10,      # Número de partidos a verificar en cada ejecución
        "leagues_to_check": ["39", "140", "78", "135", "61"],  # IDs de ligas a monitorear
        "bookmakers_to_check": [1, 6, 8, 2]  # IDs de bookmakers a verificar
    },
    # Intervalos de verificación en minutos
    "intervals": {
        "full_check": 360,             # Verificación completa cada 6 horas
        "quick_check": 60,             # Verificación rápida cada hora
        "critical_issues": 15          # Revisión de problemas críticos cada 15 minutos
    }
}

# Configuración de correo electrónico (actualizar con valores reales)
EMAIL_CONFIG = {
    "enabled": False,  # Cambiar a True para habilitar correos
    "smtp_server": "smtp.example.com",
    "smtp_port": 587,
    "username": "alertas@tuequipo.com",
    "password": "contraseña_segura",
    "from_email": "alertas@tuequipo.com",
    "to_emails": ["equipo@tuequipo.com", "admin@tuequipo.com"],
    "subject_prefix": "[ALERTA ODDS API] "
}

# Umbrales de alerta
ALERT_THRESHOLDS = {
    "simulated_data": 10,  # Alerta si más del 10% de los datos son simulados
    "error_rate": 5,       # Alerta si más del 5% de las solicitudes fallan
    "response_time": 2000  # Alerta si el tiempo promedio de respuesta supera 2000ms
}

def banner(text):
    """Mostrar un banner de texto en la consola y el log"""
    banner_text = "\n" + "=" * 70 + f"\n {text} \n" + "=" * 70
    logger.info(banner_text)
    return banner_text

def verify_odds_integration():
    """
    Ejecuta el script de verificación de integración y analiza los resultados
    
    Returns:
        dict: Resultados de la verificación
    """
    banner("INICIANDO VERIFICACIÓN DE INTEGRACIÓN")
    
    # Comprobar que existe el script de verificación
    verify_script = Path("verify_odds_integration.py")
    if not verify_script.exists():
        logger.error(f"No se encontró el script de verificación: {verify_script}")
        return {
            "success": False,
            "error": f"Script no encontrado: {verify_script}"
        }
    
    # Ejecutar script de verificación
    logger.info("Ejecutando script de verificación...")
    
    # Crear un archivo temporal para los resultados
    results_file = Path(f"odds_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    try:
        import subprocess
        cmd = [sys.executable, str(verify_script), "--output", str(results_file), "--quiet"]
        
        logger.info(f"Ejecutando: {' '.join(cmd)}")
        
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if process.returncode != 0:
            logger.error(f"El script de verificación falló con código {process.returncode}")
            logger.error(f"Error: {process.stderr}")
            return {
                "success": False,
                "error": f"Error en verificación: {process.stderr[:500]}..."
            }
        
        # Leer los resultados
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                verification_results = json.load(f)
            
            logger.info(f"Verificación completada y resultados analizados")
            return verification_results
        else:
            logger.error(f"El archivo de resultados no fue creado: {results_file}")
            return {
                "success": False,
                "error": "No se generaron resultados de verificación"
            }
            
    except Exception as e:
        logger.exception(f"Error ejecutando verificación: {e}")
        return {
            "success": False,
            "error": f"Excepción: {str(e)}"
        }
    finally:
        # Limpiar archivo temporal si existe
        if results_file.exists() and results_file.is_file():
            results_file.unlink()

def run_diagnostic_sample():
    """
    Ejecuta un diagnóstico de muestra sobre partidos específicos para medir
    la calidad actual de los datos de odds
    
    Returns:
        dict: Resultados del diagnóstico
    """
    banner("EJECUTANDO DIAGNÓSTICO DE MUESTRA")
    
    try:
        # Importar módulos necesarios
        sys.path.append('.')
        import optimize_odds_integration
        from odds_metrics import record_api_call, record_cache_event
        
        # Configurar caché
        optimize_odds_integration.setup_cache()
        
        # Lista de partidos para probar (actualizar con IDs relevantes)
        # Usar configuración de ligas a monitorear
        leagues_to_check = MONITOR_CONFIG["checks"]["leagues_to_check"]
        fixtures_to_test = []
        
        # Intentar obtener fixtures de las ligas configuradas
        if not fixtures_to_test:
            logger.info("Buscando partidos recientes para diagnosticar...")
            for league_id in leagues_to_check:
                if len(fixtures_to_test) >= MONITOR_CONFIG["checks"]["fixtures_per_check"]:
                    break
                    
                try:
                    # Intentar obtener fixtures recientes de esta liga
                    endpoint = f"{config.API_BASE_URL}/fixtures"
                    headers = {
                        "x-rapidapi-key": config.API_KEY,
                        "x-rapidapi-host": config.API_BASE_URL.replace("https://", "")
                    }
                    params = {
                        "league": league_id,
                        "last": "5"  # Últimos 5 partidos de la liga
                    }
                    
                    response = requests.get(
                        endpoint,
                        headers=headers,
                        params=params,
                        timeout=15
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("response"):
                            for fixture in data["response"]:
                                fixture_id = fixture.get("fixture", {}).get("id")
                                if fixture_id:
                                    fixtures_to_test.append(fixture_id)
                                    
                            logger.info(f"Encontrados {len(fixtures_to_test)} partidos en liga {league_id}")
                except Exception as e:
                    logger.warning(f"Error buscando partidos en liga {league_id}: {e}")
        
        # Si no se encontraron partidos, usar los IDs predeterminados
        if not fixtures_to_test:
            fixtures_to_test = [1208383, 1208384, 1208385, 1208386, 1208387]
            logger.info(f"Usando IDs de partidos predeterminados")
        
        # Limitar al máximo configurado
        fixtures_to_test = fixtures_to_test[:MONITOR_CONFIG["checks"]["fixtures_per_check"]]
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "total_fixtures": len(fixtures_to_test),
            "success_count": 0,
            "error_count": 0,
            "simulated_count": 0,
            "response_times": [],
            "details": {}
        }
        
        for fixture_id in fixtures_to_test:
            logger.info(f"Probando fixture {fixture_id}...")
            
            start_time = datetime.now()
            try:
                # Forzar refresco para obtener datos actuales
                odds_data = optimize_odds_integration.get_fixture_odds(
                    fixture_id,
                    use_cache=False,
                    force_refresh=True
                )
                  response_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
                results["response_times"].append(response_time)
                
                if odds_data:
                    results["success_count"] += 1
                    
                    # Verificar si son datos simulados
                    is_simulated = odds_data.get("simulated", True)
                    if is_simulated:
                        results["simulated_count"] += 1
                        logger.warning(f"Fixture {fixture_id}: Datos SIMULADOS")
                    else:
                        logger.info(f"Fixture {fixture_id}: Datos REALES")
                    
                    # Registrar métricas en el sistema de monitoreo
                    record_api_call(
                        endpoint="/odds",
                        params={"fixture": fixture_id},
                        success=True,
                        response_time_ms=response_time,
                        simulated=is_simulated
                    )
                    
                    # Registrar información de bookmakers si está disponible
                    if "bookmaker" in odds_data:
                        bookmaker = odds_data["bookmaker"]
                        if isinstance(bookmaker, dict) and "id" in bookmaker and "name" in bookmaker:
                            from odds_metrics import OddsAPIMetrics
                            metrics = OddsAPIMetrics()
                            metrics.record_bookmaker_coverage(bookmaker["id"], bookmaker["name"])
                    
                    results["details"][fixture_id] = {
                        "success": True,
                        "simulated": is_simulated,
                        "response_time_ms": response_time,
                        "bookmakers_count": odds_data.get("bookmakers_count", 0)
                    }
                else:
                    results["error_count"] += 1
                    logger.error(f"Fixture {fixture_id}: Error obteniendo datos")
                    results["details"][fixture_id] = {
                        "success": False,
                        "error": "No se obtuvieron datos"
                    }
            
            except Exception as e:
                results["error_count"] += 1
                logger.error(f"Fixture {fixture_id}: Excepción - {str(e)}")
                results["details"][fixture_id] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Calcular estadísticas
        if results["success_count"] > 0:
            results["simulated_percentage"] = (results["simulated_count"] / results["total_fixtures"]) * 100
        else:
            results["simulated_percentage"] = 100
            
        if results["response_times"]:
            results["avg_response_time"] = sum(results["response_times"]) / len(results["response_times"])
        else:
            results["avg_response_time"] = 0
            
        results["error_percentage"] = (results["error_count"] / results["total_fixtures"]) * 100
        
        logger.info(f"Diagnóstico completado - Resumen:")
        logger.info(f"- Total fixtures: {results['total_fixtures']}")
        logger.info(f"- Éxitos: {results['success_count']} ({100 - results['error_percentage']:.1f}%)")
        logger.info(f"- Datos simulados: {results['simulated_count']} ({results['simulated_percentage']:.1f}%)")
        logger.info(f"- Tiempo promedio: {results['avg_response_time']:.1f} ms")
        
        return results
    
    except Exception as e:
        logger.exception(f"Error ejecutando diagnóstico de muestra: {e}")
        return {
            "success": False,
            "error": str(e),
            "simulated_percentage": 100,
            "error_percentage": 100
        }

def check_historical_trend():
    """
    Analiza los resultados históricos para identificar tendencias
    en la calidad de los datos
    
    Returns:
        dict: Análisis de tendencia
    """
    banner("ANALIZANDO TENDENCIA HISTÓRICA")
    
    history_dir = Path("monitoring_history")
    history_dir.mkdir(exist_ok=True)
    
    try:
        # Buscar archivos de diagnóstico de los últimos 7 días
        now = datetime.now()
        history_files = []
        
        for i in range(7):
            date = now - timedelta(days=i)
            filename = f"diagnostic_{date.strftime('%Y%m%d')}.json"
            file_path = history_dir / filename
            
            if file_path.exists():
                history_files.append(file_path)
        
        if not history_files:
            logger.info("No se encontraron archivos históricos para analizar")
            return {
                "history_available": False,
                "message": "No hay suficientes datos históricos para analizar tendencia"
            }
            
        # Cargar y analizar datos históricos
        history_data = []
        for file_path in history_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    history_data.append({
                        "date": file_path.stem.replace("diagnostic_", ""),
                        "simulated_percentage": data.get("simulated_percentage", 100),
                        "error_percentage": data.get("error_percentage", 0),
                        "avg_response_time": data.get("avg_response_time", 0)
                    })
            except Exception as e:
                logger.warning(f"Error leyendo archivo histórico {file_path}: {e}")
        
        if not history_data:
            return {
                "history_available": False,
                "message": "No se pudieron leer los archivos históricos"
            }
            
        # Ordenar por fecha
        history_data.sort(key=lambda x: x["date"])
        
        # Calcular tendencia
        if len(history_data) >= 2:
            first = history_data[0]
            last = history_data[-1]
            
            trend = {
                "history_available": True,
                "days_analyzed": len(history_data),
                "simulated_trend": last["simulated_percentage"] - first["simulated_percentage"],
                "error_trend": last["error_percentage"] - first["error_percentage"],
                "response_trend": last["avg_response_time"] - first["avg_response_time"],
                "history": history_data
            }
            
            logger.info(f"Análisis de tendencia (últimos {len(history_data)} días):")
            logger.info(f"- Tendencia datos simulados: {trend['simulated_trend']:.1f}%")
            logger.info(f"- Tendencia errores: {trend['error_trend']:.1f}%")
            logger.info(f"- Tendencia tiempo respuesta: {trend['response_trend']:.1f} ms")
            
            return trend
        else:
            return {
                "history_available": True,
                "days_analyzed": 1,
                "message": "No hay suficientes días para calcular tendencia",
                "history": history_data
            }
    
    except Exception as e:
        logger.exception(f"Error analizando tendencia histórica: {e}")
        return {
            "history_available": False,
            "error": str(e)
        }

def detect_alerts(diagnostic_results, trend_analysis):
    """
    Detecta situaciones de alerta basadas en los umbrales configurados
    
    Args:
        diagnostic_results: Resultados del diagnóstico actual
        trend_analysis: Análisis de tendencia histórica
        
    Returns:
        list: Lista de alertas detectadas
    """
    alerts = []
    
    # Alertas basadas en diagnóstico actual
    if diagnostic_results.get("simulated_percentage", 0) > ALERT_THRESHOLDS["simulated_data"]:
        alerts.append({
            "severity": "HIGH",
            "type": "simulated_data",
            "message": f"Alto porcentaje de datos simulados: {diagnostic_results['simulated_percentage']:.1f}% (umbral: {ALERT_THRESHOLDS['simulated_data']}%)"
        })
    
    if diagnostic_results.get("error_percentage", 0) > ALERT_THRESHOLDS["error_rate"]:
        alerts.append({
            "severity": "HIGH",
            "type": "error_rate",
            "message": f"Alta tasa de error: {diagnostic_results['error_percentage']:.1f}% (umbral: {ALERT_THRESHOLDS['error_rate']}%)"
        })
        
    if diagnostic_results.get("avg_response_time", 0) > ALERT_THRESHOLDS["response_time"]:
        alerts.append({
            "severity": "MEDIUM",
            "type": "response_time",
            "message": f"Tiempo de respuesta elevado: {diagnostic_results['avg_response_time']:.1f} ms (umbral: {ALERT_THRESHOLDS['response_time']} ms)"
        })
    
    # Alertas basadas en tendencia histórica
    if trend_analysis.get("history_available") and trend_analysis.get("simulated_trend", 0) > 5:
        alerts.append({
            "severity": "MEDIUM",
            "type": "trend_simulated",
            "message": f"Tendencia creciente de datos simulados: +{trend_analysis['simulated_trend']:.1f}% en los últimos {trend_analysis['days_analyzed']} días"
        })
    
    if trend_analysis.get("history_available") and trend_analysis.get("error_trend", 0) > 5:
        alerts.append({
            "severity": "MEDIUM",
            "type": "trend_errors",
            "message": f"Tendencia creciente de errores: +{trend_analysis['error_trend']:.1f}% en los últimos {trend_analysis['days_analyzed']} días"
        })
        
    logger.info(f"Detectadas {len(alerts)} alertas")
    return alerts

def save_diagnostic_results(results):
    """Guarda los resultados del diagnóstico para análisis histórico"""
    history_dir = Path("monitoring_history")
    history_dir.mkdir(exist_ok=True)
    
    today = datetime.now().strftime("%Y%m%d")
    result_file = history_dir / f"diagnostic_{today}.json"
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Resultados guardados en {result_file}")
        return True
    except Exception as e:
        logger.error(f"Error guardando resultados: {e}")
        return False

def send_email_notification(alerts, diagnostic_results):
    """Envía notificación por correo si hay alertas"""
    if not EMAIL_CONFIG["enabled"] or not alerts:
        return
    
    try:
        # Crear mensaje
        msg = MIMEMultipart()
        
        # Determinar asunto según severidad
        has_high = any(alert["severity"] == "HIGH" for alert in alerts)
        severity_prefix = "[CRÍTICO]" if has_high else "[ADVERTENCIA]"
        
        msg['Subject'] = f"{EMAIL_CONFIG['subject_prefix']}{severity_prefix} Alertas en API de Odds ({len(alerts)})"
        msg['From'] = EMAIL_CONFIG["from_email"]
        msg['To'] = ", ".join(EMAIL_CONFIG["to_emails"])
        
        # Crear contenido HTML
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; }}
                .alert {{ padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .HIGH {{ background-color: #ffcccc; border: 1px solid #ff0000; }}
                .MEDIUM {{ background-color: #fff4cc; border: 1px solid #ffc107; }}
                .LOW {{ background-color: #e6f7ff; border: 1px solid #1890ff; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h2>Alertas de Monitoreo de API de Odds</h2>
            <p>Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h3>Alertas Detectadas:</h3>
        """
        
        # Añadir alertas
        for alert in alerts:
            html += f"""
            <div class="alert {alert['severity']}">
                <strong>{alert['severity']}:</strong> {alert['message']}
            </div>
            """
        
        # Añadir resumen
        html += f"""
            <h3>Resumen de Diagnóstico:</h3>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor</th>
                    <th>Umbral</th>
                </tr>
                <tr>
                    <td>Datos simulados</td>
                    <td>{diagnostic_results.get('simulated_percentage', 'N/A'):.1f}%</td>
                    <td>{ALERT_THRESHOLDS['simulated_data']}%</td>
                </tr>
                <tr>
                    <td>Tasa de error</td>
                    <td>{diagnostic_results.get('error_percentage', 'N/A'):.1f}%</td>
                    <td>{ALERT_THRESHOLDS['error_rate']}%</td>
                </tr>
                <tr>
                    <td>Tiempo de respuesta</td>
                    <td>{diagnostic_results.get('avg_response_time', 'N/A'):.1f} ms</td>
                    <td>{ALERT_THRESHOLDS['response_time']} ms</td>
                </tr>
            </table>
            
            <p>Para más detalles, revise los logs en el servidor.</p>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(html, 'html'))
        
        # Enviar correo
        server = smtplib.SMTP(EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"])
        server.starttls()
        server.login(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Notificación enviada a {', '.join(EMAIL_CONFIG['to_emails'])}")
        
    except Exception as e:
        logger.error(f"Error enviando notificación: {e}")

def main():
    """Función principal que ejecuta el monitoreo"""
    start_time = datetime.now()
    banner(f"MONITOREO DE API ODDS - {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Ejecutar diagnóstico
        diagnostic_results = run_diagnostic_sample()
        
        # Guardar resultados para análisis histórico
        save_diagnostic_results(diagnostic_results)
        
        # Analizar tendencia histórica
        trend_analysis = check_historical_trend()
        
        # Generar informe de salud usando el nuevo sistema de métricas
        health_report = generate_health_report(MONITOR_CONFIG["thresholds"])
        
        # Obtener alertas del informe de salud
        new_alerts = health_report.get("alerts", [])
        
        # Convertir al formato antiguo para compatibilidad
        alerts = []
        for alert in new_alerts:
            alerts.append({
                "severity": "HIGH" if alert.get("type") == "critical" else "MEDIUM",
                "type": alert.get("metric", "unknown"),
                "message": alert.get("message", "Alerta desconocida")
            })
        
        # Añadir alertas del sistema antiguo para compatibilidad
        legacy_alerts = detect_alerts(diagnostic_results, trend_analysis)
        alerts.extend(legacy_alerts)
        
        # Verificación completa de integración (más intensiva)
        integration_results = verify_odds_integration()
        
        # Generar visualización si hay alertas
        visualization_path = None
        if alerts:
            # Crear visualización de métricas
            visualization_path = create_metrics_visualization(7)  # Últimos 7 días
            
            # Enviar notificación con el gráfico adjunto
            send_email_notification(alerts, diagnostic_results)
        
        # Generar informe final
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        banner(f"MONITOREO COMPLETADO EN {elapsed_time:.2f} SEGUNDOS")
        logger.info(f"Estado: {'✓ OK' if not alerts else f'⚠ {len(alerts)} ALERTAS'}")
        logger.info(f"Estado general del sistema: {health_report.get('status', 'desconocido').upper()}")
        
        if alerts:
            for alert in alerts:
                logger.warning(f"{alert['severity']}: {alert['message']}")
            
            if visualization_path:
                logger.info(f"Visualización generada en: {visualization_path}")
        
        return 0
    
    except Exception as e:
        logger.exception(f"Error en monitoreo: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
