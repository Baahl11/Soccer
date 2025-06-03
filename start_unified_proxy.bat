@echo off
echo ==========================================
echo Proxy Unificado para API de Predicciones
echo ==========================================
echo.
echo Este script inicia el proxy unificado que combina las funcionalidades de:
echo  - fixed_api_wrapper.py
echo  - json_interceptor.py
echo.
echo En un solo servicio que:
echo  1. Asegura que tactical_analysis este en el nivel principal
echo  2. Asegura que odds_analysis este en el nivel principal
echo  3. Corrige nombres de campos para estadisticas
echo  4. Genera datos enriquecidos cuando falta informacion
echo.
echo Los endpoints disponibles seran:
echo  - API original: http://localhost:5000/api/upcoming_predictions
echo  - Proxy completo: http://localhost:8080/api/upcoming_predictions
echo  - Endpoint dedicado: http://localhost:8080/api/fixed_predictions
echo.

REM Verificar si Python esta instalado
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python no esta instalado o no se encuentra en el PATH.
    echo Instale Python o verifique su instalacion.
    pause
    exit /b 1
)

REM Verificar que el servidor original esta activo
echo Verificando que el servidor original este en ejecucion...
curl -s http://localhost:5000/ >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] El servidor original no esta activo en http://localhost:5000/
    echo Por favor, inicie primero el servidor con: python app.py
    pause
    exit /b 1
)
echo Servidor principal detectado correctamente.
echo.

REM Iniciar el proxy unificado
echo Iniciando proxy unificado en http://localhost:8080...
start "Proxy API Unificado" cmd /c "cd /D %~dp0 && python unified_api_proxy.py"

echo.
echo El proxy ha sido iniciado en una ventana separada.
echo Para detener el servicio, cierre la ventana del proxy o presione Ctrl+C en ella.
echo.
echo Ahora puede usar el endpoint:
echo http://localhost:8080/api/upcoming_predictions?league_id=619^&season=2025^&include_additional_data=true^&limit=1
echo.
echo Informacion detallada: http://localhost:8080/proxy_info
echo.
