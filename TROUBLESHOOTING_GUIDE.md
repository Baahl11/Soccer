# Guía de Resolución de Problemas

## 🚨 Troubleshooting Guide - Sistema de Predicciones Mejorado

Esta guía proporciona soluciones para problemas comunes y procedimientos de diagnóstico para el Sistema de Predicciones Mejorado.

## 📋 Índice de Problemas

### 🟢 **Problemas Resueltos**
- [Probabilidades Idénticas](#problema-1-probabilidades-idénticas-resuelto) ✅ RESUELTO
- [Conversión de Formatos](#problema-2-conversión-de-formatos-resuelto) ✅ RESUELTO
- [API No Responde](#problema-3-api-no-responde-resuelto) ✅ RESUELTO

### 🟡 **Problemas Potenciales**
- [Tiempo de Respuesta Lento](#problema-4-tiempo-de-respuesta-lento)
- [Valores xG Extremos](#problema-5-valores-xg-extremos)
- [Errores de Conexión API](#problema-6-errores-de-conexión-api)
- [Suma de Probabilidades ≠ 100%](#problema-7-suma-de-probabilidades--100)

### 🔴 **Problemas Críticos**
- [Sistema No Inicia](#problema-8-sistema-no-inicia)
- [Modelos ML No Cargan](#problema-9-modelos-ml-no-cargan)
- [Base de Datos Corrupta](#problema-10-base-de-datos-corrupta)

## ✅ Problemas Resueltos

### **Problema 1: Probabilidades Idénticas** ✅ RESUELTO

#### **Síntomas Anteriores:**
```
❌ Todas las predicciones: Home 42.1%, Draw 35.7%, Away 22.2%
```

#### **Causa Raíz Identificada:**
Valores estáticos de xG en `enhanced_match_winner.py`:
```python
# ❌ CÓDIGO PROBLEMÁTICO (YA CORREGIDO)
home_xg = kwargs.get('home_xg', 1.3)  # Siempre 1.3
away_xg = kwargs.get('away_xg', 1.1)  # Siempre 1.1
```

#### **Solución Implementada:**
```python
# ✅ CÓDIGO CORREGIDO
if 'home_xg' not in kwargs or 'away_xg' not in kwargs:
    from dynamic_xg_calculator import calculate_match_xg
    calculated_home_xg, calculated_away_xg = calculate_match_xg(
        home_team_id, away_team_id, home_form, away_form, league_id or 39, h2h
    )
    home_xg = kwargs.get('home_xg', calculated_home_xg)
    away_xg = kwargs.get('away_xg', calculated_away_xg)
```

#### **Resultado Actual:**
```
✅ Manchester United vs Liverpool: 34.0% / 36.6% / 29.4%
✅ Real Madrid vs Barcelona: 45.0% / 27.2% / 27.8%
✅ Bayern vs Dortmund: 48.8% / 29.4% / 21.8%
```

#### **Estado:** 🟢 COMPLETAMENTE RESUELTO

---

### **Problema 2: Conversión de Formatos** ✅ RESUELTO

#### **Síntomas Anteriores:**
- Sistema base devolvía porcentajes (45.2%)
- Sistema de mejora esperaba decimales (0.452)
- Incompatibilidad causaba errores o valores incorrectos

#### **Solución Implementada:**
Sistema automático de conversión bidireccional:

```python
# Conversión: Porcentajes → Decimales
if any(prob > 1 for prob in base_probs.values()):
    normalized_prediction['probabilities'] = {
        'home_win': base_probs.get('home_win', 0) / 100.0,
        'draw': base_probs.get('draw', 0) / 100.0,
        'away_win': base_probs.get('away_win', 0) / 100.0
    }

# Conversión: Decimales → Porcentajes
if all(prob <= 1 for prob in probs.values()):
    enhanced_prediction['probabilities'] = {
        'home_win': round(probs.get('home_win', 0) * 100, 1),
        'draw': round(probs.get('draw', 0) * 100, 1),
        'away_win': round(probs.get('away_win', 0) * 100, 1)
    }
```

#### **Estado:** 🟢 COMPLETAMENTE RESUELTO

---

### **Problema 3: API No Responde** ✅ RESUELTO

#### **Síntomas Anteriores:**
- Endpoints devolvían errores 500
- Falta de integración con sistema enhanced

#### **Solución Implementada:**
- Actualización de todos los endpoints para usar sistema enhanced
- Manejo robusto de errores
- Validación de parámetros mejorada

#### **Estado:** 🟢 COMPLETAMENTE RESUELTO

---

## 🟡 Problemas Potenciales

### **Problema 4: Tiempo de Respuesta Lento**

#### **Síntomas:**
- Predicciones tardan > 30 segundos
- Timeouts en API requests
- Usuario reporta lentitud

#### **Diagnóstico:**
```python
# Script de diagnóstico de rendimiento
import time
from enhanced_match_winner import predict_with_enhanced_system

def diagnose_performance():
    start_time = time.time()
    
    # Test prediction
    result = predict_with_enhanced_system(33, 40, 39)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"🔍 Performance Diagnosis:")
    print(f"   Duration: {duration:.2f} seconds")
    
    if duration > 30:
        print("❌ SLOW: Prediction took too long")
        print("💡 Check: API connections, network, server load")
    elif duration > 20:
        print("⚠️ MODERATE: Within acceptable range but monitor")
    else:
        print("✅ FAST: Performance is good")

if __name__ == "__main__":
    diagnose_performance()
```

#### **Soluciones:**

##### **Solución 1: Optimizar Llamadas API**
```python
# Implementar cache para datos frecuentes
import functools
import time

@functools.lru_cache(maxsize=128)
def cached_team_form(team_id, league_id):
    # Cache form data for 5 minutes
    return get_team_form(team_id, league_id)
```

##### **Solución 2: Timeout Configuration**
```python
# Configurar timeouts más agresivos
TIMEOUT_CONFIG = {
    'api_request_timeout': 10,     # 10s max per API call
    'total_prediction_timeout': 25, # 25s max total
    'connection_timeout': 5        # 5s connection timeout
}
```

##### **Solución 3: Fallback Mode**
```python
def predict_with_fallback(home_team_id, away_team_id, league_id, **kwargs):
    try:
        # Intento normal con todos los datos
        return predict_with_enhanced_system(home_team_id, away_team_id, league_id, **kwargs)
    except TimeoutError:
        # Fallback con datos básicos y xG por defecto
        print("⚠️ Using fallback mode due to timeout")
        return predict_with_enhanced_system(
            home_team_id, away_team_id, league_id, 
            home_xg=1.4, away_xg=1.2  # Valores por defecto
        )
```

---

### **Problema 5: Valores xG Extremos**

#### **Síntomas:**
- xG > 4.0 o xG < 0.2
- Predicciones poco realistas (>80% o <5%)
- Logs muestran valores fuera de rango

#### **Diagnóstico:**
```python
def diagnose_xg_values():
    """Diagnosticar valores xG extremos"""
    
    test_cases = [
        (33, 40, 39),
        (541, 529, 140),
        (157, 165, 78)
    ]
    
    for home_id, away_id, league_id in test_cases:
        try:
            result = predict_with_enhanced_system(home_id, away_id, league_id)
            
            # Verificar si hay logs de xG
            # (buscar en logs recientes)
            
            probs = result.get('probabilities', {})
            home_win = probs.get('home_win', 0)
            
            if home_win > 70:
                print(f"❌ EXTREME: Home win {home_win}% for teams {home_id} vs {away_id}")
                print("💡 Check: xG calculation, team form data")
            elif home_win < 10:
                print(f"❌ EXTREME: Home win {home_win}% for teams {home_id} vs {away_id}")
                print("💡 Check: xG calculation, defensive stats")
            else:
                print(f"✅ OK: Home win {home_win}% for teams {home_id} vs {away_id}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
```

#### **Soluciones:**

##### **Solución 1: Ajustar Límites xG**
```python
# En dynamic_xg_calculator.py
def apply_realistic_limits(home_xg, away_xg):
    """Aplicar límites más estrictos"""
    
    # Límites conservadores
    home_xg = max(0.7, min(2.8, home_xg))  # Entre 0.7 y 2.8
    away_xg = max(0.5, min(2.5, away_xg))  # Entre 0.5 y 2.5
    
    return home_xg, away_xg
```

##### **Solución 2: Validación de Datos de Entrada**
```python
def validate_form_data(team_form):
    """Validar datos de forma antes de cálculo"""
    
    required_fields = ['goals_for', 'goals_against']
    
    for field in required_fields:
        if field not in team_form:
            raise ValueError(f"Missing required field: {field}")
        
        values = team_form[field]
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"Invalid {field} data")
        
        # Verificar valores realistas
        for value in values:
            if value < 0 or value > 10:  # Máximo 10 goles por partido
                raise ValueError(f"Unrealistic value in {field}: {value}")
```

##### **Solución 3: Smoothing Algorithm**
```python
def smooth_extreme_xg(home_xg, away_xg, baseline_home=1.4, baseline_away=1.2, smoothing_factor=0.3):
    """Suavizar valores extremos hacia baseline"""
    
    # Si los valores son extremos, aplicar suavizado
    if home_xg > 2.5 or home_xg < 0.8:
        home_xg = home_xg * (1 - smoothing_factor) + baseline_home * smoothing_factor
    
    if away_xg > 2.3 or away_xg < 0.6:
        away_xg = away_xg * (1 - smoothing_factor) + baseline_away * smoothing_factor
    
    return home_xg, away_xg
```

---

### **Problema 6: Errores de Conexión API**

#### **Síntomas:**
- `ConnectionError` o `TimeoutError`
- Datos de equipos no disponibles
- H2H data faltante

#### **Diagnóstico:**
```python
import requests

def diagnose_api_connections():
    """Diagnosticar conexiones API externas"""
    
    apis_to_test = [
        "https://api.football-data.org/v4/",
        "https://api.sportmonks.com/v3/",
        # Añadir URLs de APIs que use el sistema
    ]
    
    for api_url in apis_to_test:
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {api_url}: OK")
            else:
                print(f"⚠️ {api_url}: Status {response.status_code}")
        except Exception as e:
            print(f"❌ {api_url}: {e}")
```

#### **Soluciones:**

##### **Solución 1: Implementar Reintentos**
```python
import time
from functools import wraps

def retry_api_call(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ConnectionError, TimeoutError) as e:
                    if attempt == max_retries - 1:
                        raise e
                    print(f"⚠️ API call failed, retrying in {delay}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
            return None
        return wrapper
    return decorator

@retry_api_call(max_retries=3, delay=2)
def get_team_form_with_retry(team_id, league_id):
    return get_team_form(team_id, league_id)
```

##### **Solución 2: Datos de Fallback**
```python
def get_team_form_with_fallback(team_id, league_id):
    """Obtener forma de equipo con fallback a datos por defecto"""
    
    try:
        return get_team_form(team_id, league_id)
    except Exception as e:
        print(f"⚠️ API Error for team {team_id}: {e}")
        print("🔄 Using fallback form data")
        
        # Datos por defecto basados en estadísticas de liga
        league_defaults = {
            39: {'goals_for': [1.5, 1.3, 1.7, 1.2, 1.6], 'goals_against': [1.1, 1.4, 1.0, 1.3, 1.2]},  # Premier League
            140: {'goals_for': [1.4, 1.2, 1.6, 1.1, 1.5], 'goals_against': [1.0, 1.3, 0.9, 1.2, 1.1]}, # La Liga
            # ... más ligas
        }
        
        return league_defaults.get(league_id, {
            'goals_for': [1.3, 1.2, 1.4, 1.1, 1.3],
            'goals_against': [1.1, 1.2, 1.0, 1.3, 1.1]
        })
```

---

### **Problema 7: Suma de Probabilidades ≠ 100%**

#### **Síntomas:**
- Suma = 99.8% o 100.2%
- Logs muestran advertencias de validación
- Respuestas API inconsistentes

#### **Diagnóstico:**
```python
def diagnose_probability_sum():
    """Diagnosticar problemas de suma de probabilidades"""
    
    test_cases = [(33, 40, 39), (541, 529, 140), (157, 165, 78)]
    
    for home_id, away_id, league_id in test_cases:
        result = predict_with_enhanced_system(home_id, away_id, league_id)
        probs = result.get('probabilities', {})
        
        total = sum(probs.values())
        
        print(f"Teams {home_id} vs {away_id}: {probs}")
        print(f"Sum: {total:.3f}%")
        
        if abs(total - 100.0) > 0.2:
            print(f"❌ PROBLEM: Sum deviation = {abs(total - 100.0):.3f}%")
        else:
            print("✅ OK: Sum within tolerance")
```

#### **Soluciones:**

##### **Solución 1: Normalización Automática**
```python
def normalize_probabilities(probs):
    """Normalizar probabilidades para que sumen exactamente 100%"""
    
    total = sum(probs.values())
    
    if total == 0:
        # Caso extremo: todas las probabilidades son 0
        return {'home_win': 33.3, 'draw': 33.3, 'away_win': 33.4}
    
    # Normalizar manteniendo proporciones
    normalized = {}
    for key, value in probs.items():
        normalized[key] = round((value * 100.0) / total, 1)
    
    # Ajustar redondeo para suma exacta
    current_sum = sum(normalized.values())
    if current_sum != 100.0:
        # Ajustar el valor más grande
        max_key = max(normalized, key=normalized.get)
        normalized[max_key] += round(100.0 - current_sum, 1)
    
    return normalized
```

##### **Solución 2: Validación Estricta**
```python
def validate_and_fix_probabilities(probs, tolerance=0.1):
    """Validar y corregir probabilidades si es necesario"""
    
    total = sum(probs.values())
    
    if abs(total - 100.0) <= tolerance:
        return probs  # Dentro de tolerancia, no cambiar
    
    print(f"⚠️ Probability sum validation failed: {total}%. Normalizing...")
    return normalize_probabilities(probs)
```

---

## 🔴 Problemas Críticos

### **Problema 8: Sistema No Inicia**

#### **Síntomas:**
- Error al importar módulos
- Flask no inicia
- Modelos ML no se cargan

#### **Diagnóstico:**
```bash
# Verificar dependencias
pip list | grep -E "(flask|sklearn|tensorflow|numpy|pandas)"

# Verificar estructura de archivos
ls -la *.py | grep -E "(enhanced_match_winner|dynamic_xg_calculator|web_dashboard_api)"

# Verificar Python path
python -c "import sys; print('\n'.join(sys.path))"
```

#### **Soluciones:**

##### **Solución 1: Reinstalar Dependencias**
```bash
# Crear nuevo entorno virtual
python -m venv .venv_new
source .venv_new/bin/activate  # Linux/Mac
# .venv_new\Scripts\activate     # Windows

# Reinstalar dependencias
pip install -r requirements.txt
```

##### **Solución 2: Verificar Archivos Críticos**
```bash
# Lista de archivos críticos que deben existir
CRITICAL_FILES=(
    "enhanced_match_winner.py"
    "dynamic_xg_calculator.py"
    "web_dashboard_api.py"
    "match_winner.py"
    "draw_prediction.py"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing critical file: $file"
    else
        echo "✅ Found: $file"
    fi
done
```

---

### **Problema 9: Modelos ML No Cargan**

#### **Síntomas:**
- Error al cargar modelos `.pkl` o `.joblib`
- Versiones incompatibles de scikit-learn
- Archivos de modelo corruptos

#### **Diagnóstico:**
```python
def diagnose_ml_models():
    """Diagnosticar carga de modelos ML"""
    
    import joblib
    import pickle
    import os
    
    model_files = [
        'model_1x2.pkl',
        'scaler.pkl',
        'calibrator.pkl'
        # Añadir nombres de archivos de modelo específicos
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                model = joblib.load(model_file)
                print(f"✅ {model_file}: Loaded successfully")
            except Exception as e:
                print(f"❌ {model_file}: Error loading - {e}")
        else:
            print(f"⚠️ {model_file}: File not found")
```

#### **Soluciones:**

##### **Solución 1: Recrear Modelos**
```python
# Script para regenerar modelos si están corruptos
def regenerate_models():
    """Regenerar modelos ML desde datos de entrenamiento"""
    
    print("🔄 Regenerating ML models...")
    
    # Implementar re-entrenamiento de modelos
    # Este código dependerá de tus datos de entrenamiento específicos
    
    # Ejemplo genérico:
    # from sklearn.ensemble import RandomForestClassifier
    # model = RandomForestClassifier()
    # model.fit(X_train, y_train)
    # joblib.dump(model, 'model_1x2.pkl')
    
    print("✅ Models regenerated")
```

---

### **Problema 10: Base de Datos Corrupta**

#### **Síntomas:**
- Error al acceder a base de datos
- Datos de monitoring perdidos
- Tablas faltantes

#### **Diagnóstico:**
```python
import sqlite3

def diagnose_database():
    """Diagnosticar estado de la base de datos"""
    
    db_files = [
        'advanced_1x2_monitoring.db',
        'predictions.db'
        # Añadir nombres de archivos de BD específicos
    ]
    
    for db_file in db_files:
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Verificar integridad
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            
            if result[0] == 'ok':
                print(f"✅ {db_file}: Database integrity OK")
            else:
                print(f"❌ {db_file}: Integrity check failed - {result[0]}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ {db_file}: Error accessing database - {e}")
```

#### **Soluciones:**

##### **Solución 1: Recrear Base de Datos**
```python
def recreate_database():
    """Recrear base de datos desde cero"""
    
    import os
    import sqlite3
    
    db_file = 'advanced_1x2_monitoring.db'
    
    # Backup si existe
    if os.path.exists(db_file):
        backup_name = f"{db_file}.backup_{int(time.time())}"
        os.rename(db_file, backup_name)
        print(f"📦 Backup created: {backup_name}")
    
    # Crear nueva BD
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Crear tablas necesarias
    cursor.execute('''
        CREATE TABLE predictions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            home_team_id INTEGER,
            away_team_id INTEGER,
            league_id INTEGER,
            home_win_prob REAL,
            draw_prob REAL,
            away_win_prob REAL
        )
    ''')
    
    conn.commit()
    conn.close()
    
    print("✅ Database recreated")
```

---

## 🛠️ Herramientas de Diagnóstico

### **Script de Diagnóstico Completo**

```python
#!/usr/bin/env python3
"""
🔧 SISTEMA DE DIAGNÓSTICO COMPLETO
=====================================
Script para diagnosticar todos los componentes del sistema
"""

import sys
import os
import time
import importlib
import sqlite3
from pathlib import Path

def full_system_diagnosis():
    """Ejecutar diagnóstico completo del sistema"""
    
    print("🔍 INICIANDO DIAGNÓSTICO COMPLETO")
    print("=" * 50)
    
    # 1. Verificar archivos críticos
    check_critical_files()
    
    # 2. Verificar dependencias
    check_dependencies()
    
    # 3. Verificar modelos ML
    check_ml_models()
    
    # 4. Verificar base de datos
    check_database()
    
    # 5. Probar predicción simple
    test_simple_prediction()
    
    # 6. Probar API
    test_api_endpoints()
    
    print("\n🏁 DIAGNÓSTICO COMPLETADO")

def check_critical_files():
    print("\n📁 Verificando archivos críticos...")
    
    critical_files = [
        'enhanced_match_winner.py',
        'dynamic_xg_calculator.py',
        'web_dashboard_api.py',
        'match_winner.py',
        'draw_prediction.py'
    ]
    
    for file in critical_files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING!")

def check_dependencies():
    print("\n📦 Verificando dependencias...")
    
    required_modules = [
        'flask', 'sklearn', 'numpy', 'pandas', 
        'requests', 'joblib', 'sqlite3'
    ]
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} - NOT INSTALLED!")

def test_simple_prediction():
    print("\n🧪 Probando predicción simple...")
    
    try:
        from enhanced_match_winner import predict_with_enhanced_system
        
        start_time = time.time()
        result = predict_with_enhanced_system(33, 40, 39)
        duration = time.time() - start_time
        
        probs = result.get('probabilities', {})
        total = sum(probs.values())
        
        print(f"   ✅ Predicción exitosa en {duration:.2f}s")
        print(f"   📊 Probabilidades: {probs}")
        print(f"   📈 Suma: {total:.1f}%")
        
        if abs(total - 100.0) > 0.2:
            print(f"   ⚠️ Suma fuera de tolerancia")
        
    except Exception as e:
        print(f"   ❌ Error en predicción: {e}")

if __name__ == "__main__":
    full_system_diagnosis()
```

### **Script de Monitoreo Continuo**

```python
#!/usr/bin/env python3
"""
📊 MONITOR DE SISTEMA EN TIEMPO REAL
===================================
"""

import time
import psutil
import threading
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.running = True
        
    def monitor_performance(self):
        """Monitor de rendimiento en tiempo real"""
        
        while self.running:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            print(f"[{timestamp}] CPU: {cpu_percent:5.1f}% | RAM: {memory.percent:5.1f}% | Available: {memory.available/1024/1024/1024:.1f}GB")
            
            if cpu_percent > 80:
                print("⚠️ HIGH CPU USAGE!")
            
            if memory.percent > 85:
                print("⚠️ HIGH MEMORY USAGE!")
            
            time.sleep(5)
    
    def start_monitoring(self):
        """Iniciar monitoreo en hilo separado"""
        monitor_thread = threading.Thread(target=self.monitor_performance)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print("📊 Sistema de monitoreo iniciado")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False
            print("\n🛑 Monitoreo detenido")

if __name__ == "__main__":
    monitor = SystemMonitor()
    monitor.start_monitoring()
```

---

## 📞 Soporte y Contacto

### **Información de Soporte**

- **Documentación completa**: Ver archivos `.md` en el directorio del proyecto
- **Logs del sistema**: Verificar archivos `.log` para detalles de errores
- **Scripts de diagnóstico**: Usar herramientas incluidas en esta guía

### **Procedimiento de Escalación**

1. **Nivel 1**: Usar scripts de diagnóstico automático
2. **Nivel 2**: Revisar logs detallados y esta guía
3. **Nivel 3**: Recrear componentes problemáticos
4. **Nivel 4**: Contactar con desarrollador del sistema

---

**Última actualización**: 30 de Mayo, 2025  
**Versión de la guía**: 2.0  
**Estado del sistema**: 🟢 OPERATIVO
