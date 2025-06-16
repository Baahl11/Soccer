# 🚨 QUICK RECOVERY GUIDE - SOCCER PREDICTION SYSTEM
**Para Uso en Emergencias - Recuperación Rápida del Sistema**

---

## ⚡ RECUPERACIÓN EN 5 MINUTOS

### **PASO 1: Verificar Estado del Sistema**
```powershell
# Navegar al directorio
cd "c:\Users\gm_me\Soccer2\Soccer"

# Verificar archivos críticos
ls app.py, data.py, final_system_test.py

# Iniciar servidor
python app.py
```

### **PASO 2: Ejecutar Tests de Diagnóstico**
```powershell
# En nueva terminal
python final_system_test.py
```

**✅ Resultado Esperado**:
```
FINAL RESULT: 5/5 tests passed
🎉 ALL TESTS PASSED! System is working correctly.
```

**❌ Si Falla, Continuar con Fixes**

---

## 🔧 FIXES ESPECÍFICOS POR ERROR

### **ERROR 1: 'FootballAPI' object has no attribute '_respect_rate_limit'**

**Solución Inmediata**:
```python
# Abrir data.py y agregar al FINAL del archivo:
FootballAPI = ApiClient
```

**Código Completo para Agregar**:
```python
# Backward compatibility: Make FootballAPI an alias to ApiClient
# This ensures all existing code that uses FootballAPI will work with the new ApiClient
FootballAPI = ApiClient
```

**Verificar que está al final del archivo después de todas las funciones**

---

### **ERROR 2: Valores de Confianza Hardcodeados (0.7, 0.5)**

**Archivo**: `app.py`  
**Buscar la función**: `normalize_prediction_structure`

**CAMBIO CRÍTICO**:
```python
# ENCONTRAR esta línea (problemática):
'score': prediction.get('confidence', 0.7),

# REEMPLAZAR por:
'score': get_or_calculate_confidence(prediction),
```

**Agregar esta función si no existe**:
```python
def get_or_calculate_confidence(prediction):
    """
    Get existing confidence or calculate dynamic confidence if needed.
    Priority: confidence_score > confidence > calculate dynamically
    """
    try:
        # Try to get existing confidence values
        existing_confidence = prediction.get('confidence_score') or prediction.get('confidence')
        
        # If we have a valid confidence value (not default 0.5 or 0.7), preserve it
        if existing_confidence and isinstance(existing_confidence, (int, float)):
            if existing_confidence != 0.5 and existing_confidence != 0.7:
                logger.debug(f"Preserving existing confidence: {existing_confidence}")
                return round(float(existing_confidence), 2)
        
        # Otherwise, calculate dynamic confidence
        logger.debug("Calculating dynamic confidence for prediction")
        return calculate_dynamic_confidence(prediction)
        
    except Exception as e:
        logger.warning(f"Error in get_or_calculate_confidence: {e}")
        return calculate_dynamic_confidence(prediction)
```

---

### **ERROR 3: Función calculate_dynamic_confidence Duplicada**

**Problema**: Dos funciones con el mismo nombre en `app.py`

**Solución**:
1. Buscar `def calculate_dynamic_confidence` en app.py
2. Si hay 2 instancias, eliminar la segunda (usualmente alrededor de línea 452-478)
3. Mantener solo la primera instancia

---

### **ERROR 4: Fallback de Confianza Hardcodeado**

**Buscar en `calculate_dynamic_confidence`**:
```python
# ENCONTRAR:
except Exception as e:
    logger.warning(f"Error calculating confidence: {e}")
    return 0.7

# REEMPLAZAR por:
except Exception as e:
    logger.warning(f"Error calculating confidence: {e}")
    # Calculate a basic fallback confidence based on available data
    try:
        home_prob = prediction.get('home_win_probability', 0.33)
        away_prob = prediction.get('away_win_probability', 0.33)
        draw_prob = prediction.get('draw_probability', 0.34)
        
        # Calculate confidence based on prediction strength
        max_prob = max(home_prob, away_prob, draw_prob)
        confidence = 0.4 + (max_prob - 0.33) * 1.5  # Scale to 0.4-0.9 range
        return round(max(0.4, min(0.9, confidence)), 2)
    except:
        return 0.6  # Final fallback
```

---

## 🧪 TESTS DE VALIDACIÓN RÁPIDA

### **Test de Confianza Dinámica (1 minuto)**:
```python
# Crear archivo test_confidence_quick.py
import sys
sys.path.append('.')
from app import calculate_dynamic_confidence

test_prediction = {
    "home_team_id": 40,
    "away_team_id": 50,
    "league_id": 39,
    "fixture_id": 12345,
    "home_win_probability": 0.65,
    "away_win_probability": 0.20,
    "draw_probability": 0.15
}

confidences = []
for i in range(5):
    test_prediction["fixture_id"] = 12345 + i
    confidence = calculate_dynamic_confidence(test_prediction)
    confidences.append(confidence)
    print(f"Test {i+1}: {confidence}")

print(f"Unique values: {len(set(confidences))}")
print(f"All in range: {all(0.4 <= c <= 0.9 for c in confidences)}")
```

### **Test de API (30 segundos)**:
```powershell
# Verificar conectividad
Test-NetConnection -ComputerName 127.0.0.1 -Port 5000

# Test básico de API (si server está ejecutándose)
Invoke-RestMethod -Uri "http://127.0.0.1:5000" -Method GET -TimeoutSec 10
```

---

## 📋 CHECKLIST DE RECUPERACIÓN

### **□ Verificaciones Previas**:
- [ ] Python ejecutándose correctamente
- [ ] Directorio correcto: `c:\Users\gm_me\Soccer2\Soccer`
- [ ] Archivos principales existen: `app.py`, `data.py`

### **□ Fix FootballAPI**:
- [ ] Abrir `data.py`
- [ ] Ir al final del archivo
- [ ] Agregar: `FootballAPI = ApiClient`
- [ ] Guardar archivo

### **□ Fix Confidence System**:
- [ ] Abrir `app.py`
- [ ] Encontrar `normalize_prediction_structure`
- [ ] Cambiar `'score': prediction.get('confidence', 0.7),` por `'score': get_or_calculate_confidence(prediction),`
- [ ] Agregar función `get_or_calculate_confidence` si no existe
- [ ] Guardar archivo

### **□ Verificar Funciones**:
- [ ] Solo 1 función `calculate_dynamic_confidence` en `app.py`
- [ ] Fallback inteligente en lugar de `return 0.7`
- [ ] Imports funcionando correctamente

### **□ Tests Finales**:
- [ ] `python app.py` inicia sin errores
- [ ] `python final_system_test.py` devuelve 5/5 passed
- [ ] Valores de confianza muestran variación
- [ ] API responde en puerto 5000

---

## 🆘 COMANDOS DE EMERGENCIA

### **Si Nada Funciona - Reset Completo**:

1. **Backup Current State**:
```powershell
cp app.py app.py.backup
cp data.py data.py.backup
```

2. **Check Git Status**:
```powershell
git status
git log --oneline -5
```

3. **Restore from Backup** (si existe):
```powershell
# Solo si hay backup conocido funcionando
git checkout HEAD~1 app.py data.py
```

### **Verificar Logs de Error**:
```powershell
# Ver últimas líneas del log cuando corre el servidor
python app.py 2>&1 | Select-Object -Last 20
```

### **Verificar Imports Manualmente**:
```powershell
python -c "from data import FootballAPI, ApiClient; print('OK')"
python -c "from app import calculate_dynamic_confidence; print('OK')"
```

---

## 🔍 SIGNOS DE QUE EL SISTEMA ESTÁ FUNCIONANDO

### **✅ Señales Positivas**:
- Server inicia con: `INFO: Starting server on http://127.0.0.1:5000`
- No errores de importación
- Tests devuelven: `🎉 ALL TESTS PASSED!`
- Confidence values varían: `[0.64, 0.66, 0.77, 0.58, 0.72]`
- API responde en menos de 60 segundos

### **❌ Señales de Problemas**:
- `'FootballAPI' object has no attribute '_respect_rate_limit'`
- `cannot import name 'ConfidenceCalculator'`
- Confidence values idénticos: `[0.7, 0.7, 0.7, 0.7, 0.7]`
- Server no inicia o se cuelga
- Tests fallan con errores de importación

---

## 📞 ARCHIVOS CRÍTICOS A VERIFICAR

### **Si Sistema No Inicia**:
1. `app.py` - Función principal y routes
2. `data.py` - API client y alias FootballAPI
3. `requirements.txt` - Dependencias
4. `.env` - Variables de entorno

### **Si Confianza No Funciona**:
1. `app.py` - Funciones de confianza
2. `confidence.py` - Sistema de cálculo
3. `predictions.py` - Pipeline de predicciones

### **Si Tests Fallan**:
1. `final_system_test.py` - Suite de tests
2. Imports en archivos principales
3. Estructura de directorios

---

**🚨 ESTE DOCUMENTO ES PARA EMERGENCIAS**  
**Si el sistema funciona, no cambiar nada**  
**Solo usar cuando el sistema falle**
