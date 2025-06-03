# Arquitectura del Sistema de Predicciones Mejorado

## 🏗️ Visión General de la Arquitectura

El Sistema de Predicciones Mejorado está diseñado con una arquitectura modular que permite flexibilidad, mantenibilidad y escalabilidad.

## 📊 Diagrama de Flujo de Datos

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   API Request   │───▶│  Web Dashboard   │───▶│   Advanced 1X2      │
│  (Team IDs)     │    │      API         │    │      System         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
                                                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   JSON Response │◀───│  Format & Return │◀───│   Enhanced Match    │
│  (Probabilities)│    │   Probabilities  │    │    Winner System    │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                           │
                       ┌──────────────────┐                │
                       │  Probability     │◀───────────────┘
                       │  Conversion      │
                       │   (% ↔ decimal)  │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Dynamic xG     │───▶│   Base Match     │───▶│   Draw Enhancement  │
│  Calculator     │    │  Winner System   │    │      System         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
         │                       │                         │
         ▼                       ▼                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│  Team Form      │    │  ML Prediction   │    │  Draw Probability   │
│  H2H Analysis   │    │  (Percentages)   │    │  Enhancement        │
│  League Data    │    │                  │    │  (Decimals)         │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
```

## 🧩 Componentes Principales

### 1. **Capa de API** (`web_dashboard_api.py`)

**Responsabilidades:**
- Manejar requests HTTP
- Validar parámetros de entrada
- Formatear respuestas JSON
- Gestionar errores y excepciones

**Endpoints principales:**
```python
POST /api/predict              # Predicción individual
POST /api/predict/formatted    # Predicción con formato hermoso
POST /api/batch_predict        # Predicciones por lotes
GET  /api/system_status        # Estado del sistema
GET  /api/performance          # Métricas de rendimiento
```

**Flujo de datos:**
```
HTTP Request → Validación → Sistema Enhanced → Formateo → HTTP Response
```

### 2. **Sistema Orquestador** (`advanced_1x2_system.py`)

**Responsabilidades:**
- Coordinar entre diferentes sistemas de predicción
- Manejar la lógica de negocio de alto nivel
- Gestionar el flujo de datos entre componentes

**Características:**
- Punto de entrada unificado para predicciones
- Manejo de diferentes tipos de predicción
- Integración con sistemas de monitoreo

### 3. **Sistema Enhanced** (`enhanced_match_winner.py`)

**Responsabilidades:**
- Integrar sistema base con sistema de mejora de empates
- Manejar conversiones de formato de probabilidad
- Gestionar cálculo dinámico de xG
- Aplicar lógica de negocio específica

**Arquitectura interna:**
```python
def predict_with_enhanced_system(home_team_id, away_team_id, league_id, **kwargs):
    # 1. Obtener datos base (forma, H2H)
    # 2. Calcular xG dinámico si no se proporciona
    # 3. Ejecutar predicción base
    # 4. Convertir formato (% → decimal)
    # 5. Aplicar mejora de empates
    # 6. Convertir formato (decimal → %)
    # 7. Retornar resultado normalizado
```

### 4. **Sistema Base** (`match_winner.py`)

**Responsabilidades:**
- Predicción core usando machine learning
- Retornar probabilidades como porcentajes (1-100)
- Manejar características base del partido

**Entrada:**
```python
{
    'home_team_id': int,
    'away_team_id': int,
    'league_id': int,
    'home_xg': float,      # Goles esperados local
    'away_xg': float,      # Goles esperados visitante
    'home_form': dict,     # Forma reciente local
    'away_form': dict,     # Forma reciente visitante
    'h2h': dict          # Head-to-head
}
```

**Salida:**
```python
{
    'probabilities': {
        'home_win': 45.2,    # Porcentaje
        'draw': 28.1,        # Porcentaje
        'away_win': 26.7     # Porcentaje
    }
}
```

### 5. **Sistema de Mejora de Empates** (`draw_prediction.py`)

**Responsabilidades:**
- Mejorar precisión de predicciones de empate
- Esperar probabilidades como decimales (0-1)
- Aplicar algoritmos especializados en empates

**Entrada:**
```python
{
    'probabilities': {
        'home_win': 0.452,   # Decimal
        'draw': 0.281,       # Decimal
        'away_win': 0.267    # Decimal
    }
}
```

**Salida:**
```python
{
    'probabilities': {
        'home_win': 0.434,   # Decimal (ajustado)
        'draw': 0.315,       # Decimal (mejorado)
        'away_win': 0.251    # Decimal (ajustado)
    }
}
```

### 6. **Calculador Dinámico de xG** (`dynamic_xg_calculator.py`)

**Responsabilidades:**
- Calcular goles esperados específicos por equipo
- Analizar forma reciente de equipos
- Considerar factores contextuales (H2H, liga, ventaja local)

**Algoritmo:**
```python
def calculate_match_xg(home_team_id, away_team_id, home_form, away_form, league_id, h2h):
    # 1. Analizar rendimiento ofensivo reciente
    # 2. Analizar rendimiento defensivo reciente
    # 3. Aplicar factor de ventaja local
    # 4. Considerar head-to-head histórico
    # 5. Ajustar por nivel de liga
    # 6. Retornar xG específico para cada equipo
    return home_xg, away_xg
```

## 🔄 Sistema de Conversión de Probabilidades

### Problema Resuelto
El sistema base retorna probabilidades como **porcentajes** (45.2%), pero el sistema de mejora de empates espera **decimales** (0.452).

### Solución Implementada
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

## 🏢 Arquitectura de Datos

### Fuentes de Datos
1. **API Externa**: Datos de equipos, ligas, partidos
2. **Cache Local**: Datos frecuentemente accedidos
3. **Base de Datos**: Predicciones históricas, métricas
4. **Modelos ML**: Modelos entrenados para predicción

### Flujo de Datos
```
API Externa → Cache → Sistema Enhanced → Base de Datos
     ↓              ↓           ↓             ↓
Datos Frescos  Datos Rápidos  Predicción  Histórico
```

## 🔧 Patrones de Diseño Utilizados

### 1. **Strategy Pattern**
- Diferentes algoritmos de predicción (base, enhanced, draw)
- Intercambiables según el contexto

### 2. **Adapter Pattern**
- Conversión entre formatos de probabilidad
- Adaptación entre sistemas con diferentes interfaces

### 3. **Facade Pattern**
- API simplificada para acceso a sistema complejo
- Ocultación de complejidad interna

### 4. **Factory Pattern**
- Creación de predictores según tipo de partido
- Configuración dinámica de componentes

## 📊 Gestión de Estado

### Estados del Sistema
```python
SYSTEM_STATES = {
    'INITIALIZING': 'Sistema iniciando',
    'READY': 'Sistema listo para predicciones',
    'PROCESSING': 'Procesando predicción',
    'ERROR': 'Error en el sistema',
    'MAINTENANCE': 'Modo mantenimiento'
}
```

### Gestión de Errores
```python
try:
    prediction = predict_with_enhanced_system(...)
except APIError as e:
    # Manejar errores de API externa
except ModelError as e:
    # Manejar errores de modelo ML
except ValidationError as e:
    # Manejar errores de validación
```

## 🚀 Escalabilidad y Rendimiento

### Optimizaciones Implementadas
1. **Cache**: Datos frecuentemente accedidos
2. **Lazy Loading**: Carga de modelos bajo demanda
3. **Connection Pooling**: Gestión eficiente de conexiones
4. **Batch Processing**: Predicciones por lotes

### Métricas de Rendimiento
- **Tiempo de respuesta**: ~20 segundos (con llamadas API externas)
- **Throughput**: Configurable según recursos
- **Memory usage**: Optimizado para modelos ML

## 🔒 Seguridad y Confiabilidad

### Validaciones
- Validación de parámetros de entrada
- Sanitización de datos
- Verificación de rangos de probabilidad

### Monitoreo
- Logging detallado de operaciones
- Métricas de rendimiento en tiempo real
- Alertas automáticas en caso de errores

### Recuperación de Errores
- Reintentos automáticos para APIs externas
- Fallback a valores por defecto
- Graceful degradation en caso de fallas

---

**Versión**: 2.0  
**Fecha**: 30 de Mayo, 2025  
**Estado**: Producción
