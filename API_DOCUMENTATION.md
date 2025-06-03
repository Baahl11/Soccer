# Documentación de la API del Sistema de Predicciones

## 🌐 API REST - Guía Completa

La API del Sistema de Predicciones Mejorado proporciona acceso programático a todas las funcionalidades del sistema a través de endpoints REST.

## 🚀 Información General

### URL Base
```
http://localhost:5000
```

### Formato de Respuesta
Todas las respuestas están en formato JSON con encoding UTF-8.

### Códigos de Estado HTTP
- `200 OK`: Operación exitosa
- `400 Bad Request`: Error en parámetros de entrada
- `500 Internal Server Error`: Error interno del servidor

## 📋 Endpoints Disponibles

### 1. **Predicción Individual**

#### `POST /api/predict`

Genera una predicción para un partido específico.

**Parámetros del Body (JSON):**
```json
{
    "home_team_id": 33,        // ID del equipo local (requerido)
    "away_team_id": 40,        // ID del equipo visitante (requerido)
    "league_id": 39,           // ID de la liga (requerido)
    "home_xg": 1.5,           // xG del equipo local (opcional)
    "away_xg": 1.2            // xG del equipo visitante (opcional)
}
```

**Ejemplo de Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "home_team_id": 33,
    "away_team_id": 40,
    "league_id": 39
  }'
```

**Ejemplo de Response:**
```json
{
  "success": true,
  "prediction": {
    "probabilities": {
      "home_win": 34.0,
      "draw": 36.6,
      "away_win": 29.4
    },
    "predicted_outcome": "Draw",
    "confidence": 36.6,
    "match_details": {
      "home_team_id": 33,
      "away_team_id": 40,
      "league_id": 39,
      "prediction_time": "2025-05-30T19:20:18.081000"
    },
    "system_info": {
      "enhanced_system": true,
      "dynamic_xg_used": true,
      "probability_conversion": "applied"
    }
  }
}
```

### 2. **Predicción con Formato Hermoso**

#### `POST /api/predict/formatted`

Genera una predicción con formato JSON hermoso y emojis.

**Parámetros:** Mismos que `/api/predict`

**Ejemplo de Response:**
```json
{
  "🏆 SOCCER MATCH PREDICTION": {
    "🎯 Prediction Result": {
      "🏅 Predicted Outcome": "Draw",
      "💡 Summary": "The model predicts a draw with 36.0% confidence.",
      "📊 Confidence Level": "36.0% (High)"
    },
    "📅 Match Details": {
      "⏰ Prediction Time": "2025-05-30T19:31:52.503316",
      "🏟️ League ID": 39,
      "🏠 Home Team": "Team 33",
      "🛣️ Away Team": "Team 34"
    },
    "📈 Probability Breakdown": {
      "🏠 Team 33 Win": "31.9%",
      "🛣️ Team 34 Win": "32.1%",
      "🤝 Draw": "36.0%"
    },
    "🔬 Advanced Metrics": {
      "🌀 Entropy (Uncertainty)": "-350.814 (lower = more certain)",
      "📊 Probability Spread": "4.1",
      "🤝 Draw Favorability": "1.0"
    },
    "🖥️ System Information": {
      "⚖️ SMOTE Balancing": "✅ Applied",
      "⚙️ Enhanced System": "✅ Active",
      "🎯 Calibration": "✅ Enabled",
      "🤖 AI Version": "2.0"
    },
    "🧠 AI Analysis": {
      "⚖️ Probability Analysis": "Very evenly matched teams with similar winning chances.",
      "💯 Confidence Explanation": "Low confidence - Close match with uncertain outcome.",
      "💼 Recommendation": "Uncertain outcome - avoid high-risk decisions."
    }
  }
}
```

### 3. **Predicciones por Lotes**

#### `POST /api/batch_predict`

Genera predicciones para múltiples partidos en una sola request.

**Parámetros del Body (JSON):**
```json
{
  "matches": [
    {
      "home_team_id": 33,
      "away_team_id": 40,
      "league_id": 39
    },
    {
      "home_team_id": 541,
      "away_team_id": 529,
      "league_id": 140
    }
  ]
}
```

**Ejemplo de Response:**
```json
{
  "success": true,
  "predictions": [
    {
      "match_index": 0,
      "probabilities": {
        "home_win": 34.0,
        "draw": 36.6,
        "away_win": 29.4
      },
      "predicted_outcome": "Draw",
      "confidence": 36.6
    },
    {
      "match_index": 1,
      "probabilities": {
        "home_win": 45.0,
        "draw": 27.2,
        "away_win": 27.8
      },
      "predicted_outcome": "Home Win",
      "confidence": 45.0
    }
  ],
  "summary": {
    "total_matches": 2,
    "processing_time": "41.2 seconds",
    "average_confidence": 40.8
  }
}
```

### 4. **Estado del Sistema**

#### `GET /api/system_status`

Obtiene información sobre el estado actual del sistema.

**Ejemplo de Response:**
```json
{
  "success": true,
  "status": {
    "system_state": "READY",
    "enhanced_system": "✅ Active",
    "dynamic_xg_calculator": "✅ Operational",
    "probability_converter": "✅ Operational",
    "base_predictor": "✅ Loaded",
    "draw_enhancer": "✅ Loaded",
    "api_connections": "✅ Healthy",
    "last_health_check": "2025-05-30T19:35:00.000000",
    "uptime": "2 hours 15 minutes",
    "version": "2.0"
  }
}
```

### 5. **Métricas de Rendimiento**

#### `GET /api/performance`

Obtiene métricas de rendimiento del sistema.

**Ejemplo de Response:**
```json
{
  "success": true,
  "performance": {
    "recent_predictions": {
      "total": 157,
      "last_24h": 23,
      "average_response_time": "19.8 seconds",
      "success_rate": "99.4%"
    },
    "system_resources": {
      "cpu_usage": "12%",
      "memory_usage": "2.1 GB",
      "disk_usage": "45%"
    },
    "prediction_accuracy": {
      "last_100_predictions": "78.2%",
      "home_win_accuracy": "82.1%",
      "draw_accuracy": "71.5%",
      "away_win_accuracy": "80.8%"
    },
    "dynamic_xg_stats": {
      "calculations_performed": 1247,
      "average_home_xg": 1.65,
      "average_away_xg": 1.32,
      "xg_variation_range": "0.8 - 2.73"
    }
  }
}
```

### 6. **Predicciones Recientes**

#### `GET /api/recent_predictions`

Obtiene las últimas predicciones realizadas.

**Parámetros de Query (opcionales):**
- `limit`: Número máximo de predicciones (default: 10)
- `league_id`: Filtrar por liga específica

**Ejemplo de Request:**
```bash
curl "http://localhost:5000/api/recent_predictions?limit=5&league_id=39"
```

**Ejemplo de Response:**
```json
{
  "success": true,
  "recent_predictions": [
    {
      "prediction_id": "pred_20250530_193152",
      "timestamp": "2025-05-30T19:31:52.503316",
      "home_team_id": 33,
      "away_team_id": 34,
      "league_id": 39,
      "probabilities": {
        "home_win": 31.9,
        "draw": 36.0,
        "away_win": 32.1
      },
      "predicted_outcome": "Draw",
      "confidence": 36.0
    }
  ],
  "summary": {
    "total_returned": 1,
    "period": "last 24 hours",
    "most_predicted_league": 39,
    "most_common_outcome": "Draw"
  }
}
```

## 🔧 Parámetros Avanzados

### Parámetros Opcionales para Predicciones

#### xG Personalizados
```json
{
  "home_xg": 1.8,    // Goles esperados equipo local
  "away_xg": 1.4     // Goles esperados equipo visitante
}
```

Si no se proporcionan, el sistema calculará automáticamente valores dinámicos basados en:
- Forma reciente de los equipos
- Análisis head-to-head
- Nivel de la liga
- Ventaja del equipo local

#### Configuración del Sistema
```json
{
  "use_enhanced": true,      // Usar sistema mejorado (default: true)
  "apply_draw_boost": true,  // Aplicar mejora de empates (default: true)
  "dynamic_xg": true        // Usar cálculo dinámico de xG (default: true)
}
```

## 🎯 Casos de Uso Comunes

### 1. **Predicción Rápida**
```javascript
// Predicción básica para un partido
const response = await fetch('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    home_team_id: 33,
    away_team_id: 40,
    league_id: 39
  })
});
const prediction = await response.json();
```

### 2. **Análisis de Liga Completa**
```javascript
// Predicciones para múltiples partidos de una jornada
const matches = [
  { home_team_id: 33, away_team_id: 40, league_id: 39 },
  { home_team_id: 47, away_team_id: 35, league_id: 39 },
  // ... más partidos
];

const response = await fetch('/api/batch_predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ matches })
});
```

### 3. **Monitoreo del Sistema**
```javascript
// Verificar estado del sistema antes de hacer predicciones
const status = await fetch('/api/system_status');
const statusData = await status.json();

if (statusData.status.system_state === 'READY') {
  // Proceder con predicciones
}
```

## ❌ Manejo de Errores

### Errores Comunes

#### 400 Bad Request
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Missing required parameter: home_team_id",
    "details": {
      "required_fields": ["home_team_id", "away_team_id", "league_id"],
      "provided_fields": ["away_team_id", "league_id"]
    }
  }
}
```

#### 500 Internal Server Error
```json
{
  "success": false,
  "error": {
    "code": "PREDICTION_ERROR",
    "message": "Unable to generate prediction",
    "details": {
      "component": "enhanced_match_winner",
      "reason": "API timeout",
      "suggestion": "Retry with default xG values"
    }
  }
}
```

### Reintentos Recomendados

1. **Error temporal**: Reintentar después de 5 segundos
2. **Error de timeout**: Reintentar con parámetros simplificados
3. **Error de validación**: Verificar parámetros y corregir

## 📊 Límites y Consideraciones

### Límites de Rate
- **Predicciones individuales**: 60 por minuto
- **Predicciones por lotes**: 10 por minuto (máximo 20 partidos por lote)
- **Consultas de estado**: 120 por minuto

### Tiempos de Respuesta Esperados
- **Predicción individual**: 15-25 segundos
- **Predicción por lotes**: 20-40 segundos por partido
- **Estado del sistema**: < 1 segundo
- **Métricas de rendimiento**: 1-3 segundos

### Consideraciones de Uso
1. **Cache**: Los resultados se cachean por 5 minutos para los mismos parámetros
2. **Horario**: Rendimiento óptimo fuera de horas pico de APIs externas
3. **Recursos**: El sistema puede manejar hasta 50 requests concurrentes

---

**Versión de API**: 2.0  
**Fecha**: 30 de Mayo, 2025  
**Documentación actualizada**: Completamente operacional
