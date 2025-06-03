# Sistema de Predicción de Fútbol - Workflows

## 1. Formato de Respuesta JSON Core

La respuesta JSON del sistema sigue una estructura estandarizada:

```json
{
    "fixture_id": number,
    "home_team_id": number,
    "away_team_id": number,
    "prediction": {
        "home_win_probability": number,
        "draw_probability": number,
        "away_win_probability": number,
        "predicted_home_goals": number,
        "predicted_away_goals": number,
        "confidence_score": number
    },
    "elo_ratings": {
        "home_elo": number,
        "away_elo": number,
        "elo_diff": number,
        "strength_comparison": string,
        "win_probabilities": {
            "home": number,
            "draw": number,
            "away": number
        },
        "expected_goal_diff": number
    },
    "tactical_analysis": {
        "tactical_style": {
            "home": {
                "possession_style": string,
                "defensive_line": string
            },
            "away": {
                "possession_style": string,
                "defensive_line": string
            }
        },
        "key_battles": [string],
        "strengths": {
            "home": [string],
            "away": [string]
        },
        "weaknesses": {
            "home": [string],
            "away": [string]
        },
        "tactical_recommendation": string,
        "expected_formations": {
            "home": string,
            "away": string
        }
    },
    "odds_analysis": {
        "market_analysis": {
            "efficiency": number,
            "margin": number
        },
        "value_opportunities": [
            {
                "market": string,
                "selection": string,
                "fair_odds": number,
                "market_odds": number,
                "value": number,
                "recommendation": string
            }
        ],
        "market_sentiment": {
            "description": string,
            "implied_probabilities": {
                "home_win": number,
                "draw": number,
                "away_win": number
            }
        }
    }
}
```

## 2. Integración de ELO

### Flujo de Trabajo ELO
1. Obtención de Ratings
   - Consulta a base de datos SQLite para ratings actuales
   - Uso de pool de conexiones para optimización
   - Sistema de respaldo en caso de fallo de DB

2. Cálculo de Probabilidades
   - Basado en diferencia de ELO entre equipos
   - Categorías de comparación de fuerza:
     * Muy Parejos: |elo_diff| < 25
     * Ligera Ventaja: 25-75 puntos
     * Clara Ventaja: 75-150 puntos
     * Ventaja Significativa: >150 puntos

3. Actualización Post-Partido
   - Actualización automática de ratings
   - Migración de datos históricos JSON a SQLite
   - Validación y corrección de errores

## 3. Proxy Unificado (Puerto 8080)

### Endpoints
- Original: `http://localhost:5000/api/upcoming_predictions`
- Proxy Completo: `http://localhost:8080/api/upcoming_predictions`
- Endpoint Dedicado: `http://localhost:8080/api/fixed_predictions`

### Flujo de Enriquecimiento de Datos
1. Recepción de Predicción Base
2. Procesamiento de Análisis Táctico:
   - Obtención de datos tácticos detallados
   - Generación de análisis predeterminado si es necesario
   - Validación de campos requeridos
   - Complementación de datos faltantes

3. Procesamiento de Análisis de Cuotas:
   - Obtención de datos reales del mercado
   - Cálculo de eficiencia y márgenes
   - Identificación de oportunidades de valor
   - Generación de datos simulados si es necesario

4. Validación y Control de Errores:
   - Logging detallado de operaciones
   - Manejo de errores por componente
   - Respuestas consistentes incluso en caso de error

## 4. Enriquecimiento de Datos

### Proceso de Análisis Táctico
1. Extracción de datos tácticos por equipo
2. Análisis de estilos de juego
3. Identificación de fortalezas/debilidades
4. Generación de recomendaciones tácticas
5. Validación de completitud de datos

### Proceso de Análisis de Cuotas
1. Obtención de cuotas de mercado
2. Cálculo de probabilidades implícitas
3. Análisis de eficiencia de mercado
4. Identificación de valor
5. Generación de recomendaciones

## 5. Manejo de Errores

### Niveles de Validación
1. Verificación de disponibilidad de servidores
2. Validación de estructura de datos
3. Control de calidad de predicciones
4. Monitoreo de rendimiento

### Respuesta a Errores
1. Uso de datos de respaldo
2. Logging detallado
3. Notificación de errores críticos
4. Mantenimiento de estructura JSON

## 6. Optimizaciones

### Caché y Rendimiento
1. Sistema de caché para ratings frecuentes
2. Optimización de consultas a base de datos
3. Procesamiento asíncrono cuando es posible
4. Monitoreo de tiempos de respuesta

### Precisión y Calidad
1. Validación continua contra resultados
2. Ajuste automático de parámetros
3. Tests unitarios automatizados
4. Monitoreo de métricas de precisión

## 7. Workflow de Predicción de Corners

### Recolección de Datos
1. Extracción de datos históricos por liga
   - Premier League (ID: 39)
   - Otras ligas principales
2. Almacenamiento en formato JSON
   - Nombre de archivo: `corner_data_{league_id}_{season}_{timestamp}.json`
   - Logging detallado de la recolección

### Variables de Predicción
1. Factores Contextuales:
   - Horario del partido (día/noche)
   - Día de la semana
   - Importancia del partido
   - Factores de derby
   
2. Factores Ambientales:
   - Temperatura
   - Condiciones climáticas
   - Velocidad del viento
   - Precipitación
   - Humedad

3. Factores de Equipo:
   - Estadísticas de corners previos
   - Ratio de asistencia al estadio
   - Distancia de viaje (equipo visitante)
   - Días de descanso entre partidos
   - Diferencial de descanso entre equipos

### Proceso de Entrenamiento
1. Preprocesamiento de Datos:
   - Validación de datos de entrada
   - Extracción de features relevantes
   - Separación de conjuntos de entrenamiento/prueba

2. Características del Modelo:
   - Variable objetivo: `total_corners`
   - Features excluidos: `home_corners`, `away_corners`
   - Validación cruzada
   - Manejo de errores y logging

### Ejecución del Workflow
1. Recolección de Datos:
   ```powershell
   # Datos reales de API-Football
   python corner_data_collector.py --days [días]
   
   # Datos simulados para testing
   python generate_sample_corners_data.py --samples 1000
   ```

2. Entrenamiento de Modelos:
   - Random Forest
   - XGBoost
   - Ensamble voting

3. Evaluación de Modelos:
   - Evaluador fijo para consistencia
   - Evaluador estándar para comparación
   - Métricas de rendimiento detalladas

### Integración de Datos
1. Enriquecimiento Contextual:
   - Datos de asistencia y capacidad del estadio
   - Factores de viaje y descanso
   - Condiciones ambientales
   - Importancia del partido

2. Normalización de Features:
   - Distancia de viaje normalizada (0-1)
   - Ratio de asistencia
   - Factores temporales
   - Impacto climático compuesto
