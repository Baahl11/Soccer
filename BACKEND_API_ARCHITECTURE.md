# Backend API Architecture - Soccer Predictions Platform

## 🏗️ **ARQUITECTURA DEL BACKEND**

### **Stack Tecnológico:**
- **Framework**: FastAPI (Python)
- **Base de Datos**: PostgreSQL + Redis (Cache)
- **Scheduler**: Celery + Redis
- **Autenticación**: JWT + OAuth2
- **Pagos**: Stripe API
- **Deploy**: Docker + AWS/Railway

### **Estructura del Proyecto:**

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── matches.py
│   │   │   │   ├── predictions.py
│   │   │   │   ├── users.py
│   │   │   │   └── subscriptions.py
│   │   │   └── api.py
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── database.py
│   ├── models/
│   │   ├── user.py
│   │   ├── match.py
│   │   ├── prediction.py
│   │   └── subscription.py
│   ├── services/
│   │   ├── prediction_service.py
│   │   ├── match_service.py
│   │   ├── notification_service.py
│   │   └── subscription_service.py
│   ├── tasks/
│   │   ├── prediction_tasks.py
│   │   ├── data_collection_tasks.py
│   │   └── notification_tasks.py
│   └── utils/
│       ├── football_api.py
│       ├── cache.py
│       └── helpers.py
├── requirements.txt
├── docker-compose.yml
└── main.py
```

## 🔌 **ENDPOINTS PRINCIPALES**

### **1. Matches API:**

```python
# app/api/v1/endpoints/matches.py
from fastapi import APIRouter, Depends, Query
from typing import List, Optional
from datetime import datetime, timedelta
from app.services.match_service import MatchService
from app.models.user import User
from app.api.deps import get_current_user, check_subscription

router = APIRouter()

@router.get("/today", response_model=List[MatchResponse])
async def get_matches_today(
    leagues: Optional[str] = Query(None, description="Comma-separated league IDs"),
    confidence: Optional[str] = Query(None, description="high, medium, low"),
    time_range: Optional[str] = Query("24h", description="2h, 6h, 12h, 24h"),
    value_bets: Optional[bool] = Query(False),
    high_scoring: Optional[bool] = Query(False),
    big_matches: Optional[bool] = Query(False),
    current_user: User = Depends(get_current_user),
    subscription_check = Depends(check_subscription)
):
    """
    Obtener todos los partidos de las próximas 24 horas con predicciones
    """
    match_service = MatchService()
    
    filters = {
        "leagues": leagues.split(",") if leagues else None,
        "confidence": confidence,
        "time_range": time_range,
        "value_bets": value_bets,
        "high_scoring": high_scoring,
        "big_matches": big_matches
    }
    
    matches = await match_service.get_matches_with_predictions(
        filters=filters,
        user_subscription=current_user.subscription_tier
    )
    
    return matches

@router.get("/live", response_model=List[MatchResponse])
async def get_live_matches(
    current_user: User = Depends(get_current_user)
):
    """
    Obtener partidos que están en vivo ahora mismo
    """
    match_service = MatchService()
    return await match_service.get_live_matches()

@router.get("/{match_id}", response_model=MatchDetailResponse)
async def get_match_detail(
    match_id: int,
    current_user: User = Depends(get_current_user),
    subscription_check = Depends(check_subscription)
):
    """
    Obtener análisis detallado de un partido específico
    """
    match_service = MatchService()
    return await match_service.get_match_detail(
        match_id=match_id,
        user_tier=current_user.subscription_tier
    )

@router.get("/leagues/active", response_model=List[LeagueResponse])
async def get_active_leagues():
    """
    Obtener todas las ligas con partidos en las próximas 24h
    """
    match_service = MatchService()
    return await match_service.get_active_leagues()
```

### **2. Predictions API:**

```python
# app/api/v1/endpoints/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from app.services.prediction_service import PredictionService
from app.models.user import User
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/batch", response_model=List[PredictionResponse])
async def create_batch_predictions(
    match_ids: List[int],
    current_user: User = Depends(get_current_user)
):
    """
    Generar predicciones para múltiples partidos
    """
    if len(match_ids) > 50:  # Limitar batch size
        raise HTTPException(status_code=400, detail="Máximo 50 partidos por batch")
    
    prediction_service = PredictionService()
    return await prediction_service.generate_batch_predictions(match_ids)

@router.get("/value-bets", response_model=List[ValueBetResponse])
async def get_value_bets(
    min_value: float = Query(10.0, description="Minimum value percentage"),
    current_user: User = Depends(get_current_user)
):
    """
    Obtener todas las value bets disponibles
    """
    prediction_service = PredictionService()
    return await prediction_service.get_value_bets(min_value=min_value)

@router.get("/confidence-analysis", response_model=ConfidenceAnalysisResponse)
async def get_confidence_analysis(
    current_user: User = Depends(get_current_user)
):
    """
    Análisis de confianza del sistema para el día actual
    """
    prediction_service = PredictionService()
    return await prediction_service.get_confidence_analysis()
```

### **3. User Management API:**

```python
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException
from app.models.user import User, UserCreate, UserUpdate
from app.services.user_service import UserService
from app.api.deps import get_current_user

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """
    Registrar nuevo usuario
    """
    user_service = UserService()
    return await user_service.create_user(user_data)

@router.get("/profile", response_model=UserProfileResponse)
async def get_user_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Obtener perfil del usuario actual
    """
    user_service = UserService()
    return await user_service.get_user_profile(current_user.id)

@router.put("/profile", response_model=UserResponse)
async def update_user_profile(
    user_data: UserUpdate,
    current_user: User = Depends(get_current_user)
):
    """
    Actualizar perfil del usuario
    """
    user_service = UserService()
    return await user_service.update_user(current_user.id, user_data)

@router.get("/betting-history", response_model=List[BettingHistoryResponse])
async def get_betting_history(
    current_user: User = Depends(get_current_user)
):
    """
    Obtener historial de picks del usuario
    """
    user_service = UserService()
    return await user_service.get_betting_history(current_user.id)

@router.post("/betting-history", response_model=BettingHistoryResponse)
async def add_betting_record(
    bet_data: BettingRecordCreate,
    current_user: User = Depends(get_current_user)
):
    """
    Agregar nuevo registro de apuesta
    """
    user_service = UserService()
    return await user_service.add_betting_record(current_user.id, bet_data)
```

## 🔄 **SERVICIOS PRINCIPALES**

### **1. Match Service:**

```python
# app/services/match_service.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from app.utils.football_api import FootballAPIClient
from app.utils.cache import cache_manager
from app.models.match import Match
from .prediction_service import PredictionService

class MatchService:
    def __init__(self):
        self.football_api = FootballAPIClient()
        self.prediction_service = PredictionService()
        self.cache = cache_manager
    
    async def get_matches_with_predictions(
        self, 
        filters: Dict,
        user_subscription: str
    ) -> List[Dict]:
        """
        Obtener partidos con predicciones aplicando filtros
        """
        cache_key = f"matches_filtered_{hash(str(filters))}"
        
        # Intentar obtener del cache
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return self._filter_by_subscription(cached_result, user_subscription)
        
        # Obtener partidos de las próximas 24h
        matches = await self._get_upcoming_matches(filters.get("time_range", "24h"))
        
        # Generar predicciones para todos los partidos
        enriched_matches = []
        for match in matches:
            try:
                # Obtener predicción del sistema existente
                prediction = await self.prediction_service.get_match_prediction(
                    home_team_id=match["home_team"]["id"],
                    away_team_id=match["away_team"]["id"],
                    league_id=match["league"]["id"],
                    match_time=match["kickoff_time"]
                )
                
                # Enriquecer partido con predicción
                enriched_match = {
                    **match,
                    "predictions": prediction,
                    "confidence": self._calculate_confidence(prediction),
                    "value_bets": self._detect_value_bets(prediction, match.get("odds", {}))
                }
                
                enriched_matches.append(enriched_match)
                
            except Exception as e:
                print(f"Error generating prediction for match {match['id']}: {e}")
                continue
        
        # Aplicar filtros
        filtered_matches = self._apply_filters(enriched_matches, filters)
        
        # Cachear resultado por 30 minutos
        await self.cache.set(cache_key, filtered_matches, ttl=1800)
        
        # Filtrar por suscripción
        return self._filter_by_subscription(filtered_matches, user_subscription)
    
    async def _get_upcoming_matches(self, time_range: str) -> List[Dict]:
        """
        Obtener partidos de la API de fútbol
        """
        now = datetime.utcnow()
        
        # Calcular tiempo final basado en el rango
        time_deltas = {
            "2h": timedelta(hours=2),
            "6h": timedelta(hours=6),
            "12h": timedelta(hours=12),
            "24h": timedelta(hours=24)
        }
        
        end_time = now + time_deltas.get(time_range, timedelta(hours=24))
        
        # Obtener fixtures de la API
        fixtures = await self.football_api.get_fixtures(
            date_from=now.strftime("%Y-%m-%d"),
            date_to=end_time.strftime("%Y-%m-%d"),
            status="NS"  # Not Started
        )
        
        return fixtures
    
    def _apply_filters(self, matches: List[Dict], filters: Dict) -> List[Dict]:
        """
        Aplicar filtros a la lista de partidos
        """
        filtered = matches.copy()
        
        # Filtrar por ligas
        if filters.get("leagues"):
            league_ids = [int(lid) for lid in filters["leagues"]]
            filtered = [m for m in filtered if m["league"]["id"] in league_ids]
        
        # Filtrar por confianza
        if filters.get("confidence"):
            confidence_ranges = {
                "high": (80, 100),
                "medium": (60, 79),
                "low": (0, 59)
            }
            min_conf, max_conf = confidence_ranges[filters["confidence"]]
            filtered = [m for m in filtered 
                       if min_conf <= m["confidence"]["overall"] <= max_conf]
        
        # Filtrar solo value bets
        if filters.get("value_bets"):
            filtered = [m for m in filtered if m.get("value_bets") and len(m["value_bets"]) > 0]
        
        # Filtrar partidos con muchos goles
        if filters.get("high_scoring"):
            filtered = [m for m in filtered 
                       if m["predictions"].get("goals", {}).get("total", 0) >= 2.5]
        
        # Filtrar partidos grandes (top teams)
        if filters.get("big_matches"):
            top_leagues = [39, 140, 135, 78, 61]  # Premier, La Liga, Serie A, Bundesliga, Ligue 1
            filtered = [m for m in filtered if m["league"]["id"] in top_leagues]
        
        return filtered
    
    def _calculate_confidence(self, prediction: Dict) -> Dict:
        """
        Calcular métricas de confianza de la predicción
        """
        # Obtener probabilidades principales
        match_probs = prediction.get("match_result", {})
        home_prob = match_probs.get("home_win", 0)
        draw_prob = match_probs.get("draw", 0)
        away_prob = match_probs.get("away_win", 0)
        
        # Calcular confianza basada en la distribución de probabilidades
        max_prob = max(home_prob, draw_prob, away_prob)
        prob_spread = max_prob - min(home_prob, draw_prob, away_prob)
        
        # Confianza general (0-100)
        overall_confidence = min(100, (max_prob * 0.7 + prob_spread * 0.3))
        
        return {
            "overall": round(overall_confidence, 1),
            "match_result": round(max_prob, 1),
            "goals": round(prediction.get("goals", {}).get("confidence", 70), 1),
            "corners": round(prediction.get("corners", {}).get("confidence", 70), 1)
        }
    
    def _detect_value_bets(self, prediction: Dict, odds: Dict) -> List[Dict]:
        """
        Detectar value bets comparando predicciones vs odds
        """
        value_bets = []
        
        if not odds:
            return value_bets
        
        # Verificar 1X2
        match_odds = odds.get("match_result", {})
        match_pred = prediction.get("match_result", {})
        
        for outcome in ["home_win", "draw", "away_win"]:
            if outcome in match_odds and outcome in match_pred:
                bookmaker_prob = 1 / match_odds[outcome] if match_odds[outcome] > 0 else 0
                our_prob = match_pred[outcome] / 100
                
                if our_prob > bookmaker_prob * 1.1:  # 10% margen mínimo
                    value_percentage = ((our_prob / bookmaker_prob) - 1) * 100
                    value_bets.append({
                        "market": "1X2",
                        "selection": outcome,
                        "our_probability": round(our_prob * 100, 1),
                        "bookmaker_probability": round(bookmaker_prob * 100, 1),
                        "value_percentage": round(value_percentage, 1),
                        "odds": match_odds[outcome],
                        "recommendation": "STRONG" if value_percentage > 20 else "MODERATE"
                    })
        
        return value_bets
    
    def _filter_by_subscription(self, matches: List[Dict], subscription_tier: str) -> List[Dict]:
        """
        Filtrar contenido basado en el nivel de suscripción
        """
        if subscription_tier == "basic":
            # Solo top 5 ligas, máximo 20 partidos
            top_leagues = [39, 140, 135, 78, 61]
            filtered = [m for m in matches if m["league"]["id"] in top_leagues]
            return filtered[:20]
        
        elif subscription_tier == "pro":
            # Todas las ligas, pero sin análisis avanzado
            for match in matches:
                # Remover análisis avanzado
                if "advanced_analysis" in match:
                    del match["advanced_analysis"]
            return matches
        
        elif subscription_tier in ["premium", "vip"]:
            # Todo el contenido
            return matches
        
        else:
            # Usuario no suscrito - solo 5 partidos
            return matches[:5]

    async def get_match_detail(self, match_id: int, user_tier: str) -> Dict:
        """
        Obtener análisis detallado de un partido
        """
        # Esta función se integraría con tu sistema existente
        # para obtener análisis más profundo usando todos los modelos
        pass
```

### **2. Prediction Service:**

```python
# app/services/prediction_service.py
import sys
import os
from typing import Dict, List
from datetime import datetime

# Agregar el directorio del proyecto al path para importar módulos existentes
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importar tu sistema existente
from enhanced_match_winner import EnhancedPredictionSystem
from dynamic_xg_calculator import DynamicXGCalculator
from web_dashboard_api import AdvancedPredictionSystem1X2

class PredictionService:
    def __init__(self):
        # Inicializar tus sistemas existentes
        self.enhanced_system = EnhancedPredictionSystem()
        self.xg_calculator = DynamicXGCalculator()
        self.advanced_1x2 = AdvancedPredictionSystem1X2()
    
    async def get_match_prediction(
        self, 
        home_team_id: int, 
        away_team_id: int, 
        league_id: int,
        match_time: datetime = None
    ) -> Dict:
        """
        Generar predicción completa para un partido usando tu sistema existente
        """
        try:
            # Usar tu sistema de predicciones mejorado
            prediction = self.enhanced_system.predict_match_outcome(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id
            )
            
            # Calcular xG dinámico
            xg_data = self.xg_calculator.calculate_dynamic_xg(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id
            )
            
            # Obtener predicción avanzada 1X2
            advanced_prediction = self.advanced_1x2.make_prediction(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id
            )
            
            # Combinar todas las predicciones
            combined_prediction = {
                "match_result": {
                    "home_win": prediction.get("home_win_probability", 33.3),
                    "draw": prediction.get("draw_probability", 33.3),
                    "away_win": prediction.get("away_win_probability", 33.3)
                },
                "goals": {
                    "total": xg_data.get("total_xg", 2.5),
                    "home": xg_data.get("home_xg", 1.25),
                    "away": xg_data.get("away_xg", 1.25),
                    "over_2_5": self._calculate_over_probability(xg_data.get("total_xg", 2.5), 2.5),
                    "over_1_5": self._calculate_over_probability(xg_data.get("total_xg", 2.5), 1.5),
                    "under_2_5": 100 - self._calculate_over_probability(xg_data.get("total_xg", 2.5), 2.5)
                },
                "corners": {
                    "total": prediction.get("total_corners", 9.5),
                    "home": prediction.get("home_corners", 5.0),
                    "away": prediction.get("away_corners", 4.5),
                    "over_9_5": prediction.get("over_9_5_corners", 50),
                    "over_8_5": prediction.get("over_8_5_corners", 60)
                },
                "both_teams_score": {
                    "yes": self._calculate_btts_probability(xg_data),
                    "no": 100 - self._calculate_btts_probability(xg_data)
                },
                "metadata": {
                    "prediction_time": datetime.utcnow().isoformat(),
                    "model_version": "enhanced_v2.0",
                    "confidence_score": advanced_prediction.get("confidence", 75)
                }
            }
            
            return combined_prediction
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
            # Fallback a predicción básica
            return self._generate_fallback_prediction()
    
    def _calculate_over_probability(self, expected_goals: float, threshold: float) -> float:
        """
        Calcular probabilidad de over usando distribución de Poisson
        """
        import math
        
        # Aproximación simple usando distribución de Poisson
        # P(X > threshold) = 1 - P(X <= threshold)
        prob_under = 0
        for k in range(int(threshold) + 1):
            prob_under += (expected_goals ** k) * math.exp(-expected_goals) / math.factorial(k)
        
        prob_over = (1 - prob_under) * 100
        return round(prob_over, 1)
    
    def _calculate_btts_probability(self, xg_data: Dict) -> float:
        """
        Calcular probabilidad de que ambos equipos anoten
        """
        home_xg = xg_data.get("home_xg", 1.25)
        away_xg = xg_data.get("away_xg", 1.25)
        
        # Probabilidad de que cada equipo anote al menos 1 gol
        # P(X >= 1) = 1 - P(X = 0) = 1 - e^(-λ)
        import math
        
        prob_home_scores = 1 - math.exp(-home_xg)
        prob_away_scores = 1 - math.exp(-away_xg)
        
        # Probabilidad de que ambos anoten (asumiendo independencia)
        prob_btts = prob_home_scores * prob_away_scores * 100
        
        return round(prob_btts, 1)
    
    def _generate_fallback_prediction(self) -> Dict:
        """
        Predicción básica de fallback si falla el sistema principal
        """
        return {
            "match_result": {"home_win": 40, "draw": 30, "away_win": 30},
            "goals": {"total": 2.5, "home": 1.3, "away": 1.2, "over_2_5": 50, "under_2_5": 50},
            "corners": {"total": 9.5, "home": 5, "away": 4.5, "over_9_5": 50},
            "both_teams_score": {"yes": 55, "no": 45},
            "metadata": {"prediction_time": datetime.utcnow().isoformat(), "model_version": "fallback"}
        }
    
    async def generate_batch_predictions(self, match_ids: List[int]) -> List[Dict]:
        """
        Generar predicciones para múltiples partidos
        """
        predictions = []
        
        for match_id in match_ids:
            try:
                # Aquí necesitarías obtener los datos del partido (home_team_id, away_team_id, league_id)
                # desde tu base de datos o API
                
                # Por ahora, ejemplo básico:
                prediction = await self.get_match_prediction(
                    home_team_id=1,  # Obtener del match_id
                    away_team_id=2,  # Obtener del match_id
                    league_id=39     # Obtener del match_id
                )
                
                predictions.append({
                    "match_id": match_id,
                    "prediction": prediction
                })
                
            except Exception as e:
                print(f"Error predicting match {match_id}: {e}")
                continue
        
        return predictions
    
    async def get_value_bets(self, min_value: float = 10.0) -> List[Dict]:
        """
        Obtener todas las value bets del día
        """
        # Esta función buscaría en tu base de datos todos los partidos
        # que tengan value bets con el porcentaje mínimo especificado
        
        # Implementación ejemplo:
        value_bets = []
        
        # Aquí integrarías con tu sistema para obtener todos los partidos del día
        # y filtrar solo aquellos con value bets
        
        return value_bets
    
    async def get_confidence_analysis(self) -> Dict:
        """
        Análisis de confianza del sistema para el día actual
        """
        return {
            "overall_confidence": 78.5,
            "high_confidence_matches": 12,
            "medium_confidence_matches": 25,
            "low_confidence_matches": 8,
            "model_performance": {
                "accuracy_last_7_days": 72.3,
                "roi_last_30_days": 8.4,
                "total_predictions_today": 45
            }
        }
```

## ⚡ **TAREAS AUTOMÁTICAS (Celery)**

```python
# app/tasks/prediction_tasks.py
from celery import Celery
from app.services.prediction_service import PredictionService
from app.services.match_service import MatchService
from app.utils.cache import cache_manager

celery_app = Celery("prediction_tasks")

@celery_app.task
def update_daily_predictions():
    """
    Tarea que se ejecuta cada 30 minutos para actualizar predicciones
    """
    print("🔄 Actualizando predicciones del día...")
    
    try:
        match_service = MatchService()
        prediction_service = PredictionService()
        
        # Obtener todos los partidos de las próximas 24h
        matches = match_service._get_upcoming_matches("24h")
        
        predictions_updated = 0
        for match in matches:
            try:
                # Generar nueva predicción
                prediction = prediction_service.get_match_prediction(
                    home_team_id=match["home_team"]["id"],
                    away_team_id=match["away_team"]["id"],
                    league_id=match["league"]["id"]
                )
                
                # Guardar en cache
                cache_key = f"prediction_{match['id']}"
                cache_manager.set(cache_key, prediction, ttl=1800)  # 30 minutos
                
                predictions_updated += 1
                
            except Exception as e:
                print(f"Error updating prediction for match {match['id']}: {e}")
                continue
        
        print(f"✅ Predicciones actualizadas: {predictions_updated}/{len(matches)}")
        
        # Limpiar cache antiguo
        cache_manager.clear_expired()
        
    except Exception as e:
        print(f"❌ Error en actualización masiva: {e}")

@celery_app.task
def send_value_bet_alerts():
    """
    Enviar alertas de value bets a usuarios suscritos
    """
    print("🎯 Enviando alertas de value bets...")
    
    try:
        prediction_service = PredictionService()
        
        # Obtener value bets con valor > 15%
        value_bets = prediction_service.get_value_bets(min_value=15.0)
        
        if value_bets:
            # Aquí integrarías con tu sistema de notificaciones
            # para enviar emails/push notifications a usuarios
            print(f"📧 {len(value_bets)} value bets encontradas, enviando alertas...")
        
    except Exception as e:
        print(f"❌ Error enviando alertas: {e}")

# Configurar tareas periódicas
from celery.schedules import crontab

celery_app.conf.beat_schedule = {
    'update-predictions-every-30-minutes': {
        'task': 'app.tasks.prediction_tasks.update_daily_predictions',
        'schedule': crontab(minute='*/30'),  # Cada 30 minutos
    },
    'send-value-bet-alerts-every-hour': {
        'task': 'app.tasks.prediction_tasks.send_value_bet_alerts',
        'schedule': crontab(minute=0),  # Cada hora
    },
}
```

## 🗄️ **MODELOS DE BASE DE DATOS**

```python
# app/models/match.py
from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import json

Base = declarative_base()

class Match(Base):
    __tablename__ = "matches"
    
    id = Column(Integer, primary_key=True, index=True)
    external_id = Column(Integer, unique=True, index=True)  # ID de la API externa
    
    # Equipos
    home_team_id = Column(Integer, nullable=False)
    away_team_id = Column(Integer, nullable=False)
    home_team_name = Column(String(100), nullable=False)
    away_team_name = Column(String(100), nullable=False)
    
    # Liga y competición
    league_id = Column(Integer, nullable=False)
    league_name = Column(String(100), nullable=False)
    season = Column(String(20), nullable=False)
    
    # Horario
    kickoff_time = Column(DateTime, nullable=False)
    status = Column(String(20), default="NS")  # NS, LIVE, FT, etc.
    
    # Metadatos
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relaciones
    predictions = relationship("Prediction", back_populates="match")

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Predicciones 1X2
    home_win_prob = Column(Float, nullable=False)
    draw_prob = Column(Float, nullable=False)
    away_win_prob = Column(Float, nullable=False)
    
    # Predicciones de goles
    total_goals = Column(Float, nullable=False)
    home_goals = Column(Float, nullable=False)
    away_goals = Column(Float, nullable=False)
    over_2_5_prob = Column(Float, nullable=False)
    btts_prob = Column(Float, nullable=False)
    
    # Predicciones de corners
    total_corners = Column(Float, nullable=False)
    home_corners = Column(Float, nullable=False)
    away_corners = Column(Float, nullable=False)
    over_9_5_corners_prob = Column(Float, nullable=False)
    
    # Métricas de confianza
    overall_confidence = Column(Float, nullable=False)
    match_confidence = Column(Float, nullable=False)
    goals_confidence = Column(Float, nullable=False)
    corners_confidence = Column(Float, nullable=False)
    
    # Value bets (JSON)
    value_bets = Column(Text)  # JSON string
    
    # Metadatos
    model_version = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    
    # Relaciones
    match = relationship("Match", back_populates="predictions")
    
    def get_value_bets(self):
        """Deserializar value bets de JSON"""
        return json.loads(self.value_bets) if self.value_bets else []
    
    def set_value_bets(self, value_bets_list):
        """Serializar value bets a JSON"""
        self.value_bets = json.dumps(value_bets_list)

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)
    
    # Perfil
    first_name = Column(String(50))
    last_name = Column(String(50))
    
    # Suscripción
    subscription_tier = Column(String(20), default="free")  # free, basic, pro, premium, vip
    subscription_status = Column(String(20), default="inactive")  # active, inactive, cancelled
    subscription_start_date = Column(DateTime)
    subscription_end_date = Column(DateTime)
    stripe_customer_id = Column(String(100))
    
    # Configuraciones
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    value_bet_alerts = Column(Boolean, default=True)
    min_value_threshold = Column(Float, default=10.0)
    
    # Metadatos
    created_at = Column(DateTime, nullable=False)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    
    # Relaciones
    betting_history = relationship("BettingHistory", back_populates="user")

class BettingHistory(Base):
    __tablename__ = "betting_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.id"), nullable=False)
    
    # Detalles de la apuesta
    bet_type = Column(String(50), nullable=False)  # 1X2, Over/Under, etc.
    selection = Column(String(100), nullable=False)
    odds = Column(Float, nullable=False)
    stake = Column(Float, nullable=False)
    
    # Resultado
    result = Column(String(20))  # win, loss, void, pending
    profit_loss = Column(Float, default=0)
    
    # Metadatos
    placed_at = Column(DateTime, nullable=False)
    settled_at = Column(DateTime)
    notes = Column(Text)
    
    # Relaciones
    user = relationship("User", back_populates="betting_history")
```

## 🐳 **DOCKER CONFIGURATION**

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/soccer_predictions
      - REDIS_URL=redis://redis:6379
      - STRIPE_SECRET_KEY=${STRIPE_SECRET_KEY}
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
    depends_on:
      - db
      - redis
    volumes:
      - .:/app
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=soccer_predictions
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"

  celery:
    build: .
    command: celery -A app.tasks.prediction_tasks worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/soccer_predictions
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

  celery-beat:
    build: .
    command: celery -A app.tasks.prediction_tasks beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/soccer_predictions
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis
    volumes:
      - .:/app

volumes:
  postgres_data:
```

## 🚀 **PRÓXIMOS PASOS**

### **Esta Semana:**
1. Implementar los endpoints básicos de matches y predictions
2. Configurar la base de datos PostgreSQL
3. Integrar con tu sistema de predicciones existente

### **Próxima Semana:**
1. Implementar autenticación y sistema de usuarios
2. Configurar Celery para tareas automáticas
3. Crear tests unitarios para la API

### **Semana 3:**
1. Integrar sistema de suscripciones con Stripe
2. Implementar sistema de notificaciones
3. Optimizar rendimiento y caching

¿Te parece bien esta arquitectura? ¿Qué parte te gustaría que desarrollemos primero?
