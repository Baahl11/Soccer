"""
Módulo para integración y análisis de datos contextuales ampliados para predicciones de fútbol.

Este módulo implementa la integración de datos contextuales detallados para mejorar las predicciones
de goles, basado en metaanálisis de Sports Medicine (2025) sobre cómo los factores ambientales
afectan el rendimiento en deportes de equipo.

Funcionalidades principales:
- Integración de datos meteorológicos detallados con efectos cuantificados
- Factores de distancia y viaje con impacto biométrico calculado
- Análisis de superficies de juego y dimensiones de campos
"""

import os
import json
import random
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from geopy.distance import geodesic
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes y datos de referencia
DATA_CACHE_PATH = "cache/contextual_data/"

# Umbrales de factores meteorológicos
WEATHER_THRESHOLDS = {
    "temperature": {
        "very_cold": 0,
        "cold": 5,
        "moderate_cool": 10,
        "moderate": 15,
        "moderate_warm": 20,
        "warm": 25,
        "hot": 30,
        "very_hot": 35
    },
    "wind": {
        "calm": 5,
        "light": 10,
        "moderate": 20,
        "strong": 30,
        "very_strong": 40
    },
    "precipitation": {
        "light": 1,  # mm/hora
        "moderate": 4,
        "heavy": 8,
        "very_heavy": 12
    },
    "humidity": {
        "dry": 30,
        "comfortable": 50,
        "humid": 70,
        "very_humid": 85
    }
}

# Factores de impacto por tipo de superficie
PITCH_SURFACE_FACTORS = {
    "natural_grass": {
        "goals_factor": 1.0,
        "speed_factor": 1.0,
        "technical_factor": 1.0,
        "injury_risk": 1.0
    },
    "hybrid_grass": {
        "goals_factor": 1.02,
        "speed_factor": 1.05,
        "technical_factor": 1.0,
        "injury_risk": 0.95
    },
    "artificial_turf": {
        "goals_factor": 1.08,
        "speed_factor": 1.1,
        "technical_factor": 0.95,
        "injury_risk": 1.15
    },
    "poor_natural": {
        "goals_factor": 0.92,
        "speed_factor": 0.85,
        "technical_factor": 0.8,
        "injury_risk": 1.2
    }
}

# Impacto del viaje
TRAVEL_IMPACT_FACTORS = {
    "short": {  # < 300km
        "fatigue": 0.02,
        "mental": 0.01,
        "performance": 0.01
    },
    "medium": {  # 300-800km
        "fatigue": 0.05,
        "mental": 0.03,
        "performance": 0.03
    },
    "long": {  # 800-1500km
        "fatigue": 0.08,
        "mental": 0.06,
        "performance": 0.06
    },
    "very_long": {  # >1500km
        "fatigue": 0.12,
        "mental": 0.09,
        "performance": 0.09
    },
    "international": {  # Vuelos internacionales
        "fatigue": 0.15,
        "mental": 0.12,
        "performance": 0.12
    }
}

# Tipos de competiciones y factores de intensidad
COMPETITION_INTENSITY = {
    "league_regular": 1.0,
    "league_decisive": 1.15,  # Final de temporada con objetivos en juego
    "cup_early": 1.05,
    "cup_late": 1.2,  # Fases finales de copa
    "european_group": 1.1,
    "european_knockout": 1.25,
    "derby": 1.15,
    "final": 1.3
}

@dataclass
class ContextualFactors:
    """Estructura de datos para factores contextuales de un partido"""
    match_id: int
    
    # Factores meteorológicos
    temperature: float = 15.0  # ºC
    weather_condition: str = "clear"  # clear, rain, snow, fog
    humidity: float = 50.0  # %
    wind_speed: float = 5.0  # km/h
    wind_direction: Optional[str] = None  # Norte, Sur, etc.
    precipitation: float = 0.0  # mm/h
    
    # Factores de campo y superficie
    pitch_type: str = "natural_grass"  # natural_grass, hybrid_grass, artificial_turf, poor_natural
    pitch_length: float = 105.0  # metros
    pitch_width: float = 68.0  # metros
    field_condition: str = "good"  # poor, average, good, excellent
    stadium_capacity: Optional[int] = None
    attendance: Optional[int] = None
    
    # Factores de viaje
    travel_distance: float = 0.0  # km
    travel_type: str = "short"  # short, medium, long, very_long, international
    days_since_last_match_home: Optional[int] = None
    days_since_last_match_away: Optional[int] = None
    timezone_change: Optional[int] = None  # Diferencia en horas
    
    # Factores de competición
    competition_type: str = "league_regular"
    match_importance: float = 1.0
    is_derby: bool = False
    
    # Factores temporales
    kickoff_hour: Optional[int] = None
    is_weekday: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el objeto a diccionario"""
        return {
            "match_id": self.match_id,
            "weather": {
                "temperature": self.temperature,
                "condition": self.weather_condition,
                "humidity": self.humidity,
                "wind_speed": self.wind_speed,
                "wind_direction": self.wind_direction,
                "precipitation": self.precipitation
            },
            "pitch": {
                "type": self.pitch_type,
                "length": self.pitch_length,
                "width": self.pitch_width,
                "condition": self.field_condition,
                "stadium_capacity": self.stadium_capacity,
                "attendance": self.attendance,
                "attendance_percentage": (self.attendance / self.stadium_capacity if 
                                         self.attendance and self.stadium_capacity else None)
            },
            "travel": {
                "distance": self.travel_distance,
                "type": self.travel_type,
                "days_since_last_match": {
                    "home": self.days_since_last_match_home,
                    "away": self.days_since_last_match_away
                },
                "timezone_change": self.timezone_change
            },
            "competition": {
                "type": self.competition_type,
                "importance": self.match_importance,
                "is_derby": self.is_derby
            },
            "temporal": {
                "kickoff_hour": self.kickoff_hour,
                "is_weekday": self.is_weekday
            }
        }

class ContextualDataIntegrator:
    """
    Clase principal para la integración y procesamiento de datos contextuales.
    
    Esta clase procesa datos meteorológicos, de superficie, viaje y otros factores
    contextuales para mejorar las predicciones de partidos de fútbol.
    """
    
    def __init__(self, data_path: str = DATA_CACHE_PATH):
        """
        Inicializa el integrador de datos contextuales.
        
        Args:
            data_path: Ruta al directorio de caché de datos contextuales
        """
        self.data_path = data_path
        self._ensure_data_directory()
        
        # Inicializar escaladores para normalización
        self.feature_scaler = StandardScaler()
        
        # Base de datos de estadios y sus características (simplificada)
        # En implementación real, cargaría desde archivo
        self.stadium_database = {}
        
        # Base de datos de distancias entre ciudades (simplificada)
        # En implementación real, usaría API o base de datos geográfica
        self.city_distances = {}
        
    def _ensure_data_directory(self) -> None:
        """Asegura que exista el directorio para cache de datos"""
        os.makedirs(self.data_path, exist_ok=True)
        
    def process_contextual_data(self, match_id: int, match_data: Dict[str, Any]) -> ContextualFactors:
        """
        Procesa datos contextuales de un partido.
        
        Args:
            match_id: ID del partido
            match_data: Datos del partido incluyendo información contextual
            
        Returns:
            Objeto ContextualFactors con los factores procesados
        """
        logger.info(f"Procesando datos contextuales para partido {match_id}")
        
        try:
            # Inicializar con valores por defecto
            factors = ContextualFactors(match_id=match_id)
            
            # Procesar datos meteorológicos
            weather_data = match_data.get("weather", {})
            if weather_data:
                factors.temperature = weather_data.get("temperature", factors.temperature)
                factors.weather_condition = weather_data.get("condition", factors.weather_condition)
                factors.humidity = weather_data.get("humidity", factors.humidity)
                factors.wind_speed = weather_data.get("wind_speed", factors.wind_speed)
                factors.wind_direction = weather_data.get("wind_direction", factors.wind_direction)
                factors.precipitation = weather_data.get("precipitation", factors.precipitation)
            
            # Procesar datos de campo
            pitch_data = match_data.get("pitch", {})
            if pitch_data:
                factors.pitch_type = pitch_data.get("type", factors.pitch_type)
                factors.pitch_length = pitch_data.get("length", factors.pitch_length)
                factors.pitch_width = pitch_data.get("width", factors.pitch_width)
                factors.field_condition = pitch_data.get("condition", factors.field_condition)
                factors.stadium_capacity = pitch_data.get("stadium_capacity", factors.stadium_capacity)
                factors.attendance = pitch_data.get("attendance", factors.attendance)
            
            # Procesar datos de viaje
            travel_data = match_data.get("travel", {})
            if travel_data:
                factors.travel_distance = travel_data.get("distance", factors.travel_distance)
                factors.travel_type = self._determine_travel_type(factors.travel_distance)
                factors.days_since_last_match_home = travel_data.get("days_since_last_match_home", 
                                                                  factors.days_since_last_match_home)
                factors.days_since_last_match_away = travel_data.get("days_since_last_match_away", 
                                                                  factors.days_since_last_match_away)
                factors.timezone_change = travel_data.get("timezone_change", factors.timezone_change)
            
            # Procesar datos de competición
            competition_data = match_data.get("competition", {})
            if competition_data:
                factors.competition_type = competition_data.get("type", factors.competition_type)
                factors.match_importance = competition_data.get("importance", factors.match_importance)
                factors.is_derby = competition_data.get("is_derby", factors.is_derby)
            
            # Procesar datos temporales
            datetime_str = match_data.get("datetime")
            if datetime_str:
                match_date = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                factors.kickoff_hour = match_date.hour
                factors.is_weekday = match_date.weekday() < 5  # Lunes a Viernes son días de semana
            
            # Guardar en caché
            self._cache_contextual_data(match_id, factors)
            
            return factors
            
        except Exception as e:
            logger.error(f"Error procesando datos contextuales para partido {match_id}: {e}")
            return ContextualFactors(match_id=match_id)
    
    def _determine_travel_type(self, distance: float) -> str:
        """
        Determina el tipo de viaje basado en la distancia.
        
        Args:
            distance: Distancia en kilómetros
            
        Returns:
            Tipo de viaje (short, medium, long, very_long, international)
        """
        if distance < 300:
            return "short"
        elif distance < 800:
            return "medium"
        elif distance < 1500:
            return "long"
        else:
            return "very_long"
    
    def _cache_contextual_data(self, match_id: int, data: ContextualFactors) -> None:
        """Guarda datos contextuales en caché"""
        try:
            file_path = os.path.join(self.data_path, f"match_{match_id}_contextual.json")
            with open(file_path, 'w') as f:
                json.dump(data.to_dict(), f, indent=2)
            logger.debug(f"Datos contextuales guardados para partido {match_id}")
        except Exception as e:
            logger.error(f"Error guardando datos contextuales en caché: {e}")
    
    def get_cached_contextual_data(self, match_id: int) -> Optional[ContextualFactors]:
        """Recupera datos contextuales de caché si existen"""
        try:
            file_path = os.path.join(self.data_path, f"match_{match_id}_contextual.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Reconstruir objeto ContextualFactors
                factors = ContextualFactors(match_id=match_id)
                
                # Procesar datos meteorológicos
                weather = data.get("weather", {})
                factors.temperature = weather.get("temperature", factors.temperature)
                factors.weather_condition = weather.get("condition", factors.weather_condition)
                factors.humidity = weather.get("humidity", factors.humidity)
                factors.wind_speed = weather.get("wind_speed", factors.wind_speed)
                factors.wind_direction = weather.get("wind_direction", factors.wind_direction)
                factors.precipitation = weather.get("precipitation", factors.precipitation)
                
                # Procesar datos de campo
                pitch = data.get("pitch", {})
                factors.pitch_type = pitch.get("type", factors.pitch_type)
                factors.pitch_length = pitch.get("length", factors.pitch_length)
                factors.pitch_width = pitch.get("width", factors.pitch_width)
                factors.field_condition = pitch.get("condition", factors.field_condition)
                factors.stadium_capacity = pitch.get("stadium_capacity", factors.stadium_capacity)
                factors.attendance = pitch.get("attendance", factors.attendance)
                
                # Procesar datos de viaje
                travel = data.get("travel", {})
                factors.travel_distance = travel.get("distance", factors.travel_distance)
                factors.travel_type = travel.get("type", factors.travel_type)
                days = travel.get("days_since_last_match", {})
                factors.days_since_last_match_home = days.get("home", factors.days_since_last_match_home)
                factors.days_since_last_match_away = days.get("away", factors.days_since_last_match_away)
                factors.timezone_change = travel.get("timezone_change", factors.timezone_change)
                
                # Procesar datos de competición
                competition = data.get("competition", {})
                factors.competition_type = competition.get("type", factors.competition_type)
                factors.match_importance = competition.get("importance", factors.match_importance)
                factors.is_derby = competition.get("is_derby", factors.is_derby)
                
                # Procesar datos temporales
                temporal = data.get("temporal", {})
                factors.kickoff_hour = temporal.get("kickoff_hour", factors.kickoff_hour)
                factors.is_weekday = temporal.get("is_weekday", factors.is_weekday)
                
                return factors
            
            return None
            
        except Exception as e:
            logger.error(f"Error leyendo datos contextuales de caché: {e}")
            return None
    
    def calculate_travel_impact(self, travel_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula el impacto del viaje en el rendimiento del equipo.
        
        Args:
            travel_data: Datos de viaje del equipo
            
        Returns:
            Diccionario con factores de impacto calculados
        """
        impact = {
            "fatigue": 0.0,
            "mental": 0.0,
            "performance": 0.0
        }
        
        try:
            distance = travel_data.get("distance", 0)
            travel_type = self._determine_travel_type(distance)
            
            # Factores base por tipo de viaje
            impact["fatigue"] = TRAVEL_IMPACT_FACTORS[travel_type]["fatigue"]
            impact["mental"] = TRAVEL_IMPACT_FACTORS[travel_type]["mental"]
            impact["performance"] = TRAVEL_IMPACT_FACTORS[travel_type]["performance"]
            
            # Ajuste por cambio de zona horaria
            timezone_change = abs(travel_data.get("timezone_change", 0))
            if timezone_change > 0:
                timezone_factor = min(timezone_change * 0.015, 0.06)  # Max 6% por zona horaria
                impact["fatigue"] += timezone_factor
                impact["mental"] += timezone_factor
                impact["performance"] += timezone_factor
            
            # Ajuste por descanso
            days_rest = travel_data.get("days_since_last_match", 3)
            if days_rest < 3:
                rest_penalty = (3 - days_rest) * 0.03  # 3% por cada día menos de 3
                impact["fatigue"] += rest_penalty
                impact["performance"] += rest_penalty * 0.7
            elif days_rest > 5:
                # Demasiado descanso puede ser negativo (pérdida de ritmo)
                rhythm_penalty = min((days_rest - 5) * 0.01, 0.03)  # Max 3%
                impact["performance"] += rhythm_penalty
            
        except Exception as e:
            logger.error(f"Error calculando impacto de viaje: {e}")
            
        return impact
    
    def calculate_weather_impact(self, weather_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula el impacto de las condiciones meteorológicas en el partido.
        
        Args:
            weather_data: Datos meteorológicos
            
        Returns:
            Diccionario con factores de impacto calculados
        """
        impact = {
            "goals_factor": 1.0,
            "pace_factor": 1.0,
            "technical_factor": 1.0,
            "player_performance": 1.0,
            "description": "Condiciones normales"
        }
        
        try:
            temperature = weather_data.get("temperature", 15)
            condition = weather_data.get("condition", "clear").lower()
            wind_speed = weather_data.get("wind_speed", 5)
            precipitation = weather_data.get("precipitation", 0)
            humidity = weather_data.get("humidity", 50)
            
            # Factor de temperatura
            if temperature <= WEATHER_THRESHOLDS["temperature"]["very_cold"]:
                impact["goals_factor"] *= 0.9
                impact["pace_factor"] *= 0.85
                impact["player_performance"] *= 0.9
                impact["description"] = "Condiciones muy frías"
            elif temperature <= WEATHER_THRESHOLDS["temperature"]["cold"]:
                impact["goals_factor"] *= 0.95
                impact["pace_factor"] *= 0.9
                impact["player_performance"] *= 0.95
                impact["description"] = "Condiciones frías"
            elif temperature >= WEATHER_THRESHOLDS["temperature"]["very_hot"]:
                impact["goals_factor"] *= 0.92
                impact["pace_factor"] *= 0.8
                impact["player_performance"] *= 0.85
                impact["description"] = "Condiciones muy calurosas"
            elif temperature >= WEATHER_THRESHOLDS["temperature"]["hot"]:
                impact["goals_factor"] *= 0.95
                impact["pace_factor"] *= 0.9
                impact["player_performance"] *= 0.92
                impact["description"] = "Condiciones calurosas"
                
            # Factor de condición meteorológica
            if condition == "rain":
                if precipitation > WEATHER_THRESHOLDS["precipitation"]["heavy"]:
                    impact["goals_factor"] *= 0.85
                    impact["pace_factor"] *= 0.8
                    impact["technical_factor"] *= 0.75
                    impact["description"] = "Lluvia intensa"
                elif precipitation > WEATHER_THRESHOLDS["precipitation"]["moderate"]:
                    impact["goals_factor"] *= 0.9
                    impact["pace_factor"] *= 0.85
                    impact["technical_factor"] *= 0.85
                    impact["description"] = "Lluvia moderada"
                else:
                    impact["goals_factor"] *= 0.95
                    impact["technical_factor"] *= 0.95
                    impact["description"] = "Lluvia ligera"
            elif condition == "snow":
                impact["goals_factor"] *= 0.8
                impact["pace_factor"] *= 0.7
                impact["technical_factor"] *= 0.7
                impact["player_performance"] *= 0.85
                impact["description"] = "Nevando"
            elif condition == "fog":
                impact["goals_factor"] *= 0.9
                impact["technical_factor"] *= 0.85
                impact["description"] = "Neblina"
                
            # Factor de viento
            if wind_speed >= WEATHER_THRESHOLDS["wind"]["very_strong"]:
                impact["goals_factor"] *= 0.85
                impact["technical_factor"] *= 0.8
                impact["description"] = f"{impact['description']}, viento muy fuerte"
            elif wind_speed >= WEATHER_THRESHOLDS["wind"]["strong"]:
                impact["goals_factor"] *= 0.9
                impact["technical_factor"] *= 0.85
                impact["description"] = f"{impact['description']}, viento fuerte"
            elif wind_speed >= WEATHER_THRESHOLDS["wind"]["moderate"]:
                impact["goals_factor"] *= 0.95
                impact["technical_factor"] *= 0.95
                
            # Factor de humedad
            if humidity >= WEATHER_THRESHOLDS["humidity"]["very_humid"] and temperature > 20:
                impact["player_performance"] *= 0.9
                impact["pace_factor"] *= 0.92
                impact["description"] = f"{impact['description']}, muy húmedo"
            
            # Condiciones extremas combinadas
            if (temperature <= WEATHER_THRESHOLDS["temperature"]["very_cold"] and 
                wind_speed >= WEATHER_THRESHOLDS["wind"]["strong"]):
                impact["goals_factor"] *= 0.95
                impact["player_performance"] *= 0.95
                impact["description"] = "Condiciones extremas: muy frío con viento fuerte"
            elif (temperature >= WEATHER_THRESHOLDS["temperature"]["very_hot"] and 
                  humidity >= WEATHER_THRESHOLDS["humidity"]["very_humid"]):
                impact["goals_factor"] *= 0.9
                impact["player_performance"] *= 0.85
                impact["pace_factor"] *= 0.85
                impact["description"] = "Condiciones extremas: muy caluroso y húmedo"
                
        except Exception as e:
            logger.error(f"Error calculando impacto meteorológico: {e}")
            
        return impact
    
    def calculate_pitch_impact(self, pitch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula el impacto de las características del campo en el partido.
        
        Args:
            pitch_data: Datos del campo
            
        Returns:
            Diccionario con factores de impacto calculados
        """
        impact = {
            "goals_factor": 1.0,
            "pace_factor": 1.0,
            "technical_factor": 1.0,
            "description": "Campo estándar"
        }
        
        try:
            pitch_type = pitch_data.get("type", "natural_grass")
            condition = pitch_data.get("condition", "good")
            length = pitch_data.get("length", 105)
            width = pitch_data.get("width", 68)
            area = length * width
            standard_area = 105 * 68
            
            # Impacto por tipo de superficie
            if pitch_type in PITCH_SURFACE_FACTORS:
                surface = PITCH_SURFACE_FACTORS[pitch_type]
                impact["goals_factor"] *= surface["goals_factor"]
                impact["pace_factor"] *= surface["speed_factor"]
                impact["technical_factor"] *= surface["technical_factor"]
                
                if pitch_type == "artificial_turf":
                    impact["description"] = "Césped artificial"
                elif pitch_type == "hybrid_grass":
                    impact["description"] = "Césped híbrido"
                elif pitch_type == "poor_natural":
                    impact["description"] = "Césped natural en mal estado"
            
            # Ajustar por condición del campo
            if condition == "poor":
                impact["goals_factor"] *= 0.9
                impact["pace_factor"] *= 0.85
                impact["technical_factor"] *= 0.8
                impact["description"] = f"{impact['description']} en mal estado"
            elif condition == "excellent":
                impact["technical_factor"] *= 1.05
                impact["pace_factor"] *= 1.05
                impact["description"] = f"{impact['description']} en excelente estado"
            
            # Impacto por dimensiones del campo
            size_ratio = area / standard_area
            
            if size_ratio > 1.1:  # Campo grande
                impact["goals_factor"] *= 1.05
                impact["pace_factor"] *= 0.95  # Más espacio pero más distancias
                impact["description"] = f"{impact['description']} de grandes dimensiones"
            elif size_ratio < 0.9:  # Campo pequeño
                impact["goals_factor"] *= 0.95
                impact["technical_factor"] *= 0.95
                impact["pace_factor"] *= 1.05  # Menos espacio pero menos distancias
                impact["description"] = f"{impact['description']} de dimensiones reducidas"
                
            # Asistencia y factor de presión
            capacity = pitch_data.get("stadium_capacity", 0)
            attendance = pitch_data.get("attendance", 0)
            
            if capacity > 0 and attendance > 0:
                attendance_ratio = attendance / capacity
                if attendance_ratio > 0.9:
                    impact["home_advantage"] = 1.1
                    impact["description"] = f"{impact['description']} con estadio lleno"
                elif attendance_ratio < 0.5:
                    impact["home_advantage"] = 0.95
            
        except Exception as e:
            logger.error(f"Error calculando impacto de campo: {e}")
            
        return impact
    
    def fetch_stadium_data(self, stadium_id: int) -> Dict[str, Any]:
        """
        Obtiene datos de un estadio.
        
        Args:
            stadium_id: ID del estadio
            
        Returns:
            Datos del estadio
        """
        # Esta función simularía la obtención de datos desde una API o base de datos
        
        # Comprobación en caché local
        if stadium_id in self.stadium_database:
            return self.stadium_database[stadium_id]
        
        # Valores por defecto si no se encuentra
        return {
            "name": f"Stadium {stadium_id}",
            "capacity": 25000,
            "pitch_type": "natural_grass",
            "pitch_dimensions": {
                "length": 105,
                "width": 68
            },
            "location": {
                "city": "Unknown",
                "country": "Unknown",
                "latitude": 0,
                "longitude": 0
            },
            "average_attendance": 15000
        }
    
    def calculate_travel_distance(self, origin_city: str, destination_city: str) -> float:
        """
        Calcula la distancia de viaje entre ciudades.
        
        Args:
            origin_city: Ciudad de origen
            destination_city: Ciudad de destino
            
        Returns:
            Distancia en kilómetros
        """
        # Comprobación en caché local
        cache_key = f"{origin_city}_{destination_city}"
        if cache_key in self.city_distances:
            return self.city_distances[cache_key]
        
        # Esta función simularía la obtención de distancias mediante geolocalización
        # En una implementación real conectaría con API de geolocalización

        # Simulamos una distancia aleatoria para fines de demostración
        import random
        distance = random.uniform(50, 2000)
        
        # Guardamos en caché
        self.city_distances[cache_key] = distance
        
        return distance
    
    def extract_prediction_features(self, match_id: int) -> Dict[str, float]:
        """
        Extrae características contextuales para modelos predictivos.
        
        Args:
            match_id: ID del partido
            
        Returns:
            Diccionario con características normalizadas para predicción
        """
        features = {}
        
        # Intentar obtener datos contextuales de caché
        contextual_data = self.get_cached_contextual_data(match_id)
        
        if not contextual_data:
            logger.warning(f"No se encontraron datos contextuales para el partido {match_id}")
            return features
        
        try:
            # Factores meteorológicos
            features["temperature"] = contextual_data.temperature
            features["is_raining"] = 1.0 if contextual_data.weather_condition == "rain" else 0.0
            features["is_snowing"] = 1.0 if contextual_data.weather_condition == "snow" else 0.0
            features["wind_speed"] = contextual_data.wind_speed
            features["precipitation"] = contextual_data.precipitation
            
            # Normalizar factores meteorológicos
            features["norm_temperature"] = features["temperature"] / 35  # Normalizar a escala 0-1
            features["norm_wind"] = features["wind_speed"] / 60  # Normalizar a escala 0-1
            
            # Factores de campo
            features["is_artificial"] = 1.0 if contextual_data.pitch_type == "artificial_turf" else 0.0
            features["is_poor_pitch"] = 1.0 if contextual_data.field_condition == "poor" else 0.0
            
            pitch_area = contextual_data.pitch_length * contextual_data.pitch_width
            standard_area = 105 * 68
            features["pitch_size_ratio"] = pitch_area / standard_area
            
            # Factores de asistencia
            if contextual_data.stadium_capacity and contextual_data.attendance:
                features["attendance_ratio"] = contextual_data.attendance / contextual_data.stadium_capacity
            else:
                features["attendance_ratio"] = 0.5  # Valor por defecto
            
            # Factores de viaje
            features["travel_distance_away"] = contextual_data.travel_distance
            features["norm_travel_distance"] = min(contextual_data.travel_distance / 2000, 1.0)  # Normalizar
            
            features["days_rest_home"] = contextual_data.days_since_last_match_home if contextual_data.days_since_last_match_home else 3
            features["days_rest_away"] = contextual_data.days_since_last_match_away if contextual_data.days_since_last_match_away else 3
            features["rest_differential"] = features["days_rest_home"] - features["days_rest_away"]
            
            # Factores temporales
            features["is_night_match"] = 1.0 if (contextual_data.kickoff_hour and contextual_data.kickoff_hour >= 19) else 0.0
            features["is_weekday"] = 1.0 if contextual_data.is_weekday else 0.0
            
            # Factores de competición
            features["is_derby"] = 1.0 if contextual_data.is_derby else 0.0
            features["is_high_importance"] = 1.0 if contextual_data.match_importance > 1.1 else 0.0
            
            # Calcular impacto compuesto
            weather_impact = self.calculate_weather_impact({
                "temperature": contextual_data.temperature,
                "condition": contextual_data.weather_condition,
                "wind_speed": contextual_data.wind_speed,
                "precipitation": contextual_data.precipitation,
                "humidity": contextual_data.humidity
            })
            
            pitch_impact = self.calculate_pitch_impact({
                "type": contextual_data.pitch_type,
                "condition": contextual_data.field_condition,
                "length": contextual_data.pitch_length,
                "width": contextual_data.pitch_width
            })
            
            travel_impact = self.calculate_travel_impact({
                "distance": contextual_data.travel_distance,
                "days_since_last_match": features["days_rest_away"]
            })
            
            # Añadir factores de impacto
            features["weather_goals_factor"] = weather_impact["goals_factor"]
            features["weather_technical_factor"] = weather_impact["technical_factor"]
            features["pitch_goals_factor"] = pitch_impact["goals_factor"]
            features["pitch_technical_factor"] = pitch_impact["technical_factor"]
            features["travel_performance_factor"] = 1.0 - travel_impact["performance"]
            
            # Factor global de impacto contextual
            features["contextual_goals_modifier"] = (
                weather_impact["goals_factor"] * 
                pitch_impact["goals_factor"]
            )
            
        except Exception as e:
            logger.error(f"Error extrayendo características contextuales: {e}")
            
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()}
    
    def integrate_with_prediction(self, match_id: int, base_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integra factores contextuales con una predicción base.
        
        Args:
            match_id: ID del partido
            base_prediction: Predicción base a modificar
            
        Returns:
            Predicción modificada con factores contextuales
        """
        if not base_prediction or "goals" not in base_prediction:
            logger.warning("Predicción base inválida para integración contextual")
            return base_prediction
            
        # Crear copia para modificar
        modified = base_prediction.copy()
        
        # Obtener características contextuales
        features = self.extract_prediction_features(match_id)
        
        if not features:
            logger.info(f"No hay características contextuales para partido {match_id}")
            return base_prediction
        
        try:
            # Extraer factores de impacto
            contextual_modifier = features.get("contextual_goals_modifier", 1.0)
            weather_factor = features.get("weather_goals_factor", 1.0)
            pitch_factor = features.get("pitch_goals_factor", 1.0)
            
            # Calcular impacto de viaje en equipo visitante
            travel_factor = features.get("travel_performance_factor", 1.0)
            
            # Aplicar modificadores a predicción de goles
            if "predicted_home_goals" in modified["goals"]:
                # Para equipo local aplicamos factores de clima y campo
                current = modified["goals"]["predicted_home_goals"]
                modified["goals"]["predicted_home_goals"] = max(0, current * weather_factor * pitch_factor)
                
            if "predicted_away_goals" in modified["goals"]:
                # Para equipo visitante aplicamos factores de clima, campo y viaje
                current = modified["goals"]["predicted_away_goals"]
                modified["goals"]["predicted_away_goals"] = max(0, current * weather_factor * pitch_factor * travel_factor)
                
            # Añadir metadatos contextuales
            if "metadata" not in modified:
                modified["metadata"] = {}
                
            modified["metadata"]["contextual_factors"] = {
                "applied": True,
                "weather_factor": round(weather_factor, 3),
                "pitch_factor": round(pitch_factor, 3),
                "travel_factor": round(travel_factor, 3),
                "combined_modifier": round(contextual_modifier, 3)
            }
            
            # Incluir descripción de factores
            contextual_data = self.get_cached_contextual_data(match_id)
            if contextual_data:
                weather_impact = self.calculate_weather_impact({
                    "temperature": contextual_data.temperature,
                    "condition": contextual_data.weather_condition,
                    "wind_speed": contextual_data.wind_speed,
                    "precipitation": contextual_data.precipitation,
                    "humidity": contextual_data.humidity
                })
                
                pitch_impact = self.calculate_pitch_impact({
                    "type": contextual_data.pitch_type,
                    "condition": contextual_data.field_condition,
                    "length": contextual_data.pitch_length,
                    "width": contextual_data.pitch_width
                })
                
                modified["contextual_insights"] = [
                    {
                        "type": "weather",
                        "description": weather_impact["description"],
                        "impact": "high" if abs(weather_factor - 1.0) > 0.1 else "medium" if abs(weather_factor - 1.0) > 0.05 else "low"
                    },
                    {
                        "type": "pitch",
                        "description": pitch_impact["description"],
                        "impact": "high" if abs(pitch_factor - 1.0) > 0.1 else "medium" if abs(pitch_factor - 1.0) > 0.05 else "low"
                    }
                ]
                
                if features.get("travel_distance_away", 0) > 500:
                    modified["contextual_insights"].append({
                        "type": "travel",
                        "description": f"El equipo visitante ha viajado {int(features['travel_distance_away'])} km",
                        "impact": "high" if abs(travel_factor - 1.0) > 0.1 else "medium" if abs(travel_factor - 1.0) > 0.05 else "low"
                    })
            
        except Exception as e:
            logger.error(f"Error integrando factores contextuales: {e}")
            
        return modified
      def visualize_contextual_factors(self, match_id: int) -> Optional[Figure]:
        """
        Genera visualización de factores contextuales del partido.
        
        Args:
            match_id: ID del partido
            
        Returns:
            Figura de matplotlib con visualización o None si hay error
        """
        contextual_data = self.get_cached_contextual_data(match_id)
        features = self.extract_prediction_features(match_id)
        
        if not contextual_data or not features:
            logger.warning(f"No hay datos suficientes para visualizar partido {match_id}")
            return None
            
        try:
            # Crear figura
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f"Análisis Contextual - Partido {match_id}", fontsize=16)
            
            # Gráfico 1: Factores meteorológicos
            ax1 = axes[0, 0]
            weather_factors = [
                features.get("temperature", 15),
                features.get("wind_speed", 5),
                features.get("precipitation", 0) * 10,  # Escalar para visualización
                features.get("humidity", 50) / 10       # Escalar para visualización
            ]
            weather_labels = ["Temperatura (°C)", "Viento (km/h)", "Precipitación (×10)", "Humedad (/10)"]
            
            ax1.bar(weather_labels, weather_factors, color='skyblue')
            ax1.set_title("Condiciones Meteorológicas")
            ax1.set_ylim(0, max(50, max(weather_factors) * 1.2))
            
            # Añadir etiqueta de condición
            condition = contextual_data.weather_condition.capitalize()
            ax1.annotate(f"Condición: {condition}", xy=(0.5, 0.95), xycoords='axes fraction', 
                        ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white"))
            
            # Gráfico 2: Impacto en factores de juego
            ax2 = axes[0, 1]
            impact_factors = [
                features.get("weather_goals_factor", 1.0),
                features.get("weather_technical_factor", 1.0),
                features.get("pitch_goals_factor", 1.0),
                features.get("travel_performance_factor", 1.0)
            ]
            impact_labels = ["Clima-Goles", "Clima-Técnica", "Campo-Goles", "Viaje-Rendimiento"]
            
            # Definir colores según impacto (rojo para reducción, verde para aumento)
            colors = ['red' if x < 1.0 else 'green' for x in impact_factors]
            
            ax2.bar(impact_labels, impact_factors, color=colors)
            ax2.set_title("Factores de Impacto")
            ax2.set_ylim(0.5, 1.5)
            ax2.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
            
            # Gráfico 3: Factores de campo
            ax3 = axes[1, 0]
            
            # Visualización del campo y dimensiones
            field_width = contextual_data.pitch_width / 105 * 100  # Porcentaje respecto a estándar
            field_length = contextual_data.pitch_length / 68 * 100  # Porcentaje respecto a estándar
            
            field_factors = [field_width, field_length]
            field_labels = ["Ancho (%)", "Largo (%)"]
            
            ax3.bar(field_labels, field_factors, color='green')
            ax3.set_title("Dimensiones del Campo")
            ax3.set_ylim(80, 120)
            ax3.axhline(y=100, color='gray', linestyle='-', alpha=0.3)
            
            # Añadir información de superficie y condición
            pitch_info = f"Superficie: {contextual_data.pitch_type.replace('_', ' ').title()}\n"
            pitch_info += f"Condición: {contextual_data.field_condition.title()}"
            ax3.annotate(pitch_info, xy=(0.5, 0.8), xycoords='axes fraction', 
                        ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white"))
            
            # Gráfico 4: Factores adicionales
            ax4 = axes[1, 1]
            
            # Factores varios
            other_factors = [
                features.get("attendance_ratio", 0.5) * 100,
                features.get("days_rest_home", 3),
                features.get("days_rest_away", 3),
                features.get("travel_distance_away", 0) / 100  # Escalar para visualización
            ]
            other_labels = ["Asistencia (%)", "Descanso Local (días)", 
                          "Descanso Visit. (días)", "Distancia Viaje (/100 km)"]
            
            ax4.bar(other_labels, other_factors, color='orange')
            ax4.set_title("Factores Adicionales")
            
            # Añadir información de competición
            competition_info = "Partido de alta importancia" if contextual_data.match_importance > 1.1 else "Partido regular"
            if contextual_data.is_derby:
                competition_info += "\nDerby local"
                
            ax4.annotate(competition_info, xy=(0.5, 0.9), xycoords='axes fraction', 
                        ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white"))
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error generando visualización: {e}")
            return None

def create_sample_contextual_data(match_id: int, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Crea datos contextuales de muestra para pruebas.
    
    Args:
        match_id: ID del partido
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        
    Returns:
        Datos contextuales sintéticos para prueba
    """
    import random
    random.seed(match_id)
    
    # Generar datos meteorológicos
    weather_conditions = ["clear", "rain", "snow", "fog"]
    condition_weights = [0.6, 0.3, 0.05, 0.05]
    
    weather = {
        "temperature": random.uniform(5, 30),
        "condition": random.choices(weather_conditions, weights=condition_weights)[0],
        "humidity": random.uniform(30, 90),
        "wind_speed": random.uniform(0, 40),
        "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        "precipitation": 0
    }
    
    # Ajustar precipitación según condición
    if weather["condition"] == "rain":
        weather["precipitation"] = random.uniform(0.5, 15)
    elif weather["condition"] == "snow":
        weather["precipitation"] = random.uniform(1, 5)
        weather["temperature"] = random.uniform(-5, 5)  # Ajustar temperatura para nieve
    
    # Generar datos de campo
    pitch_types = ["natural_grass", "hybrid_grass", "artificial_turf", "poor_natural"]
    pitch_weights = [0.7, 0.15, 0.1, 0.05]
    
    field_conditions = ["poor", "average", "good", "excellent"]
    condition_weights = [0.05, 0.2, 0.6, 0.15]
    
    capacity = random.randint(5000, 80000)
    
    pitch = {
        "type": random.choices(pitch_types, weights=pitch_weights)[0],
        "length": random.uniform(100, 110),
        "width": random.uniform(64, 75),
        "condition": random.choices(field_conditions, weights=condition_weights)[0],
        "stadium_capacity": capacity,
        "attendance": int(capacity * random.uniform(0.3, 1.0))
    }
    
    # Generar datos de viaje
    travel_distance = random.uniform(0, 2000)
    
    travel = {
        "distance": travel_distance,
        "days_since_last_match_home": random.randint(2, 7),
        "days_since_last_match_away": random.randint(2, 7),
        "timezone_change": int(travel_distance / 500)  # Aproximación simple
    }
    
    # Generar datos de competición
    is_derby = random.random() < 0.1
    is_high_importance = random.random() < 0.2
    
    competition_types = ["league_regular", "league_decisive", "cup_early", 
                       "cup_late", "european_group", "european_knockout"]
    
    competition = {
        "type": random.choice(competition_types),
        "importance": 1.25 if is_high_importance else 1.0,
        "is_derby": is_derby
    }
    
    # Generar fecha y hora
    from datetime import datetime, timedelta
    
    base_date = datetime.now()
    days_offset = random.randint(1, 30)
    hours = [13, 16, 19, 21]
    
    match_date = base_date + timedelta(days=days_offset)
    match_date = match_date.replace(hour=random.choice(hours), minute=0)
    
    # Integrar todos los datos
    return {
        "match_id": match_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "datetime": match_date.isoformat(),
        "weather": weather,
        "pitch": pitch,
        "travel": travel,
        "competition": competition
    }

def main():
    """Función principal para demostración"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Crear integrador
    integrator = ContextualDataIntegrator()
    
    # Generar datos de ejemplo
    match_id = 12345
    home_team_id = 55
    away_team_id = 65
    
    print(f"Generando datos contextuales para partido {match_id}")
    contextual_data = create_sample_contextual_data(match_id, home_team_id, away_team_id)
    
    # Procesar datos
    print("Procesando datos contextuales...")
    factors = integrator.process_contextual_data(match_id, contextual_data)
    
    # Mostrar factores procesados
    print("\nFactores contextuales procesados:")
    for k, v in factors.to_dict().items():
        if isinstance(v, dict):
            print(f"\n{k.upper()}:")
            for sub_k, sub_v in v.items():
                print(f"  - {sub_k}: {sub_v}")
        else:
            print(f"{k}: {v}")
    
    # Calcular impactos
    weather_impact = integrator.calculate_weather_impact({
        "temperature": factors.temperature,
        "condition": factors.weather_condition,
        "wind_speed": factors.wind_speed,
        "precipitation": factors.precipitation,
        "humidity": factors.humidity
    })
    
    print("\nImpacto meteorológico:")
    print(f"- Descripción: {weather_impact['description']}")
    print(f"- Factor de goles: {weather_impact['goals_factor']}")
    print(f"- Factor técnico: {weather_impact['technical_factor']}")
    print(f"- Factor de ritmo: {weather_impact['pace_factor']}")
    
    # Aplicar a predicción
    base_prediction = {
        "match_id": match_id,
        "goals": {
            "predicted_home_goals": 1.8,
            "predicted_away_goals": 1.2
        },
        "probabilities": {
            "home_win": 0.45,
            "draw": 0.25,
            "away_win": 0.3
        }
    }
    
    modified = integrator.integrate_with_prediction(match_id, base_prediction)
    
    print("\nPredicción original vs. modificada:")
    print(f"Home goals: {base_prediction['goals']['predicted_home_goals']} → {modified['goals']['predicted_home_goals']}")
    print(f"Away goals: {base_prediction['goals']['predicted_away_goals']} → {modified['goals']['predicted_away_goals']}")
    
    # Generar visualización
    print("\nGenerando visualización de factores contextuales...")
    fig = integrator.visualize_contextual_factors(match_id)
    if fig:
        plt.tight_layout()
        plt.savefig(f"match_{match_id}_contextual.png")
        print(f"Visualización guardada como match_{match_id}_contextual.png")
    
if __name__ == "__main__":
    main()
