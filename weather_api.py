# weather_api.py
import logging
import pandas as pd
from data import FootballAPI
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import requests
import json
import os
from pathlib import Path
import random
import config

logger = logging.getLogger(__name__)

# Crear directorio de caché si no existe
WEATHER_CACHE_DIR = Path("cache/weather")
WEATHER_CACHE_DIR.mkdir(exist_ok=True, parents=True)

class WeatherConditions:
    """
    Obtiene y procesa información sobre el clima y las condiciones del partido
    que pueden afectar el rendimiento de los equipos.
    """
    
    def __init__(self):
        self.api = FootballAPI()
        
    def get_match_conditions(self, fixture_id: int) -> Dict:
        """
        Obtiene las condiciones del partido (clima, estado del campo, etc.)
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Diccionario con información sobre las condiciones del partido
        """
        try:
            # Obtener datos del partido
            fixture_data = self.api._make_request('fixtures', {'id': fixture_id})
            fixture = fixture_data.get('response', [{}])[0] if fixture_data.get('response') else {}
            
            if not fixture:
                return self._default_conditions()
            
            # Extraer información del venue (estadio)
            venue_info = fixture.get('fixture', {}).get('venue', {})
            venue_id = venue_info.get('id')
            
            # Extraer información del clima si está disponible
            weather_info = fixture.get('fixture', {}).get('weather', {})
            
            # Compilar información de condiciones
            conditions = {
                'temperature': self._parse_temperature(weather_info.get('temperature', '')),
                'weather_code': self._parse_weather_code(weather_info.get('description', '')),
                'humidity': self._parse_humidity(weather_info.get('humidity', '')),
                'wind': self._parse_wind(weather_info.get('wind', '')),
                'stadium_info': self._get_stadium_info(venue_id) if venue_id else {},
                'is_rainy': 'rain' in (weather_info.get('description', '') or '').lower(),
                'is_windy': self._is_windy(weather_info.get('wind', '')),
                'is_extreme_temp': self._is_extreme_temperature(weather_info.get('temperature', '')),
                'precipitation': self._estimate_precipitation(weather_info.get('description', ''))
            }
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error obteniendo condiciones para partido {fixture_id}: {e}")
            return self._default_conditions()
    
    def _default_conditions(self) -> Dict:
        """Retorna valores predeterminados para condiciones desconocidas"""
        return {
            'temperature': 15.0,  # Temperatura moderada en °C
            'weather_code': 0,    # Despejado
            'humidity': 50,       # Humedad media
            'wind': 5.0,          # Viento medio
            'stadium_info': {},
            'is_rainy': False,
            'is_windy': False,
            'is_extreme_temp': False,
            'precipitation': 0.0
        }
    
    def _parse_temperature(self, temp_str: str) -> float:
        """Convierte temperatura de string a float en °C"""
        try:
            # Ejemplo: "22°C" o "72°F"
            temp_str_clean = temp_str.replace('°C', '').replace('°F', '')
            temp = float(temp_str_clean)
            
            # Convertir de Fahrenheit a Celsius si es necesario
            if '°F' in temp_str:
                temp = (temp - 32) * 5 / 9
                
            return round(temp, 1)
        except (ValueError, TypeError, AttributeError):
            return 15.0  # Valor por defecto
    
    def _parse_humidity(self, humidity_str: str) -> int:
        """Convierte humedad de string a entero en %"""
        try:
            # Ejemplo: "85%"
            humidity = int(humidity_str.replace('%', ''))
            return humidity
        except (ValueError, TypeError, AttributeError):
            return 50  # Valor por defecto
    
    def _parse_wind(self, wind_str: str) -> float:
        """Convierte velocidad del viento de string a float en km/h"""
        try:
            # Ejemplo: "5 m/s" o "10 km/h"
            parts = wind_str.split()
            if len(parts) >= 2:
                speed = float(parts[0])
                unit = parts[1].lower()
                
                # Convertir a km/h si es necesario
                if 'm/s' in unit:
                    speed = speed * 3.6
                elif 'mph' in unit:
                    speed = speed * 1.60934
                    
                return round(speed, 1)
            return 5.0
        except (ValueError, TypeError, AttributeError, IndexError):
            return 5.0  # Valor por defecto
    
    def _parse_weather_code(self, description: str) -> int:
        """
        Convierte descripción del clima a código numérico
        
        0: Despejado/Soleado
        1: Parcialmente nublado
        2: Nublado
        3: Lluvia ligera
        4: Lluvia intensa
        5: Tormenta/Truenos
        6: Nieve
        7: Niebla
        8: Otro
        """
        description = description.lower() if description else ''
        
        if any(x in description for x in ['sunny', 'clear', 'despejado']):
            return 0
        elif any(x in description for x in ['partly cloudy', 'parcialmente nublado']):
            return 1
        elif any(x in description for x in ['cloudy', 'overcast', 'nublado']):
            return 2
        elif any(x in description for x in ['light rain', 'drizzle', 'lluvia ligera']):
            return 3
        elif any(x in description for x in ['rain', 'heavy rain', 'lluvia']):
            return 4
        elif any(x in description for x in ['storm', 'thunder', 'tormenta', 'truenos']):
            return 5
        elif any(x in description for x in ['snow', 'nieve']):
            return 6
        elif any(x in description for x in ['fog', 'mist', 'niebla']):
            return 7
        else:
            return 8
    
    def _is_windy(self, wind_str: str) -> bool:
        """Determina si las condiciones son ventosas (>15 km/h)"""
        wind_speed = self._parse_wind(wind_str)
        return wind_speed > 15.0
    
    def _is_extreme_temperature(self, temp_str: str) -> bool:
        """Determina si la temperatura es extrema (<5°C o >30°C)"""
        temp = self._parse_temperature(temp_str)
        return temp < 5.0 or temp > 30.0
    
    def _estimate_precipitation(self, description: str) -> float:
        """
        Estima la precipitación en mm basada en la descripción
        
        Args:
            description: Descripción del clima
            
        Returns:
            Precipitación estimada en mm
        """
        description = description.lower() if description else ''
        
        if 'heavy rain' in description:
            return 8.0
        elif 'rain' in description:
            return 3.0
        elif 'light rain' in description or 'drizzle' in description:
            return 1.0
        elif 'shower' in description:
            return 2.0
        else:
            return 0.0
    
    def _get_stadium_info(self, venue_id: int) -> Dict:
        """
        Obtiene información sobre el estadio
        
        Args:
            venue_id: ID del estadio
            
        Returns:
            Diccionario con información del estadio
        """
        try:
            # Obtener datos del estadio
            venue_data = self.api._make_request('venues', {'id': venue_id})
            venue = venue_data.get('response', [{}])[0] if venue_data.get('response') else {}
            
            if not venue:
                return {}
            
            # Extraer información relevante
            stadium_info = {
                'name': venue.get('name', ''),
                'city': venue.get('city', ''),
                'capacity': venue.get('capacity', 0),
                'surface': venue.get('surface', ''),
                'altitude': venue.get('altitude', 0)  # metros sobre el nivel del mar
            }
            
            return stadium_info
            
        except Exception as e:
            logger.error(f"Error obteniendo información del estadio {venue_id}: {e}")
            return {}
    
    def get_weather_impact(self, fixture_id: int) -> Dict[str, float]:
        """
        Estimates the impact of weather on match performance with enhanced analysis
        
        Args:
            fixture_id: ID of the match
            
        Returns:
            Dictionary with detailed weather impact analysis
        """
        conditions = self.get_match_conditions(fixture_id)
        
        # Initialize base impact factors
        impact = {
            'attack_impact': 0.0,
            'defense_impact': 0.0,
            'overall_impact': 0.0,
            'possession_impact': 0.0,
            'passing_impact': 0.0,
            'physical_impact': 0.0,
            'technical_impact': 0.0
        }
        
        try:
            # Temperature impact analysis
            temp = conditions['temperature']
            if temp < 5:  # Cold conditions
                impact['physical_impact'] -= 0.05
                impact['technical_impact'] -= 0.03
            elif temp > 30:  # Hot conditions
                impact['physical_impact'] -= 0.08
                impact['possession_impact'] -= 0.04
                
            # Wind impact analysis
            wind_speed = conditions.get('wind', 0)
            if wind_speed > 20:  # Strong wind
                impact['passing_impact'] -= 0.06 * (wind_speed / 20)
                impact['technical_impact'] -= 0.04 * (wind_speed / 20)
            
            # Rain/precipitation impact
            precipitation = conditions.get('precipitation', 0)
            if precipitation > 0:
                # Impact increases with precipitation intensity
                rain_factor = min(precipitation / 10.0, 1.0)
                impact['technical_impact'] -= 0.05 * rain_factor
                impact['possession_impact'] -= 0.04 * rain_factor
                impact['passing_impact'] -= 0.03 * rain_factor
            
            # Stadium conditions impact (if available)
            stadium_info = conditions.get('stadium_info', {})
            altitude = stadium_info.get('altitude', 0)
            if altitude > 1500:  # High altitude impact
                impact['physical_impact'] -= 0.02 * (altitude / 1500)
            
            # Calculate derived impacts
            # Attack impact affected by technical and physical factors
            impact['attack_impact'] = (
                impact['technical_impact'] * 0.4 +
                impact['physical_impact'] * 0.3 +
                impact['possession_impact'] * 0.3
            )
            
            # Defense impact affected by physical and tactical factors
            impact['defense_impact'] = (
                impact['physical_impact'] * 0.4 +
                impact['technical_impact'] * 0.3 +
                impact['passing_impact'] * 0.3
            )
            
            # Overall impact considers all factors
            impact['overall_impact'] = (
                impact['attack_impact'] * 0.4 +
                impact['defense_impact'] * 0.4 +
                impact['possession_impact'] * 0.2
            )
            
            # Add confidence level based on data completeness
            impact['confidence'] = self._calculate_impact_confidence(conditions)
            
            return impact
            
        except Exception as e:
            logger.error(f"Error calculating weather impact: {e}")
            return {k: 0.0 for k in impact.keys()}  # Return neutral impact
            
    def _calculate_impact_confidence(self, conditions: Dict) -> float:
        """
        Calculate confidence level in weather impact assessment
        """
        try:
            confidence = 0.7  # Base confidence
            
            # Reduce confidence for missing data
            required_fields = ['temperature', 'wind', 'precipitation']
            for field in required_fields:
                if field not in conditions or conditions[field] is None:
                    confidence *= 0.8
                    
            # Reduce confidence for extreme values
            if conditions.get('temperature', 20) > 35 or conditions.get('temperature', 20) < 0:
                confidence *= 0.9
                
            if conditions.get('wind', 0) > 50:
                confidence *= 0.9
                
            return max(0.3, min(0.95, confidence))  # Keep confidence between 0.3 and 0.95
            
        except Exception as e:
            logger.error(f"Error calculating impact confidence: {e}")
            return 0.5  # Return moderate confidence on error

    def get_weather_data(self, match_date: str) -> Dict[str, float]:
        """Get weather data for a specific match date."""
        try:
            params = {'date': match_date}
            data = self.api._make_request('weather', params)
            weather_data = data.get('response', {})
            
            return {
                'precipitation': float(weather_data.get('precipitation', 0)),
                'temperature': float(weather_data.get('temperature', 20)),
                'wind_speed': float(weather_data.get('wind_speed', 0))
            }
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return {
                'precipitation': 0.0,
                'temperature': 20.0,
                'wind_speed': 0.0
            }

    def get_weather_for_date(self, date_str: str) -> Dict[str, Any]:
        """
        Obtiene información meteorológica para una fecha específica.
        
        Args:
            date_str: Fecha en formato 'YYYY-MM-DD'
            
        Returns:
            Diccionario con información meteorológica
        """
        # En una implementación real, aquí se haría una llamada a la API
        # de pronóstico del tiempo. Para este ejemplo, devolvemos datos ficticios.
        return {
            'temperature': 18,  # Celsius
            'humidity': 65,     # Porcentaje
            'wind_speed': 12,   # km/h
            'precipitation': 0, # mm
            'weather_code': 'clear',
            'description': 'Mostly clear'
        }

def get_weather_forecast(city: str, country: str = "", match_date: str = "") -> Dict[str, Any]:
    """
    Get weather forecast for a specific city and date.
    
    Args:
        city: City name
        country: Country code (optional)
        match_date: Date in ISO format (YYYY-MM-DD)
        
    Returns:
        Dictionary with weather information
    """
    try:
        # This would normally call a weather API
        # For now we'll return mock data
        logger.info(f"Getting weather forecast for {city}, {country} on {match_date}")
        
        # Mock weather data
        return {
            "city": city,
            "country": country,
            "date": match_date,
            "condition": "Clear",
            "temperature": 22,
            "wind_speed": 5,
            "precipitation": 0,
            "humidity": 65
        }
    except Exception as e:
        logger.error(f"Error getting weather forecast: {e}")
        return {}

def get_weather_impact(weather_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze impact of weather conditions on a football match.
    
    Args:
        weather_data: Dictionary with weather information
        
    Returns:
        Dictionary with analysis of weather impact
    """
    try:
        if not weather_data:
            return {"impact": "unknown", "description": "No weather data available"}
            
        condition = weather_data.get("condition", "").lower()
        temp = weather_data.get("temperature", 20)
        wind = weather_data.get("wind_speed", 0)
        precipitation = weather_data.get("precipitation", 0)
        
        impact = {
            "overall_impact": "neutral",
            "goal_probability_modifier": 0.0,
            "factors": []
        }
        
        # Analyze temperature
        if temp < 5:
            impact["factors"].append("Cold conditions may affect player performance")
            impact["goal_probability_modifier"] -= 0.05
        elif temp > 30:
            impact["factors"].append("Hot conditions may slow down the game")
            impact["goal_probability_modifier"] -= 0.05
            
        # Analyze precipitation/rain
        if precipitation > 0 or "rain" in condition:
            impact["factors"].append("Wet conditions may lead to more mistakes")
            impact["goal_probability_modifier"] += 0.1
            
        # Analyze wind
        if wind > 20:
            impact["factors"].append("Strong wind may affect long passes and shots")
            impact["goal_probability_modifier"] -= 0.1
            
        # Set overall impact based on modifiers
        if impact["goal_probability_modifier"] > 0.05:
            impact["overall_impact"] = "positive"
        elif impact["goal_probability_modifier"] < -0.05:
            impact["overall_impact"] = "negative"
        return impact
    except Exception as e:
        logger.error(f"Error analyzing weather impact: {e}")
        return {"impact": "unknown", "description": "Error analyzing weather data"}
