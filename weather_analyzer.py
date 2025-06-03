"""
Weather Impact Analysis Module

This module analyzes weather conditions and their impact on soccer match outcomes.
It integrates weather data, historical performance under different conditions,
and provides weather-adjusted prediction features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import requests
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

@dataclass
class WeatherCondition:
    """Weather condition data structure"""
    temperature: float  # Celsius
    humidity: float  # Percentage
    wind_speed: float  # km/h
    precipitation: float  # mm
    pressure: float  # hPa
    visibility: float  # km
    condition: str  # sunny, cloudy, rainy, snowy, etc.
    weather_severity: float  # 0-1 scale

@dataclass
class WeatherImpact:
    """Weather impact analysis results"""
    temperature_impact: float
    precipitation_impact: float
    wind_impact: float
    visibility_impact: float
    overall_impact: float
    favored_team: Optional[str]  # home/away/none
    impact_confidence: float

class WeatherAnalyzer:
    """
    Analyzes weather impact on soccer match outcomes
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.scaler = StandardScaler()
        self.weather_cache = {}
        
        # Weather impact parameters
        self.temperature_thresholds = {
            'extreme_cold': -5,  # Below -5°C
            'cold': 5,          # 5°C and below  
            'cool': 15,         # 5-15°C
            'optimal': 25,      # 15-25°C
            'hot': 35,          # 25-35°C
            'extreme_hot': 35   # Above 35°C
        }
        
        self.wind_thresholds = {
            'calm': 10,         # 0-10 km/h
            'moderate': 25,     # 10-25 km/h
            'strong': 40,       # 25-40 km/h
            'very_strong': 40   # 40+ km/h
        }
        
        # Team playing style preferences (would be loaded from DB)
        self.team_weather_preferences = {}
        
    def get_weather_forecast(self, latitude: float, longitude: float, 
                           match_datetime: datetime) -> Optional[WeatherCondition]:
        """
        Get weather forecast for match location and time
        
        Args:
            latitude: Venue latitude
            longitude: Venue longitude
            match_datetime: Match date and time
            
        Returns:
            WeatherCondition object or None if unavailable
        """
        try:
            # Check cache first
            cache_key = f"{latitude}_{longitude}_{match_datetime.isoformat()}"
            if cache_key in self.weather_cache:
                return self.weather_cache[cache_key]
            
            # Get weather data (using OpenWeatherMap API format)
            weather_data = self._fetch_weather_data(latitude, longitude, match_datetime)
            
            if weather_data:
                weather_condition = self._parse_weather_data(weather_data)
                self.weather_cache[cache_key] = weather_condition
                return weather_condition
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return None
    
    def analyze_weather_impact(self, weather: WeatherCondition, 
                             home_team_id: int, away_team_id: int) -> WeatherImpact:
        """
        Analyze weather impact on match outcome
        
        Args:
            weather: Weather conditions
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            WeatherImpact analysis
        """
        try:
            # Analyze individual weather factors
            temp_impact = self._analyze_temperature_impact(weather.temperature)
            precip_impact = self._analyze_precipitation_impact(weather.precipitation)
            wind_impact = self._analyze_wind_impact(weather.wind_speed)
            visibility_impact = self._analyze_visibility_impact(weather.visibility)
            
            # Get team-specific weather preferences
            home_preference = self._get_team_weather_preference(home_team_id, weather)
            away_preference = self._get_team_weather_preference(away_team_id, weather)
            
            # Determine overall impact and favored team
            overall_impact = self._calculate_overall_impact(
                temp_impact, precip_impact, wind_impact, visibility_impact
            )
            
            favored_team = self._determine_favored_team(
                home_preference, away_preference, overall_impact
            )
            
            # Calculate confidence based on weather severity and historical data
            confidence = self._calculate_impact_confidence(weather, overall_impact)
            
            return WeatherImpact(
                temperature_impact=temp_impact,
                precipitation_impact=precip_impact,
                wind_impact=wind_impact,
                visibility_impact=visibility_impact,
                overall_impact=overall_impact,
                favored_team=favored_team,
                impact_confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error analyzing weather impact: {e}")
            return self._default_weather_impact()
    
    def get_weather_features(self, latitude: float, longitude: float,
                           match_datetime: datetime, home_team_id: int, 
                           away_team_id: int) -> Dict[str, float]:
        """
        Generate weather-related features for match prediction
        
        Args:
            latitude: Venue latitude
            longitude: Venue longitude
            match_datetime: Match date and time
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            
        Returns:
            Dict containing weather features
        """
        try:
            # Get weather forecast
            weather = self.get_weather_forecast(latitude, longitude, match_datetime)
            
            if not weather:
                return self._default_weather_features()
            
            # Analyze weather impact
            impact = self.analyze_weather_impact(weather, home_team_id, away_team_id)
            
            # Generate features
            features = {
                # Raw weather conditions
                'temperature': weather.temperature,
                'humidity': weather.humidity / 100.0,  # Normalize to 0-1
                'wind_speed': min(weather.wind_speed / 50.0, 1.0),  # Normalize, cap at 50 km/h
                'precipitation': min(weather.precipitation / 20.0, 1.0),  # Normalize, cap at 20mm
                'pressure': (weather.pressure - 980) / 60.0,  # Normalize around typical range
                'visibility': min(weather.visibility / 10.0, 1.0),  # Normalize, cap at 10km
                
                # Weather condition categories (one-hot encoding)
                'is_sunny': 1.0 if 'sun' in weather.condition.lower() else 0.0,
                'is_cloudy': 1.0 if 'cloud' in weather.condition.lower() else 0.0,
                'is_rainy': 1.0 if 'rain' in weather.condition.lower() else 0.0,
                'is_snowy': 1.0 if 'snow' in weather.condition.lower() else 0.0,
                
                # Weather severity and comfort
                'weather_severity': weather.weather_severity,
                'temperature_comfort': self._calculate_temperature_comfort(weather.temperature),
                'playing_conditions': self._calculate_playing_conditions_score(weather),
                
                # Impact features
                'temperature_impact': impact.temperature_impact,
                'precipitation_impact': impact.precipitation_impact,
                'wind_impact': impact.wind_impact,
                'visibility_impact': impact.visibility_impact,
                'overall_weather_impact': impact.overall_impact,
                
                # Team advantage features
                'home_weather_advantage': 1.0 if impact.favored_team == 'home' else 0.0,
                'away_weather_advantage': 1.0 if impact.favored_team == 'away' else 0.0,
                'weather_neutral': 1.0 if impact.favored_team == 'none' else 0.0,
                'weather_impact_confidence': impact.impact_confidence,
                
                # Derived features
                'extreme_weather': 1.0 if weather.weather_severity > 0.7 else 0.0,
                'adverse_conditions': 1.0 if (weather.precipitation > 5 or 
                                             weather.wind_speed > 30 or 
                                             weather.temperature < 0) else 0.0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating weather features: {e}")
            return self._default_weather_features()
    
    def analyze_historical_weather_performance(self, team_id: int, 
                                             weather_type: str) -> Dict[str, float]:
        """
        Analyze team's historical performance under specific weather conditions
        
        Args:
            team_id: Team identifier
            weather_type: Type of weather condition
            
        Returns:
            Dict containing performance metrics
        """
        try:
            # This would query historical match data with weather conditions
            # For now, simulate based on typical patterns
            
            performance_patterns = {
                'sunny': {'win_rate': 0.45, 'draw_rate': 0.25, 'goals_per_game': 1.8},
                'rainy': {'win_rate': 0.35, 'draw_rate': 0.35, 'goals_per_game': 1.4},
                'windy': {'win_rate': 0.38, 'draw_rate': 0.32, 'goals_per_game': 1.5},
                'cold': {'win_rate': 0.40, 'draw_rate': 0.30, 'goals_per_game': 1.6},
                'hot': {'win_rate': 0.42, 'draw_rate': 0.28, 'goals_per_game': 1.7}
            }
            
            base_performance = performance_patterns.get(weather_type, 
                                                       performance_patterns['sunny'])
            
            # Add some team-specific variation
            variation = np.random.uniform(-0.05, 0.05)
            
            return {
                'win_rate': max(0.0, min(1.0, base_performance['win_rate'] + variation)),
                'draw_rate': max(0.0, min(1.0, base_performance['draw_rate'] + variation)),
                'goals_per_game': max(0.0, base_performance['goals_per_game'] + variation),
                'matches_played': np.random.randint(10, 50)  # Simulate sample size
            }
            
        except Exception as e:
            logger.error(f"Error analyzing historical weather performance: {e}")
            return {'win_rate': 0.4, 'draw_rate': 0.3, 'goals_per_game': 1.6, 'matches_played': 20}
    
    def _fetch_weather_data(self, latitude: float, longitude: float, 
                          match_datetime: datetime) -> Optional[Dict]:
        """Fetch weather data from API"""
        try:
            if not self.api_key:
                # Return mock data for testing
                return self._generate_mock_weather_data()
            
            # OpenWeatherMap API call (example)
            url = f"http://api.openweathermap.org/data/2.5/forecast"
            params = {
                'lat': latitude,
                'lon': longitude,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching weather data: {e}")
            return self._generate_mock_weather_data()
    
    def _parse_weather_data(self, weather_data: Dict) -> WeatherCondition:
        """Parse weather API response into WeatherCondition"""
        try:
            # Parse OpenWeatherMap response format
            if 'list' in weather_data and weather_data['list']:
                forecast = weather_data['list'][0]  # Get first forecast
                main = forecast.get('main', {})
                weather_desc = forecast.get('weather', [{}])[0]
                wind = forecast.get('wind', {})
                
                temp = main.get('temp', 20)
                humidity = main.get('humidity', 60)
                pressure = main.get('pressure', 1013)
                wind_speed = wind.get('speed', 0) * 3.6  # Convert m/s to km/h
                precipitation = forecast.get('rain', {}).get('3h', 0)
                visibility = forecast.get('visibility', 10000) / 1000  # Convert m to km
                condition = weather_desc.get('description', 'clear')
                
                # Calculate weather severity
                severity = self._calculate_weather_severity(
                    temp, humidity, wind_speed, precipitation, visibility
                )
                
                return WeatherCondition(
                    temperature=temp,
                    humidity=humidity,
                    wind_speed=wind_speed,
                    precipitation=precipitation,
                    pressure=pressure,
                    visibility=visibility,
                    condition=condition,
                    weather_severity=severity
                )
            else:
                return self._default_weather_condition()
                
        except Exception as e:
            logger.error(f"Error parsing weather data: {e}")
            return self._default_weather_condition()
    
    def _calculate_weather_severity(self, temp: float, humidity: float, 
                                  wind_speed: float, precipitation: float, 
                                  visibility: float) -> float:
        """Calculate overall weather severity score (0-1)"""
        severity_factors = []
        
        # Temperature severity
        if temp < -5 or temp > 35:
            severity_factors.append(0.8)
        elif temp < 0 or temp > 30:
            severity_factors.append(0.6)
        elif temp < 5 or temp > 25:
            severity_factors.append(0.3)
        else:
            severity_factors.append(0.1)
        
        # Precipitation severity
        if precipitation > 10:
            severity_factors.append(0.9)
        elif precipitation > 5:
            severity_factors.append(0.6)
        elif precipitation > 1:
            severity_factors.append(0.3)
        else:
            severity_factors.append(0.0)
        
        # Wind severity
        if wind_speed > 40:
            severity_factors.append(0.8)
        elif wind_speed > 25:
            severity_factors.append(0.5)
        elif wind_speed > 15:
            severity_factors.append(0.2)
        else:
            severity_factors.append(0.0)
        
        # Visibility severity
        if visibility < 1:
            severity_factors.append(0.9)
        elif visibility < 3:
            severity_factors.append(0.6)
        elif visibility < 5:
            severity_factors.append(0.3)
        else:
            severity_factors.append(0.0)
          # Humidity severity (extreme values)
        if humidity > 90 or humidity < 20:
            severity_factors.append(0.4)
        else:
            severity_factors.append(0.0)
        
        return min(1.0, float(np.mean(severity_factors)) * 1.2)
    
    def _analyze_temperature_impact(self, temperature: float) -> float:
        """Analyze temperature impact on match (0-1 scale)"""
        if temperature < -5:
            return 0.8  # Extreme cold impact
        elif temperature < 0:
            return 0.6  # Cold impact
        elif temperature < 5:
            return 0.4  # Cool impact
        elif temperature > 35:
            return 0.8  # Extreme heat impact
        elif temperature > 30:
            return 0.6  # Hot impact
        elif temperature > 25:
            return 0.3  # Warm impact
        else:
            return 0.1  # Optimal temperature range
    
    def _analyze_precipitation_impact(self, precipitation: float) -> float:
        """Analyze precipitation impact on match (0-1 scale)"""
        if precipitation > 15:
            return 0.9  # Heavy rain/snow
        elif precipitation > 10:
            return 0.7  # Moderate rain
        elif precipitation > 5:
            return 0.5  # Light rain
        elif precipitation > 1:
            return 0.3  # Drizzle
        else:
            return 0.0  # No precipitation
    
    def _analyze_wind_impact(self, wind_speed: float) -> float:
        """Analyze wind impact on match (0-1 scale)"""
        if wind_speed > 40:
            return 0.8  # Very strong wind
        elif wind_speed > 25:
            return 0.6  # Strong wind
        elif wind_speed > 15:
            return 0.3  # Moderate wind
        else:
            return 0.1  # Calm conditions
    
    def _analyze_visibility_impact(self, visibility: float) -> float:
        """Analyze visibility impact on match (0-1 scale)"""
        if visibility < 1:
            return 0.9  # Very poor visibility
        elif visibility < 3:
            return 0.7  # Poor visibility
        elif visibility < 5:
            return 0.4  # Reduced visibility
        else:
            return 0.0  # Good visibility
    
    def _calculate_overall_impact(self, temp_impact: float, precip_impact: float,
                                wind_impact: float, visibility_impact: float) -> float:
        """Calculate overall weather impact"""
        # Weighted combination of individual impacts
        weights = [0.25, 0.35, 0.25, 0.15]  # precipitation has highest weight
        impacts = [temp_impact, precip_impact, wind_impact, visibility_impact]
        
        return min(1.0, sum(w * i for w, i in zip(weights, impacts)))
    
    def _get_team_weather_preference(self, team_id: int, weather: WeatherCondition) -> float:
        """Get team's preference/performance under current weather conditions"""
        # This would load from database in real implementation
        # For now, simulate based on team characteristics
        
        if team_id in self.team_weather_preferences:
            preferences = self.team_weather_preferences[team_id]
        else:
            # Generate random preferences for simulation
            preferences = {
                'cold_tolerance': np.random.uniform(0.3, 0.8),
                'heat_tolerance': np.random.uniform(0.3, 0.8),
                'rain_performance': np.random.uniform(0.2, 0.7),
                'wind_adaptation': np.random.uniform(0.3, 0.7)
            }
            self.team_weather_preferences[team_id] = preferences
        
        # Calculate preference score based on current weather
        preference_score = 0.5  # Neutral baseline
        
        if weather.temperature < 5:
            preference_score = preferences['cold_tolerance']
        elif weather.temperature > 25:
            preference_score = preferences['heat_tolerance']
        
        if weather.precipitation > 5:
            preference_score *= preferences['rain_performance']
        
        if weather.wind_speed > 20:
            preference_score *= preferences['wind_adaptation']
        
        return preference_score
    
    def _determine_favored_team(self, home_preference: float, away_preference: float,
                              overall_impact: float) -> str:
        """Determine which team is favored by weather conditions"""
        preference_diff = home_preference - away_preference
        
        # Only declare advantage if difference is significant and weather has impact
        if overall_impact > 0.3 and abs(preference_diff) > 0.15:
            return 'home' if preference_diff > 0 else 'away'
        else:
            return 'none'
    
    def _calculate_impact_confidence(self, weather: WeatherCondition, 
                                   overall_impact: float) -> float:
        """Calculate confidence in weather impact assessment"""
        # Higher confidence for more extreme weather conditions
        base_confidence = min(1.0, weather.weather_severity + 0.3)
        
        # Adjust based on overall impact magnitude
        impact_confidence = min(1.0, overall_impact + 0.4)
        
        return (base_confidence + impact_confidence) / 2.0
    
    def _calculate_temperature_comfort(self, temperature: float) -> float:
        """Calculate temperature comfort score (0-1, higher = more comfortable)"""
        optimal_temp = 20  # Optimal temperature for soccer
        temp_diff = abs(temperature - optimal_temp)
        
        if temp_diff <= 5:
            return 1.0
        elif temp_diff <= 10:
            return 0.8
        elif temp_diff <= 15:
            return 0.6
        elif temp_diff <= 20:
            return 0.4
        else:
            return 0.2
    
    def _calculate_playing_conditions_score(self, weather: WeatherCondition) -> float:
        """Calculate overall playing conditions score (0-1)"""
        temp_score = self._calculate_temperature_comfort(weather.temperature)
        
        # Precipitation score
        if weather.precipitation == 0:
            precip_score = 1.0
        elif weather.precipitation <= 2:
            precip_score = 0.8
        elif weather.precipitation <= 5:
            precip_score = 0.6
        elif weather.precipitation <= 10:
            precip_score = 0.4
        else:
            precip_score = 0.2
        
        # Wind score
        if weather.wind_speed <= 10:
            wind_score = 1.0
        elif weather.wind_speed <= 20:
            wind_score = 0.8
        elif weather.wind_speed <= 30:
            wind_score = 0.6
        else:
            wind_score = 0.4
        
        # Visibility score
        visibility_score = min(1.0, weather.visibility / 10.0)
        
        # Weighted average
        return (temp_score * 0.3 + precip_score * 0.4 + 
                wind_score * 0.2 + visibility_score * 0.1)
    
    def _generate_mock_weather_data(self) -> Dict:
        """Generate mock weather data for testing"""
        return {
            'list': [{
                'main': {
                    'temp': np.random.uniform(5, 25),
                    'humidity': np.random.uniform(40, 80),
                    'pressure': np.random.uniform(995, 1025)
                },
                'weather': [{
                    'description': np.random.choice(['clear sky', 'few clouds', 'scattered clouds', 
                                                   'light rain', 'moderate rain'])
                }],
                'wind': {
                    'speed': np.random.uniform(0, 8)  # m/s
                },
                'rain': {
                    '3h': np.random.uniform(0, 5) if np.random.random() > 0.7 else 0
                },
                'visibility': np.random.uniform(5000, 10000)  # meters
            }]
        }
    
    def _default_weather_condition(self) -> WeatherCondition:
        """Default weather condition when data is unavailable"""
        return WeatherCondition(
            temperature=18.0,
            humidity=60.0,
            wind_speed=10.0,
            precipitation=0.0,
            pressure=1013.0,
            visibility=10.0,
            condition='clear',
            weather_severity=0.1
        )
    
    def _default_weather_impact(self) -> WeatherImpact:
        """Default weather impact when analysis fails"""
        return WeatherImpact(
            temperature_impact=0.1,
            precipitation_impact=0.0,
            wind_impact=0.1,
            visibility_impact=0.0,
            overall_impact=0.1,
            favored_team='none',
            impact_confidence=0.5
        )
    
    def _default_weather_features(self) -> Dict[str, float]:
        """Default weather features when data is unavailable"""
        return {
            'temperature': 0.6,  # Normalized ~18°C
            'humidity': 0.6,
            'wind_speed': 0.2,
            'precipitation': 0.0,
            'pressure': 0.55,  # Normalized ~1013 hPa
            'visibility': 1.0,
            'is_sunny': 1.0,
            'is_cloudy': 0.0,
            'is_rainy': 0.0,
            'is_snowy': 0.0,
            'weather_severity': 0.1,
            'temperature_comfort': 0.9,
            'playing_conditions': 0.9,
            'temperature_impact': 0.1,
            'precipitation_impact': 0.0,
            'wind_impact': 0.1,
            'visibility_impact': 0.0,
            'overall_weather_impact': 0.1,
            'home_weather_advantage': 0.0,
            'away_weather_advantage': 0.0,
            'weather_neutral': 1.0,
            'weather_impact_confidence': 0.5,
            'extreme_weather': 0.0,
            'adverse_conditions': 0.0
        }

def demonstrate_weather_analysis():
    """Demonstrate weather analysis functionality"""
    print("=== Weather Impact Analysis Demo ===")
    
    analyzer = WeatherAnalyzer()
    
    # Get weather forecast
    print("\n1. Weather Forecast:")
    weather = analyzer.get_weather_forecast(
        latitude=51.5074, 
        longitude=-0.1278, 
        match_datetime=datetime.now() + timedelta(days=1)
    )
    if weather:
        print(f"   Temperature: {weather.temperature}°C")
        print(f"   Humidity: {weather.humidity}%")
        print(f"   Wind Speed: {weather.wind_speed} km/h")
        print(f"   Precipitation: {weather.precipitation} mm")
        print(f"   Condition: {weather.condition}")
        print(f"   Severity: {weather.weather_severity:.2f}")
    
    # Analyze weather impact
    print("\n2. Weather Impact Analysis:")
    if weather:
        impact = analyzer.analyze_weather_impact(weather, home_team_id=123, away_team_id=456)
        print(f"   Temperature Impact: {impact.temperature_impact:.2f}")
        print(f"   Precipitation Impact: {impact.precipitation_impact:.2f}")
        print(f"   Wind Impact: {impact.wind_impact:.2f}")
        print(f"   Overall Impact: {impact.overall_impact:.2f}")
        print(f"   Favored Team: {impact.favored_team}")
        print(f"   Confidence: {impact.impact_confidence:.2f}")
    
    # Generate weather features
    print("\n3. Weather Features for Prediction:")
    features = analyzer.get_weather_features(
        latitude=51.5074,
        longitude=-0.1278,
        match_datetime=datetime.now() + timedelta(days=1),
        home_team_id=123,
        away_team_id=456
    )
    for feature, value in list(features.items())[:10]:  # Show first 10 features
        print(f"   {feature}: {value:.3f}")
    print(f"   ... and {len(features) - 10} more features")

if __name__ == "__main__":
    demonstrate_weather_analysis()
