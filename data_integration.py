import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
import sqlite3
import json
import requests
from time import sleep
from feature_engineering import AdvancedFeatureEngineering
from weather_api import WeatherConditions
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

def make_request(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Make a request to the API with rate limiting and error handling"""
    base_url = os.getenv('API_BASE_URL', 'https://api-football-v1.p.rapidapi.com/v3')
    headers = {
        "x-rapidapi-host": os.getenv('API_HOST', 'api-football-v1.p.rapidapi.com'),
        "x-rapidapi-key": os.getenv('API_KEY')
    }
    
    if not headers["x-rapidapi-key"]:
        logger.error("API key not found in environment variables")
        return {"errors": "API key not configured"}
    
    try:
        response = requests.get(
            f"{base_url}/{endpoint}",
            headers=headers,
            params=params
        )
        response.raise_for_status()
        
        # Enhanced rate limiting
        remaining = int(response.headers.get('x-ratelimit-remaining', 0))
        if remaining < 10:
            logger.warning(f"Low on API requests: {remaining} remaining")
            sleep(2)
        else:
            sleep(1)
        
        data = response.json()
        return data if isinstance(data, dict) else {"data": data}
        
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return {"errors": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"errors": str(e)}

class DataIntegrator:
    def __init__(self):
        self.feature_engineering = AdvancedFeatureEngineering()
        self.weather_conditions = WeatherConditions()
        
    def prepare_training_data(self, 
                            historical_matches: pd.DataFrame,
                            weather_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Prepares comprehensive training data by combining all available data sources
        """
        # Generate advanced features
        features_df = self.feature_engineering.create_advanced_features(historical_matches)
        
        # Add weather impact if available
        if weather_data is not None:
            features_df = self._integrate_weather_data(features_df, weather_data)
        
        # Add interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        return features_df
    
    def _integrate_weather_data(self, 
                              match_data: pd.DataFrame, 
                              weather_data: pd.DataFrame) -> pd.DataFrame:
        """Integrates weather data with match features with enhanced impact analysis"""
        df = match_data.copy()
        
        # Merge weather data
        df = df.merge(weather_data, on=['fixture_id', 'date'], how='left')
        
        # Calculate base weather impacts
        df['temp_impact'] = df.apply(self._calculate_temperature_impact, axis=1)
        df['wind_impact'] = df.apply(self._calculate_wind_impact, axis=1)
        df['rain_impact'] = df.apply(self._calculate_rain_impact, axis=1)
        
        # Calculate team-specific weather adjustments
        for team_type in ['home', 'away']:
            # Team's historical performance in similar conditions
            df[f'{team_type}_weather_adaptation'] = df.apply(
                lambda x: self._calculate_weather_adaptation(x, team_type), axis=1
            )
            
            # Playing style weather impact
            df[f'{team_type}_style_weather_impact'] = df.apply(
                lambda x: self._calculate_style_weather_impact(x, team_type), axis=1
            )
            
            # Combined weather effect
            df[f'{team_type}_weather_effect'] = (
                df['weather_score'] * 
                df[f'{team_type}_weather_adaptation'] * 
                df[f'{team_type}_style_weather_impact']
            )
        
        return df
        
    def _calculate_weather_adaptation(self, row: pd.Series, team_type: str) -> float:
        """Calculate how well a team adapts to current weather conditions"""
        try:
            # Base adaptation score
            adaptation = 1.0
            
            # Adjust for historical performance in similar conditions
            if row['temp_impact'] < 0.7:  # Cold/hot conditions
                adaptation *= 1 + (row[f'{team_type}_extreme_temp_performance'] - 0.5)
            
            if row['rain_impact'] > 0.3:  # Rainy conditions
                adaptation *= 1 + (row[f'{team_type}_wet_weather_performance'] - 0.5)
            
            if row['wind_impact'] > 0.3:  # Windy conditions
                adaptation *= 1 + (row[f'{team_type}_wind_performance'] - 0.5)
            
            return float(adaptation)
        except Exception as e:
            logger.error(f"Error calculating weather adaptation: {e}")
            return 1.0
            
    def _calculate_style_weather_impact(self, row: pd.Series, team_type: str) -> float:
        """Calculate how weather affects team's playing style"""
        try:
            impact = 1.0
            
            # Impact on possession-based teams
            if row[f'{team_type}_possession_ma'] > 55:
                impact *= (1 - 0.1 * row['rain_impact'])  # Rain affects possession
            
            # Impact on direct play teams
            if row[f'{team_type}_direct_play_index'] > 60:
                impact *= (1 - 0.15 * row['wind_impact'])  # Wind affects long balls
            
            # Impact on high-pressing teams
            if row[f'{team_type}_pressing_intensity'] > 1.2:
                impact *= (1 - 0.1 * (1 - row['temp_impact']))  # Temperature affects pressing
            
            return float(impact)
        except Exception as e:
            logger.error(f"Error calculating style weather impact: {e}")
            return 1.0
    
    def _calculate_temperature_impact(self, row: pd.Series) -> float:
        """Calculate temperature impact on performance"""
        temp = row['temperature']
        optimal_temp = 20  # Optimal playing temperature
        
        # Impact increases as temperature deviates from optimal
        impact = 1 - min(abs(temp - optimal_temp) / 30, 1)  # Normalized to [0,1]
        return impact
    
    def _calculate_wind_impact(self, row: pd.Series) -> float:
        """Calculate wind impact on performance"""
        wind_speed = row['wind_speed']
        
        # Impact increases with wind speed
        impact = 1 - min(wind_speed / 50, 1)  # Normalized to [0,1]
        return impact
    
    def _calculate_rain_impact(self, row: pd.Series) -> float:
        """Calculate rain impact on performance"""
        precipitation = row['precipitation']
        
        # Impact increases with precipitation
        impact = 1 - min(precipitation / 10, 1)  # Normalized to [0,1]
        return impact
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced interaction features between different factors"""
        # Team form × Weather interaction with style consideration
        df['form_weather_interaction'] = df.apply(
            lambda x: (x['home_form'] * x['weather_score'] * x['home_style_weather_impact'] +
                      x['away_form'] * x['weather_score'] * x['away_style_weather_impact']) / 2,
            axis=1
        )
        
        # Team quality × Match importance with form weight
        df['quality_importance_interaction'] = df.apply(
            lambda x: (x['home_avg_rating'] * x['match_importance'] * (1 + x['home_form']) +
                      x['away_avg_rating'] * x['match_importance'] * (1 + x['away_form'])) / 4,
            axis=1
        )
        
        # Weather × Playing style interaction with adaptation
        df['weather_style_interaction'] = df.apply(
            lambda x: (x['weather_score'] * x['home_possession_ma'] * x['home_weather_adaptation'] +
                      x['weather_score'] * x['away_possession_ma'] * x['away_weather_adaptation']) / 2,
            axis=1
        )
        
        # H2H dominance × Current form with rivalry impact
        df['h2h_form_interaction'] = df.apply(
            lambda x: x['h2h_dominance'] * x['home_form'] * (1 + x['rivalry_factor']),
            axis=1
        )
        
        # Match importance × Team momentum
        df['importance_momentum_interaction'] = df.apply(
            lambda x: x['match_importance'] * (x['home_momentum_score'] + x['away_momentum_score']) / 2,
            axis=1
        )
        
        # European competition fatigue × Form
        if 'european_distraction' in df.columns:
            df['european_form_interaction'] = df.apply(
                lambda x: (x['home_form'] * (1 - 0.1 * x['european_distraction']) +
                          x['away_form'] * (1 - 0.1 * x['european_distraction'])) / 2,
                axis=1
            )
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with intelligent imputation"""
        # For numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            # Use median for outlier-sensitive features
            if 'score' in col or 'ratio' in col:
                df[col] = df[col].fillna(df[col].median())
            # Use mean for regular numerical features
            else:
                df[col] = df[col].fillna(df[col].mean())
        
        # For categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            
        return df

    def prepare_prediction_features(self, 
                                  match_data: Dict,
                                  historical_data: pd.DataFrame,
                                  weather_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepares features for a single match prediction
        """
        # Convert match data to DataFrame
        match_df = pd.DataFrame([match_data])
        
        # Add historical context
        full_df = pd.concat([historical_data, match_df])
        
        # Generate advanced features
        features = self.feature_engineering.create_advanced_features(full_df).tail(1)
        
        # Add weather data if available
        if weather_data is not None:
            weather_df = pd.DataFrame([weather_data])
            features = self._integrate_weather_data(features, weather_df)
        
        # Add interaction features
        features = self._add_interaction_features(features)
        
        # Handle any missing values
        features = self._handle_missing_values(features)
        
        return features

def prepare_batch_predictions(matches: List[Dict],
                            historical_data: pd.DataFrame,
                            weather_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Prepares features for batch prediction of multiple matches
    """
    integrator = DataIntegrator()
    
    # Convert matches to DataFrame
    matches_df = pd.DataFrame(matches)
    
    # Prepare features for all matches
    features = integrator.prepare_training_data(
        pd.concat([historical_data, matches_df]),
        weather_data
    ).tail(len(matches))
    
    return features

def get_team_stats_from_db(team_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Obtiene estadísticas históricas de un equipo desde la base de datos
    """
    try:
        with sqlite3.connect('cache/football_data.db') as conn:
            query = """
                SELECT m.*
                FROM match_statistics m
                WHERE m.home_team_id = ? OR m.away_team_id = ?
                ORDER BY m.timestamp DESC
                LIMIT ?
            """
            
            cursor = conn.execute(query, (team_id, team_id, limit))
            matches = []
            
            for row in cursor.fetchall():
                fixture_id, _, _, home_team_id, away_team_id, home_goals, away_goals, home_stats_json, away_stats_json, _ = row
                
                # Convertir JSON a diccionarios
                home_stats = json.loads(home_stats_json)
                away_stats = json.loads(away_stats_json)
                
                # Determinar si el equipo jugó como local o visitante
                is_home = home_team_id == team_id
                team_stats = home_stats if is_home else away_stats
                opponent_stats = away_stats if is_home else home_stats
                
                # Calcular estadísticas específicas del equipo
                match_stats = {
                    "corners_for": team_stats.get("corner_kicks", 5),
                    "corners_against": opponent_stats.get("corner_kicks", 4),
                    "cards": team_stats.get("yellow_cards", 0) + team_stats.get("red_cards", 0),
                    "fouls": team_stats.get("fouls", 10),
                    "possession": team_stats.get("ball_possession", 50),
                    "shots": team_stats.get("total_shots", 12),
                    "shots_on_target": team_stats.get("shots_on_goal", 4),
                    "goals_scored": home_goals if is_home else away_goals,
                    "goals_conceded": away_goals if is_home else home_goals
                }
                
                matches.append(match_stats)
            
            return matches
            
    except Exception as e:
        logger.error(f"Error getting team stats from DB for team {team_id}: {e}")
        return []

def get_player_statistics(player_id: int, league_id: int, season: int) -> Dict[str, Any]:
    """Get detailed player statistics"""
    endpoint = "players/statistics"
    params = {
        "player": player_id,
        "league": league_id,
        "season": season
    }
    return make_request(endpoint, params)

def get_team_injuries(team_id: int, league_id: int) -> Dict[str, Any]:
    """Get current team injuries"""
    endpoint = "injuries"
    params = {
        "team": team_id,
        "league": league_id
    }
    return make_request(endpoint, params)

def get_fixture_events(fixture_id: int) -> Dict[str, Any]:
    """Get detailed match events"""
    endpoint = "fixtures/events"
    params = {"fixture": fixture_id}
    return make_request(endpoint, params)

def get_live_odds(fixture_id: Optional[int] = None) -> Dict[str, Any]:
    """Get live odds for ongoing matches"""
    endpoint = "odds/live"
    params = {"fixture": fixture_id} if fixture_id else {}
    return make_request(endpoint, params)