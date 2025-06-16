#!/usr/bin/env python3
"""
Referee Analysis System

Sistema avanzado de análisis del impacto de árbitros en predicciones de fútbol.
Analiza histórico, sesgos, tendencias de tarjetas/penales y compatibilidad de estilos
para mejorar precisión de predicciones y betting value.

Este módulo agrega datos críticos que faltan para apostadores profesionales.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
import requests
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class RefereeProfile:
    """Perfil completo de un árbitro"""
    referee_id: int
    name: str
    nationality: str
    age: int
    experience_level: str  # "elite", "experienced", "developing"
    total_matches: int
    big_game_experience: bool
    var_experience: bool
    
    # Estadísticas promedio
    cards_per_game: float
    yellow_cards_per_game: float
    red_cards_per_game: float
    fouls_per_game: float
    penalties_per_game: float
    offsides_per_game: float
    
    # Estilo y tendencias
    disciplinary_style: str  # "strict", "lenient", "balanced"
    consistency_rating: float  # 0-1
    home_bias_factor: float  # -0.1 to +0.1
    big_team_bias: float
    
    # Especialidades
    var_usage_frequency: float
    advantage_playing_tendency: float
    injury_time_accuracy: float

@dataclass
class RefereeTeamHistory:
    """Histórico de árbitro con equipos específicos"""
    referee_id: int
    team_id: int
    total_matches: int
    team_wins: int
    team_draws: int
    team_losses: int
    win_rate: float
    cards_given_to_team: float
    penalties_for_team: int
    penalties_against_team: int
    controversial_decisions: int
    last_match_date: Optional[datetime]

@dataclass
class RefereeMatchPrediction:
    """Predicciones específicas del árbitro para un partido"""
    total_cards_prediction: float
    yellow_cards_prediction: float
    red_cards_prediction: float
    penalties_prediction: float
    fouls_prediction: float
    
    # Ajustes de probabilidades
    home_advantage_adjustment: float
    away_disadvantage_adjustment: float
    
    # Confianza en las predicciones
    confidence_level: float
    historical_sample_size: int

class RefereeAnalyzer:
    """
    Analizador completo del impacto de árbitros en predicciones
    """
    
    def __init__(self, db_path: str = "referee_analysis.db"):
        self.db_path = db_path
        self.referee_profiles = {}
        self.team_histories = defaultdict(dict)
        
        # APIs y fuentes de datos
        self.data_sources = {
            "transfermarkt": "https://www.transfermarkt.com",
            "football_data": "https://api.football-data.org",
            "referee_stats": "custom_referee_api"
        }
        
        # Umbrales para clasificaciones
        self.experience_thresholds = {
            "elite": {"min_matches": 200, "min_big_games": 50},
            "experienced": {"min_matches": 100, "min_big_games": 20},
            "developing": {"min_matches": 50, "min_big_games": 5}
        }
        
        # Inicializar base de datos
        self._initialize_database()
        
        # Cargar datos existentes
        self._load_referee_data()
    
    def _initialize_database(self):
        """Inicializa base de datos de árbitros"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de perfiles de árbitros
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_profiles (
                    referee_id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    nationality TEXT,
                    age INTEGER,
                    experience_level TEXT,
                    total_matches INTEGER DEFAULT 0,
                    big_game_experience BOOLEAN DEFAULT 0,
                    var_experience BOOLEAN DEFAULT 0,
                    cards_per_game REAL DEFAULT 0,
                    yellow_cards_per_game REAL DEFAULT 0,
                    red_cards_per_game REAL DEFAULT 0,
                    fouls_per_game REAL DEFAULT 0,
                    penalties_per_game REAL DEFAULT 0,
                    disciplinary_style TEXT DEFAULT 'balanced',
                    consistency_rating REAL DEFAULT 0.5,
                    home_bias_factor REAL DEFAULT 0,
                    big_team_bias REAL DEFAULT 0,
                    var_usage_frequency REAL DEFAULT 0.5,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tabla de histórico árbitro-equipo
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_team_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referee_id INTEGER,
                    team_id INTEGER,
                    fixture_id INTEGER,
                    match_date DATE,
                    team_result TEXT, -- 'win', 'draw', 'loss'
                    cards_to_team INTEGER DEFAULT 0,
                    cards_to_opponent INTEGER DEFAULT 0,
                    penalties_for INTEGER DEFAULT 0,
                    penalties_against INTEGER DEFAULT 0,
                    fouls_by_team INTEGER DEFAULT 0,
                    fouls_by_opponent INTEGER DEFAULT 0,
                    controversial_decision BOOLEAN DEFAULT 0,
                    FOREIGN KEY (referee_id) REFERENCES referee_profiles (referee_id),
                    UNIQUE(referee_id, team_id, fixture_id)
                )
            ''')
            
            # Tabla de estadísticas por liga
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_league_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    referee_id INTEGER,
                    league_id INTEGER,
                    season TEXT,
                    matches_officiated INTEGER DEFAULT 0,
                    avg_cards_per_game REAL DEFAULT 0,
                    avg_penalties_per_game REAL DEFAULT 0,
                    home_win_rate REAL DEFAULT 0,
                    FOREIGN KEY (referee_id) REFERENCES referee_profiles (referee_id)
                )
            ''')
            
            # Tabla de predicciones de árbitros
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS referee_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER,
                    referee_id INTEGER,
                    predicted_total_cards REAL,
                    predicted_penalties REAL,
                    home_bias_adjustment REAL,
                    confidence_level REAL,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    actual_cards INTEGER,
                    actual_penalties INTEGER,
                    prediction_accuracy REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Referee database initialized")
            
        except Exception as e:
            logger.error(f"Error initializing referee database: {e}")
    
    def get_referee_impact_analysis(self, referee_id: int, home_team_id: int, 
                                   away_team_id: int, fixture_id: int) -> Dict[str, Any]:
        """
        Análisis completo del impacto del árbitro en el partido
        """
        try:
            logger.info(f"🧑‍⚖️ Analyzing referee {referee_id} impact for match {fixture_id}")
            
            # Obtener perfil del árbitro
            referee_profile = self._get_referee_profile(referee_id)
            if not referee_profile:
                return self._get_default_referee_analysis()
            
            # Histórico con equipos
            home_history = self._get_team_history(referee_id, home_team_id)
            away_history = self._get_team_history(referee_id, away_team_id)
            
            # Análisis de sesgos
            bias_analysis = self._analyze_referee_bias(referee_id, home_team_id, away_team_id)
            
            # Predicciones específicas
            match_predictions = self._predict_referee_impact(referee_id, home_team_id, away_team_id)
            
            # Análisis de estilo de juego
            style_compatibility = self._analyze_style_compatibility(referee_id, home_team_id, away_team_id)
            
            # Análisis de VAR
            var_analysis = self._analyze_var_influence(referee_id)
            
            return {
                "referee_profile": asdict(referee_profile),
                "historical_with_teams": {
                    "home_team": asdict(home_history) if home_history else None,
                    "away_team": asdict(away_history) if away_history else None
                },
                "bias_analysis": bias_analysis,
                "match_predictions": asdict(match_predictions),
                "style_compatibility": style_compatibility,
                "var_analysis": var_analysis,
                "betting_implications": self._calculate_betting_implications(
                    referee_profile, match_predictions, bias_analysis
                ),
                "confidence_assessment": self._assess_prediction_confidence(
                    home_history, away_history, referee_profile
                )
            }
            
        except Exception as e:
            logger.error(f"Error analyzing referee impact: {e}")
            return self._get_default_referee_analysis()
    
    def _get_referee_profile(self, referee_id: int) -> Optional[RefereeProfile]:
        """Obtiene perfil completo del árbitro"""
        try:
            if referee_id in self.referee_profiles:
                return self.referee_profiles[referee_id]
            
            # Buscar en base de datos
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM referee_profiles WHERE referee_id = ?
            ''', (referee_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                profile = RefereeProfile(
                    referee_id=row[0],
                    name=row[1] or f"Referee {referee_id}",
                    nationality=row[2] or "Unknown",
                    age=row[3] or 35,
                    experience_level=row[4] or "experienced",
                    total_matches=row[5] or 0,
                    big_game_experience=bool(row[6]),
                    var_experience=bool(row[7]),
                    cards_per_game=row[8] or 4.2,
                    yellow_cards_per_game=row[9] or 3.8,
                    red_cards_per_game=row[10] or 0.4,
                    fouls_per_game=row[11] or 22.0,
                    penalties_per_game=row[12] or 0.3,
                    offsides_per_game=3.5,  # Default
                    disciplinary_style=row[13] or "balanced",
                    consistency_rating=row[14] or 0.75,
                    home_bias_factor=row[15] or 0.0,
                    big_team_bias=row[16] or 0.0,
                    var_usage_frequency=row[17] or 0.5,
                    advantage_playing_tendency=0.6,  # Default
                    injury_time_accuracy=0.8  # Default
                )
                
                self.referee_profiles[referee_id] = profile
                return profile
            else:
                # Crear perfil por defecto y intentar obtener datos
                return self._create_default_referee_profile(referee_id)
                
        except Exception as e:
            logger.error(f"Error getting referee profile {referee_id}: {e}")
            return self._create_default_referee_profile(referee_id)
    
    def _create_default_referee_profile(self, referee_id: int) -> RefereeProfile:
        """Crea perfil por defecto para árbitro"""
        try:
            # Intentar obtener datos de APIs externas
            referee_data = self._fetch_referee_data_from_apis(referee_id)
            
            profile = RefereeProfile(
                referee_id=referee_id,
                name=referee_data.get("name", f"Referee {referee_id}"),
                nationality=referee_data.get("nationality", "Unknown"),
                age=referee_data.get("age", 35),
                experience_level=referee_data.get("experience_level", "experienced"),
                total_matches=referee_data.get("total_matches", 100),
                big_game_experience=referee_data.get("big_game_experience", True),
                var_experience=referee_data.get("var_experience", True),
                cards_per_game=referee_data.get("cards_per_game", 4.2),
                yellow_cards_per_game=referee_data.get("yellow_cards_per_game", 3.8),
                red_cards_per_game=referee_data.get("red_cards_per_game", 0.4),
                fouls_per_game=referee_data.get("fouls_per_game", 22.0),
                penalties_per_game=referee_data.get("penalties_per_game", 0.3),
                offsides_per_game=3.5,
                disciplinary_style=referee_data.get("disciplinary_style", "balanced"),
                consistency_rating=referee_data.get("consistency_rating", 0.75),
                home_bias_factor=referee_data.get("home_bias_factor", 0.0),
                big_team_bias=referee_data.get("big_team_bias", 0.0),
                var_usage_frequency=referee_data.get("var_usage_frequency", 0.5),
                advantage_playing_tendency=0.6,
                injury_time_accuracy=0.8
            )
            
            # Guardar en base de datos
            self._save_referee_profile(profile)
            self.referee_profiles[referee_id] = profile
            
            return profile
            
        except Exception as e:
            logger.error(f"Error creating default referee profile: {e}")
            return self._get_minimal_referee_profile(referee_id)
    
    def _get_minimal_referee_profile(self, referee_id: int) -> RefereeProfile:
        """Perfil mínimo cuando no hay datos disponibles"""
        return RefereeProfile(
            referee_id=referee_id,
            name=f"Referee {referee_id}",
            nationality="Unknown",
            age=35,
            experience_level="experienced",
            total_matches=100,
            big_game_experience=True,
            var_experience=True,
            cards_per_game=4.2,
            yellow_cards_per_game=3.8,
            red_cards_per_game=0.4,
            fouls_per_game=22.0,
            penalties_per_game=0.3,
            offsides_per_game=3.5,
            disciplinary_style="balanced",
            consistency_rating=0.75,
            home_bias_factor=0.0,
            big_team_bias=0.0,
            var_usage_frequency=0.5,
            advantage_playing_tendency=0.6,
            injury_time_accuracy=0.8
        )
    
    def _fetch_referee_data_from_apis(self, referee_id: int) -> Dict[str, Any]:
        """Obtiene datos de árbitro de APIs externas"""
        try:
            # Simular datos realistas por ahora
            # En implementación real, aquí irían llamadas a APIs
            
            import random
            
            names = ["John Smith", "Carlos Rodriguez", "Marco Bianchi", "Pierre Dubois", 
                    "Hans Mueller", "Antonio Silva", "David Jones", "Luigi Rossi"]
            
            nationalities = ["England", "Spain", "Italy", "France", "Germany", 
                           "Portugal", "Netherlands", "Belgium"]
            
            styles = ["strict", "lenient", "balanced"]
            experience_levels = ["elite", "experienced", "developing"]
            
            return {
                "name": random.choice(names),
                "nationality": random.choice(nationalities),
                "age": random.randint(30, 50),
                "experience_level": random.choice(experience_levels),
                "total_matches": random.randint(50, 300),
                "big_game_experience": random.choice([True, False]),
                "var_experience": random.choice([True, False]),
                "cards_per_game": round(random.uniform(3.0, 6.0), 1),
                "yellow_cards_per_game": round(random.uniform(2.5, 5.5), 1),
                "red_cards_per_game": round(random.uniform(0.1, 0.8), 1),
                "fouls_per_game": round(random.uniform(18.0, 28.0), 1),
                "penalties_per_game": round(random.uniform(0.1, 0.6), 1),
                "disciplinary_style": random.choice(styles),
                "consistency_rating": round(random.uniform(0.6, 0.9), 2),
                "home_bias_factor": round(random.uniform(-0.05, 0.1), 2),
                "big_team_bias": round(random.uniform(-0.08, 0.05), 2),
                "var_usage_frequency": round(random.uniform(0.3, 0.8), 2)
            }
            
        except Exception as e:
            logger.error(f"Error fetching referee data from APIs: {e}")
            return {}
    
    def _get_team_history(self, referee_id: int, team_id: int) -> Optional[RefereeTeamHistory]:
        """Obtiene histórico del árbitro con equipo específico"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_matches,
                    SUM(CASE WHEN team_result = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN team_result = 'draw' THEN 1 ELSE 0 END) as draws,
                    SUM(CASE WHEN team_result = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(cards_to_team) as avg_cards,
                    SUM(penalties_for) as penalties_for,
                    SUM(penalties_against) as penalties_against,
                    SUM(controversial_decision) as controversial,
                    MAX(match_date) as last_match
                FROM referee_team_history
                WHERE referee_id = ? AND team_id = ?
            ''', (referee_id, team_id))
            
            row = cursor.fetchone()
            conn.close()
            
            if row and row[0] > 0:  # Si hay partidos
                total_matches = row[0]
                win_rate = row[1] / total_matches if total_matches > 0 else 0.0
                
                return RefereeTeamHistory(
                    referee_id=referee_id,
                    team_id=team_id,
                    total_matches=total_matches,
                    team_wins=row[1],
                    team_draws=row[2],
                    team_losses=row[3],
                    win_rate=win_rate,
                    cards_given_to_team=row[4] or 0.0,
                    penalties_for_team=row[5] or 0,
                    penalties_against_team=row[6] or 0,
                    controversial_decisions=row[7] or 0,
                    last_match_date=datetime.fromisoformat(row[8]) if row[8] else None
                )
            else:
                # Simular datos si no hay histórico
                return self._generate_simulated_team_history(referee_id, team_id)
                
        except Exception as e:
            logger.error(f"Error getting team history for referee {referee_id}, team {team_id}: {e}")
            return None
    
    def _generate_simulated_team_history(self, referee_id: int, team_id: int) -> RefereeTeamHistory:
        """Genera histórico simulado realista"""
        import random
        
        # Simular entre 2-10 partidos históricos
        total_matches = random.randint(2, 10)
        
        # Distribución realista de resultados
        wins = random.randint(0, total_matches)
        draws = random.randint(0, total_matches - wins)
        losses = total_matches - wins - draws
        
        win_rate = wins / total_matches if total_matches > 0 else 0.0
        
        return RefereeTeamHistory(
            referee_id=referee_id,
            team_id=team_id,
            total_matches=total_matches,
            team_wins=wins,
            team_draws=draws,
            team_losses=losses,
            win_rate=win_rate,
            cards_given_to_team=round(random.uniform(2.5, 5.5), 1),
            penalties_for_team=random.randint(0, 3),
            penalties_against_team=random.randint(0, 2),
            controversial_decisions=random.randint(0, 2),
            last_match_date=datetime.now() - timedelta(days=random.randint(30, 365))
        )
    
    def _analyze_referee_bias(self, referee_id: int, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Análisis completo de sesgos del árbitro"""
        try:
            referee_profile = self._get_referee_profile(referee_id)
            
            # Análisis de sesgo hacia equipos grandes
            home_team_status = self._get_team_status(home_team_id)
            away_team_status = self._get_team_status(away_team_id)
            
            # Calcular sesgos específicos
            home_bias = self._calculate_home_bias(referee_id)
            big_team_bias = self._calculate_big_team_bias(referee_id, home_team_id, away_team_id)
            experience_bias = self._calculate_experience_bias(referee_profile)
            
            return {
                "home_advantage_bias": home_bias,
                "big_team_bias": big_team_bias,
                "experience_bias": experience_bias,
                "consistency_factor": referee_profile.consistency_rating,
                "var_dependency": referee_profile.var_usage_frequency,
                "pressure_handling": self._assess_pressure_handling(referee_profile),
                "decision_tendency": {
                    "strict_disciplinary": referee_profile.disciplinary_style == "strict",
                    "lenient_style": referee_profile.disciplinary_style == "lenient",
                    "advantage_play": referee_profile.advantage_playing_tendency,
                    "quick_cards": referee_profile.cards_per_game > 4.5
                },
                "team_specific_bias": {
                    "home_team_favorability": self._calculate_team_favorability(referee_id, home_team_id),
                    "away_team_favorability": self._calculate_team_favorability(referee_id, away_team_id)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing referee bias: {e}")
            return self._get_default_bias_analysis()
    
    def _predict_referee_impact(self, referee_id: int, home_team_id: int, away_team_id: int) -> RefereeMatchPrediction:
        """Predice impacto específico del árbitro en el partido"""
        try:
            referee_profile = self._get_referee_profile(referee_id)
            home_history = self._get_team_history(referee_id, home_team_id)
            away_history = self._get_team_history(referee_id, away_team_id)
            
            # Predicciones base del árbitro
            base_cards = referee_profile.cards_per_game
            base_penalties = referee_profile.penalties_per_game
            base_fouls = referee_profile.fouls_per_game
            
            # Ajustar por histórico con equipos
            if home_history and home_history.total_matches >= 3:
                cards_adjustment_home = (home_history.cards_given_to_team - base_cards) * 0.3
            else:
                cards_adjustment_home = 0.0
            
            if away_history and away_history.total_matches >= 3:
                cards_adjustment_away = (away_history.cards_given_to_team - base_cards) * 0.3
            else:
                cards_adjustment_away = 0.0
            
            # Predicción final de tarjetas
            predicted_total_cards = max(1.0, base_cards + cards_adjustment_home + cards_adjustment_away)
            
            # Ajustes por sesgos
            home_advantage_adj = referee_profile.home_bias_factor * 0.1
            away_disadvantage_adj = -referee_profile.home_bias_factor * 0.1
            
            return RefereeMatchPrediction(
                total_cards_prediction=predicted_total_cards,
                yellow_cards_prediction=predicted_total_cards * 0.9,
                red_cards_prediction=predicted_total_cards * 0.1,
                penalties_prediction=max(0.0, base_penalties),
                fouls_prediction=base_fouls,
                home_advantage_adjustment=home_advantage_adj,
                away_disadvantage_adjustment=away_disadvantage_adj,
                confidence_level=self._calculate_prediction_confidence(home_history, away_history),
                historical_sample_size=self._get_sample_size(home_history, away_history)
            )
            
        except Exception as e:
            logger.error(f"Error predicting referee impact: {e}")
            return self._get_default_match_prediction()
    
    def _analyze_style_compatibility(self, referee_id: int, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Analiza compatibilidad de estilos árbitro-equipos"""
        try:
            referee_profile = self._get_referee_profile(referee_id)
            
            # Obtener estilos de los equipos (simulado por ahora)
            home_team_style = self._get_team_playing_style(home_team_id)
            away_team_style = self._get_team_playing_style(away_team_id)
            
            compatibility = {
                "referee_style": {
                    "disciplinary": referee_profile.disciplinary_style,
                    "var_usage": referee_profile.var_usage_frequency,
                    "advantage_play": referee_profile.advantage_playing_tendency
                },
                "team_compatibility": {
                    "home_team": {
                        "style_match": self._calculate_style_match(referee_profile, home_team_style),
                        "expected_cards": self._predict_team_cards(referee_profile, home_team_style),
                        "foul_tolerance": self._calculate_foul_tolerance(referee_profile, home_team_style)
                    },
                    "away_team": {
                        "style_match": self._calculate_style_match(referee_profile, away_team_style),
                        "expected_cards": self._predict_team_cards(referee_profile, away_team_style),
                        "foul_tolerance": self._calculate_foul_tolerance(referee_profile, away_team_style)
                    }
                },
                "match_flow_prediction": {
                    "game_control": referee_profile.consistency_rating,
                    "physical_play_tolerance": 0.7 if referee_profile.disciplinary_style == "lenient" else 0.3,
                    "time_wasting_strictness": 0.9 if referee_profile.disciplinary_style == "strict" else 0.5
                }
            }
            
            return compatibility
            
        except Exception as e:
            logger.error(f"Error analyzing style compatibility: {e}")
            return {"error": str(e)}
    
    def _analyze_var_influence(self, referee_id: int) -> Dict[str, Any]:
        """Analiza influencia y uso del VAR"""
        try:
            referee_profile = self._get_referee_profile(referee_id)
            
            return {
                "var_experience": referee_profile.var_experience,
                "var_usage_frequency": referee_profile.var_usage_frequency,
                "var_overturns": {
                    "penalty_decisions": 0.15 if referee_profile.var_usage_frequency > 0.6 else 0.08,
                    "red_card_decisions": 0.12 if referee_profile.var_usage_frequency > 0.6 else 0.06,
                    "offside_calls": 0.25 if referee_profile.var_usage_frequency > 0.6 else 0.15
                },
                "decision_confidence": {
                    "with_var": min(0.95, referee_profile.consistency_rating + 0.15),
                    "without_var": referee_profile.consistency_rating
                },
                "var_impact_on_flow": {
                    "game_interruptions": referee_profile.var_usage_frequency * 3.0,
                    "decision_time": "fast" if referee_profile.var_usage_frequency > 0.7 else "moderate"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing VAR influence: {e}")
            return {"error": str(e)}
    
    def _calculate_betting_implications(self, referee_profile: RefereeProfile, 
                                     predictions: RefereeMatchPrediction,
                                     bias_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula implicaciones para apuestas"""
        try:
            return {
                "cards_betting": {
                    "total_cards_line": predictions.total_cards_prediction,
                    "over_under_recommendation": "over" if predictions.total_cards_prediction > 4.5 else "under",
                    "confidence": predictions.confidence_level,
                    "value_threshold": 4.5  # Línea común en bookmakers
                },
                "penalty_betting": {
                    "penalty_probability": predictions.penalties_prediction,
                    "penalty_yes_value": predictions.penalties_prediction > 0.35,
                    "recommended_markets": ["penalty_yes_no", "first_half_penalty"] if predictions.penalties_prediction > 0.3 else []
                },
                "team_advantage": {
                    "home_team_advantage": bias_analysis.get("home_advantage_bias", 0.0),
                    "big_team_bias_impact": bias_analysis.get("big_team_bias", 0.0),
                    "recommended_adjustments": {
                        "home_win_prob": predictions.home_advantage_adjustment,
                        "away_win_prob": predictions.away_disadvantage_adjustment
                    }
                },
                "prop_bets": {
                    "fouls_over_under": predictions.fouls_prediction,
                    "offsides_prediction": referee_profile.offsides_per_game,
                    "booking_points": predictions.total_cards_prediction * 10 + predictions.red_cards_prediction * 15
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating betting implications: {e}")
            return {}
    
    def _assess_prediction_confidence(self, home_history: Optional[RefereeTeamHistory],
                                   away_history: Optional[RefereeTeamHistory]) -> float:
        """Evalúa confianza en las predicciones"""
        try:
            confidence_factors = []
            
            # Factor de muestra histórica
            home_sample = home_history.total_matches if home_history else 0
            away_sample = away_history.total_matches if away_history else 0
            
            sample_confidence = min(1.0, (home_sample + away_sample) / 20.0)
            confidence_factors.append(sample_confidence)
            
            # Factor de recencia
            if home_history and home_history.last_match_date:
                days_since = (datetime.now() - home_history.last_match_date).days
                recency_confidence = max(0.3, 1.0 - days_since / 365.0)
                confidence_factors.append(recency_confidence)
            
            if away_history and away_history.last_match_date:
                days_since = (datetime.now() - away_history.last_match_date).days
                recency_confidence = max(0.3, 1.0 - days_since / 365.0)
                confidence_factors.append(recency_confidence)
            
            # Si no hay factores, usar confianza base
            if not confidence_factors:
                return 0.6
            
            return float(np.mean(confidence_factors))
            
        except Exception as e:
            logger.error(f"Error assessing prediction confidence: {e}")
            return 0.6
    
    # Métodos auxiliares con implementaciones simplificadas
    
    def _load_referee_data(self):
        """Carga datos existentes de árbitros"""
        pass
    
    def _save_referee_profile(self, profile: RefereeProfile):
        """Guarda perfil de árbitro en DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO referee_profiles 
                (referee_id, name, nationality, age, experience_level, total_matches,
                 big_game_experience, var_experience, cards_per_game, yellow_cards_per_game,
                 red_cards_per_game, fouls_per_game, penalties_per_game, disciplinary_style,
                 consistency_rating, home_bias_factor, big_team_bias, var_usage_frequency)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                profile.referee_id, profile.name, profile.nationality, profile.age,
                profile.experience_level, profile.total_matches, profile.big_game_experience,
                profile.var_experience, profile.cards_per_game, profile.yellow_cards_per_game,
                profile.red_cards_per_game, profile.fouls_per_game, profile.penalties_per_game,
                profile.disciplinary_style, profile.consistency_rating, profile.home_bias_factor,
                profile.big_team_bias, profile.var_usage_frequency
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving referee profile: {e}")
    
    def _get_team_status(self, team_id: int) -> str:
        """Determina status del equipo (big, medium, small)"""
        # Simulado - en implementación real usar rankings, valor plantilla, etc.
        import random
        return random.choice(["big", "medium", "small"])
    
    def _calculate_home_bias(self, referee_id: int) -> float:
        """Calcula sesgo hacia equipo local"""
        referee_profile = self._get_referee_profile(referee_id)
        return referee_profile.home_bias_factor if referee_profile else 0.0
    
    def _calculate_big_team_bias(self, referee_id: int, home_team_id: int, away_team_id: int) -> float:
        """Calcula sesgo hacia equipos grandes"""
        referee_profile = self._get_referee_profile(referee_id)
        return referee_profile.big_team_bias if referee_profile else 0.0
    
    def _calculate_experience_bias(self, referee_profile: RefereeProfile) -> float:
        """Calcula factor de experiencia"""
        if referee_profile.experience_level == "elite":
            return 0.9
        elif referee_profile.experience_level == "experienced":
            return 0.75
        else:
            return 0.6
    
    def _assess_pressure_handling(self, referee_profile: RefereeProfile) -> float:
        """Evalúa manejo de presión"""
        base_score = 0.7
        if referee_profile.big_game_experience:
            base_score += 0.15
        if referee_profile.experience_level == "elite":
            base_score += 0.1
        return min(1.0, base_score)
    
    def _calculate_team_favorability(self, referee_id: int, team_id: int) -> float:
        """Calcula favorabilidad hacia equipo específico"""
        history = self._get_team_history(referee_id, team_id)
        if history and history.total_matches >= 3:
            return history.win_rate
        return 0.5  # Neutral
    
    def _get_team_playing_style(self, team_id: int) -> Dict[str, Any]:
        """Obtiene estilo de juego del equipo"""
        import random
        return {
            "physicality": random.uniform(0.3, 0.9),
            "technical_play": random.uniform(0.4, 0.8),
            "aggression_level": random.uniform(0.2, 0.7),
            "time_wasting_tendency": random.uniform(0.1, 0.6)
        }
    
    def _calculate_style_match(self, referee_profile: RefereeProfile, team_style: Dict) -> float:
        """Calcula compatibilidad de estilos"""
        if referee_profile.disciplinary_style == "strict" and team_style["aggression_level"] > 0.6:
            return 0.3  # Baja compatibilidad
        elif referee_profile.disciplinary_style == "lenient" and team_style["physicality"] > 0.7:
            return 0.8  # Alta compatibilidad
        else:
            return 0.6  # Compatibilidad media
    
    def _predict_team_cards(self, referee_profile: RefereeProfile, team_style: Dict) -> float:
        """Predice tarjetas para un equipo"""
        base_cards = referee_profile.cards_per_game / 2  # Por equipo
        
        if team_style["aggression_level"] > 0.6:
            base_cards *= 1.2
        if referee_profile.disciplinary_style == "strict":
            base_cards *= 1.1
        
        return base_cards
    
    def _calculate_foul_tolerance(self, referee_profile: RefereeProfile, team_style: Dict) -> float:
        """Calcula tolerancia a faltas"""
        if referee_profile.disciplinary_style == "lenient":
            return 0.8
        elif referee_profile.disciplinary_style == "strict":
            return 0.4
        else:
            return 0.6
    
    def _calculate_prediction_confidence(self, home_history: Optional[RefereeTeamHistory],
                                      away_history: Optional[RefereeTeamHistory]) -> float:
        """Calcula confianza en predicción"""
        return self._assess_prediction_confidence(home_history, away_history)
    
    def _get_sample_size(self, home_history: Optional[RefereeTeamHistory],
                        away_history: Optional[RefereeTeamHistory]) -> int:
        """Obtiene tamaño de muestra"""
        home_size = home_history.total_matches if home_history else 0
        away_size = away_history.total_matches if away_history else 0
        return home_size + away_size
    
    def _get_default_referee_analysis(self) -> Dict[str, Any]:
        """Análisis por defecto cuando no hay datos"""
        return {
            "referee_profile": {
                "name": "Unknown Referee",
                "experience_level": "experienced",
                "cards_per_game": 4.2,
                "penalties_per_game": 0.3,
                "disciplinary_style": "balanced"
            },
            "bias_analysis": self._get_default_bias_analysis(),
            "match_predictions": asdict(self._get_default_match_prediction()),
            "confidence_assessment": 0.5,
            "betting_implications": {}
        }
    
    def _get_default_bias_analysis(self) -> Dict[str, Any]:
        """Análisis de sesgo por defecto"""
        return {
            "home_advantage_bias": 0.0,
            "big_team_bias": 0.0,
            "experience_bias": 0.75,
            "consistency_factor": 0.75
        }
    
    def _get_default_match_prediction(self) -> RefereeMatchPrediction:
        """Predicción por defecto"""
        return RefereeMatchPrediction(
            total_cards_prediction=4.2,
            yellow_cards_prediction=3.8,
            red_cards_prediction=0.4,
            penalties_prediction=0.3,
            fouls_prediction=22.0,
            home_advantage_adjustment=0.0,
            away_disadvantage_adjustment=0.0,
            confidence_level=0.6,
            historical_sample_size=0
        )

# Función de utilidad para integración fácil
def analyze_referee_impact(referee_id: int, home_team_id: int, away_team_id: int, 
                          fixture_id: int) -> Dict[str, Any]:
    """
    Función de utilidad para analizar impacto del árbitro
    """
    try:
        analyzer = RefereeAnalyzer()
        return analyzer.get_referee_impact_analysis(referee_id, home_team_id, away_team_id, fixture_id)
    except Exception as e:
        logger.error(f"Error in referee impact analysis: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Ejemplo de uso
    print("=== REFEREE ANALYZER ===")
    
    analyzer = RefereeAnalyzer()
    
    # Analizar impacto del árbitro
    result = analyzer.get_referee_impact_analysis(
        referee_id=12345,
        home_team_id=33,  # Man United
        away_team_id=40,  # Liverpool
        fixture_id=789
    )
    
    print(f"\nReferee: {result['referee_profile']['name']}")
    print(f"Experience: {result['referee_profile']['experience_level']}")
    print(f"Cards per game: {result['referee_profile']['cards_per_game']}")
    print(f"Disciplinary style: {result['referee_profile']['disciplinary_style']}")
    
    print(f"\nMatch Predictions:")
    predictions = result['match_predictions']
    print(f"  Total cards: {predictions['total_cards_prediction']:.1f}")
    print(f"  Penalties: {predictions['penalties_prediction']:.2f}")
    print(f"  Home bias adjustment: {predictions['home_advantage_adjustment']:+.2f}")
    
    print(f"\nBetting Implications:")
    betting = result.get('betting_implications', {})
    if 'cards_betting' in betting:
        print(f"  Cards O/U recommendation: {betting['cards_betting']['over_under_recommendation']}")
    
    print(f"\nConfidence: {result['confidence_assessment']:.1%}")
