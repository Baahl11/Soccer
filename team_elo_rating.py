"""
Elo rating system for soccer teams based on academic research.

This module implements an enhanced Elo rating system that is specifically calibrated for 
soccer/football match outcomes, incorporating goal difference, home advantage, and other factors.

References:
- Hvattum, L. M., & Arntzen, H. (2010). "Using ELO ratings for match result prediction in association football"
- Lasek, J., Szlávik, Z., & Bhulai, S. (2013). "The predictive power of ranking systems in association football"
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import math
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib
from collections import defaultdict
matplotlib.use('Agg')  # Use non-interactive backend

logger = logging.getLogger(__name__)

class TeamEloRating:
    """
    Implements an enhanced Elo rating system for soccer teams based on academic research.
    Features include time decay, goal difference adjustment, and game importance weighting.
    """
    
    def __init__(
        self, 
        initial_rating: float = 1500, 
        k_factor: float = 32,
        home_advantage: float = 70,
        ratings_file: str = 'data/team_elo_ratings.json',
        league_adjustments: Optional[Dict[int, Dict[str, float]]] = None
    ):
        """
        Initialize the Elo rating system
        
        Args:
            initial_rating: Default rating for new teams
            k_factor: K-factor controlling rating adjustments
            home_advantage: Home advantage in Elo points
            ratings_file: File to store/load ratings
            league_adjustments: League-specific adjustments for K-factor and home advantage
        """
        self.ratings_file = ratings_file
        self.ratings = self._load_ratings()
        self.wins = defaultdict(int)
        self.losses = defaultdict(int)
        self.draws = defaultdict(int)
        
        self.initial_rating = initial_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.last_updated = datetime.now().strftime("%Y-%m-%d")
          # League-specific adjustments
        self.league_adjustments = league_adjustments or {
            # European Top Leagues
            # Premier League - Very competitive, higher K-factor for faster adaptation
            39: {"k_factor_mult": 1.1, "home_advantage_mult": 0.95},
            # La Liga - Less home advantage historically
            140: {"k_factor_mult": 1.0, "home_advantage_mult": 0.9},
            # Serie A - Strong home advantage
            135: {"k_factor_mult": 1.0, "home_advantage_mult": 1.1},
            # Bundesliga - Highly competitive, faster rating changes
            78: {"k_factor_mult": 1.15, "home_advantage_mult": 1.0},
            # Ligue 1 - Standard parameters
            61: {"k_factor_mult": 1.0, "home_advantage_mult": 1.0},
            
            # European Competitions
            # Champions League - Important matches, stronger effect, less home advantage
            2: {"k_factor_mult": 1.2, "home_advantage_mult": 0.9},
            # Europa League - Important but less than Champions
            3: {"k_factor_mult": 1.1, "home_advantage_mult": 0.9},            # Conference League - New competition
            848: {"k_factor_mult": 1.05, "home_advantage_mult": 0.95},
            
            # Other European Leagues
            # Eredivisie (Netherlands) - Offensive league, strong home teams
            88: {"k_factor_mult": 1.05, "home_advantage_mult": 1.1},
            # Primeira Liga (Portugal) - Strong home advantage
            94: {"k_factor_mult": 1.0, "home_advantage_mult": 1.15},
            # Super Lig (Turkey) - Very strong home advantage, passionate fans
            203: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # Russian Premier League - Strong home advantage, travel factor
            235: {"k_factor_mult": 1.0, "home_advantage_mult": 1.15},
            # Belgian Pro League - Competitive
            144: {"k_factor_mult": 1.05, "home_advantage_mult": 1.0},
            # Scottish Premiership - Strong home advantage
            179: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
              # American Leagues
            # MLS (USA) - Long travel distances affect home advantage
            253: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # USL Championship (USA) - Second division
            254: {"k_factor_mult": 0.95, "home_advantage_mult": 1.25},
            # USL League One (USA) - Third division
            255: {"k_factor_mult": 0.9, "home_advantage_mult": 1.3},
            # USL League Two (USA) - Fourth division
            256: {"k_factor_mult": 0.85, "home_advantage_mult": 1.35},
            # Liga MX (Mexico) - Strong home advantage (altitude factors in some stadiums)
            262: {"k_factor_mult": 1.0, "home_advantage_mult": 1.15},
            # Ascenso MX (Mexico) - Second division
            263: {"k_factor_mult": 0.95, "home_advantage_mult": 1.2},
            # Segunda División (Mexico) - Third division
            264: {"k_factor_mult": 0.9, "home_advantage_mult": 1.25},
            # Tercera División (Mexico) - Fourth division
            265: {"k_factor_mult": 0.85, "home_advantage_mult": 1.3},
            # Brasileirão (Brazil) - Very competitive, strong home advantage
            71: {"k_factor_mult": 1.1, "home_advantage_mult": 1.1},
            # Série B (Brazil) - Second division
            72: {"k_factor_mult": 1.05, "home_advantage_mult": 1.15},
            # Série C (Brazil) - Third division
            73: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # Série D (Brazil) - Fourth division
            74: {"k_factor_mult": 0.95, "home_advantage_mult": 1.25},
            # Argentine Primera División - Strong home advantage
            128: {"k_factor_mult": 1.05, "home_advantage_mult": 1.15},
            # Primera Nacional (Argentina) - Second division
            129: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # Primera B (Argentina) - Third division
            130: {"k_factor_mult": 0.95, "home_advantage_mult": 1.25},
            # Primera C (Argentina) - Fourth division
            131: {"k_factor_mult": 0.9, "home_advantage_mult": 1.3},# Asian Leagues
            # J1 League (Japan) - Balanced league
            98: {"k_factor_mult": 1.0, "home_advantage_mult": 1.0},
            # J2 League (Japan) - Second division
            99: {"k_factor_mult": 0.95, "home_advantage_mult": 1.05},
            # J3 League (Japan) - Third division
            100: {"k_factor_mult": 0.9, "home_advantage_mult": 1.1},
            # JFL (Japan) - Fourth division
            101: {"k_factor_mult": 0.85, "home_advantage_mult": 1.15},
            # K League 1 (South Korea) - Standard parameters
            292: {"k_factor_mult": 1.0, "home_advantage_mult": 1.05},
            # K League 2 (South Korea) - Second division
            293: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # K3 League (South Korea) - Third division
            294: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # K4 League (South Korea) - Fourth division
            295: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # Chinese Super League - Strong home advantage due to travel
            169: {"k_factor_mult": 0.95, "home_advantage_mult": 1.2},
            # China League One - Second division
            170: {"k_factor_mult": 0.9, "home_advantage_mult": 1.25},
            # China League Two - Third division
            171: {"k_factor_mult": 0.85, "home_advantage_mult": 1.3},
            # Saudi Pro League - Strong home advantage, growing league
            307: {"k_factor_mult": 1.15, "home_advantage_mult": 1.1},
            # Saudi First Division League
            308: {"k_factor_mult": 1.1, "home_advantage_mult": 1.15},
            # Saudi Second Division League
            309: {"k_factor_mult": 1.05, "home_advantage_mult": 1.2},
            # Saudi Third Division League
            310: {"k_factor_mult": 1.0, "home_advantage_mult": 1.25},
            # A-League (Australia) - Long travel distances
            188: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # A-League Second Division
            189: {"k_factor_mult": 0.95, "home_advantage_mult": 1.25},
            # National Premier Leagues (Australia) - Regional third tier
            190: {"k_factor_mult": 0.9, "home_advantage_mult": 1.3},
              # African Leagues
            # Egyptian Premier League - Strong home advantage
            233: {"k_factor_mult": 0.95, "home_advantage_mult": 1.2},
            # Egyptian Second Division
            234: {"k_factor_mult": 0.9, "home_advantage_mult": 1.25},
            # Egyptian Third Division
            235: {"k_factor_mult": 0.85, "home_advantage_mult": 1.3},
            # Egyptian Fourth Division
            236: {"k_factor_mult": 0.8, "home_advantage_mult": 1.35},
            # South African Premier Division - Balanced
            289: {"k_factor_mult": 1.0, "home_advantage_mult": 1.05},
            # South African National First Division
            290: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # ABC Motsepe League (South Africa) - Third division
            291: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # SAB League (South Africa) - Fourth division
            292: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # Botola (Morocco) - Strong home advantage
            200: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # Botola 2 (Morocco)
            201: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # Division Amateur (Morocco) - Third division
            202: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            
            # Additional European Leagues
            # Swiss Super League
            207: {"k_factor_mult": 1.0, "home_advantage_mult": 1.05},            # Swiss Challenge League (Second division)
            208: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # Promotion League (Switzerland)
            209: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # Allsvenskan (Sweden)
            113: {"k_factor_mult": 1.0, "home_advantage_mult": 1.1},
            # Superettan (Sweden Second division)
            114: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # Ettan (Sweden Third division)
            115: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # Division 2 (Sweden Fourth division)
            116: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            # Israeli Premier League
            271: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # Liga Leumit (Israel Second division)
            272: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # Liga Alef (Israel Third division)
            273: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            # Danish Superliga
            119: {"k_factor_mult": 1.0, "home_advantage_mult": 1.05},
            # Danish 1st Division
            120: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # Danish 2nd Division
            121: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # Danish 3rd Division
            122: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # First Professional Football League (Bulgaria)
            172: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},            # Second Professional Football League (Bulgaria)
            173: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # Third Professional Football League (Bulgaria)
            174: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            # Veikkausliiga (Finland)
            143: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # Ykkönen (Finland Second division)
            144: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # Kakkonen (Finland Third division)
            145: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # Kolmonen (Finland Fourth division)
            146: {"k_factor_mult": 0.8, "home_advantage_mult": 1.25},
            # Úrvalsdeild (Iceland)
            238: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # 1. deild karla (Iceland Second division)
            239: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # 2. deild karla (Iceland Third division)
            240: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # 3. deild karla (Iceland Fourth division)
            241: {"k_factor_mult": 0.8, "home_advantage_mult": 1.25},
            # Eliteserien (Norway)
            103: {"k_factor_mult": 1.0, "home_advantage_mult": 1.1},
            # OBOS-ligaen (Norway Second division)
            104: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # PostNord-ligaen (Norway Third division)
            105: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},            # Ekstraklasa (Poland)
            106: {"k_factor_mult": 1.0, "home_advantage_mult": 1.1},
            # I Liga (Poland Second division)
            107: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # II Liga (Poland Third division)
            108: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # III Liga (Poland Fourth division)
            109: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            # Liga Portugal (Portugal) - Using original name
            94: {"k_factor_mult": 1.0, "home_advantage_mult": 1.15},
            # Liga Portugal 2 (Portugal Second division)
            95: {"k_factor_mult": 0.95, "home_advantage_mult": 1.2},
            # Liga 3 (Portugal Third division)
            96: {"k_factor_mult": 0.9, "home_advantage_mult": 1.25},
            # Campeonato de Portugal (Portugal Fourth division)
            97: {"k_factor_mult": 0.85, "home_advantage_mult": 1.3},
            # Süper Lig (Turkey)
            203: {"k_factor_mult": 1.0, "home_advantage_mult": 1.2},
            # TFF 1. Lig (Turkey Second division)
            204: {"k_factor_mult": 0.95, "home_advantage_mult": 1.25},
            # TFF 2. Lig (Turkey Third division)
            205: {"k_factor_mult": 0.9, "home_advantage_mult": 1.3},
            # TFF 3. Lig (Turkey Fourth division)
            206: {"k_factor_mult": 0.85, "home_advantage_mult": 1.35},
            # Croatian Football League
            210: {"k_factor_mult": 0.95, "home_advantage_mult": 1.15},
            # First NL (Croatia Second division)
            211: {"k_factor_mult": 0.9, "home_advantage_mult": 1.2},
            # Second NL (Croatia Third division)
            212: {"k_factor_mult": 0.85, "home_advantage_mult": 1.25},
            # Third NL (Croatia Fourth division)
            213: {"k_factor_mult": 0.8, "home_advantage_mult": 1.3},
            # Erovnuli Liga (Georgia)
            265: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15},
            # Erovnuli Liga 2 (Georgia Second division)
            266: {"k_factor_mult": 0.85, "home_advantage_mult": 1.2},
            # Liga 3 (Georgia Third division)
            267: {"k_factor_mult": 0.8, "home_advantage_mult": 1.25},
            # Meistriliiga (Estonia)
            329: {"k_factor_mult": 0.9, "home_advantage_mult": 1.1},
            # Esiliiga (Estonia Second division)
            330: {"k_factor_mult": 0.85, "home_advantage_mult": 1.15},
            # Esiliiga B (Estonia Third division)
            331: {"k_factor_mult": 0.8, "home_advantage_mult": 1.2},
            # II Liiga (Estonia Fourth division)
            332: {"k_factor_mult": 0.75, "home_advantage_mult": 1.25},
              # Second divisions of existing major leagues
            # Championship (England)
            40: {"k_factor_mult": 1.05, "home_advantage_mult": 1.0},
            # League One (England)
            41: {"k_factor_mult": 1.0, "home_advantage_mult": 1.05},
            # League Two (England)
            42: {"k_factor_mult": 0.95, "home_advantage_mult": 1.1},
            # National League (England)
            43: {"k_factor_mult": 0.9, "home_advantage_mult": 1.15}
        }
        
        # Load existing ratings
        self._load_ratings()
    
    def _load_ratings(self) -> Dict[int, float]:
        """Load team ratings from file"""
        try:
            if os.path.exists(self.ratings_file):
                with open(self.ratings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    ratings = {int(team_id): rating for team_id, rating in data.get('ratings', {}).items()}
                    self.last_updated = data.get('last_updated', self.last_updated)
                    logger.info(f"Loaded {len(ratings)} team ratings from {self.ratings_file}")
                    return ratings
            else:
                logger.warning(f"Ratings file {self.ratings_file} not found, using empty ratings")
                return {}
        except Exception as e:
            logger.error(f"Error loading ratings: {e}")
            # Continue with empty ratings rather than crash
            return {}
    
    def force_load_ratings(self) -> None:
        """
        Fuerza la carga de ratings desde el archivo.
        Útil cuando necesitamos asegurarnos de tener los datos más recientes.
        """
        self._load_ratings()
        logger.info("Forced reload of ELO ratings from file")
    
    def save_ratings(self) -> None:
        """Save current ratings to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ratings_file), exist_ok=True)
            
            data = {
                'ratings': self.ratings,
                'last_updated': datetime.now().strftime("%Y-%m-%d")
            }
            
            with open(self.ratings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"Saved {len(self.ratings)} team ratings to {self.ratings_file}")
        except Exception as e:
            logger.error(f"Error saving ratings: {e}")
    
    def get_rating(self, team_id: int) -> float:
        """Get team's Elo rating, or default if not found"""
        return self.ratings.get(team_id, self.initial_rating)
    
    def set_rating(self, team_id: int, rating: float) -> None:
        """Set a team's ELO rating"""
        self.ratings[team_id] = rating

    def _calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using ELO formula"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def _calculate_k_factor(self, rating_diff: float) -> float:
        """Calculate dynamic K-factor based on rating difference"""
        if abs(rating_diff) > 400:
            return 16  # Lower K for mismatched teams
        elif abs(rating_diff) > 200:
            return 24
        else:
            return 32  # Higher K for evenly matched teams

    def predict_match(self, home_id: int, away_id: int, league_id: Optional[int] = None) -> Dict[str, float]:
        """
        Predict match outcome probabilities
        
        Args:
            home_id: Home team ID  
            away_id: Away team ID
            league_id: Optional league ID for adjustments
            
        Returns:
            Dict with win, draw and loss probabilities
        """
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)

        # Get edge case adjustments if edge case handler is available
        try:
            from edge_case_handler import EdgeCaseHandler
            edge_handler = EdgeCaseHandler()
            
            # Validate ratings
            home_rating = edge_handler.validate_rating(home_rating)
            away_rating = edge_handler.validate_rating(away_rating)
            
            # Get match data for special cases
            match_data = {
                "home_team_id": home_id,
                "away_team_id": away_id,
                "league_id": league_id,
                "prediction_time": datetime.now().isoformat()
            }
            
            # Get special case adjustments
            home_adjustments = edge_handler.handle_special_cases(home_id, match_data)
            away_adjustments = edge_handler.handle_special_cases(away_id, match_data)
            
            # Apply adjustments
            home_rating_adjusted = home_rating * home_adjustments["rating_factor"]
            away_rating_adjusted = away_rating * away_adjustments["rating_factor"]
            
            # Get uncertainty factor
            uncertainty = max(home_adjustments["uncertainty_factor"], away_adjustments["uncertainty_factor"])
            
        except ImportError:
            # Edge case handler not available, use base ratings
            home_rating_adjusted = home_rating
            away_rating_adjusted = away_rating
            uncertainty = 1.0
            
        # Apply home advantage    
        home_rating_final = home_rating_adjusted + 100
        
        # Calculate raw win probability
        win_prob = self._calculate_expected_score(home_rating_final, away_rating_adjusted)
        
        # Adjust for uncertainty
        if uncertainty > 1.0:
            # Move probabilities closer to even when uncertainty is high
            win_prob = 0.33 + (win_prob - 0.33) / uncertainty
            
        # Convert to 3-way probabilities with dynamic draw probability
        rating_diff = abs(home_rating_final - away_rating_adjusted)
        if rating_diff < 100:
            draw_prob = 0.30  # Higher draw probability for close matches
        elif rating_diff < 300:
            draw_prob = 0.25  # Medium draw probability
        else:
            draw_prob = 0.20  # Lower draw probability for mismatched teams
            
        # Apply draw probability
        win_prob = win_prob * (1 - draw_prob)
        loss_prob = (1 - win_prob) * (1 - draw_prob)
        
        # Validate probabilities sum to 1
        total = win_prob + draw_prob + loss_prob
        if total != 1.0:
            factor = 1.0 / total
            win_prob *= factor
            draw_prob *= factor
            loss_prob *= factor
            
        return {
            "win_probability": win_prob,
            "draw_probability": draw_prob,
            "loss_probability": loss_prob
        }

    def update_ratings_for_match(self, home_id: int, away_id: int, home_score: int, away_score: int) -> Tuple[float, float]:
        """
        Update ratings for both teams based on match result
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            home_score: Goals scored by home team
            away_score: Goals scored by away team
            
        Returns:
            Tuple of (new home rating, new away rating)
        """
        # Get current ratings
        home_rating = self.get_rating(home_id)
        away_rating = self.get_rating(away_id)
        
        # Calculate expected scores
        home_expected = self._calculate_expected_score(home_rating + 100, away_rating)  # Home advantage
        away_expected = 1 - home_expected
        
        # Calculate actual scores
        if home_score > away_score:
            home_actual = 1.0
            away_actual = 0.0
            self.wins[home_id] += 1
            self.losses[away_id] += 1
        elif away_score > home_score:
            home_actual = 0.0
            away_actual = 1.0
            self.wins[away_id] += 1 
            self.losses[home_id] += 1
        else:
            home_actual = 0.5
            away_actual = 0.5
            self.draws[home_id] += 1
            self.draws[away_id] += 1
        
        # Calculate K-factor
        k = self._calculate_k_factor(home_rating - away_rating)
        
        # Calculate goal difference factor
        goal_diff = abs(home_score - away_score)
        goal_factor = math.log(goal_diff + 1) if goal_diff > 0 else 1.0
        
        # Update ratings
        home_change = k * goal_factor * (home_actual - home_expected)
        away_change = k * goal_factor * (away_actual - away_expected)
        
        new_home = home_rating + home_change
        new_away = away_rating + away_change
        
        # Save new ratings
        self.set_rating(home_id, new_home)
        self.set_rating(away_id, new_away)
        
        return new_home, new_away

    def update_ratings(self, home_id: int, away_id: int, home_goals: int, away_goals: int, match_importance: float = 1.0) -> Tuple[float, float]:
        """
        Update ratings for both teams based on match result (adapter method for auto_updating_elo.py)
        
        Args:
            home_id: Home team ID
            away_id: Away team ID
            home_goals: Goals scored by home team
            away_goals: Goals scored by away team
            match_importance: Match importance factor (currently not used but kept for compatibility)
            
        Returns:
            Tuple of (new home rating, new away rating)
        """
        # Use the existing update_ratings_for_match method
        return self.update_ratings_for_match(home_id, away_id, home_goals, away_goals)

    def get_league_adjustments(self, league_id: int) -> Dict[str, float]:
        """Get league-specific adjustments, or defaults if not found"""
        return self.league_adjustments.get(
            league_id, 
            {"k_factor_mult": 1.0, "home_advantage_mult": 1.0}
        )
    
    def get_team_features(self, home_team_id: Optional[int], away_team_id: Optional[int]) -> Dict[str, float]:
        """
        Get features for a match based on team Elo ratings
        
        Args:
            home_team_id: Home team ID (can be None)
            away_team_id: Away team ID (can be None)
            
        Returns:
            Dictionary with features like home_elo, away_elo, elo_diff
        """
        safe_home_team_id = home_team_id if home_team_id is not None else 0
        safe_away_team_id = away_team_id if away_team_id is not None else 0
        
        home_elo = self.get_rating(safe_home_team_id)
        away_elo = self.get_rating(safe_away_team_id)
        elo_diff = home_elo - away_elo
        
        return {
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': elo_diff,
            'elo_sum': home_elo + away_elo,
            'elo_avg': (home_elo + away_elo) / 2,
            'elo_ratio': home_elo / away_elo if away_elo > 0 else 1.0
        }
    
    def get_win_probability(self, home_team_id: Optional[int], away_team_id: Optional[int], league_id: Optional[int] = None) -> float:
        """
        Calculate probability of home team winning based on Elo ratings
        
        Args:
            home_team_id: Home team ID (can be None)
            away_team_id: Away team ID (can be None)
            league_id: Optional league ID for league-specific adjustments
            
        Returns:
            Probability of home team winning [0-1]
        """
        safe_home_team_id = home_team_id if home_team_id is not None else 0
        safe_away_team_id = away_team_id if away_team_id is not None else 0
        
        home_rating = self.get_rating(safe_home_team_id)
        away_rating = self.get_rating(safe_away_team_id)
        
        # Apply league-specific home advantage if league_id provided
        if league_id and league_id in self.league_adjustments:
            home_adv_mult = self.league_adjustments[league_id].get("home_advantage_mult", 1.0)
            effective_home_rating = home_rating + (self.home_advantage * home_adv_mult)
        else:
            effective_home_rating = home_rating + self.home_advantage
        
        # Calculate win probability using the standard Elo formula
        win_probability = 1.0 / (1.0 + 10 ** ((away_rating - effective_home_rating) / 400.0))
        
        return win_probability


# Adapter class para facilitar el uso en app.py
class EloRating:
    """
    Adapter class para facilitar el uso del sistema ELO en otras partes del código.
    Esta clase simplifica la interfaz de TeamEloRating.
    """
    
    def __init__(self, ratings_file: str = 'data/team_elo_ratings.json'):
        """Initialize EloRating adapter with default settings"""
        self.elo_system = TeamEloRating(ratings_file=ratings_file)
        self._ensure_ratings_initialized()
        
    def _ensure_ratings_initialized(self) -> None:
        """
        Asegura que los ratings ELO estén inicializados con valores realistas para equipos importantes.
        En un sistema de producción, esto se cargaría desde una base de datos y se actualizaría después
        de cada partido, pero para nuestra demostración, establecemos valores iniciales.
        """
        # Valores ELO preestablecidos para equipos conocidos
        ratings_data = {
            # Equipos españoles
            529: 1566.0,  # Barcelona
            541: 1602.0,  # Real Madrid
            530: 1479.7,  # Atlético Madrid
            543: 1488.2,  # Real Betis
            546: 1459.0,  # Getafe
            533: 1490.5,  # Villarreal
            532: 1505.0,  # Valencia
            538: 1462.3,  # Sevilla
            540: 1443.0,  # Espanyol
            727: 1476.7,  # Osasuna
            724: 1432.5,  # Mallorca
            542: 1441.2,  # Celta de Vigo
            539: 1438.0,  # Athletic Club
            728: 1435.6,  # Rayo Vallecano
            
            # Equipos italianos
            489: 1590.0,  # Juventus
            505: 1550.0,  # Inter
            497: 1545.0,  # Roma
            496: 1565.0,  # AC Milan
            492: 1535.0,  # Napoli
            487: 1525.0,  # Lazio
            
            # Equipos ingleses
            33: 1585.0,   # Manchester United
            42: 1598.0,   # Arsenal
            40: 1610.0,   # Liverpool
            50: 1605.0,   # Manchester City
            49: 1560.0,   # Chelsea
            47: 1535.0,   # Tottenham
            
            # Equipos alemanes
            157: 1595.0,  # Bayern Munich
            165: 1570.0,  # Borussia Dortmund
            159: 1525.0,  # Eintracht Frankfurt
            169: 1515.0,  # RB Leipzig
            
            # Equipos franceses
            85: 1580.0,   # Paris Saint-Germain
            80: 1525.0,   # Olympique de Marseille
            95: 1510.0,   # Olympique Lyonnais
            
            # Equipos georgianos (valores más conservadores ya que son menos conocidos)
            3504: 1470.0,  # WIT Georgia
            14861: 1445.0,  # Gori 
            13853: 1435.0,  # Kolkheti Khobi
            16507: 1452.0,  # Irao
            10015: 1465.0,  # Aragvi Dusheti
            13860: 1430.0,  # Betlemi
            13856: 1455.0,  # Odishi 1919
            13848: 1440.0,  # Borjomi
            14857: 1462.0,  # Didube 2014
            13851: 1441.0,  # Gardabani
        }
        
        # Aplicar estos ratings solo si no existen ya
        for team_id, rating in ratings_data.items():
            if team_id not in self.elo_system.ratings:
                self.elo_system.ratings[team_id] = rating
        
        # Guardar los ratings actualizados
        self.elo_system.save_ratings()
        
    def force_load_ratings(self) -> None:
        """Force reload ratings from file"""
        self.elo_system.force_load_ratings()
        # Asegurarse de que los equipos clave tengan ratings realistas
        self._ensure_ratings_initialized()
    
    
    def _get_smart_default_rating(self, team_id: int, league_id: Optional[int] = None) -> float:
        """
        Calculate a smart default rating for a new team based on available context.
        
        Args:
            team_id: The team ID
            league_id: Optional league ID to provide context
            
        Returns:
            A realistic starting ELO rating
        """
        import random
        import statistics
        
        # Start with the base default rating
        base_rating = self.elo_system.initial_rating
        
        # If we have league information, use it for better defaults
        if league_id is not None:
            # Get all teams from this league
            league_teams = []
            for existing_id, rating in self.elo_system.ratings.items():
                # We'd need a mapping of teams to leagues which we don't have
                # This is a placeholder for that logic
                # For now, just use the existing teams as reference
                league_teams.append(rating)
            
            # If we have other teams in this league, base new rating on their average
            if league_teams:
                try:
                    # Use median to avoid outliers skewing the result
                    league_median = statistics.median(league_teams)
                    # Use the median as the center, but push towards the lower end
                    # New teams are usually slightly weaker than established teams
                    base_rating = league_median * 0.97
                    
                    # Add some randomness to avoid all new teams having identical ratings
                    # More variance for lower-tier leagues
                    if league_id > 100:  # Assuming higher IDs = lower tier leagues
                        variance = 40.0
                    else:
                        variance = 25.0
                        
                    adjustment = random.uniform(-variance, variance)
                    return base_rating + adjustment
                except Exception as e:
                    logging.getLogger(__name__).warning(f"Error calculating league-based rating: {e}")
                    
        # If we couldn't calculate a league-based rating, use a slightly randomized default
        adjustment = random.uniform(-30, 30)
        return base_rating + adjustment
        
    def get_team_rating(self, team_id: int, league_id: Optional[int] = None) -> float:
        """
        Get rating for a specific team, with smart fallback.
        If the team doesn't exist in our database, this method will:
        1. Add the team to our database with a smart default rating
        2. Save the updated database
        3. Return the new rating
        
        Args:
            team_id: The team ID
            league_id: Optional league ID to provide context for new teams
        """
        if not team_id:
            return self.elo_system.initial_rating
        
        # Check if team exists in our ratings
        if team_id in self.elo_system.ratings:
            return self.elo_system.get_rating(team_id)
        
        # If team doesn't exist, add it with a smart default rating
        logger = logging.getLogger(__name__)
        logger.info(f"Adding new team ID: {team_id} to ELO ratings database")
        
        # Get a smart default rating
        new_rating = self._get_smart_default_rating(team_id, league_id)
        
        # Add the new team to our ratings
        self.elo_system.ratings[team_id] = new_rating
        
        # Save the updated ratings to persist this new team
        self.elo_system.save_ratings()
        
        logger.info(f"Added new team ID: {team_id} with initial rating: {new_rating}")
        
        return new_rating
    def get_match_probabilities(self, home_team_id: Optional[int], away_team_id: Optional[int], league_id: Optional[int] = None) -> Tuple[float, float, float]:
        """Get win, draw, loss probabilities for a match"""
        safe_home_team_id = home_team_id if home_team_id is not None else 0
        safe_away_team_id = away_team_id if away_team_id is not None else 0
        
        # Pass the league_id for smarter default rating calculation if teams don't exist
        home_rating = self.get_team_rating(safe_home_team_id, league_id)
        away_rating = self.get_team_rating(safe_away_team_id, league_id)
        
        # Apply league-specific home advantage if available
        if league_id is not None and league_id in self.elo_system.league_adjustments:
            home_adv_mult = self.elo_system.league_adjustments[league_id].get("home_advantage_mult", 1.0)
            effective_home_rating = home_rating + (self.elo_system.home_advantage * home_adv_mult)
        else:
            effective_home_rating = home_rating + self.elo_system.home_advantage
        
        # Calculate win probability using standard Elo formula
        win_prob = 1.0 / (1.0 + 10 ** ((away_rating - effective_home_rating) / 400.0))
        
        # Draw probability is highest when teams are evenly matched
        # This is a simplified model based on empirical data
        draw_prob = 0.32 - (abs(win_prob - 0.5) * 0.2)
        
        # Loss probability (remaining probability)
        loss_prob = 1.0 - win_prob - draw_prob
        
        # Ensure probabilities are in valid range
        win_prob = max(0.01, min(0.99, win_prob))
        draw_prob = max(0.01, min(0.98, draw_prob))
        loss_prob = max(0.01, min(0.99, loss_prob))
        
        # Normalize to ensure they sum to 1
        total = win_prob + draw_prob + loss_prob
        win_prob /= total
        draw_prob /= total
        loss_prob /= total
        
        return win_prob, draw_prob, loss_prob 
       
    def get_expected_goal_diff(self, home_team_id: Optional[int], away_team_id: Optional[int], league_id: Optional[int] = None) -> float:
        """Get expected goal difference based on Elo ratings"""
        safe_home_team_id = home_team_id if home_team_id is not None else 0
        safe_away_team_id = away_team_id if away_team_id is not None else 0
        
        # Pass the league_id for smarter default rating calculation if teams don't exist
        home_rating = self.get_team_rating(safe_home_team_id, league_id)
        away_rating = self.get_team_rating(safe_away_team_id, league_id)
        
        # Raw Elo difference
        raw_elo_diff = home_rating - away_rating
        
        # Convert to expected goal difference (empirical conversion factor)
        # Each 100 Elo points is roughly 0.15 goals difference
        expected_goal_diff = raw_elo_diff * 0.15 / 100.0
        
        return expected_goal_diff

def get_elo_ratings_for_match(home_team_id: Optional[int], away_team_id: Optional[int], league_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Obtiene los ratings ELO y datos relacionados para un partido.
    Esta función es usada por prediction_integration.py y test_elo_integration.py.
    
    Args:
        home_team_id: ID del equipo local (puede ser None)
        away_team_id: ID del equipo visitante (puede ser None)
        league_id: ID de la liga (opcional, para ajustes específicos)
        
    Returns:
        Dict con los ratings ELO, probabilidades y diferencia de goles esperada
    """
    try:
        # Use the new auto-updating ELO system that automatically adds missing teams
        from auto_updating_elo import get_elo_data_with_auto_rating
        return get_elo_data_with_auto_rating(home_team_id, away_team_id, league_id)
    except ImportError:
        # Fallback to original implementation if the new module isn't available
        logger.warning("AutoUpdatingELO module not available. Using legacy ELO implementation.")
        
        # Create EloRating adapter instance with default settings
        elo_system = EloRating()
        
        # Use default team ID 0 if None to satisfy type checker and avoid errors
        safe_home_team_id = home_team_id if home_team_id is not None else 0
        safe_away_team_id = away_team_id if away_team_id is not None else 0
        
        # Get ratings for both teams with safe fallback for None
        home_elo = elo_system.get_team_rating(safe_home_team_id)
        away_elo = elo_system.get_team_rating(safe_away_team_id)
        elo_diff = home_elo - away_elo
        
        # Apply home advantage for calculating win probability
        effective_home_rating = home_elo + elo_system.elo_system.home_advantage
        
        # Calculate win probability using standard ELO formula
        win_prob = 1.0 / (1.0 + 10 ** ((away_elo - effective_home_rating) / 400.0))
        
        # Draw probability model (based on empirical soccer data)
        # Draw probability is highest when teams are evenly matched
        draw_prob = 0.33 - (abs(win_prob - 0.5) * 0.2)
        
        # Loss probability
        loss_prob = 1.0 - win_prob - draw_prob
        
        # Normalize probabilities to ensure they sum to 1
        total = win_prob + draw_prob + loss_prob
        win_prob /= total
        draw_prob /= total
        loss_prob /= total
        
        # Calculate expected goal difference
        # Based on empirical data: ~0.15 goals per 100 ELO points difference
        expected_goal_diff = elo_diff * 0.15 / 100.0
        
        # Return results dictionary
        return {
            'home_elo': round(home_elo, 1),
            'away_elo': round(away_elo, 1),
            'elo_diff': round(elo_diff, 1),
            'elo_win_probability': round(win_prob, 3),
            'elo_draw_probability': round(draw_prob, 3),
            'elo_loss_probability': round(loss_prob, 3),
            'elo_expected_goal_diff': round(expected_goal_diff, 2),
            'league_id': league_id
        }
