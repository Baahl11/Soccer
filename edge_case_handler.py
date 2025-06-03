"""
Edge Case Handler for ELO Rating System

This module provides comprehensive edge case handling for the ELO rating system.
It handles various scenarios that might impact rating calculations and predictions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EdgeCaseConfig:
    """Configuration for edge case handling parameters"""
    min_elo_rating: int = 100  # Minimum possible ELO rating
    max_elo_rating: int = 3000  # Maximum possible ELO rating
    max_rating_change: int = 200  # Maximum allowed change in a single match
    default_rating: int = 1500  # Default rating for new teams
    min_matches_for_reliability: int = 5  # Minimum matches needed for reliable rating
    max_inactivity_days: int = 365  # Maximum days before rating decay
    rating_decay_factor: float = 0.95  # Factor to decay ratings after inactivity

class EdgeCaseHandler:
    """
    Handles edge cases in the ELO rating system to ensure robustness and reliability.
    """
    def __init__(self, config: Optional[EdgeCaseConfig] = None):
        """Initialize the edge case handler with optional custom configuration."""
        self.config = config or EdgeCaseConfig()
        logger.info("Initialized EdgeCaseHandler with config: %s", self.config)

    def validate_rating(self, rating: float) -> float:
        """
        Validates and adjusts an ELO rating to ensure it's within acceptable bounds.
        
        Args:
            rating: The ELO rating to validate
            
        Returns:
            float: The validated and potentially adjusted rating
        """
        if not isinstance(rating, (int, float)):
            logger.warning(f"Invalid rating type: {type(rating)}. Using default rating.")
            return self.config.default_rating
            
        if math.isnan(rating) or math.isinf(rating):
            logger.warning(f"Invalid rating value: {rating}. Using default rating.")
            return self.config.default_rating
            
        rating = max(self.config.min_elo_rating, min(self.config.max_elo_rating, rating))
        return float(rating)

    def handle_rating_update(self, old_rating: float, new_rating: float) -> float:
        """
        Handles rating updates, ensuring changes are within acceptable limits.
        
        Args:
            old_rating: Previous ELO rating
            new_rating: Proposed new ELO rating
            
        Returns:
            float: The validated new rating
        """
        old_rating = self.validate_rating(old_rating)
        new_rating = self.validate_rating(new_rating)
        
        # Limit maximum rating change
        max_change = self.config.max_rating_change
        rating_change = new_rating - old_rating
        if abs(rating_change) > max_change:
            logger.warning(f"Rating change of {rating_change} exceeds maximum of {max_change}")
            new_rating = old_rating + (max_change if rating_change > 0 else -max_change)
        
        return self.validate_rating(new_rating)

    def calculate_rating_reliability(self, matches_played: int, days_inactive: int) -> float:
        """
        Calculates the reliability factor for a team's rating based on match history.
        
        Args:
            matches_played: Number of matches the team has played
            days_inactive: Number of days since the team's last match
            
        Returns:
            float: Reliability factor between 0 and 1
        """
        # Calculate base reliability from matches played
        base_reliability = min(1.0, matches_played / self.config.min_matches_for_reliability)
        
        # Calculate inactivity penalty
        if days_inactive > self.config.max_inactivity_days:
            inactivity_penalty = self.config.rating_decay_factor ** (days_inactive / 365)
        else:
            inactivity_penalty = 1.0
            
        return base_reliability * inactivity_penalty

    def handle_missing_data(self, league_id: Optional[int] = None) -> Dict:
        """
        Provides fallback values when required data is missing.
        
        Args:
            league_id: Optional league ID to determine appropriate defaults
            
        Returns:
            Dict: Default values for missing data
        """
        defaults = {
            "rating": self.config.default_rating,
            "reliability": 0.5,
            "win_probability": 0.5,
            "draw_probability": 0.25,
            "loss_probability": 0.25,
            "expected_goal_diff": 0.0
        }
        
        # Adjust defaults based on league tier if available
        if league_id:
            try:
                # Implement league-specific adjustments here
                pass
            except Exception as e:
                logger.warning(f"Error adjusting defaults for league {league_id}: {e}")
                
        return defaults

    def validate_prediction(self, prediction: Dict) -> Tuple[Dict, List[str]]:
        """
        Validates prediction data and ensures all required fields are present and valid.
        
        Args:
            prediction: Dictionary containing prediction data
            
        Returns:
            Tuple[Dict, List[str]]: Validated prediction and list of warnings
        """
        validated = prediction.copy()
        warnings = []
        
        required_fields = {
            "elo_ratings": dict,
            "probabilities": dict,
            "expected_goal_diff": (int, float)
        }
        
        for field, field_type in required_fields.items():
            if field not in validated:
                warnings.append(f"Missing required field: {field}")
                validated[field] = self.handle_missing_data().get(field)
            elif not isinstance(validated[field], field_type):
                warnings.append(f"Invalid type for {field}: {type(validated[field])}")
                validated[field] = self.handle_missing_data().get(field)
                
        # Validate probabilities sum to 1
        if "probabilities" in validated:
            probs = validated["probabilities"]
            total = sum(probs.values())
            if not math.isclose(total, 1.0, rel_tol=1e-9):
                warnings.append(f"Probabilities sum to {total}, normalizing")
                factor = 1.0 / total
                validated["probabilities"] = {k: v * factor for k, v in probs.items()}
                
        return validated, warnings

    def handle_special_cases(self, team_id: int, match_data: Dict) -> Dict[str, float]:
        """
        Handles special cases like derbies, cup matches, or other unusual situations.
        
        Args:
            team_id: ID of the team
            match_data: Dictionary containing match information
            
        Returns:
            Dict[str, float]: Adjustment factors for the match
        """
        adjustments = {
            "rating_factor": 1.0,
            "importance_factor": 1.0,
            "uncertainty_factor": 1.0
        }
        
        try:
            # Handle derby matches
            if self._is_derby(match_data):
                adjustments["uncertainty_factor"] = 1.2
                
            # Handle cup matches
            if match_data.get("is_cup_match"):
                adjustments["importance_factor"] = 1.5
                
            # Handle matches after long breaks
            if match_data.get("days_since_last_match", 0) > 60:
                adjustments["uncertainty_factor"] = 1.1
                
        except Exception as e:
            logger.error(f"Error handling special cases for team {team_id}: {e}")
            
        return adjustments

    def _is_derby(self, match_data: Dict) -> bool:
        """
        Determines if a match is a derby based on team locations and history.
        
        Args:
            match_data: Dictionary containing match information
            
        Returns:
            bool: True if the match is a derby
        """
        try:
            home_city = match_data.get("home_team_city")
            away_city = match_data.get("away_team_city")
            
            if home_city and away_city and home_city == away_city:
                return True
                
            # Add additional derby detection logic here
            return False
            
        except Exception as e:
            logger.error(f"Error detecting derby status: {e}")
            return False
