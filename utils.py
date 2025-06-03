from typing import Dict, Any, Optional, List, Union, Tuple
import logging
import numpy as np  # Add numpy import

# Create logger for this module
logger = logging.getLogger(__name__)

def calculate_over_probability(expected_value: float, line: float, std_dev: float) -> float:
    """
    Calculate probability of going over a line given expected value and standard deviation
    
    Handles edge cases safely and avoids division by zero or other numerical issues
    """
    # Ensure we have valid inputs
    if std_dev <= 0:
        std_dev = 1.0
    
    # Safety check for expected_value
    if not isinstance(expected_value, (int, float)) or np.isnan(expected_value):
        expected_value = line
    
    try:
        # Calculate z-score and probability
        from scipy.stats import norm
        z = (line - expected_value) / std_dev
        probability = 1 - norm.cdf(z)
        # Convert numpy types to float to avoid type issues with min/max
        probability_float = float(probability)
        return min(0.99, max(0.01, probability_float))
    except (ImportError, ValueError, ZeroDivisionError, TypeError):
        # Fallback calculation if scipy not available or error occurs
        if expected_value > line + std_dev:
            return 0.8
        elif expected_value > line:
            return 0.65
        elif expected_value > line - std_dev:
            return 0.35
        else:
            return 0.2
    except Exception as e:
        logger.error(f"Error calculating probability: {e}")
        return 0.5

def get_referee_strictness(fixture_id: int, default: float = 1.0) -> float:
    """Get referee strictness coefficient based on historical data"""
    try:
        # Placeholder for actual implementation
        return default
    except Exception:
        return default

def get_fixture_referee(fixture_id: int) -> Optional[Dict[str, Any]]:
    """Get referee information for a fixture"""
    try:
        # Placeholder for actual implementation
        return {"name": "Unknown", "strictness": 1.0}
    except Exception:
        return None

def get_league_averages(league_id: int, season: int) -> Dict[str, float]:
    """Get average statistics for a league in a season"""
    try:
        # Placeholder - would normally retrieve from database or API
        return {
            "goals_per_game": 2.7,
            "cards_per_game": 3.8,
            "corners_per_game": 9.8,
            "fouls_per_game": 21.5
        }
    except Exception:
        return {
            "goals_per_game": 2.7,
            "cards_per_game": 3.8,
            "corners_per_game": 9.8,
            "fouls_per_game": 21.5
        }
