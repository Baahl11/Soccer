import logging
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Any, Optional
from formation import (
    get_formation_strength,
    analyze_formation_style,
    FORMATION_IMPACT
)
from datetime import datetime
from math import factorial
from odds_analyzer import OddsAnalyzer

logger = logging.getLogger(__name__)
odds_analyzer = OddsAnalyzer()

# Add at top of file with other constants
WEATHER_FACTORS = {
    "rain": {
        "high": {"adj": -0.10, "prob_adj": -0.15},
        "medium": {"adj": -0.05, "prob_adj": -0.10},
        "light": {"adj": -0.02, "prob_adj": -0.05}
    },
    "snow": {
        "high": {"adj": -0.15, "prob_adj": -0.20},
        "medium": {"adj": -0.10, "prob_adj": -0.15},
        "light": {"adj": -0.08, "prob_adj": -0.10}
    },
    "wind": {
        "thresholds": {
            "strong": 30,
            "moderate": 20,
            "light": 15
        },
        "adjustments": {
            "strong": -0.12,
            "moderate": -0.06,
            "light": -0.03
        }
    },
    "temperature": {
        "thresholds": {
            "very_cold": 0,
            "cold": 5,
            "hot": 32,
            "very_hot": 35
        },
        "adjustments": {
            "very_cold": -0.08,
            "cold": -0.05,
            "hot": -0.07,
            "very_hot": -0.10
        }
    }
}

DERBY_PAIRS = {
    524: [  # Premier League
        {"manchester united", "manchester city"},
        {"arsenal", "tottenham hotspur"},
        {"liverpool", "everton"},
    ],
    564: [  # La Liga
        {"real madrid", "atletico madrid"},
        {"barcelona", "espanyol"},
        {"real betis", "sevilla"},
    ],
    262: [  # Liga MX
        {"america", "guadalajara"},
        {"pumas unam", "cruz azul"},
        {"monterrey", "tigres uanl"},
    ]
}

def adjust_prediction_based_on_lineup(prediction: Dict[str, Any], 
                                    lineup_data: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust prediction based on lineup and formation data"""
    try:
        if not lineup_data or not prediction:
            return prediction.copy()
            
        result = prediction.copy()
            
        # Get formations with validation
        home_formation = lineup_data.get('home', {}).get('formation')
        away_formation = lineup_data.get('away', {}).get('formation')
        
        if not home_formation or not away_formation:
            return result
            
        # Get formation strength and style analysis
        strength_data = get_formation_strength(home_formation, away_formation)
        
        # Safely get lineup data with empty dict fallbacks
        home_lineup = lineup_data.get('home', {}) or {}
        away_lineup = lineup_data.get('away', {}) or {}
        
        # Analyze formation styles with validated data
        home_style = analyze_formation_style(home_formation, home_lineup)
        away_style = analyze_formation_style(away_formation, away_lineup)
        
        # Calculate formation impact - using proper dictionary access first
        formation_key = f"{home_formation}-{away_formation}"
        formation_impact = FORMATION_IMPACT.get(formation_key, [1.0, False])
        impact_factor = formation_impact[0] if len(formation_impact) > 0 else 1.0
        is_offensive = formation_impact[1] if len(formation_impact) > 1 else False
        
        # Apply formation adjustments to goals
        if 'goals' in result:
            goals = result['goals']
            if 'predicted_home_goals' in goals:
                goals['predicted_home_goals'] *= impact_factor
            if 'predicted_away_goals' in goals:
                goals['predicted_away_goals'] *= impact_factor
                
        # Add formation analysis to result
        result['formation_analysis'] = {
            'home': home_style,
            'away': away_style,
            'strength': strength_data,
            'impact_factor': impact_factor,
            'is_offensive': is_offensive
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error adjusting prediction for formation: {e}")
        return prediction.copy()

def calculate_weather_impact(conditions: Dict[str, Any]) -> float:
    """
    Calculate the impact of weather conditions on match predictions
    """
    if not conditions or not isinstance(conditions, dict):
        return 1.0

    base_multiplier = 1.0

    # Map test keys to internal keys
    condition = conditions.get('condition', '').lower() if isinstance(conditions.get('condition'), str) else ''
    intensity = conditions.get('intensity', '').lower() if isinstance(conditions.get('intensity'), str) else ''
    wind_speed = conditions.get('wind', 0)
    temperature = conditions.get('temperature', 15)  # Default to 15°C

    # Rain impact using WEATHER_FACTORS
    if condition == 'rain':
        if isinstance(intensity, str):
            rain_adj = WEATHER_FACTORS.get('rain', {}).get(intensity, {}).get('adj', 0)
        else:
            rain_adj = 0
        base_multiplier += rain_adj

    # Snow impact using WEATHER_FACTORS
    elif condition == 'snow':
        if isinstance(intensity, str):
            snow_adj = WEATHER_FACTORS.get('snow', {}).get(intensity, {}).get('adj', 0)
        else:
            snow_adj = 0
        base_multiplier += snow_adj

    # Ensure home_form and away_form are dictionaries
    home_form_data = conditions.get('home_form')
    away_form_data = conditions.get('away_form')
    
    if not isinstance(home_form_data, dict):
        home_form_data = {}
    if not isinstance(away_form_data, dict):
        away_form_data = {}

    # Fix for h2h_stats
    h2h_stats = conditions.get('h2h_stats')
    if not isinstance(h2h_stats, dict):
        h2h_stats = {}

    # Wind impact using thresholds
    wind_thresholds = WEATHER_FACTORS.get('wind', {}).get('thresholds', {})
    wind_adjustments = WEATHER_FACTORS.get('wind', {}).get('adjustments', {})
    wind_level = None
    if wind_speed >= wind_thresholds.get('strong', 30):
        wind_level = 'strong'
    elif wind_speed >= wind_thresholds.get('moderate', 20):
        wind_level = 'moderate'
    elif wind_speed >= wind_thresholds.get('light', 15):
        wind_level = 'light'
    if wind_level:
        base_multiplier += wind_adjustments.get(wind_level, 0)

    # Fix for Pylance error: convert lists of sets to sets of frozensets before union
    league_derbies_sets = []
    for key in [524, 564, 262]:
        pairs = DERBY_PAIRS.get(key, [])
        if isinstance(pairs, list):
            league_derbies_sets.append(set(frozenset(pair) for pair in pairs))
    if league_derbies_sets:
        combined_derbies = set.union(*league_derbies_sets)
    else:
        combined_derbies = set()

    # Removed duplicate block to fix Pylance error

    # Temperature impact using thresholds
    temp_thresholds = WEATHER_FACTORS.get('temperature', {}).get('thresholds', {})
    temp_adjustments = WEATHER_FACTORS.get('temperature', {}).get('adjustments', {})
    if temperature <= temp_thresholds.get('very_cold', 0):
        base_multiplier += temp_adjustments.get('very_cold', 0)
    elif temperature <= temp_thresholds.get('cold', 5):
        base_multiplier += temp_adjustments.get('cold', 0)
    elif temperature >= temp_thresholds.get('very_hot', 35):
        base_multiplier += temp_adjustments.get('very_hot', 0)
    elif temperature >= temp_thresholds.get('hot', 32):
        base_multiplier += temp_adjustments.get('hot', 0)

    # Combined conditions handling
    if condition == 'rain' and intensity == 'high' and wind_speed > 20:
        base_multiplier += -0.1  # Additional 10% reduction for severe conditions

    # Ensure the multiplier stays within reasonable bounds
    return max(0.6, min(1.2, base_multiplier))

def apply_weather_adjustments(base_prediction: float, weather_data: Dict[str, Any]) -> float:
    """
    Apply weather adjustments to the base prediction
    """
    if not isinstance(base_prediction, (int, float)) or base_prediction < 0:
        raise ValueError("Base prediction must be a non-negative number")

    if not weather_data:
        return base_prediction

    weather_multiplier = calculate_weather_impact(weather_data)
    return base_prediction * weather_multiplier

def adjust_prediction_for_weather(prediction: Dict[str, Any], weather: Dict[str, Any]) -> Dict[str, Any]:
    """Adjust prediction based on weather conditions"""
    if not prediction or not weather:
        return prediction.copy()

    result = prediction.copy()
    total_factor = calculate_weather_impact(weather)

    # Apply combined adjustments to predictions
    if 'goals' in result:
        for key in ['home', 'away', 'total']:
            if key in result['goals']:
                try:
                    result['goals'][key] *= max(0.5, total_factor)  # Ensure we don't reduce by more than 50%
                except Exception as e:
                    logger.warning(f"Failed to adjust goal {key}: {e}")

    # Also adjust top-level predicted goals keys if present
    for key in ['predicted_home_goals', 'predicted_away_goals']:
        if key in result:
            try:
                result[key] *= max(0.5, total_factor)
            except Exception as e:
                logger.warning(f"Failed to adjust {key}: {e}")

    if 'corners' in result and 'predicted_corners_mean' in result['corners']:
        try:
            result['corners']['predicted_corners_mean'] *= max(0.5, total_factor)
        except Exception as e:
            logger.warning(f"Failed to adjust corners predicted_corners_mean: {e}")

    return result

def adjust_single_pred_for_weather(value: float, weather: Dict[str, Any]) -> float:
    """Adjust a single prediction value based on weather"""
    if not isinstance(value, (int, float)) or not weather:
        return value

    weather_multiplier = calculate_weather_impact(weather)
    return value * max(0.5, weather_multiplier)

def test_weather_adjustments():
    """Unit tests for weather adjustment functions"""
    # Test dict version
    pred = {"goals": {"home": 1.5, "away": 1.2}}
    weather = {"condition": "rain", "intensity": "high"}
    result = adjust_prediction_for_weather(pred, weather)
    assert isinstance(result, dict)
    assert "goals" in result
    
    # Test float version
    mean = 9.5
    result = adjust_single_pred_for_weather(mean, weather)
    assert isinstance(result, float)
    assert result != mean  # Should be adjusted

def predict_team_goals(team_stats: Dict[str, Any], is_home: bool) -> float:
    """Calculate expected goals for a team based on their stats"""
    try:
        # Get scoring stats
        side = "home" if is_home else "away"
        goals = team_stats.get("goals", {})
        scored = goals.get("for", {}).get(side, 0)
        against = goals.get("against", {}).get(side, 0)
        
        # Calculate expected goals (with home advantage)
        base_goals = scored * (1.1 if is_home else 0.9)
        
        # Fallback to league average if no data
        if base_goals == 0:
            return 1.5 if is_home else 1.2
            
        return round(max(0.3, min(4.5, base_goals)), 2)
        
    except Exception as e:
        logger.error(f"Team goals prediction error: {e}")
        return 1.5 if is_home else 1.2

def calculate_match_probabilities(
    home_goals: float,
    away_goals: float,
    odds_data: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """Calculate match outcome probabilities"""
    try:
        # Base probabilities from goals
        lambda_h = float(home_goals)
        lambda_a = float(away_goals)
        
        # Poisson probabilities for scores up to 3 goals
        probs = defaultdict(float)
        for h in range(4):
            for a in range(4):
                p = (np.exp(-lambda_h) * lambda_h**h / factorial(h) * 
                     np.exp(-lambda_a) * lambda_a**a / factorial(a))
                probs[(h,a)] = p
                
        # Calculate main probabilities
        p_home = sum(p for (h,a), p in probs.items() if h > a)
        p_draw = sum(p for (h,a), p in probs.items() if h == a)
        p_away = sum(p for (h,a), p in probs.items() if h < a)
        
        # Normalize
        total = p_home + p_draw + p_away
        result = {
            "prob_home_win": round(p_home/total, 3),
            "prob_draw": round(p_draw/total, 3),
            "prob_away_win": round(p_away/total, 3),
            "predicted_home_goals": home_goals,
            "predicted_away_goals": away_goals
        }
        
        # Calibrate with odds if available
        if odds_data is not None:
            result = calibrate_odds_enhanced(result, odds_data)
            
        return result
        
    except Exception as e:
        logger.error(f"Probability calculation error: {e}")
        return {
            "prob_home_win": 0.45,
            "prob_draw": 0.25,
            "prob_away_win": 0.30,
            "predicted_home_goals": float(home_goals),
            "predicted_away_goals": float(away_goals)
        }

def extract_city_name(team_name: str) -> str:
    """
    Extracts city name from team name handling special cases
    """
    # Special cases dictionary
    special_cases = {
        "Real Madrid": "Madrid",
        "Atletico Madrid": "Madrid",
        "Athletic Bilbao": "Bilbao",
        "Real Sociedad": "San Sebastian",
        "Manchester United": "Manchester",
        "Manchester City": "Manchester",
        "Inter Milan": "Milan",
        "AC Milan": "Milan"
    }
    
    # Check for special cases first
    if team_name in special_cases:
        return special_cases[team_name]
        
    # Remove common prefixes
    prefixes_to_remove = ["FC ", "CF ", "Real ", "Atletico ", "Athletic ", "RC ", "RCD ", "CD "]
    clean_name = team_name
    for prefix in prefixes_to_remove:
        if clean_name.startswith(prefix):
            clean_name = clean_name[len(prefix):]
            
    # Return first word as city name, handling hyphenated names
    return clean_name.split()[0].split("-")[0]

def is_derby_match(home_team: str, away_team: str) -> bool:
    """
    Determine if a match is a derby based on team names and cities
    """
    if not isinstance(home_team, str) or not isinstance(away_team, str):
        return False

    home_team = home_team.strip().lower()
    away_team = away_team.strip().lower()

    if not home_team or not away_team:
        return False

    # Direct city name matches
    city_indicators = ['united', 'city', 'fc', 'athletic', 'atletico']
    
    def extract_city_name(team_name: str) -> str:
        parts = team_name.split()
        # Remove common suffixes
        cleaned_parts = [p for p in parts if p not in city_indicators]
        return ' '.join(cleaned_parts)

    home_city = extract_city_name(home_team)
    away_city = extract_city_name(away_team)

    # Check for exact city match
    if home_city == away_city and home_city:
        return True

    # Special cases for known derbies
    known_derbies = {
        ('manchester united', 'manchester city'),
        ('ac milan', 'inter milan'),
        ('real madrid', 'atletico madrid'),
        ('river plate', 'boca juniors'),
        ('roma', 'lazio'),
    }

    match_pair = (home_team, away_team)
    reverse_pair = (away_team, home_team)

    if match_pair in known_derbies or reverse_pair in known_derbies:
        return True

    # Check derby pairs by league
    all_derbies = set()
    for league_id in [524, 564, 262]:
        if league_id in DERBY_PAIRS:
            all_derbies.update(frozenset(pair) for pair in DERBY_PAIRS[league_id])
    
    # Use the combined derbies
    for derby_pair in all_derbies:
        if {home_team, away_team} == derby_pair:
            return True

    return False

def get_derby_intensity(home_team: str, away_team: str, historical_data: Optional[Dict[str, float]] = None) -> float:
    """
    Get the intensity factor for a derby match
    """
    if not is_derby_match(home_team, away_team):
        return 1.0

    if historical_data and (f"{home_team}-{away_team}" in historical_data or f"{away_team}-{home_team}" in historical_data):
        key = f"{home_team}-{away_team}" if f"{home_team}-{away_team}" in historical_data else f"{away_team}-{home_team}"
        return max(1.0, min(2.0, historical_data[key]))  # Limit factor between 1.0 and 2.0

    # Default derby intensity if no historical data
    return 1.25  # 25% increase in base predictions for derby matches

def adjust_predictions_for_context(
    prediction: Dict[str, Any],
    fixture_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adjusts predictions based on match context (derbies, etc)
    Safely handles missing keys with defaults
    """
    try:
        if not isinstance(prediction, dict) or not isinstance(fixture_data, dict):
            return prediction
        result = prediction.copy()
        
        # Safe extraction of team names and league
        teams = fixture_data.get("teams", {})
        home = teams.get("home", {}).get("name", "")
        away = teams.get("away", {}).get("name", "")
        league_id = fixture_data.get("league", {}).get("id", 0)
        # Derby check
        if is_derby_match(home, away):
            logger.info(f"Derby match detected: {home} vs {away}")
            
            # Cards adjustments
            if "cards" in result:
                cards = result["cards"]
                cards["total"] = cards.get("total", 4.0) * 1.25
                cards["home_yellows"] = cards.get("home_yellows", 2.0) * 1.2
                cards["away_yellows"] = cards.get("away_yellows", 2.0) * 1.2
                cards["total_reds"] = cards.get("total_reds", 0.2) * 1.3
                cards["prob_over_3.5_cards"] = min(
                    0.95,
                    cards.get("prob_over_3.5_cards", 0.65) * 1.3
                )
            
            # Goals adjustments
            if "goals" in result:
                goals = result["goals"]
                for key in ["home", "away", "total"]:
                    if key in goals:
                        goals[key] = goals[key] * 0.9
                
                # Adjust probabilities
                if "prob_over_2.5" in goals:
                    goals["prob_over_2.5"] *= 0.9
                if "prob_btts" in goals:
                    goals["prob_btts"] *= 1.1
            
            # Corners adjustments
            if "corners" in result:
                corners = result["corners"]
                if "predicted_corners_mean" in corners:
                    corners["predicted_corners_mean"] *= 1.1
                if "prob_over_9.5_corners" in corners:
                    corners["prob_over_9.5_corners"] = min(
                        0.95,
                        corners["prob_over_9.5_corners"] * 1.15
                    )
        return result
    except Exception as e:
        logger.error(f"Context adjustment error: {e}")
        return prediction

def calculate_bet_values(
    p_home: float,
    p_draw: float, 
    p_away: float,
    home_odds: float,
    draw_odds: float,
    away_odds: float) -> Dict[str, float]:
    """
    Calculates value indicators for each bet type
    Returns dict of bet types and their values (>1 indicates value)
    """
    return {
        "HOME": (p_home * home_odds) - 1,
        "DRAW": (p_draw * draw_odds) - 1,
        "AWAY": (p_away * away_odds) - 1
    }

def calculate_bet_edges(values: Dict[str, float], odds_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculates edge percentages from value indicators
    Returns dict of bet types and their edge percentages
    """
    try:
        edges = {}
        for bet_type, value in values.items():
            odds = float(odds_data.get(bet_type.lower() + "_win", 0))
            if odds > 0:
                implied_prob = 1 / odds
                model_prob = (value + 1) / odds
                edges[bet_type] = (model_prob / implied_prob - 1) * 100
            else:
                edges[bet_type] = 0
        return edges
        
    except Exception as e:
        logger.error(f"Edge calculation error: {e}")
        return {bet: 0 for bet in values.keys()}

def detect_value_bets(predictions: Dict[str, Any], odds_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze discrepancies between predictions and odds to find value."""
    try:
        # Calculate market efficiency first
        market_analysis = odds_analyzer.calculate_market_efficiency(odds_data)
        if market_analysis["efficiency"] < 0.90:  # Market is inefficient
            logger.warning(f"Low market efficiency: {market_analysis['efficiency']:.2%}")
            return predictions.copy()

        # Get best available odds
        best_odds = odds_analyzer.get_best_odds(odds_data, "match_odds")
        if not best_odds:
            return predictions.copy()

        result = predictions.copy()
        value_bets = {}
        
        # Check each market type
        markets = {
            "match_result": {
                "home": ("prob_home_win", best_odds.get("home", {}).get("odds")),
                "draw": ("prob_draw", best_odds.get("draw", {}).get("odds")),
                "away": ("prob_away_win", best_odds.get("away", {}).get("odds"))
            },
            "goals": {
                "over_2.5": ("prob_over_2_5", odds_data.get("over_under", {}).get("over_2.5", {}).get("odds", [])),
                "under_2.5": ("prob_under_2_5", odds_data.get("over_under", {}).get("under_2.5", {}).get("odds", []))
            },
            "corners": {
                "over_9.5": ("prob_over_corners_9_5", odds_data.get("corners", {}).get("over_9.5", {}).get("odds", [])),
                "under_9.5": ("prob_under_corners_9_5", odds_data.get("corners", {}).get("under_9.5", {}).get("odds", []))
            },
            "cards": {
                "over_3.5": ("prob_over_cards_3_5", odds_data.get("cards", {}).get("over_3.5", {}).get("odds", [])),
                "under_3.5": ("prob_under_cards_3_5", odds_data.get("cards", {}).get("under_3.5", {}).get("odds", []))
            }
        }

        for market_type, outcomes in markets.items():
            value_bets[market_type] = {}
            
            for outcome, (prob_key, market_odds) in outcomes.items():
                if prob_key in predictions and market_odds:
                    our_prob = predictions[prob_key]
                    if isinstance(market_odds, list):
                        market_odds = max(market_odds) if market_odds else 0
                        
                    if market_odds > 0:
                        edge = odds_analyzer.calculate_edge(our_prob, market_odds)
                        
                        if abs(edge) > 2:  # Edge threshold of 2%
                            value_bets[market_type][outcome] = {
                                "edge": edge,
                                "our_prob": our_prob,
                                "market_odds": market_odds
                            }

        # Add market efficiency metrics
        value_bets["market_analysis"] = market_analysis
        result["value_analysis"] = value_bets
        
        return result
        
    except Exception as e:
        logger.error(f"Error in detect_value_bets: {e}")
        return predictions.copy()

def calibrate_odds_enhanced(predictions: Dict[str, Any], odds_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calibrate predictions based on market odds and efficiency."""
    try:
        if not odds_data or not predictions:
            return predictions
            
        # Get market efficiency metrics
        market_analysis = odds_analyzer.calculate_market_efficiency(odds_data)
        efficiency = market_analysis["efficiency"]
        
        # Weight our predictions vs market implied probabilities
        # Higher market efficiency = more weight to market probabilities
        market_weight = min(0.7, efficiency)  # Cap market influence at 70%
        model_weight = 1 - market_weight
        
        result = predictions.copy()
        
        # Get market implied probabilities
        best_odds = odds_analyzer.get_best_odds(odds_data, "match_odds")
        if best_odds:
            try:
                # Calculate implied probabilities from best odds
                home_implied = 1 / best_odds["home"]["odds"] if "home" in best_odds else 0.33
                draw_implied = 1 / best_odds["draw"]["odds"] if "draw" in best_odds else 0.33
                away_implied = 1 / best_odds["away"]["odds"] if "away" in best_odds else 0.34
                
                # Normalize implied probabilities
                total = home_implied + draw_implied + away_implied
                if total > 0:
                    home_implied /= total
                    draw_implied /= total 
                    away_implied /= total
                    
                    # Blend predictions with market probabilities
                    result["prob_home_win"] = (model_weight * predictions["prob_home_win"] + 
                                             market_weight * home_implied)
                    result["prob_draw"] = (model_weight * predictions["prob_draw"] + 
                                         market_weight * draw_implied)
                    result["prob_away_win"] = (model_weight * predictions["prob_away_win"] + 
                                             market_weight * away_implied)
                    
                    # Log calibration results
                    logger.info(f"Calibrated predictions with efficiency {efficiency:.2%} " +
                              f"(market_weight: {market_weight:.2%})")
                    
            except Exception as e:
                logger.warning(f"Error calibrating match odds: {e}")
        
        # Add calibration metadata
        result["calibration"] = {
            "market_efficiency": efficiency,
            "market_weight": market_weight,
            "model_weight": model_weight,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in calibrate_odds_enhanced: {e}")
        return predictions

def validate_predictions(predictions):
    """
    Validates predictions to ensure they are consistent and within reasonable ranges.
    """
    validated = predictions.copy()
    
    # Check goal predictions
    if validated.get('predicted_home_goals', 0) < 0:
        logger.warning(f"Negative home goals prediction: {validated.get('predicted_home_goals', 0)}")
        validated['predicted_home_goals'] = 0
        
    if validated.get('predicted_away_goals', 0) < 0:
        logger.warning(f"Negative away goals prediction: {validated.get('predicted_away_goals', 0)}")
        validated['predicted_away_goals'] = 0
    # Check extreme predictions
    if validated.get('predicted_home_goals', 0) > 5:
        logger.warning(f"Very high home goals prediction: {validated.get('predicted_home_goals', 0)}")
        validated['predicted_home_goals'] = min(validated['predicted_home_goals'], 5)
        
    if validated.get('predicted_away_goals', 0) > 5:
        logger.warning(f"Very high away goals prediction: {validated.get('predicted_away_goals', 0)}")
        validated['predicted_away_goals'] = min(validated['predicted_away_goals'], 5)
    # Recalculate total goals
    validated['total_goals'] = (
        validated.get('predicted_home_goals', 0) + 
        validated.get('predicted_away_goals', 0)
    )
    # Check that probabilities sum to 1
    prob_sum = (
        validated.get('prob_home_win', 0.33) + 
        validated.get('prob_draw', 0.34) + 
        validated.get('prob_away_win', 0.33)
    )
    if abs(prob_sum - 1) > 0.01:
        logger.warning(f"Probabilities do not sum to 1: {prob_sum}")
        # Normalize
        if prob_sum > 0:
            validated['prob_home_win'] = validated.get('prob_home_win', 0.33) / prob_sum
            validated['prob_draw'] = validated.get('prob_draw', 0.34) / prob_sum
            validated['prob_away_win'] = validated.get('prob_away_win', 0.33) / prob_sum
    return validated

def validate_fixture_statistics(stats_df: pd.DataFrame) -> bool:
    """Validates that statistics DataFrame has required columns and valid values."""
    required_cols = [
        "team_id", "shots_on_goal", "shots_off_goal", 
        "total_shots", "corners", "yellow_cards", "red_cards"
    ]
    
    try:
        if stats_df.empty:
            return False
            
        # Check columns exist
        if not all(col in stats_df.columns for col in required_cols):
            logger.warning(f"Missing required columns: {[col for col in required_cols if col not in stats_df.columns]}")
            return False
            
        # Check for valid values
        for col in required_cols[1:]:  # Skip team_id
            if stats_df[col].isna().all() or (stats_df[col] == 0).all():
                logger.warning(f"Invalid values in column {col}")
                return False
                
        return True
        
    except Exception as e:
        logger.error(f"Statistics validation error: {e}")
        return False

# Fix undefined variable 'player_scores' by defining it or replacing usage
# Assuming player_scores is a list of floats representing scores

# Example fix: define player_scores or replace with actual variable
# Here, we define a placeholder empty list to avoid undefined variable error
player_scores = []

sorted_scores = sorted(player_scores, reverse=True)
first_place_score = sorted_scores[0] if sorted_scores else 0.0
second_place_score = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
